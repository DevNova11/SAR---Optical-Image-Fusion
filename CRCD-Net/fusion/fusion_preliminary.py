"""CRCD-Net preliminary SAR + optical deep-fusion training prototype.

Quick CPU smoke test (synthetic data; validates code, NOT scientific performance):
    python fusion_preliminary.py --smoke-test --epochs 2

Train on aligned patch arrays saved as NPZ with keys:
    s1: (N, H, W, 2), Sentinel-1 [VV, VH] in dB
    s2: (N, H, W, 4), Sentinel-2 [B2, B3, B4, B8] reflectance in [0, 1]
    python fusion_preliminary.py --data data/fusion_patches.npz --epochs 10

NODATA sentinel -9999 is masked from the reconstruction loss.
This is a preliminary trainer; use AOI-level train/validation/test splits before reporting results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

NODATA = -9999.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SpatialChannelGating(nn.Module):
    """Learns channel and pixel-wise weights for SAR and optical features."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, sar_features: torch.Tensor, optical_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([sar_features, optical_features], dim=1)
        channel_weights = self.channel_gate(combined)
        sar_c, opt_c = torch.split(channel_weights, sar_features.shape[1], dim=1)
        spatial_weights = self.spatial_gate(combined)
        sar_s, opt_s = spatial_weights[:, 0:1], spatial_weights[:, 1:2]
        # Normalize modality weights at each pixel to sum to one.
        sar_w = sar_c.mean(dim=1, keepdim=True) * sar_s
        opt_w = opt_c.mean(dim=1, keepdim=True) * opt_s
        denom = sar_w + opt_w + 1e-6
        sar_w, opt_w = sar_w / denom, opt_w / denom
        return sar_features * sar_w + optical_features * opt_w


class CRCDNet(nn.Module):
    """Two-branch CNN: 2 SAR channels + 4 optical channels -> 4-band output."""
    def __init__(self, sar_channels: int = 2, optical_channels: int = 4,
                 output_channels: int = 4, feature_dim: int = 32):
        super().__init__()
        def encoder(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim), nn.LeakyReLU(0.2, inplace=True),
            )
        self.sar_encoder = encoder(sar_channels)
        self.opt_encoder = encoder(optical_channels)
        self.gating_fusion = SpatialChannelGating(feature_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim // 2, output_channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, s1: torch.Tensor, s2_masked: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.gating_fusion(self.sar_encoder(s1), self.opt_encoder(s2_masked)))


class PatchDataset(Dataset):
    def __init__(self, s1: np.ndarray, s2: np.ndarray, sar_mean: np.ndarray,
                 sar_std: np.ndarray):
        # Convert HWC arrays to NCHW tensors; NODATA must not enter the network.
        s1_valid = np.isfinite(s1) & (s1 != NODATA)
        s2_valid = np.isfinite(s2) & (s2 != NODATA)
        pixel_valid = s1_valid.all(axis=-1) & s2_valid.all(axis=-1)
        s1_clean = np.where(s1_valid, s1, sar_mean.reshape(1, 1, 1, 2)).astype(np.float32)
        s1_norm = (s1_clean - sar_mean.reshape(1, 1, 1, 2)) / sar_std.reshape(1, 1, 1, 2)
        s2_clean = np.where(s2_valid, s2, 0.0).astype(np.float32)
        s2_clean = np.clip(s2_clean, 0.0, 1.0)
        self.s1 = torch.from_numpy(np.transpose(s1_norm, (0, 3, 1, 2)).copy())
        self.s2 = torch.from_numpy(np.transpose(s2_clean, (0, 3, 1, 2)).copy())
        self.valid = torch.from_numpy(pixel_valid[:, None, :, :].astype(np.float32))

    def __len__(self) -> int:
        return self.s1.shape[0]

    def __getitem__(self, index: int):
        return self.s1[index], self.s2[index], self.valid[index]


def make_masked_optical(optical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blank random rectangles covering about 10-40% of each patch; return input and mask."""
    batch, _, height, width = optical.shape
    hidden = torch.zeros((batch, 1, height, width), device=optical.device)
    masked = optical.clone()
    for i in range(batch):
        fraction = random.uniform(0.10, 0.40)
        target_area = int(height * width * fraction)
        rect_h = max(1, min(height, int(np.sqrt(target_area))))
        rect_w = max(1, min(width, target_area // rect_h))
        top = random.randint(0, max(0, height - rect_h))
        left = random.randint(0, max(0, width - rect_w))
        hidden[i, :, top:top + rect_h, left:left + rect_w] = 1.0
        # Occasionally blank one optical band over the same region.
        masked[i, :, top:top + rect_h, left:left + rect_w] = 0.0
        if random.random() < 0.25:
            band = random.randrange(optical.shape[1])
            masked[i, band, top:top + rect_h, left:left + rect_w] = 0.0
    return masked, hidden


def gradient_loss(prediction: torch.Tensor, sar: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Light structural regularizer: compare image gradients, not SAR/optical intensities."""
    pred_gray = prediction.mean(dim=1, keepdim=True)
    sar_gray = sar[:, :1]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=prediction.device).view(1, 1, 3, 3) / 8
    ky = kx.transpose(2, 3)
    px, py = nn.functional.conv2d(pred_gray, kx, padding=1), nn.functional.conv2d(pred_gray, ky, padding=1)
    sx, sy = nn.functional.conv2d(sar_gray, kx, padding=1), nn.functional.conv2d(sar_gray, ky, padding=1)
    # Standardized SAR gradients are only a weak structural cue; small coefficient is used.
    valid = valid_mask.expand_as(px)
    return (((px - sx).abs() + (py - sy).abs()) * valid).sum() / (valid.sum() + 1e-6)


def synthetic_data(count: int = 8, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic arrays solely to smoke-test forward/backward training code."""
    rng = np.random.default_rng(123)
    yy, xx = np.mgrid[0:size, 0:size]
    base = (0.25 + 0.25 * (xx / max(size - 1, 1)) + 0.10 * np.sin(yy / 5.0)).astype(np.float32)
    s2 = np.stack([base, np.clip(base + 0.05, 0, 1), np.clip(base - 0.03, 0, 1),
                   np.clip(base + 0.12 * np.cos(xx / 7.0), 0, 1)], axis=-1)
    s2 = np.stack([np.clip(s2 + rng.normal(0, 0.015, s2.shape), 0, 1) for _ in range(count)]).astype(np.float32)
    s1 = np.stack([np.stack([-14 + 4 * base + rng.normal(0, 1, base.shape),
                              -19 + 3 * base + rng.normal(0, 1, base.shape)], axis=-1)
                   for _ in range(count)]).astype(np.float32)
    return s1, s2


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "s1" not in archive or "s2" not in archive:
            raise ValueError("NPZ must contain arrays named 's1' and 's2'.")
        s1, s2 = archive["s1"], archive["s2"]
    if s1.ndim != 4 or s2.ndim != 4:
        raise ValueError(f"Expected s1/s2 as (N,H,W,C); received {s1.shape} and {s2.shape}.")
    if s1.shape[:3] != s2.shape[:3] or s1.shape[-1] != 2 or s2.shape[-1] != 4:
        raise ValueError("Expected aligned s1=(N,H,W,2) and s2=(N,H,W,4) arrays.")
    return s1.astype(np.float32), s2.astype(np.float32)


def train(args: argparse.Namespace) -> None:
    if args.smoke_test:
        print("SMOKE TEST: using synthetic data only; this does not train a scientific model.")
        s1, s2 = synthetic_data(count=8, size=args.patch_size)
    else:
        if not args.data:
            raise SystemExit("Provide --data path/to/patches.npz or use --smoke-test.")
        s1, s2 = load_arrays(Path(args.data))

    valid_s1 = np.isfinite(s1) & (s1 != NODATA)
    means, stds = [], []
    for channel in range(2):
        vals = s1[..., channel][valid_s1[..., channel]]
        if vals.size == 0:
            raise ValueError(f"SAR channel {channel} has no valid pixels.")
        means.append(float(vals.mean()))
        stds.append(max(float(vals.std()), 1e-6))
    sar_mean, sar_std = np.asarray(means, np.float32), np.asarray(stds, np.float32)

    dataset = PatchDataset(s1, s2, sar_mean, sar_std)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                        num_workers=0, drop_last=False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = CRCDNet(feature_dim=args.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Device: {device} | patches: {len(dataset)} | patch size: {s1.shape[1]}x{s1.shape[2]}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, batches = 0.0, 0
        for sar, optical, valid in loader:
            sar, optical, valid = sar.to(device), optical.to(device), valid.to(device)
            masked_optical, hidden = make_masked_optical(optical)
            prediction = model(sar, masked_optical)
            loss_mask = hidden * valid
            # L1 reconstruction only where optical information was hidden and data are valid.
            recon = ((prediction - optical).abs() * loss_mask).sum() / (loss_mask.sum() * optical.shape[1] + 1e-6)
            # Use a small edge term only on valid pixels; keep reconstruction as the main objective.
            edge = gradient_loss(prediction, sar, valid)
            loss = recon + 0.02 * edge
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={epoch_loss / max(batches, 1):.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "crcdnet_preliminary.pt"
    torch.save({"model_state_dict": model.cpu().state_dict(),
                "model_config": {"sar_channels": 2, "optical_channels": 4,
                                 "output_channels": 4, "feature_dim": args.feature_dim}}, checkpoint)
    stats_path = output_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps({"sar_mean": sar_mean.tolist(), "sar_std": sar_std.tolist(),
                                      "optical_range": [0.0, 1.0], "nodata": NODATA}, indent=2))
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved normalization stats: {stats_path}")

    # Check inference output shape and finite values before exiting.
    model.eval()
    with torch.no_grad():
        sample_sar, sample_opt, _ = dataset[0]
        output = model(sample_sar.unsqueeze(0).to(device), sample_opt.unsqueeze(0).to(device))
    print(f"Inference output shape (B,C,H,W): {tuple(output.shape)}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}] | all finite: {bool(torch.isfinite(output).all())}")
    print("Prototype run completed. Use real aligned Sentinel patches and AOI-level evaluation for research results.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preliminary CRCD-Net deep-fusion trainer")
    parser.add_argument("--data", type=str, default=None, help="NPZ with s1 and s2 arrays")
    parser.add_argument("--smoke-test", action="store_true", help="run a code test with synthetic data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64, help="synthetic smoke-test patch size")
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/deep_fusion_preliminary")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    seed_everything(args.seed)
    return args


if __name__ == "__main__":
    train(parse_args())
