"""Self-supervised training of CRCDNet by mask-and-reconstruct.

There is no ground-truth "fused image". Instead: hide random regions of the
optical image (synthetic clouds), and train the network to rebuild the TRUE
optical pixels there from SAR + the remaining optical. The target is real
data, and the only way to do well in the hidden region is to use SAR.

Inputs to the network
    SAR      (B, 2, H, W)  [VV, VH]   z-scored with FIXED train-set statistics
    optical  (B, 9, H, W)  [B2,B3,B4,B8 z-scored, hidden/nodata -> 0] + validity-mask channel
                           + context-fill prior (Gaussian normalized-convolution of the VISIBLE optical,
                           z-scored). CRCDNet's receptive field is only ~13 px, far less than the size of a
                           cloud, so without this prior it never sees the surrounding optical level.
Output (B, 4, H, W) in [0, 1] = reflectance / S2_SCALE.

Normalization uses dataset-level constants saved with the weights (never
per-image statistics, which would make the same ground look different on two
dates and create fake change downstream).

Run from CRCD-Net/ :
    python -m fusion.train_fusion --iters 2000 --out fusion/weights/crcdnet_maskrec.pt
    python -m fusion.train_fusion --eval-only --out fusion/weights/crcdnet_maskrec.pt --split test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion import metrics as M  # noqa: E402
from local_preprocessing import NODATA, build_date_stack  # noqa: E402
from validate_export import DataValidationError  # noqa: E402

S2_SCALE = 0.5  # reflectance -> [0, 1] network target (sigmoid output); data_range for metrics
OPT_IN_CH = 9  # 4 optical bands (hidden->0) + validity-mask channel + 4-band context-fill prior
SPLITS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "splits.json")

SceneKey = Tuple[str, str]


# ---------------------------------------------------------------- data
def list_scenes(data_dir: str) -> Dict[str, List[str]]:
    """aoi_name -> sorted dates that have BOTH an S1 and an S2 GeoTIFF."""
    out: Dict[str, List[str]] = {}
    for p in glob.glob(os.path.join(data_dir, "*_S1.tif")):
        base = os.path.basename(p)[: -len("_S1.tif")]
        m = re.match(r"(.+)_(\d{4}-\d{2}-\d{2})$", base)
        if m and os.path.exists(p.replace("_S1.tif", "_S2.tif")):
            out.setdefault(m.group(1), []).append(m.group(2))
    return {k: sorted(v) for k, v in out.items()}


def load_splits(path: str = SPLITS_PATH) -> Dict[str, List[str]]:
    with open(path) as f:
        s = json.load(f)
    return {k: s[k] for k in ("train", "val", "test")}


def load_store(data_dir: str, aois: List[str]) -> Dict[SceneKey, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(aoi, date) -> (s1 (H,W,2), s2 (H,W,4), valid (H,W) bool). Goes through the validated loader."""
    avail = list_scenes(data_dir)
    store = {}
    for aoi in aois:
        for date in avail.get(aoi, []):
            s1_p = os.path.join(data_dir, f"{aoi}_{date}_S1.tif")
            s2_p = os.path.join(data_dir, f"{aoi}_{date}_S2.tif")
            try:
                s1, s2 = build_date_stack(s1_p, s2_p)
            except DataValidationError as exc:
                print(f"SKIP {aoi} {date}: {str(exc).split(': ', 1)[-1][:90]}")
                continue
            valid = (s1 != NODATA).all(-1) & (s2 != NODATA).all(-1) & np.isfinite(s1).all(-1) & np.isfinite(s2).all(-1)
            store[(aoi, date)] = (s1, s2, valid)
    return store


def compute_norm_stats(store) -> Dict[str, List[float]]:
    """Per-band mean/std over VALID train pixels only. Frozen into the checkpoint."""
    s1_all = np.concatenate([s1[v] for s1, _, v in store.values()])
    s2_all = np.concatenate([s2[v] for _, s2, v in store.values()])
    return {
        "s1_mean": s1_all.mean(0).tolist(), "s1_std": (s1_all.std(0) + 1e-6).tolist(),
        "s2_mean": s2_all.mean(0).tolist(), "s2_std": (s2_all.std(0) + 1e-6).tolist(),
    }


def normalize(s1, s2, valid, norm):
    s1n = (s1 - np.array(norm["s1_mean"], np.float32)) / np.array(norm["s1_std"], np.float32)
    s2n = (s2 - np.array(norm["s2_mean"], np.float32)) / np.array(norm["s2_std"], np.float32)
    s1n = np.where(valid[..., None], s1n, 0.0).astype(np.float32)
    s2n = np.where(valid[..., None], s2n, 0.0).astype(np.float32)
    return s1n, s2n


def random_hide_mask(h: int, w: int, rng: np.random.Generator, frac_lo=0.10, frac_hi=0.40) -> np.ndarray:
    """True = hidden. Union of random rectangles ("clouds") covering ~frac_lo..frac_hi of the patch."""
    target = rng.uniform(frac_lo, frac_hi) * h * w
    hide = np.zeros((h, w), bool)
    guard = 0
    while hide.sum() < target and guard < 200:
        rh = int(rng.integers(max(4, h // 8), max(5, h // 2)))
        rw = int(rng.integers(max(4, w // 8), max(5, w // 2)))
        y = int(rng.integers(0, h - rh + 1))
        x = int(rng.integers(0, w - rw + 1))
        hide[y : y + rh, x : x + rw] = True
        guard += 1
    return hide


def context_prior(s2_raw, vis, norm):
    """Z-scored Gaussian fill of the visible optical (no SAR). Identity-like where everything is visible."""
    fill = _gaussian_fill(s2_raw, vis).astype(np.float32)
    return (fill - np.array(norm["s2_mean"], np.float32)) / np.array(norm["s2_std"], np.float32)


def build_example(s1n, s2n, s2_raw, valid, hide, norm):
    """-> network inputs/targets as float32 (C,H,W) arrays + boolean loss masks."""
    vis = ~hide & valid
    opt_in = np.concatenate(
        [s2n * vis[..., None], vis[..., None].astype(np.float32), context_prior(s2_raw, vis, norm)], axis=-1
    )
    target = np.clip(s2_raw / S2_SCALE, 0.0, 1.0).astype(np.float32)
    chw = lambda a: np.ascontiguousarray(np.transpose(a, (2, 0, 1)), dtype=np.float32)
    return chw(s1n), chw(opt_in), chw(target), (hide & valid), (~hide & valid)


def make_batch(store, keys, norm, rng, patch: int, batch: int):
    import torch

    xs, os_, ts, hms, vms = [], [], [], [], []
    for _ in range(batch):
        for _try in range(10):
            key = keys[int(rng.integers(len(keys)))]
            s1, s2, valid = store[key]
            h, w = valid.shape
            if h < patch or w < patch:
                continue
            y, x = int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))
            v = valid[y : y + patch, x : x + patch]
            if v.mean() >= 0.9:
                break
        sl = (slice(y, y + patch), slice(x, x + patch))
        s1p, s2p, vp = s1[sl], s2[sl], v
        s1n, s2n = normalize(s1p, s2p, vp, norm)
        hide = random_hide_mask(patch, patch, rng)
        arrs = [s1n, s2n, s2p, vp, hide]
        k = int(rng.integers(4))  # rot90 + flip augmentation applied identically to everything
        flip = bool(rng.integers(2))
        arrs = [np.rot90(a, k) for a in arrs]
        if flip:
            arrs = [a[:, ::-1] for a in arrs]
        s1n, s2n, s2p, vp, hide = [np.ascontiguousarray(a) for a in arrs]
        x_s, x_o, t, hm, vm = build_example(s1n, s2n, s2p, vp, hide, norm)
        xs.append(x_s), os_.append(x_o), ts.append(t), hms.append(hm), vms.append(vm)
    T = lambda a: torch.from_numpy(np.stack(a))
    return T(xs), T(os_), T(ts), T(hms).float(), T(vms).float()


# ---------------------------------------------------------------- model / loss
def build_model(feature_dim: int = 64):
    from fusion.deep_model import CRCDNet

    return CRCDNet(sar_channels=2, optical_channels=OPT_IN_CH, output_channels=4, feature_dim=feature_dim)


def _masked_l1(out, tgt, m):
    return ((out - tgt).abs() * m[:, None]).sum() / (m.sum() * out.shape[1] + 1e-8)


def _grad(a):
    return a[..., :, 1:] - a[..., :, :-1], a[..., 1:, :] - a[..., :-1, :]


def fusion_loss(out, tgt, hidden_m, visible_m, w_vis=0.2, w_grad=0.1):
    l_hidden = _masked_l1(out, tgt, hidden_m)
    l_vis = _masked_l1(out, tgt, visible_m)
    (ox, oy), (tx, ty) = _grad(out), _grad(tgt)
    hx, hy = hidden_m[..., :, 1:], hidden_m[..., 1:, :]
    l_grad = _masked_l1(ox, tx, hx) + _masked_l1(oy, ty, hy)
    return l_hidden + w_vis * l_vis + w_grad * l_grad, {"hidden": float(l_hidden.detach()), "visible": float(l_vis.detach())}


# ---------------------------------------------------------------- evaluation
def _predict_scene(model, norm, s1, s2, valid, hide, device, zero_sar=False):
    import torch

    s1n, s2n = normalize(s1, s2, valid, norm)
    if zero_sar:
        s1n = np.zeros_like(s1n)
    x_s, x_o, _, _, _ = build_example(s1n, s2n, s2, valid, hide, norm)
    with torch.no_grad():
        out = model(torch.from_numpy(x_s)[None].to(device), torch.from_numpy(x_o)[None].to(device))
    return np.transpose(out[0].cpu().numpy(), (1, 2, 0)) * S2_SCALE  # reflectance


def _gaussian_fill(s2, vis, sigma=12.0):
    """Normalized-convolution inpainting from visible pixels only (a strong spatial baseline that uses NO SAR)."""
    from scipy.ndimage import gaussian_filter

    w = gaussian_filter(vis.astype(np.float64), sigma)
    out = np.empty_like(s2, dtype=np.float64)
    for c in range(s2.shape[-1]):
        out[..., c] = gaussian_filter(np.where(vis, s2[..., c], 0.0), sigma) / np.maximum(w, 1e-6)
    gmean = s2[vis].mean(0) if vis.any() else np.zeros(s2.shape[-1])
    return np.where(w[..., None] > 1e-3, out, gmean)


def evaluate(model, store, norm, device="cpu", seed=1234, frac=(0.25, 0.25)) -> Dict[str, Dict[str, float]]:
    """Deterministic hidden-region reconstruction on whole held-out scenes. Returns mean metrics per method."""
    model.eval()
    methods: Dict[str, List[Dict[str, float]]] = {
        "trained_crcdnet": [], "trained_no_sar": [], "gaussian_fill_no_sar": [], "mean_fill": [],
    }
    for i, key in enumerate(sorted(store)):
        s1, s2, valid = store[key]
        rng = np.random.default_rng(seed + i)
        hide = random_hide_mask(*valid.shape, rng, *frac)
        hm = hide & valid
        if hm.sum() < 100:
            continue
        vis = ~hide & valid
        tgt = np.clip(s2, 0, S2_SCALE)
        preds = {
            "trained_crcdnet": _predict_scene(model, norm, s1, s2, valid, hide, device),
            "trained_no_sar": _predict_scene(model, norm, s1, s2, valid, hide, device, zero_sar=True),
            "gaussian_fill_no_sar": _gaussian_fill(s2, vis),
            "mean_fill": np.broadcast_to(s2[vis].mean(0), s2.shape),
        }
        for name, p in preds.items():
            methods[name].append(M.all_metrics(np.clip(p, 0, S2_SCALE), tgt, hm, data_range=S2_SCALE))
    return {n: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for n, rs in methods.items() if rs}


def print_table(res):
    print(f"{'method':24s} {'PSNR':>7s} {'SSIM':>7s} {'RMSE':>8s} {'SAM(deg)':>9s}")
    for n, r in res.items():
        print(f"{n:24s} {r['psnr']:7.2f} {r['ssim']:7.3f} {r['rmse']:8.4f} {r['sam_deg']:9.2f}")


# ---------------------------------------------------------------- train
def save_checkpoint(path, model, norm, cfg, meta):
    import torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "norm": norm, "config": cfg, "meta": meta}, path)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "fusion", "weights", "crcdnet_maskrec.pt"))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4, help="evals without val improvement before stopping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="split used by --eval-only")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    splits = load_splits()

    if a.eval_only:
        ck = torch.load(a.out, map_location=a.device)
        model = build_model(ck["config"]["feature_dim"]).to(a.device)
        model.load_state_dict(ck["state_dict"])
        store = load_store(a.data_dir, splits[a.split])
        print(f"eval on {a.split}: {sorted({k[0] for k in store})}  ({len(store)} scenes)")
        print_table(evaluate(model, store, ck["norm"], a.device))
        return

    train_store = load_store(a.data_dir, splits["train"])
    val_store = load_store(a.data_dir, splits["val"])
    norm = compute_norm_stats(train_store)
    keys = sorted(train_store)
    print(f"train scenes={len(train_store)} val scenes={len(val_store)} device={a.device}")

    model = build_model().to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
    best, bad, t0 = -1e9, 0, time.time()
    cfg = {"feature_dim": 64, "opt_in_ch": OPT_IN_CH, "s2_scale": S2_SCALE, "patch": a.patch, "context_prior": True}

    for it in range(1, a.iters + 1):
        model.train()
        xs, xo, tg, hm, vm = [t.to(a.device) for t in make_batch(train_store, keys, norm, rng, a.patch, a.batch)]
        loss, parts = fusion_loss(model(xs, xo), tg, hm, vm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 25 == 0:
            print(f"it {it:5d} loss={loss.item():.4f} hidden_l1={parts['hidden']:.4f} ({time.time() - t0:.0f}s)")
        if it % a.eval_every == 0 or it == a.iters:
            res = evaluate(model, val_store, norm, a.device)
            print("--- val (hidden-region reconstruction) ---")
            print_table(res)
            score = res["trained_crcdnet"]["psnr"]
            if score > best:
                best, bad = score, 0
                save_checkpoint(a.out, model, norm, cfg, {"iter": it, "val_psnr": score, "seed": a.seed})
                print(f"saved best (val PSNR {score:.2f}) -> {a.out}")
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop")
                    break
    print(f"done in {time.time() - t0:.0f}s, best val PSNR {best:.2f}")


if __name__ == "__main__":
    main()
