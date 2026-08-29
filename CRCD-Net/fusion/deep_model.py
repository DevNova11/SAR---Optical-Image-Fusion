"""
Deep Fusion Model for SAR (Sentinel-1) and Optical (Sentinel-2) Imagery.

Implements a lightweight two-branch PyTorch Convolutional Neural Network with
Spatial-Channel Gating Attention mechanism for adaptive feature fusion.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelGating(nn.Module):
    """Spatial and Channel Attention Gating Module.

    Dynamically computes importance weights for SAR and Optical feature maps.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Channel Attention (Squeeze-and-Excitation)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial Attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f_sar: torch.Tensor, f_opt: torch.Tensor) -> torch.Tensor:
        """Forward pass fusing SAR and Optical features adaptively.

        Parameters
        ----------
        f_sar : torch.Tensor
            SAR feature tensor of shape (B, C, H, W).
        f_opt : torch.Tensor
            Optical feature tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Fused feature map of shape (B, C, H, W).
        """
        concat_feat = torch.cat([f_sar, f_opt], dim=1)  # (B, 2C, H, W)

        # Channel weights
        channel_weights = self.channel_gate(concat_feat)  # (B, 2C, 1, 1)
        w_sar_c, w_opt_c = torch.split(channel_weights, self.channels, dim=1)

        # Spatial weights
        spatial_weights = self.spatial_gate(concat_feat)  # (B, 2, H, W)
        w_sar_s = spatial_weights[:, 0:1, :, :]
        w_opt_s = spatial_weights[:, 1:2, :, :]

        # Combine channel and spatial gating weights
        gated_sar = f_sar * w_sar_c * w_sar_s
        gated_opt = f_opt * w_opt_c * w_opt_s

        # Adaptive fused representation
        fused_features = gated_sar + gated_opt
        return fused_features


class CRCDNet(nn.Module):
    """CRCD-Net Two-Branch Deep SAR-Optical Fusion Architecture.

    Parameters
    ----------
    sar_channels : int
        Number of input channels for Sentinel-1 (e.g. 2).
    optical_channels : int
        Number of input channels for Sentinel-2 (e.g. 4).
    output_channels : int
        Number of output channels in fused representation (e.g. 4).
    feature_dim : int
        Internal feature map dimensions for branch encoders (default: 64).
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 4,
        output_channels: int = 4,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.output_channels = output_channels
        self.feature_dim = feature_dim

        # SAR Branch Encoder
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Optical Branch Encoder
        self.opt_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Attention Gating Fusion
        self.gating_fusion = SpatialChannelGating(channels=feature_dim)

        # Reconstruction Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim // 2, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Standardized output range [0, 1]
        )

    def forward(self, x_sar: torch.Tensor, x_opt: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_sar : torch.Tensor
            SAR tensor of shape (B, C_sar, H, W).
        x_opt : torch.Tensor
            Optical tensor of shape (B, C_opt, H, W).

        Returns
        -------
        torch.Tensor
            Fused image tensor of shape (B, C_out, H, W).
        """
        f_sar = self.sar_encoder(x_sar)
        f_opt = self.opt_encoder(x_opt)
        f_fused = self.gating_fusion(f_sar, f_opt)
        out = self.decoder(f_fused)
        return out


def fuse_deep(
    s1_image: np.ndarray,
    s2_image: np.ndarray,
    model: Optional[CRCDNet] = None,
    device: str = "cpu",
    data_layout: str = "CHW",
) -> np.ndarray:
    """Inference wrapper for deep PyTorch model SAR-Optical image fusion.

    Accepts NumPy arrays, converts to PyTorch tensors, runs forward pass,
    and returns fused output array matching input layout.
    """
    from fusion.baseline import validate_and_format_inputs

    s1_formatted, s2_formatted, was_hwc = validate_and_format_inputs(
        s1_image, s2_image, data_layout=data_layout
    )

    c1, h, w = s1_formatted.shape
    c2, _, _ = s2_formatted.shape

    if model is None:
        model = CRCDNet(
            sar_channels=c1,
            optical_channels=c2,
            output_channels=c2,
        )

    model = model.to(device)
    model.eval()

    # Add batch dimension: (1, C, H, W)
    t_sar = torch.from_numpy(s1_formatted).unsqueeze(0).to(device)
    t_opt = torch.from_numpy(s2_formatted).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(t_sar, t_opt)

    # Remove batch dimension and convert to NumPy float32
    fused_chw = out_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    if was_hwc or data_layout.upper() == "HWC":
        fused_out = np.transpose(fused_chw, (1, 2, 0))
    else:
        fused_out = fused_chw

    return fused_out


def train_fusion_model(
    model: CRCDNet,
    s1_data: np.ndarray,
    s2_data: np.ndarray,
    target_reference: np.ndarray,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[CRCDNet, List[float]]:
    """Simple training loop for fine-tuning or training the deep fusion model on paired synthetic

    or real reference data.

    Parameters
    ----------
    model : CRCDNet
        PyTorch fusion model instance.
    s1_data : np.ndarray
        SAR training batch array (B, C1, H, W) or single (C1, H, W).
    s2_data : np.ndarray
        Optical training batch array (B, C2, H, W) or single (C2, H, W).
    target_reference : np.ndarray
        Target reference array (B, C_out, H, W) or single (C_out, H, W).
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam optimizer.
    device : str
        Target compute device ('cpu' or 'cuda').

    Returns
    -------
    Tuple[CRCDNet, List[float]]
        (trained_model, loss_history)
    """
    if s1_data.ndim == 3:
        s1_data = s1_data[np.newaxis, ...]
        s2_data = s2_data[np.newaxis, ...]
        target_reference = target_reference[np.newaxis, ...]

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t_sar = torch.from_numpy(s1_data).to(device)
    t_opt = torch.from_numpy(s2_data).to(device)
    t_target = torch.from_numpy(target_reference).to(device)

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(t_sar, t_opt)
        loss = criterion(out, t_target)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

    return model, loss_history
