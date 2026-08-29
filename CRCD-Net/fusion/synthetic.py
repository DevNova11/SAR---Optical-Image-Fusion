"""
Synthetic Data Generator for SAR (Sentinel-1) and Optical (Sentinel-2) Imagery.

Used during development and testing before Person 1 delivers real preprocessed satellite data.
No Google Earth Engine or external dataset downloading required.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def generate_synthetic_sar_optical_pair(
    height: int = 128,
    width: int = 128,
    sar_channels: int = 2,
    optical_channels: int = 4,
    output_channels: int = 4,
    data_layout: str = "CHW",
    noise_level: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """Generates synthetic Sentinel-1 (SAR) and Sentinel-2 (Optical) image pairs along with a

    synthetic reference image for evaluation testing.

    Parameters
    ----------
    height : int
        Spatial height of synthetic images.
    width : int
        Spatial width of synthetic images.
    sar_channels : int
        Number of SAR bands (e.g. 2 for VV, VH).
    optical_channels : int
        Number of Optical bands (e.g. 4 for Blue, Green, Red, NIR).
    output_channels : int
        Target output fused channels (e.g. 4).
    data_layout : str
        'CHW' (default) for (Channels, Height, Width) or 'HWC' for (Height, Width, Channels).
    noise_level : float
        Standard deviation of Gaussian/speckle noise added.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - "s1": SAR image array [float32, normalized 0..1]
        - "s2": Optical image array [float32, normalized 0..1]
        - "reference": Clean ground truth array [float32, normalized 0..1]
    """
    if seed is not None:
        np.random.seed(seed)

    # Grid coordinates normalized to [-1, 1]
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)

    # Base spatial structures (synthetic terrain/features)
    base_structure_1 = np.sin(3 * xx) * np.cos(3 * yy)
    base_structure_2 = np.exp(-(xx**2 + yy**2) / 0.5)
    base_structure_3 = 0.5 * (np.sin(6 * xx + 4 * yy) + 1.0)

    # Synthesize clean ground truth reference representation (multispectral)
    ref_bands = []
    for c in range(output_channels):
        freq = (c + 1) * 2.0
        band_signal = (
            np.sin(freq * xx + c) * np.cos(freq * yy)
            + 0.5 * base_structure_1
            + 0.3 * base_structure_2
        )
        band_signal = (band_signal - band_signal.min()) / (
            band_signal.max() - band_signal.min() + 1e-8
        )
        ref_bands.append(band_signal)

    reference = np.stack(ref_bands, axis=0).astype(np.float32)

    # Synthesize Sentinel-1 SAR (Speckled intensity/amplitude responses)
    sar_bands = []
    for c in range(sar_channels):
        # SAR texture dominated by structural edges and multiplicative noise
        texture = base_structure_1 + (c + 1) * 0.2 * base_structure_3
        # Speckle noise model (Rayleigh / Gamma-like noise)
        speckle = np.random.gamma(shape=2.0, scale=0.5, size=(height, width))
        sar_band = texture * speckle + noise_level * np.random.randn(height, width)
        sar_band = (sar_band - sar_band.min()) / (
            sar_band.max() - sar_band.min() + 1e-8
        )
        sar_bands.append(sar_band)

    s1_image = np.stack(sar_bands, axis=0).astype(np.float32)

    # Synthesize Sentinel-2 Optical (Smoother multispectral reflections)
    opt_bands = []
    for c in range(optical_channels):
        opt_band = (
            reference[c % output_channels]
            + 0.1 * base_structure_2
            + noise_level * np.random.randn(height, width)
        )
        opt_band = (opt_band - opt_band.min()) / (
            opt_band.max() - opt_band.min() + 1e-8
        )
        opt_bands.append(opt_band)

    s2_image = np.stack(opt_bands, axis=0).astype(np.float32)

    if data_layout.upper() == "HWC":
        s1_image = np.transpose(s1_image, (1, 2, 0))
        s2_image = np.transpose(s2_image, (1, 2, 0))
        reference = np.transpose(reference, (1, 2, 0))

    return {
        "s1": s1_image,
        "s2": s2_image,
        "reference": reference,
    }


def generate_two_date_synthetic_data(
    height: int = 128,
    width: int = 128,
    sar_channels: int = 2,
    optical_channels: int = 4,
    output_channels: int = 4,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generates synthetic SAR and Optical pairs for Date 1 and Date 2, simulating land cover change

    between dates for testing end-to-end Person 2 -> Person 3 pipeline.

    Returns
    -------
    Tuple[dict, dict]
        (date1_dict, date2_dict) where each contains "s1", "s2", "reference" arrays.
    """
    date1_data = generate_synthetic_sar_optical_pair(
        height=height,
        width=width,
        sar_channels=sar_channels,
        optical_channels=optical_channels,
        output_channels=output_channels,
        seed=seed,
    )

    # Date 2 with a simulated patch change (e.g. deforestation or urbanization in center)
    date2_data = generate_synthetic_sar_optical_pair(
        height=height,
        width=width,
        sar_channels=sar_channels,
        optical_channels=optical_channels,
        output_channels=output_channels,
        seed=seed + 100,
    )

    # Modify center patch in Date 2 to create a localized change pattern
    cy, cx = height // 2, width // 2
    r = min(height, width) // 6

    y, x = np.ogrid[:height, :width]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= r**2

    for c in range(sar_channels):
        date2_data["s1"][c, mask] = np.clip(
            date1_data["s1"][c, mask] * 1.8 + 0.1, 0.0, 1.0
        )
    for c in range(optical_channels):
        date2_data["s2"][c, mask] = np.clip(
            date1_data["s2"][c, mask] * 0.4, 0.0, 1.0
        )
    for c in range(output_channels):
        date2_data["reference"][c, mask] = np.clip(
            date1_data["reference"][c, mask] * 0.5, 0.0, 1.0
        )

    return date1_data, date2_data
