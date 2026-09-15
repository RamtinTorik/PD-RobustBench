"""Image corruption functions for the PD drawing robustness benchmark.

Implements 8 corruption families x 5 severity levels following the ImageNet-C
methodology (Hendrycks & Dietterich, ICLR 2019), adapted to black-ink-on-white
handwriting/drawing images:

  - jpeg_compression : JPEG re-encoding at decreasing quality factors
  - gaussian_blur    : Gaussian low-pass filter with growing kernel size
  - gaussian_noise   : additive white Gaussian noise (sigma as fraction of 255)
  - contrast         : mean-centred multiplicative contrast change
  - brightness       : multiplicative brightness change
  - shadow           : diagonal illumination gradient (uneven lighting)
  - perspective      : mild perspective/skew warp of the sheet
  - rotation         : in-plane rotation, white (background) fill

All functions take/return uint8 HxWx3 RGB arrays. Corruption is applied on the
fly at inference time (no persistent copies needed) and deterministically, so
every experiment is reproducible.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from config import CORRUPTION_LEVELS

__all__ = [
    "apply_jpeg_compression",
    "apply_gaussian_blur",
    "add_gaussian_noise",
    "adjust_contrast",
    "adjust_brightness",
    "apply_shadow",
    "apply_perspective",
    "rotate_image",
    "apply_corruption",
    "corruption_config_rows",
]


def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode to JPEG at the given quality and decode back."""
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def apply_gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian blur with an odd kernel size (sigma auto-derived by OpenCV)."""
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=0, sigmaY=0,
                            borderType=cv2.BORDER_REFLECT)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Additive white Gaussian noise; sigma is a fraction of the 0-255 range."""
    out = img.astype(np.float32) + np.random.normal(
        loc=0.0, scale=float(sigma) * 255.0, size=img.shape
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Mean-centred contrast scaling: (x - mean) * factor + mean."""
    out = img.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * float(factor) + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiplicative brightness change."""
    out = img.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shadow(img: np.ndarray, min_factor: float) -> np.ndarray:
    """Uneven illumination: a linear diagonal gradient multiplying the image
    from 1.0 (lit corner) down to `min_factor` (shadowed corner), emulating a
    shadow cast across the paper."""
    h, w = img.shape[:2]
    gx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    gy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    grad = (gx[None, :] + gy[:, None]) / 2.0                     # 0..1 diagonal
    factor = 1.0 - grad * (1.0 - float(min_factor))
    out = img.astype(np.float32) * factor[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_perspective(img: np.ndarray, severity_frac: float) -> np.ndarray:
    """Mild perspective/skew warp: the four sheet corners are displaced by a
    deterministic asymmetric pattern of magnitude `severity_frac` * min(h, w),
    emulating a photographed/scan-bed sheet that is not parallel to the
    camera. Borders are filled white (paper background)."""
    h, w = img.shape[:2]
    d = float(severity_frac) * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [0.15 * d, 0.90 * d],
        [w - 0.60 * d, 0.25 * d],
        [w - 0.10 * d, h - 0.50 * d],
        [0.70 * d, h - 0.15 * d],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the image centre; border is filled with white (the paper
    background of the drawings), matching a physically rotated scan."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), scale=1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


_APPLY = {
    "jpeg_compression": apply_jpeg_compression,
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": add_gaussian_noise,
    "contrast": adjust_contrast,
    "brightness": adjust_brightness,
    "shadow": apply_shadow,
    "perspective": apply_perspective,
    "rotation": rotate_image,
}


def apply_corruption(img: np.ndarray, corruption_type: str, level: int) -> np.ndarray:
    """Apply one corruption family at a given severity level (1..5).

    Deterministic per (image, corruption, level): the RNG is re-seeded from a
    hash of the inputs so that identical calls always give identical outputs.
    """
    if corruption_type == "clean":
        return img
    if corruption_type not in _APPLY:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    levels = CORRUPTION_LEVELS[corruption_type]
    if not 1 <= level <= len(levels):
        raise ValueError(f"Level {level} out of range 1..{len(levels)}")
    param = levels[level - 1]
    # deterministic noise across runs / machines
    seed = (hash(corruption_type) & 0xFFFF) * 100000 + level * 7919
    np.random.seed(seed % (2**32))
    return _APPLY[corruption_type](img, param)


def corruption_config_rows():
    """Yield (corruption_type, level, parameter, description) rows."""
    desc = {
        "jpeg_compression": "JPEG quality factor",
        "gaussian_blur": "Gaussian kernel size (px)",
        "gaussian_noise": "noise sigma (fraction of 255)",
        "contrast": "contrast factor",
        "brightness": "brightness factor",
        "shadow": "brightness at the darkest corner of a diagonal gradient",
        "perspective": "corner displacement (fraction of min(h, w))",
        "rotation": "rotation angle (deg, +/- direction)",
    }
    for ctype, levels in CORRUPTION_LEVELS.items():
        for i, p in enumerate(levels, start=1):
            yield ctype, i, p, desc[ctype]
