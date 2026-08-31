"""Image loading/saving helpers shared by the runner and the factory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str | Path) -> tuple[np.ndarray, float]:
    """Load an image as float32 grayscale in [0,1]; return (image, dpi)."""
    with Image.open(path) as im:
        dpi = float(im.info.get("dpi", (300, 300))[0]) or 300.0
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return gray, dpi


def save_image(path: str | Path, arr: np.ndarray) -> None:
    """Save a debug/state array as PNG.

    Accepts 2-D float [0,1] (grayscale), 2-D bool (ink=True rendered black),
    or (H, W, 3) uint8 (color).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype == bool:
        out = np.where(arr, 0, 255).astype(np.uint8)  # ink black on white
    elif arr.ndim == 2:
        out = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    elif arr.ndim == 3 and arr.dtype == np.uint8:
        out = arr
    else:
        raise TypeError(f"unsupported debug image: shape={arr.shape} dtype={arr.dtype}")
    Image.fromarray(out).save(path)
