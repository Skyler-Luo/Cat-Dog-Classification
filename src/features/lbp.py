"""
LBP（Local Binary Pattern）纹理特征。

实现：
- compute_lbp_hist(image_bgr, P, R, method) -> L2 归一化直方图
- extract_lbp_from_path(image_path, ...) -> 1D np.ndarray
- batch_extract_lbp(image_paths, ...) -> 2D np.ndarray
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def _safe_l2_normalize(vector):
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def compute_lbp_hist(image_bgr, image_size=224, P=8, R=1, method="uniform", num_bins=None):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(img_gray, P=P, R=R, method=method)
    if num_bins is None:
        num_bins = P + 2 if method == "uniform" else int(2 ** P)
    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    hist = hist.astype(np.float32)
    hist = _safe_l2_normalize(hist)
    return hist


def extract_lbp_from_path(image_path, image_size=224, P=8, R=1, method="uniform", num_bins=None):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_lbp_hist(
        image_bgr,
        image_size=image_size,
        P=P,
        R=R,
        method=method,
        num_bins=num_bins,
    )


def batch_extract_lbp(image_paths, image_size=224, P=8, R=1, method="uniform", num_bins=None):
    features = []
    if num_bins is None:
        num_bins = P + 2 if method == "uniform" else int(2 ** P)
    for p in image_paths:
        try:
            feat = extract_lbp_from_path(
                p,
                image_size=image_size,
                P=P,
                R=R,
                method=method,
                num_bins=num_bins,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, num_bins), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
