"""
HOG（Histogram of Oriented Gradients）特征提取。

提供三个接口：
- compute_hog(image_bgr, ...) -> 1D np.ndarray
- extract_hog_from_path(image_path, ...) -> 1D np.ndarray
- batch_extract_hog(image_paths, ...) -> 2D np.ndarray
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog as sk_hog


def compute_hog(image_bgr, image_size=224, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys"):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    feat = sk_hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_hog_from_path(image_path, image_size=224, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys"):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_hog(
        image_bgr,
        image_size=image_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
    )


def batch_extract_hog(image_paths, image_size=224, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys"):
    features = []
    for p in image_paths:
        try:
            feat = extract_hog_from_path(
                p,
                image_size=image_size,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm=block_norm,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        hog_dim = ((image_size // pixels_per_cell[0] - cells_per_block[0] + 1) * 
                   (image_size // pixels_per_cell[1] - cells_per_block[1] + 1) * 
                   cells_per_block[0] * cells_per_block[1] * orientations)
        return np.zeros((0, hog_dim), dtype=np.float32)
    # HOG 每张图维度一致，可直接堆叠
    return np.vstack(features).astype(np.float32)
