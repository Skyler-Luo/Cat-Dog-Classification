"""
GLCM（灰度共生矩阵）/ Haralick 纹理特征。

输出常用统计：contrast, dissimilarity, homogeneity, ASM, energy, correlation
按多距离、多角度计算并取均值拼接。
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def compute_glcm_features(
    image_bgr,
    image_size=224,
    distances=(1, 2, 4),
    angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels=256,
    symmetric=True,
    normed=True,
):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # skimage 期望灰度值在 [0, levels-1] 的整数
    img_gray = np.clip(img_gray, 0, levels - 1).astype(np.uint8)

    glcm = graycomatrix(img_gray, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    props = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "ASM",
        "energy",
        "correlation",
    ]
    feat_list = []
    for prop in props:
        vals = graycoprops(glcm, prop)  # shape: (len(distances), len(angles))
        feat_list.append(vals.mean(axis=(0, 1)))
    feature = np.asarray(feat_list, dtype=np.float32).reshape(-1)
    return feature


def extract_glcm_from_path(
    image_path,
    image_size=224,
    distances=(1, 2, 4),
    angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels=256,
    symmetric=True,
    normed=True,
):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_glcm_features(
        image_bgr,
        image_size=image_size,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )


def batch_extract_glcm(
    image_paths,
    image_size=224,
    distances=(1, 2, 4),
    angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels=256,
    symmetric=True,
    normed=True,
):
    features = []
    for p in image_paths:
        try:
            feat = extract_glcm_from_path(
                p,
                image_size=image_size,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=symmetric,
                normed=normed,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, 6), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
