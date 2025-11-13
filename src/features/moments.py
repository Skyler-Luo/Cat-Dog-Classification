"""
颜色矩（Color Moments）与 Hu 不变矩（形状）特征。

color_moments: 每个通道的 mean/std/skew，共 9 维（BGR 或 LAB/HSV 等）
hu_moments: 7 维，对灰度或边缘图计算
"""

from pathlib import Path

import cv2
import numpy as np


def _compute_color_moments(image_cs):
    # 每通道：均值、标准差、偏度
    chans = cv2.split(image_cs)
    feats = []
    for c in chans:
        c = c.astype(np.float32)
        mean = float(c.mean())
        std = float(c.std() + 1e-12)
        skew = float(((c - mean) ** 3).mean() / (std ** 3 + 1e-12))
        feats.extend([mean, std, skew])
    return np.asarray(feats, dtype=np.float32)


def _to_color_space(image_bgr, color_space):
    if color_space is None or color_space.upper() == "BGR":
        return image_bgr
    if color_space.upper() == "RGB":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if color_space.upper() == "HSV":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if color_space.upper() == "LAB":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    raise ValueError(f"Unsupported color_space: {color_space}")


def compute_color_moments(image_bgr, image_size=224, color_space="HSV"):
    img = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img_cs = _to_color_space(img, color_space)
    return _compute_color_moments(img_cs)


def compute_hu_moments(image_bgr, image_size=224, use_edges=False, canny_threshold1=100, canny_threshold2=200):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if use_edges:
        edges = cv2.Canny(img_gray, threshold1=canny_threshold1, threshold2=canny_threshold2)
        src = edges
    else:
        _, src = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    m = cv2.moments(src)
    hu = cv2.HuMoments(m).reshape(-1).astype(np.float32)
    # log 变换可提升数值稳定性
    hu = np.sign(hu) * np.log(np.abs(hu) + 1e-12)
    return hu


def extract_color_moments_from_path(image_path, image_size=224, color_space="HSV"):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_color_moments(image_bgr, image_size=image_size, color_space=color_space)


def extract_hu_moments_from_path(image_path, image_size=224, use_edges=False, canny_threshold1=100, canny_threshold2=200):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_hu_moments(
        image_bgr,
        image_size=image_size,
        use_edges=use_edges,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
    )


def batch_extract_color_moments(image_paths, image_size=224, color_space="HSV"):
    features = []
    for p in image_paths:
        try:
            feat = extract_color_moments_from_path(p, image_size=image_size, color_space=color_space)
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, 9), dtype=np.float32)
    return np.vstack(features).astype(np.float32)


def batch_extract_hu_moments(image_paths, image_size=224, use_edges=False, canny_threshold1=100, canny_threshold2=200):
    features = []
    for p in image_paths:
        try:
            feat = extract_hu_moments_from_path(
                p,
                image_size=image_size,
                use_edges=use_edges,
                canny_threshold1=canny_threshold1,
                canny_threshold2=canny_threshold2,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, 7), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
