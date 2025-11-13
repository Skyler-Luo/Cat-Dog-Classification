"""
角点与边缘密度特征。

组成：
- Harris 角点密度（角点数 / 像素数）；
- Shi-Tomasi (goodFeaturesToTrack) 角点密度；
- Canny 边缘密度（已有 edge_hist 中包含，此处独立提供更轻量统计）；
- Sobel 梯度幅值均值与标准差；
- Laplacian 绝对响应均值与标准差（细节强度）。

输出：固定长度 7 维向量。
"""

from pathlib import Path

import cv2
import numpy as np


def compute_corner_edge_density(
    image_bgr,
    image_size=224,
    harris_block_size=2,
    harris_ksize=3,
    harris_k=0.04,
    shi_max_corners=500,
    shi_quality_level=0.01,
    shi_min_distance=5,
    canny_threshold1=100,
    canny_threshold2=200,
):
    # 统一尺寸
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)

    h, w = img_gray.shape[:2]
    num_pixels = float(h * w)

    # Harris 角点
    harris = cv2.cornerHarris(img_gray, blockSize=harris_block_size, ksize=harris_ksize, k=harris_k)
    # 以相对阈值计数
    harris_norm = cv2.normalize(harris, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    harris_count = int((harris_norm > 0.1).sum())
    harris_density = np.array([harris_count / max(num_pixels, 1.0)], dtype=np.float32)

    # Shi-Tomasi 角点
    shi_corners = cv2.goodFeaturesToTrack(
        img_gray,
        maxCorners=int(shi_max_corners),
        qualityLevel=float(shi_quality_level),
        minDistance=float(shi_min_distance),
    )
    shi_count = 0 if shi_corners is None else int(len(shi_corners))
    shi_density = np.array([shi_count / max(num_pixels, 1.0)], dtype=np.float32)

    # Canny 边缘密度
    edges = cv2.Canny(img_gray, threshold1=canny_threshold1, threshold2=canny_threshold2)
    edge_density = np.array([(edges > 0).mean()], dtype=np.float32)

    # Sobel 梯度幅值统计
    sobel_dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_dx, sobel_dy)
    grad_mean = np.array([float(magnitude.mean())], dtype=np.float32)
    grad_std = np.array([float(magnitude.std())], dtype=np.float32)

    # Laplacian 细节响应统计
    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_mean = np.array([float(lap_abs.mean())], dtype=np.float32)
    lap_std = np.array([float(lap_abs.std())], dtype=np.float32)

    feature = np.concatenate([
        harris_density,
        shi_density,
        edge_density,
        grad_mean,
        grad_std,
        lap_mean,
        lap_std,
    ], axis=0)
    return feature.astype(np.float32)


def extract_corner_edge_density_from_path(
    image_path,
    image_size=224,
    harris_block_size=2,
    harris_ksize=3,
    harris_k=0.04,
    shi_max_corners=500,
    shi_quality_level=0.01,
    shi_min_distance=5,
    canny_threshold1=100,
    canny_threshold2=200,
):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_corner_edge_density(
        image_bgr,
        image_size=image_size,
        harris_block_size=harris_block_size,
        harris_ksize=harris_ksize,
        harris_k=harris_k,
        shi_max_corners=shi_max_corners,
        shi_quality_level=shi_quality_level,
        shi_min_distance=shi_min_distance,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
    )


def batch_extract_corner_edge_density(
    image_paths,
    image_size=224,
    harris_block_size=2,
    harris_ksize=3,
    harris_k=0.04,
    shi_max_corners=500,
    shi_quality_level=0.01,
    shi_min_distance=5,
    canny_threshold1=100,
    canny_threshold2=200,
):
    features = []
    feature_dim = 7
    for p in image_paths:
        try:
            feat = extract_corner_edge_density_from_path(
                p,
                image_size=image_size,
                harris_block_size=harris_block_size,
                harris_ksize=harris_ksize,
                harris_k=harris_k,
                shi_max_corners=shi_max_corners,
                shi_quality_level=shi_quality_level,
                shi_min_distance=shi_min_distance,
                canny_threshold1=canny_threshold1,
                canny_threshold2=canny_threshold2,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
