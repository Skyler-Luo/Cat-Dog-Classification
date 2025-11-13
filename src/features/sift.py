"""
SIFT 统计特征提取。

思路：
- 使用 OpenCV SIFT 提取关键点与描述子；
- 为保持维度固定且适合传统分类器，统计以下直方图/聚合量：
  1) 关键点尺度（sigma）直方图；
  2) 关键点响应值直方图；
  3) 关键点方向（[0, 360)）直方图；
  4) 关键点密度（数量/像素数，标量）；
  5) 描述子分量的全局均值与标准差（128 维各 2 个统计，共 256 维）。

输出：一维 np.ndarray。
"""

from pathlib import Path

import cv2
import numpy as np


def _safe_l2_normalize(vector):
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def _build_sift(image_bgr, n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma):
    # OpenCV SIFT 需要 contrib 版本
    sift = cv2.SIFT_create(
        nfeatures=int(n_features),
        nOctaveLayers=int(n_octave_layers),
        contrastThreshold=float(contrast_threshold),
        edgeThreshold=float(edge_threshold),
        sigma=float(sigma),
    )
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints or [], descriptors


def compute_sift_stats(
    image_bgr,
    image_size=224,
    n_features=0,
    n_octave_layers=3,
    contrast_threshold=0.04,
    edge_threshold=10,
    sigma=1.6,
    num_scale_bins=8,
    num_response_bins=8,
    num_angle_bins=18,
):
    """从图像计算 SIFT 统计特征。

    参数说明
    ----
    - n_features=0 表示不限制数量（OpenCV 默认）。
    - 直方图分箱可按需调整；会做 L2 归一化。
    - 对描述子 (N,128)，计算每维均值与标准差，若 N=0 则置零。
    """
    # 统一缩放尺寸，保证密度尺度可比
    image_bgr = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    keypoints, descriptors = _build_sift(
        image_bgr,
        n_features=n_features,
        n_octave_layers=n_octave_layers,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
    )

    h, w = image_bgr.shape[:2]
    num_pixels = float(h * w)

    if len(keypoints) == 0:
        scale_hist = np.zeros((num_scale_bins,), dtype=np.float32)
        response_hist = np.zeros((num_response_bins,), dtype=np.float32)
        angle_hist = np.zeros((num_angle_bins,), dtype=np.float32)
        kp_density = np.array([0.0], dtype=np.float32)
        desc_mean = np.zeros((128,), dtype=np.float32)
        desc_std = np.zeros((128,), dtype=np.float32)
    else:
        sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
        angles = np.array([(kp.angle if kp.angle >= 0 else 0.0) for kp in keypoints], dtype=np.float32)
        # 角度范围 [0, 360)
        angles = angles % 360.0

        # 尺度/响应采用自适应范围
        s_min, s_max = float(sizes.min()), float(sizes.max())
        r_min, r_max = float(responses.min()), float(responses.max())
        if not np.isfinite(s_min):
            s_min, s_max = 0.0, 1.0
        if not np.isfinite(r_min):
            r_min, r_max = 0.0, 1.0
        if abs(s_max - s_min) < 1e-6:
            s_min, s_max = 0.0, s_min + 1.0
        if abs(r_max - r_min) < 1e-12:
            r_min, r_max = 0.0, r_min + 1.0

        scale_hist, _ = np.histogram(sizes, bins=num_scale_bins, range=(s_min, s_max))
        response_hist, _ = np.histogram(responses, bins=num_response_bins, range=(r_min, r_max))
        angle_hist, _ = np.histogram(angles, bins=num_angle_bins, range=(0.0, 360.0))

        scale_hist = _safe_l2_normalize(scale_hist.astype(np.float32))
        response_hist = _safe_l2_normalize(response_hist.astype(np.float32))
        angle_hist = _safe_l2_normalize(angle_hist.astype(np.float32))

        kp_density = np.array([len(keypoints) / max(num_pixels, 1.0)], dtype=np.float32)

        if descriptors is None or len(descriptors) == 0:
            desc_mean = np.zeros((128,), dtype=np.float32)
            desc_std = np.zeros((128,), dtype=np.float32)
        else:
            # OpenCV SIFT 描述子为 float32 (N,128)
            descriptors = descriptors.astype(np.float32)
            desc_mean = descriptors.mean(axis=0)
            desc_std = descriptors.std(axis=0)

    feature = np.concatenate([
        scale_hist,
        response_hist,
        angle_hist,
        kp_density,
        desc_mean,
        desc_std,
    ], axis=0).astype(np.float32)
    return feature


def extract_sift_from_path(
    image_path,
    image_size=224,
    n_features=0,
    n_octave_layers=3,
    contrast_threshold=0.04,
    edge_threshold=10,
    sigma=1.6,
    num_scale_bins=8,
    num_response_bins=8,
    num_angle_bins=18,
):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_sift_stats(
        image_bgr,
        image_size=image_size,
        n_features=n_features,
        n_octave_layers=n_octave_layers,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
        num_scale_bins=num_scale_bins,
        num_response_bins=num_response_bins,
        num_angle_bins=num_angle_bins,
    )


def batch_extract_sift(
    image_paths,
    image_size=224,
    n_features=0,
    n_octave_layers=3,
    contrast_threshold=0.04,
    edge_threshold=10,
    sigma=1.6,
    num_scale_bins=8,
    num_response_bins=8,
    num_angle_bins=18,
):
    features = []
    feature_dim = num_scale_bins + num_response_bins + num_angle_bins + 1 + 128 * 2
    for p in image_paths:
        try:
            feat = extract_sift_from_path(
                p,
                image_size=image_size,
                n_features=n_features,
                n_octave_layers=n_octave_layers,
                contrast_threshold=contrast_threshold,
                edge_threshold=edge_threshold,
                sigma=sigma,
                num_scale_bins=num_scale_bins,
                num_response_bins=num_response_bins,
                num_angle_bins=num_angle_bins,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
