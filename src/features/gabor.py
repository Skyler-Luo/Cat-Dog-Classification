"""
Gabor 滤波器组纹理特征。

对多尺度多方向的 Gabor 核进行卷积，提取响应的统计量（均值、方差）。
"""

from pathlib import Path

import cv2
import numpy as np


def _build_gabor_kernels(ksizes, sigmas, thetas, lambdas, gammas, psis):
    kernels = []
    for k in ksizes:
        for sigma in sigmas:
            for theta in thetas:
                for lambd in lambdas:
                    for gamma in gammas:
                        for psi in psis:
                            kernel = cv2.getGaborKernel((k, k), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                            kernels.append(kernel)
    return kernels


def compute_gabor_features(
    image_bgr,
    image_size=224,
    ksizes=(15,),
    sigmas=(4.0,),
    thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    lambdas=(10.0, 20.0),
    gammas=(0.5, 0.8),
    psis=(0, np.pi / 2),
):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)

    kernels = _build_gabor_kernels(ksizes, sigmas, thetas, lambdas, gammas, psis)
    feats = []
    for ker in kernels:
        resp = cv2.filter2D(img_gray, cv2.CV_32F, ker)
        feats.append(resp.mean())
        feats.append(resp.std())
    if not feats:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def extract_gabor_from_path(
    image_path,
    image_size=224,
    ksizes=(15,),
    sigmas=(4.0,),
    thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    lambdas=(10.0, 20.0),
    gammas=(0.5, 0.8),
    psis=(0, np.pi / 2),
):
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_gabor_features(
        image_bgr,
        image_size=image_size,
        ksizes=ksizes,
        sigmas=sigmas,
        thetas=thetas,
        lambdas=lambdas,
        gammas=gammas,
        psis=psis,
    )


def batch_extract_gabor(
    image_paths,
    image_size=224,
    ksizes=(15,),
    sigmas=(4.0,),
    thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    lambdas=(10.0, 20.0),
    gammas=(0.5, 0.8),
    psis=(0, np.pi / 2),
):
    features = []
    # 维度固定：每个 kernel 2 维（均值、标准差）
    num_kernels = len(ksizes) * len(sigmas) * len(thetas) * len(lambdas) * len(gammas) * len(psis)
    feat_dim = num_kernels * 2
    for p in image_paths:
        try:
            feat = extract_gabor_from_path(
                p,
                image_size=image_size,
                ksizes=ksizes,
                sigmas=sigmas,
                thetas=thetas,
                lambdas=lambdas,
                gammas=gammas,
                psis=psis,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, feat_dim), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
