"""
猫狗分类 - 边缘特征提取

本模块实现基于图像边缘的特征提取方法。
边缘特征能够捕获图像中的轮廓和形状信息，是区分不同物体的重要特征。

提取的特征包括:
    1. 梯度方向直方图（基于Sobel算子）
       - 统计不同方向梯度的分布
       - 使用梯度幅值加权，强调显著边缘
       - 方向范围：[0°, 180°)，无符号方向
    
    2. 梯度幅值直方图
       - 统计边缘强度的分布
       - 反映图像中边缘的整体强度特性
    
    3. Canny边缘密度
       - 使用Canny算子检测边缘
       - 计算边缘像素占总像素的比例
       - 反映图像中边缘的丰富程度

特征处理:
    - 每个部分独立进行L2归一化
    - 最终特征 = [方向直方图(归一化) + 幅值直方图(归一化) + 边缘密度]
    - 特征维度 = num_orientation_bins + num_magnitude_bins + 1

技术说明:
    - 使用Sobel算子计算梯度（dx, dy）
    - 方向直方图使用幅值加权，突出显著边缘
    - 归一化提高特征的鲁棒性和稳定性
"""
from pathlib import Path

import cv2
import numpy as np


def _safe_l2_normalize(vector):
    """安全的L2归一化，避免除零错误
    
    参数:
        vector: 待归一化的向量
        
    返回:
        L2归一化后的向量（单位长度）
    """
    norm = np.linalg.norm(vector) + 1e-12  # 添加小常数避免除零
    return vector / norm


def compute_edge_histogram(
    image_bgr,
    image_size=224,
    num_orientation_bins=9,
    num_magnitude_bins=32,
    canny_threshold1=100,
    canny_threshold2=200,
):
    """从图像计算完整的边缘特征向量
    
    该函数提取三类边缘相关特征并拼接成一个特征向量。
    
    处理流程:
        1. 图像预处理：转换为灰度图并调整大小
        2. 梯度计算：使用Sobel算子计算x和y方向梯度
        3. 方向直方图：统计梯度方向分布（幅值加权）
        4. 幅值直方图：统计梯度强度分布
        5. Canny边缘检测：计算边缘像素密度
        6. 特征拼接：组合所有特征并归一化
    
    参数:
        image_bgr: BGR格式的输入图像
        image_size: 处理前将图像缩放到的尺寸（正方形）
        num_orientation_bins: 方向直方图的bin数量（推荐9）
        num_magnitude_bins: 幅值直方图的bin数量（推荐32）
        canny_threshold1: Canny算子的低阈值
        canny_threshold2: Canny算子的高阈值
        
    返回:
        一维特征向量，长度为 num_orientation_bins + num_magnitude_bins + 1
        包含：[方向直方图(9维) + 幅值直方图(32维) + 边缘密度(1维)]
    """
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)

    sobel_dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobel_dx, sobel_dy)
    angle = cv2.phase(sobel_dx, sobel_dy, angleInDegrees=True)  # 角度范围为 [0, 360)
    angle = angle % 180.0  # 取无符号方向，[0, 180)

    # 使用幅值加权的方向直方图
    orientation_hist = np.zeros((num_orientation_bins,), dtype=np.float32)
    bin_width = 180.0 / num_orientation_bins
    bin_indices = np.floor(angle / bin_width).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, num_orientation_bins - 1)
    for b in range(num_orientation_bins):
        orientation_hist[b] = magnitude[bin_indices == b].sum()
    orientation_hist = _safe_l2_normalize(orientation_hist)

    # 幅值直方图
    max_mag = float(magnitude.max())
    if max_mag <= 1e-6:
        mag_hist = np.zeros((num_magnitude_bins,), dtype=np.float32)
    else:
        mag_hist, _ = np.histogram(
            magnitude.flatten(),
            bins=num_magnitude_bins,
            range=(0.0, max_mag),
        )
        mag_hist = mag_hist.astype(np.float32)
    mag_hist = _safe_l2_normalize(mag_hist)

    # Canny 边缘密度
    edges = cv2.Canny(img_gray, threshold1=canny_threshold1, threshold2=canny_threshold2)
    edge_density = float((edges > 0).mean())  # 边缘像素占比

    feature = np.concatenate([orientation_hist, mag_hist, np.array([edge_density], dtype=np.float32)], axis=0)
    return feature


def extract_edge_hist_from_path(
    image_path,
    image_size=224,
    num_orientation_bins=9,
    num_magnitude_bins=32,
    canny_threshold1=100,
    canny_threshold2=200,
):
    """从图像路径提取边缘特征
    
    参数:
        image_path: 图像文件路径
        image_size: 目标图像尺寸
        num_orientation_bins: 方向直方图bin数
        num_magnitude_bins: 幅值直方图bin数
        canny_threshold1: Canny低阈值
        canny_threshold2: Canny高阈值
        
    返回:
        边缘特征向量（一维numpy数组）
        
    异常:
        FileNotFoundError: 如果无法读取图像文件
    """
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return compute_edge_histogram(
        image_bgr,
        image_size=image_size,
        num_orientation_bins=num_orientation_bins,
        num_magnitude_bins=num_magnitude_bins,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
    )


def batch_extract_edge_hist(
    image_paths,
    image_size=224,
    num_orientation_bins=9,
    num_magnitude_bins=32,
    canny_threshold1=100,
    canny_threshold2=200,
):
    """批量提取多张图像的边缘特征
    
    对图像路径列表中的每张图像提取边缘特征，
    自动跳过无法读取的图像。
    
    参数:
        image_paths: 图像文件路径列表
        image_size: 目标图像尺寸
        num_orientation_bins: 方向直方图bin数
        num_magnitude_bins: 幅值直方图bin数
        canny_threshold1: Canny低阈值
        canny_threshold2: Canny高阈值
        
    返回:
        特征矩阵，形状为 (成功处理的图像数, feature_dim)
        feature_dim = num_orientation_bins + num_magnitude_bins + 1
        
    注意:
        如果所有图像都无法处理，返回形状为 (0, feature_dim) 的空矩阵
    """
    features = []
    for p in image_paths:
        try:
            feat = extract_edge_hist_from_path(
                p,
                image_size=image_size,
                num_orientation_bins=num_orientation_bins,
                num_magnitude_bins=num_magnitude_bins,
                canny_threshold1=canny_threshold1,
                canny_threshold2=canny_threshold2,
            )
            features.append(feat)
        except Exception:
            continue
    if not features:
        return np.zeros((0, num_orientation_bins + num_magnitude_bins + 1), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
