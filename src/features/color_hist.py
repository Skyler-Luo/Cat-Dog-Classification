"""
猫狗分类 - 颜色直方图特征提取

本模块实现基于颜色直方图的图像特征提取方法。
颜色直方图是一种简单而有效的图像表示方法，通过统计不同颜色在图像中出现的频率来描述图像的颜色分布特征。

主要功能:
    - 支持多种颜色空间（HSV, LAB, RGB, BGR）
    - 3D颜色直方图计算
    - L2归一化确保特征尺度一致
    - 批量提取接口

颜色空间选择:
    - HSV: 更符合人类视觉感知，对光照变化较为稳健，推荐用于一般场景
    - LAB: 感知均匀色彩空间，适合需要精确颜色表示的场景
    - RGB/BGR: 简单直接，但对光照变化敏感

技术说明:
    - OpenCV默认以BGR顺序读取图像
    - 直方图使用L2归一化，使不同尺寸图像的特征可比较
    - 特征维度 = hist_size[0] × hist_size[1] × hist_size[2]
"""

from pathlib import Path

import cv2
import numpy as np


def _convert_color_space(image_bgr, color_space):
    """将 BGR 图像转换到指定颜色空间
    
    参数:
        image_bgr: 输入图像，BGR 格式（cv2.imread 默认格式）
        color_space: 颜色空间名称，可选值 {"HSV", "LAB", "RGB"}，为 None 或 "BGR" 时不做转换
        
    返回:
        转换后的图像数组
    """
    if color_space is None or color_space.upper() == "BGR":
        return image_bgr
    if color_space.upper() == "RGB":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if color_space.upper() == "HSV":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if color_space.upper() == "LAB":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    raise ValueError(f"Unsupported color_space: {color_space}")


def compute_color_histogram(image_bgr, color_space="HSV", hist_size=(8, 8, 8)):
    """计算 3D 颜色直方图并返回展平后的 L2 归一化向量
    
    参数:
        image_bgr: 输入图像，BGR 格式
        color_space: 计算直方图所用颜色空间，可选值 {"HSV", "LAB", "RGB", "BGR"}
        hist_size: 每个通道的直方图分箱数，元组格式 (int, int, int)
        
    返回:
        展平且 L2 归一化后的直方图数组，长度为 hist_size[0]*hist_size[1]*hist_size[2]
    """
    image_cs = _convert_color_space(image_bgr, color_space)

    # 直方图取值范围取决于颜色空间
    if color_space and color_space.upper() == "HSV":
        ranges = [0, 180, 0, 256, 0, 256]  # OpenCV HSV: H[0,180)、S[0,255]、V[0,255]
        channels = [0, 1, 2]
    else:
        # 对于 BGR、RGB、LAB，三个通道范围均为 0-255
        ranges = [0, 256, 0, 256, 0, 256]
        channels = [0, 1, 2]

    hist = cv2.calcHist([image_cs], channels, None, hist_size, ranges)
    hist = hist.astype(np.float32).flatten()

    # 使用 L2 归一化到单位长度
    norm = np.linalg.norm(hist) + 1e-12
    hist /= norm
    return hist


def extract_color_hist_from_path(image_path, image_size=224, color_space="HSV", hist_size=(8, 8, 8)):
    """读取图像、按指定尺寸缩放，并计算颜色直方图特征
    
    参数:
        image_path: 图像文件路径，支持字符串或 Path 对象
        image_size: 目标边长，缩放为正方形 (image_size x image_size)
        color_space: 用于计算直方图的颜色空间
        hist_size: 每个通道的分箱数，元组格式 (int, int, int)
        
    返回:
        颜色直方图特征向量
        
    异常:
        FileNotFoundError: 如果无法读取图像文件
    """
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_bgr = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return compute_color_histogram(image_bgr, color_space=color_space, hist_size=hist_size)


def batch_extract_color_hist(image_paths, image_size=224, color_space="HSV", hist_size=(8, 8, 8)):
    """批量为一组图像路径提取颜色直方图特征
    
    参数:
        image_paths: 图像文件路径列表
        image_size: 目标图像尺寸
        color_space: 颜色空间名称
        hist_size: 每个通道的分箱数
        
    返回:
        特征矩阵，形状为 (成功处理的图像数, feature_dim)
        
    注意:
        自动跳过无法读取的图像文件
    """
    features = []
    for p in image_paths:
        try:
            feat = extract_color_hist_from_path(p, image_size=image_size, color_space=color_space, hist_size=hist_size)
            features.append(feat)
        except Exception:
            # 跳过无法读取的图像
            continue
    if not features:
        return np.zeros((0, hist_size[0] * hist_size[1] * hist_size[2]), dtype=np.float32)
    return np.vstack(features).astype(np.float32)
