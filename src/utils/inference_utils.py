"""推理脚本通用工具函数

该模块提供推理脚本中常用的工具函数，包括：
- 训练配置加载
- 图像路径收集
- 预测结果保存
- 特征提取辅助函数
"""

import json
import csv
from pathlib import Path


def load_training_config(weights_path):
    """加载训练时保存的配置文件
    
    参数:
        weights_path: 模型权重路径（str 或 Path）
        
    返回:
        dict 或 None: 训练配置字典，如果文件不存在或解析失败则返回 None
        
    说明:
        该函数会在模型权重同目录下查找 'training_results.json' 文件，
        并从中提取配置信息。
        
    示例:
        >>> config = load_training_config('runs/sklearn_svm/best.joblib')
        >>> if config:
        ...     print(config['svm']['C'])
    """
    try:
        cfg_path = Path(weights_path).parent / "training_results.json"
        
        if not cfg_path.exists():
            return None
        
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 兼容不同的配置格式
        return data.get("config") or data
        
    except Exception as e:
        print(f"⚠️  无法加载训练配置: {e}")
        return None


def gather_image_paths(path, extensions=None, recursive=True, verbose=True):
    """收集待推理的图片路径
    
    参数:
        path: 文件或目录路径（str 或 Path）
        extensions: 支持的扩展名集合（set 或 None）
                   默认: {'.jpg', '.jpeg', '.png', '.bmp'}
        recursive: 是否递归搜索子目录（bool，默认: True）
        verbose: 是否打印收集信息（bool，默认: True）
        
    返回:
        list[str]: 图片路径列表（已排序）
        
    说明:
        - 如果 path 是文件，检查扩展名并返回单元素列表
        - 如果 path 是目录，搜索所有符合条件的图片
        - 支持递归和非递归两种搜索模式
        
    示例:
        >>> # 收集单个文件
        >>> paths = gather_image_paths('test.jpg')
        ['test.jpg']
        
        >>> # 收集目录下所有图片
        >>> paths = gather_image_paths('dataset/test')
        📸 收集图片: dataset/test
           • 找到 1000 张图片
        
        >>> # 仅收集 jpg 和 png
        >>> paths = gather_image_paths('dataset', extensions={'.jpg', '.png'})
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # 确保扩展名为小写
    extensions = {ext.lower() for ext in extensions}
    
    p = Path(path)
    
    # 处理单个文件
    if p.is_file():
        if p.suffix.lower() in extensions:
            if verbose:
                print(f"📸 单个文件: {p}")
            return [str(p)]
        else:
            print(f"⚠️  不支持的文件扩展名: {p.suffix}（支持: {extensions}）")
            return []
    
    # 处理目录
    if p.is_dir():
        if verbose:
            print(f"📸 收集图片: {p}")
        
        paths = []
        
        if recursive:
            # 递归搜索
            for ext in extensions:
                paths.extend([str(q) for q in p.rglob(f"*{ext}")])
        else:
            # 仅搜索当前目录
            for ext in extensions:
                paths.extend([str(q) for q in p.glob(f"*{ext}")])
        
        paths = sorted(paths)
        
        if verbose:
            print(f"   • 找到 {len(paths)} 张图片")
        
        return paths
    
    # 路径不存在
    print(f"❌ 路径不存在: {path}")
    return []


def save_predictions_to_csv(results, csv_path, verbose=True):
    """保存预测结果到 CSV 文件
    
    参数:
        results: 预测结果列表，每项为 (image_path, prediction, confidence) 元组
        csv_path: CSV 文件保存路径（str 或 Path）
        verbose: 是否打印保存信息（bool，默认: True）
        
    说明:
        CSV 文件格式：
        - 第一行：列标题（Image, Prediction, Confidence）
        - 后续行：图像路径、预测类别、置信度
        
    示例:
        >>> results = [
        ...     ('img1.jpg', 'cat', 0.95),
        ...     ('img2.jpg', 'dog', 0.88),
        ... ]
        >>> save_predictions_to_csv(results, 'predictions.csv')
        💾 预测结果已保存至: predictions.csv
           • 共 2 条记录
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['Image', 'Prediction', 'Confidence'])
        
        # 写入数据
        for img_path, pred, conf in results:
            writer.writerow([img_path, pred, f'{conf:.4f}'])
    
    if verbose:
        print(f"💾 预测结果已保存至: {csv_path}")
        print(f"   • 共 {len(results)} 条记录")


def print_predictions(results, max_display=10):
    """打印预测结果到控制台
    
    参数:
        results: 预测结果列表，每项为 (image_path, prediction, confidence) 元组
        max_display: 最多显示的记录数（int，默认: 10）
        
    示例:
        >>> results = [('img1.jpg', 'cat', 0.95), ('img2.jpg', 'dog', 0.88)]
        >>> print_predictions(results)
        
        📊 预测结果:
        ────────────────────────────────────────────────────────
        1. img1.jpg
           预测: 🐱 cat (置信度: 95.00%)
        2. img2.jpg
           预测: 🐶 dog (置信度: 88.00%)
    """
    if not results:
        print("❌ 没有预测结果")
        return
    
    print(f"\n📊 预测结果:")
    print("─" * 60)
    
    # 显示前 max_display 条
    for i, (img_path, pred, conf) in enumerate(results[:max_display], 1):
        # 获取文件名
        filename = Path(img_path).name
        
        # 选择表情符号
        emoji = "🐱" if pred.lower() in ['cat', '0', 'cats'] else "🐶"
        
        print(f"{i}. {filename}")
        print(f"   预测: {emoji} {pred} (置信度: {conf*100:.2f}%)")
    
    # 如果有更多结果，显示省略信息
    if len(results) > max_display:
        print(f"\n... 还有 {len(results) - max_display} 条记录（省略显示）")
    
    print("─" * 60)
    print(f"总计: {len(results)} 张图片")


def format_prediction_summary(results):
    """格式化预测结果摘要
    
    参数:
        results: 预测结果列表，每项为 (image_path, prediction, confidence) 元组
        
    返回:
        dict: 摘要统计，包含：
            - total: 总预测数量
            - cat_count: 预测为猫的数量
            - dog_count: 预测为狗的数量
            - avg_confidence: 平均置信度
            - high_confidence: 高置信度（>0.9）样本数
            - low_confidence: 低置信度（<0.6）样本数
            
    示例:
        >>> summary = format_prediction_summary(results)
        >>> print(summary['cat_count'])
        45
    """
    if not results:
        return {
            'total': 0,
            'cat_count': 0,
            'dog_count': 0,
            'avg_confidence': 0.0,
            'high_confidence': 0,
            'low_confidence': 0
        }
    
    cat_count = 0
    dog_count = 0
    confidences = []
    
    for _, pred, conf in results:
        confidences.append(conf)
        
        if pred.lower() in ['cat', '0', 'cats']:
            cat_count += 1
        else:
            dog_count += 1
    
    avg_conf = sum(confidences) / len(confidences)
    high_conf = sum(1 for c in confidences if c > 0.9)
    low_conf = sum(1 for c in confidences if c < 0.6)
    
    return {
        'total': len(results),
        'cat_count': cat_count,
        'dog_count': dog_count,
        'avg_confidence': avg_conf,
        'high_confidence': high_conf,
        'low_confidence': low_conf
    }


def print_prediction_summary(summary):
    """打印预测结果摘要
    
    参数:
        summary: 由 format_prediction_summary() 返回的摘要字典
        
    示例:
        >>> summary = format_prediction_summary(results)
        >>> print_prediction_summary(summary)
        
        📈 预测摘要:
        ────────────────────────────────────────
        总数量: 100
        🐱 猫: 45 (45.0%)
        🐶 狗: 55 (55.0%)
        平均置信度: 87.5%
        高置信度 (>90%): 60
        低置信度 (<60%): 5
    """
    total = summary['total']
    
    if total == 0:
        print("❌ 没有预测结果")
        return
    
    print(f"\n📈 预测摘要:")
    print("─" * 40)
    print(f"总数量: {total}")
    print(f"🐱 猫: {summary['cat_count']} ({summary['cat_count']/total*100:.1f}%)")
    print(f"🐶 狗: {summary['dog_count']} ({summary['dog_count']/total*100:.1f}%)")
    print(f"平均置信度: {summary['avg_confidence']*100:.1f}%")
    print(f"高置信度 (>90%): {summary['high_confidence']}")
    print(f"低置信度 (<60%): {summary['low_confidence']}")


def create_feature_extractor_from_config(config, fallback_preset="balanced", fallback_size=64, n_jobs=8):
    """从训练配置创建特征提取器
    
    参数:
        config: 训练配置字典（可为None）
        fallback_preset: 默认预设（str，默认: 'balanced'）
        fallback_size: 默认图像尺寸（int，默认: 64）
        n_jobs: 并行线程数（int，默认: 8）
        
    返回:
        UnifiedFeatureExtractor实例
        
    说明:
        该函数尝试从训练配置中读取特征提取参数，如果配置不存在或解析失败，
        则使用fallback参数。这确保了推理时的特征提取与训练时保持一致。
        
    示例:
        >>> config = load_training_config('runs/sklearn_svm/best.joblib')
        >>> extractor = create_feature_extractor_from_config(config)
        📋 使用训练时的特征配置
        🧩 特征配置: preset=balanced, image_size=64
    """
    from src.data.feature_extract import UnifiedFeatureExtractor
    
    # 从配置中提取参数
    if config and isinstance(config, dict):
        feat_cfg = config.get("features", {})
        preset = feat_cfg.get("preset", fallback_preset)
        image_size = feat_cfg.get("image_size", fallback_size)
        enable_preproc = bool(feat_cfg.get("enable_extractor_preprocessing", False))
        print(f"📋 使用训练时的特征配置")
    else:
        preset = fallback_preset
        image_size = fallback_size
        enable_preproc = False
        print(f"⚠️  未找到训练配置，使用默认参数")
    
    print(f"🧩 特征配置: preset={preset}, image_size={image_size}")
    
    return UnifiedFeatureExtractor(
        feature_config=preset,
        image_size=image_size,
        enable_preprocessing=enable_preproc,
        n_jobs=n_jobs,
        verbose=False
    )


def extract_features_for_inference(image_paths, extractor, show_progress=True):
    """提取推理特征（带错误处理）
    
    参数:
        image_paths: 图像路径列表
        extractor: 特征提取器实例（UnifiedFeatureExtractor）
        show_progress: 是否显示进度条（bool，默认: True）
        
    返回:
        tuple: (features, valid_paths) - 特征矩阵和有效路径列表
        
    异常:
        RuntimeError: 如果特征提取失败或没有有效图片
        
    示例:
        >>> extractor = UnifiedFeatureExtractor(...)
        >>> features, valid_paths = extract_features_for_inference(image_paths, extractor)
        🎨 提取特征...
        100%|████████████| 1000/1000 [00:30<00:00, 33.33it/s]
           ✅ 成功提取 1000 个样本的特征
    """
    print(f"\n🎨 提取特征...")
    features, _, valid_indices = extractor.extract_features_batch(
        image_paths,
        labels=None,
        show_progress=show_progress
    )
    
    if len(features) == 0:
        raise RuntimeError("❌ 特征提取失败，没有有效的图片")
    
    valid_paths = [image_paths[i] for i in valid_indices]
    print(f"   ✅ 成功提取 {len(features)} 个样本的特征")
    
    return features, valid_paths

