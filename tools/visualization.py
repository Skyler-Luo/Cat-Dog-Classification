"""可视化与图形生成工具集。"""

from pathlib import Path
import platform

import numpy as np
import matplotlib
from matplotlib import font_manager
from sklearn import metrics

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _configure_fonts():
    """配置 Matplotlib 字体以支持中文显示。"""
    system_name = platform.system().lower()
    fallback_font_files = []
    linux_fonts = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    windows_fonts = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/msyhbd.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simkai.ttf',
        'C:/Windows/Fonts/mingliu.ttc',
    ]
    mac_fonts = [
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        '/Library/Fonts/Hiragino Sans GB W3.otf',
    ]
    if 'windows' in system_name:
        fallback_font_files = windows_fonts + linux_fonts + mac_fonts
    elif 'darwin' in system_name or 'mac' in system_name:
        fallback_font_files = mac_fonts + linux_fonts + windows_fonts
    else:
        fallback_font_files = linux_fonts + mac_fonts + windows_fonts
    preferred_fonts = [
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'SimHei',
        'Microsoft YaHei',
        'Arial Unicode MS',
    ]
    fallback_font_files = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    detected_fonts = []
    for font_path in fallback_font_files:
        try:
            path = Path(font_path)
            if path.exists():
                font_manager.fontManager.addfont(str(path))
                detected_name = font_manager.FontProperties(fname=str(path)).get_name()
                if detected_name:
                    detected_fonts.append(detected_name)
        except Exception:
            continue
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidate_fonts = []
    for name in preferred_fonts + detected_fonts:
        if name in available and name not in candidate_fonts:
            candidate_fonts.append(name)
    if not candidate_fonts:
        candidate_fonts = ['DejaVu Sans']
    current = [font for font in plt.rcParams.get('font.sans-serif', []) if font not in candidate_fonts]
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = candidate_fonts + current
    plt.rcParams['axes.unicode_minus'] = False


_configure_fonts()


METRIC_NAME_MAP = {
    'accuracy': '准确率',
    'precision': '精确率',
    'recall': '召回率',
    'f1': 'F1 值',
    'auc': 'AUC',
    'loss': '损失',
    'average_precision': '平均精确率',
}

SPLIT_NAME_MAP = {
    'train': '训练集',
    'val': '验证集',
    'test': '测试集',
}


def translate_metric_name(metric):
    """将常用指标名称转换为中文描述。"""
    if metric is None:
        return ''
    name = str(metric)
    return METRIC_NAME_MAP.get(name.lower(), name)


def _prepare_output_path(save_path, default_name):
    """准备输出路径。
    
    参数:
        save_path: 图像保存路径，支持 str 或 Path。如果为 None，则使用默认文件名。
        default_name: 默认文件名（例如 "accuracy.png"）
        
    返回:
        Path: 可用于保存图像的绝对路径
    """
    if save_path is None:
        save_path = Path(default_name)
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_split_metrics(results, metric='pr', title=None, save_path=None):
    """绘制训练/验证/测试集上的 PR 曲线图。
    
    参数:
        results: 评估结果字典，键为数据集划分（"train"/"val"/"test"），
            值为包含 precision_curve/recall_curve 列表或 pr_curve 子字典的结构
        metric: 用于输出文件命名的标识（默认: "pr"）
        title: 图标题（可选）
        save_path: 图像保存路径（可选）
        
    返回:
        Path: 图像保存路径
        
    示例:
        >>> results = {
        ...     "train": {"precision_curve": [1.0, 0.9], "recall_curve": [0.0, 1.0]},
        ...     "val": {"precision_curve": [1.0, 0.85], "recall_curve": [0.0, 1.0]}
        ... }
        >>> plot_split_metrics(results, save_path="runs/pr_curve.png")
    """
    if not results or not isinstance(results, dict):
        raise ValueError("results 必须是包含 PR 曲线信息的字典。")
    
    plt.figure(figsize=(6, 4))
    split_colors = {
        'train': '#4c78a8',
        'val': '#f58518',
        'test': '#54a24b',
    }
    has_data = False
    for split in ['train', 'val', 'test']:
        split_metrics = results.get(split)
        if not isinstance(split_metrics, dict):
            continue
        precision_curve = split_metrics.get('precision_curve')
        recall_curve = split_metrics.get('recall_curve')
        if precision_curve is None or recall_curve is None:
            pr_curve = split_metrics.get('pr_curve')
            if isinstance(pr_curve, dict):
                precision_curve = pr_curve.get('precision')
                recall_curve = pr_curve.get('recall')
        if precision_curve is None or recall_curve is None:
            continue
        precision_curve = np.array(precision_curve, dtype=float)
        recall_curve = np.array(recall_curve, dtype=float)
        if precision_curve.size == 0 or recall_curve.size == 0:
            continue
        label = SPLIT_NAME_MAP.get(split, split.capitalize())
        average_precision = split_metrics.get('average_precision')
        if average_precision is not None:
            label = "{} (AP = {:.3f})".format(label, average_precision)
        color = split_colors.get(split)
        plt.step(recall_curve, precision_curve, where='post', label=label, color=color, linewidth=1.8)
        if color is not None:
            plt.fill_between(recall_curve, precision_curve, step='post', alpha=0.12, color=color)
        has_data = True
    if not has_data:
        raise ValueError("未找到可用的 PR 曲线数据。")
    
    plt.xlabel("召回率")
    plt.ylabel("精确率")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    if title:
        plt.title(title)
    else:
        plt.title("训练/验证/测试集 PR 曲线")
    plt.grid(ls='--', alpha=0.3)
    plt.legend(loc='lower left')
    
    default_name = "pr_curve.png" if metric == 'pr' else "{}.png".format(metric)
    output_path = _prepare_output_path(save_path, default_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_confusion_matrix(confusion_matrix, class_names=None, normalize=False, title=None, save_path=None):
    """绘制混淆矩阵。
    
    参数:
        confusion_matrix: 混淆矩阵数组，形状为 (n_classes, n_classes)
        class_names: 类别名称列表，例如 ["Cat", "Dog"]
        normalize: 是否按行归一化混淆矩阵
        title: 图标题（可选）
        save_path: 图像保存路径（可选）
        
    返回:
        Path: 图像保存路径
    """
    matrix = np.array(confusion_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix 必须是方阵。")
    
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
    
    plt.figure(figsize=(5, 4))
    cmap = plt.get_cmap('Blues')
    im = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    num_classes = matrix.shape[0]
    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(num_classes)]
    plt.xticks(np.arange(num_classes), class_names, rotation=45)
    plt.yticks(np.arange(num_classes), class_names)
    
    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(matrix[i, j], fmt),
                     ha="center", va="center",
                     color="white" if matrix[i, j] > thresh else "black")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    if title:
        plt.title(title)
    else:
        plt.title("Confusion Matrix")
    
    output_path = _prepare_output_path(save_path, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_metric_curves(history, metric_keys=None, title=None, xlabel="训练轮次", ylabel=None, save_path=None):
    """绘制训练过程中的指标曲线。
    
    参数:
        history: 指标历史记录字典，键为指标名称，值为列表
        metric_keys: 需要绘制的指标名称列表（可选，默认为 history 中的所有键）
        title: 图标题（可选）
        xlabel: 横轴标签（默认: Epoch）
        ylabel: 纵轴标签（可选，为空时自动设置为 'Metric Value'）
        save_path: 图像保存路径（可选）
        
    返回:
        Path: 图像保存路径
    """
    if not history or not isinstance(history, dict):
        raise ValueError("history 必须是包含曲线数据的字典。")
    
    if metric_keys is None:
        metric_keys = [key for key in history if isinstance(history[key], (list, tuple, np.ndarray))]
    if not metric_keys:
        raise ValueError("未找到可绘制的指标曲线。")
    
    plt.figure(figsize=(7, 4))
    epochs = None
    for key in metric_keys:
        values = history.get(key)
        if values is None:
            continue
        values = np.array(values, dtype=float)
        if epochs is None:
            epochs = np.arange(1, len(values) + 1)
        plt.plot(epochs, values, marker='o', label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or "指标值")
    if title:
        plt.title(title)
    else:
        plt.title("训练指标曲线")
    plt.grid(ls='--', alpha=0.3)
    plt.legend()
    output_path = _prepare_output_path(save_path, "metric_curves.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_roc_curve(y_true, y_scores, title=None, save_path=None):
    """绘制 ROC 曲线。
    
    参数:
        y_true: 真实标签数组
        y_scores: 正类得分或概率数组
        title: 图标题（可选）
        save_path: 图像保存路径（可选）
        
    返回:
        Path: 图像保存路径
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    auc_value = metrics.auc(fpr, tpr)
    
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#4c78a8", label="AUC = {:.4f}".format(auc_value))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    if title:
        plt.title(title)
    else:
        plt.title("ROC 曲线")
    plt.legend(loc="lower right")
    plt.grid(ls='--', alpha=0.3)
    output_path = _prepare_output_path(save_path, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_cv_results(cv_results, metric_key='mean_test_score', std_key='std_test_score',
                    top_n=20, title=None, save_path=None):
    """绘制超参数搜索结果对比图。
    
    参数:
        cv_results: GridSearchCV.cv_results_ 字典
        metric_key: 使用的评分指标键（默认: mean_test_score）
        std_key: 评分标准差的键（默认: std_test_score）
        top_n: 展示的前 N 个配置
        title: 图标题（可选）
        save_path: 图像保存路径（可选）
        
    返回:
        Path: 图像保存路径
    """
    if cv_results is None or metric_key not in cv_results:
        raise ValueError("cv_results 中缺少 {}。".format(metric_key))
    
    params_list = cv_results.get('params')
    scores = cv_results.get(metric_key)
    stds = cv_results.get(std_key)
    if params_list is None or scores is None:
        raise ValueError("cv_results 缺少必要的参数或分数信息。")
    
    entries = []
    for params, score, std in zip(params_list, scores, stds if stds is not None else [0] * len(scores)):
        params_str = ", ".join("{}={}".format(k, v) for k, v in sorted(params.items()))
        entries.append((params_str, score, std))
    entries.sort(key=lambda item: item[1], reverse=True)
    entries = entries[:top_n]
    
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    errors = [e[2] for e in entries]
    
    plt.figure(figsize=(8, max(4, len(entries) * 0.4)))
    y_pos = np.arange(len(entries))
    plt.barh(y_pos, values, xerr=errors, color="#4c78a8", alpha=0.8)
    plt.yticks(y_pos, labels)
    plt.xlabel(translate_metric_name(metric_key) or metric_key)
    if title:
        plt.title(title)
    else:
        plt.title("超参数搜索结果 ({})".format(translate_metric_name(metric_key) or metric_key))
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    output_path = _prepare_output_path(save_path, "cv_results.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_torch_training_curves(train_history, val_history, learning_rates=None, save_dir=None):
    """绘制 PyTorch 训练过程中的损失曲线、精度曲线等。
    
    参数:
        train_history: 训练集历史记录字典，包含 loss, accuracy, precision, recall, f1 等键
        val_history: 验证集历史记录字典，包含 loss, accuracy, precision, recall, f1 等键
        learning_rates: 学习率历史列表（可选）
        save_dir: 图像保存目录（可选）
        
    返回:
        dict: 包含生成图像路径的字典
    """
    if save_dir is None:
        save_dir = Path(".")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    epochs = np.arange(1, len(train_history.get("loss", [])) + 1)
    
    # 1. 损失曲线
    if "loss" in train_history and "loss" in val_history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_history["loss"], label="训练损失", color="#4c78a8", linewidth=2)
        plt.plot(epochs, val_history["loss"], label="验证损失", color="#f58518", linewidth=2)
        plt.xlabel("训练轮次")
        plt.ylabel("损失")
        plt.title("训练/验证损失曲线")
        plt.legend()
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        loss_path = save_dir / "loss_curve.png"
        plt.savefig(loss_path, dpi=200)
        plt.close()
        outputs['loss_curve'] = loss_path
    
    # 2. 准确率曲线
    if "accuracy" in train_history and "accuracy" in val_history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, np.array(train_history["accuracy"]) * 100, label="训练准确率", color="#4c78a8", linewidth=2)
        plt.plot(epochs, np.array(val_history["accuracy"]) * 100, label="验证准确率", color="#f58518", linewidth=2)
        plt.xlabel("训练轮次")
        plt.ylabel("准确率 (%)")
        plt.title("训练/验证准确率曲线")
        plt.legend()
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        acc_path = save_dir / "accuracy_curve.png"
        plt.savefig(acc_path, dpi=200)
        plt.close()
        outputs['accuracy_curve'] = acc_path
    
    # 3. F1 分数曲线
    if "f1" in train_history and "f1" in val_history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_history["f1"], label="训练 F1", color="#4c78a8", linewidth=2)
        plt.plot(epochs, val_history["f1"], label="验证 F1", color="#f58518", linewidth=2)
        plt.xlabel("训练轮次")
        plt.ylabel("F1 分数")
        plt.title("训练/验证 F1 分数曲线")
        plt.legend()
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        f1_path = save_dir / "f1_curve.png"
        plt.savefig(f1_path, dpi=200)
        plt.close()
        outputs['f1_curve'] = f1_path
    
    # 4. 精确率和召回率曲线
    if "precision" in train_history and "recall" in train_history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_history["precision"], label="训练精确率", color="#4c78a8", linewidth=2, linestyle='-')
        plt.plot(epochs, train_history["recall"], label="训练召回率", color="#4c78a8", linewidth=2, linestyle='--')
        plt.plot(epochs, val_history.get("precision", []), label="验证精确率", color="#f58518", linewidth=2, linestyle='-')
        plt.plot(epochs, val_history.get("recall", []), label="验证召回率", color="#f58518", linewidth=2, linestyle='--')
        plt.xlabel("训练轮次")
        plt.ylabel("分数")
        plt.title("训练/验证精确率与召回率曲线")
        plt.legend()
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        pr_path = save_dir / "precision_recall_curve.png"
        plt.savefig(pr_path, dpi=200)
        plt.close()
        outputs['precision_recall_curve'] = pr_path
    
    # 5. 学习率曲线
    if learning_rates is not None and len(learning_rates) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, learning_rates, label="学习率", color="#54a24b", linewidth=2)
        plt.xlabel("训练轮次")
        plt.ylabel("学习率")
        plt.title("学习率变化曲线")
        plt.legend()
        plt.grid(ls='--', alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        lr_path = save_dir / "learning_rate_curve.png"
        plt.savefig(lr_path, dpi=200)
        plt.close()
        outputs['learning_rate_curve'] = lr_path
    
    # 6. 综合指标对比图（子图形式）
    if len(outputs) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失
        if "loss" in train_history:
            axes[0, 0].plot(epochs, train_history["loss"], label="训练", color="#4c78a8", linewidth=1.5)
            axes[0, 0].plot(epochs, val_history["loss"], label="验证", color="#f58518", linewidth=1.5)
            axes[0, 0].set_xlabel("训练轮次")
            axes[0, 0].set_ylabel("损失")
            axes[0, 0].set_title("损失曲线")
            axes[0, 0].legend()
            axes[0, 0].grid(ls='--', alpha=0.3)
        
        # 准确率
        if "accuracy" in train_history:
            axes[0, 1].plot(epochs, np.array(train_history["accuracy"]) * 100, label="训练", color="#4c78a8", linewidth=1.5)
            axes[0, 1].plot(epochs, np.array(val_history["accuracy"]) * 100, label="验证", color="#f58518", linewidth=1.5)
            axes[0, 1].set_xlabel("训练轮次")
            axes[0, 1].set_ylabel("准确率 (%)")
            axes[0, 1].set_title("准确率曲线")
            axes[0, 1].legend()
            axes[0, 1].grid(ls='--', alpha=0.3)
        
        # F1
        if "f1" in train_history:
            axes[1, 0].plot(epochs, train_history["f1"], label="训练", color="#4c78a8", linewidth=1.5)
            axes[1, 0].plot(epochs, val_history["f1"], label="验证", color="#f58518", linewidth=1.5)
            axes[1, 0].set_xlabel("训练轮次")
            axes[1, 0].set_ylabel("F1 分数")
            axes[1, 0].set_title("F1 分数曲线")
            axes[1, 0].legend()
            axes[1, 0].grid(ls='--', alpha=0.3)
        
        # 学习率
        if learning_rates is not None and len(learning_rates) > 0:
            axes[1, 1].plot(epochs, learning_rates, label="学习率", color="#54a24b", linewidth=1.5)
            axes[1, 1].set_xlabel("训练轮次")
            axes[1, 1].set_ylabel("学习率")
            axes[1, 1].set_title("学习率变化")
            axes[1, 1].legend()
            axes[1, 1].grid(ls='--', alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        summary_path = save_dir / "training_summary.png"
        plt.savefig(summary_path, dpi=200)
        plt.close()
        outputs['training_summary'] = summary_path
    
    return outputs
