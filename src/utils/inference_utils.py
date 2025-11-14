"""
推理脚本通用工具函数集合。

该模块提供加载模型、准备数据与执行批量推理的工具函数，
供命令行推理脚本和其他上层应用复用。
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from src.models.cnn import create_CatDogCNNv1, create_CatDogCNNv2
from src.models.resnet import PretrainedResNet


CLASS_NAMES = ["cats", "dogs"]
LOGGER = logging.getLogger(__name__)


def collect_image_paths(input_path, recursive=True, allowed_suffixes=None):
    """收集待推理的图像路径列表。
    
    参数:
        input_path: 输入资源路径，可以是图片文件、目录、TXT/CSV 列表文件。
        recursive: 是否递归搜索目录（bool，默认: True）。
        allowed_suffixes: 允许的文件后缀列表（可选，默认包含常见图像格式）。
        
    返回:
        list: 排序后的图像绝对路径列表（str）。
    """
    if allowed_suffixes is None:
        allowed_suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError("输入路径不存在: {}".format(input_path))
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix in allowed_suffixes:
            resolved = str(input_path.resolve())
            LOGGER.debug("收集单张图像: %s", resolved)
            return [resolved]
        if suffix in [".txt", ".csv", ".tsv"]:
            paths = []
            for line in input_path.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if not item:
                    continue
                if suffix == ".csv":
                    item = item.split(",")[0].strip()
                elif suffix == ".tsv":
                    item = item.split("\t")[0].strip()
                paths.append(item)
            if not paths:
                raise ValueError("列表文件未包含有效的图像路径: {}".format(input_path))
            gathered = []
            for path in paths:
                p = Path(path).expanduser()
                if p.exists() and p.suffix.lower() in allowed_suffixes:
                    gathered.append(str(p.resolve()))
            if not gathered:
                raise ValueError("未在列表文件中找到有效图像: {}".format(input_path))
            gathered = sorted(gathered)
            LOGGER.info("从列表加载 %d 张图像: %s", len(gathered), input_path)
            return gathered
        raise ValueError("不支持的文件类型: {}".format(input_path.suffix))
    pattern = "**/*" if recursive else "*"
    files = []
    for file_path in input_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in allowed_suffixes:
            files.append(str(file_path.resolve()))
    if not files:
        raise ValueError("未在目录中找到符合条件的图像文件: {}".format(input_path))
    files = sorted(files)
    LOGGER.info("从目录收集 %d 张图像: %s", len(files), input_path)
    return files


def load_checkpoint(checkpoint_path, device=None):
    """加载训练保存的模型检查点。
    
    参数:
        checkpoint_path: 检查点文件路径（str 或 Path）。
        device: 目标设备标识（str，可选）。若为 None，将自动推断。
        
    返回:
        dict: 包含 state_dict、config、metrics 等字段的检查点字典。
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError("未找到检查点文件: {}".format(checkpoint_path))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    map_location = torch.device(device)
    LOGGER.info("加载模型检查点: %s", checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(payload, dict):
        raise RuntimeError("检查点格式无效: {}".format(checkpoint_path))
    return payload


def _build_model_from_config(config, arch=None):
    """根据配置构建模型实例。
    
    参数:
        config: 训练时保存的配置字典。
        arch: 模型架构名称（str，可选），默认读取 config['model_version']。
        
    返回:
        torch.nn.Module: 加载完成的模型实例（尚未载入权重）。
    """
    arch = arch or config.get("model_version")
    dropout = config.get("dropout", 0.0)
    if arch == "cnn_v1":
        return create_CatDogCNNv1(num_classes=1, in_channels=3, dropout_p=dropout)
    if arch == "cnn_v2":
        return create_CatDogCNNv2(num_classes=1, in_channels=3, dropout_p=dropout)
    if arch in ["resnet18", "resnet34", "resnet50"]:
        return PretrainedResNet(
            model_name=arch,
            num_classes=1,
            pretrained=False,
            freeze_backbone=False,
            dropout_p=dropout,
        )
    raise ValueError("不支持的模型架构: {}".format(arch))


def prepare_model(checkpoint_path, device=None, arch=None):
    """载入模型检查点并准备好用于推理的模型。
    
    参数:
        checkpoint_path: 检查点文件路径。
        device: 目标设备标识（str，可选），默认自动选择。
        arch: 模型架构名称（str，可选），用于覆盖检查点配置。
        
    返回:
        tuple: (model, config) 模型实例与配置字典。
    
    示例:
        >>> model, cfg = prepare_model("runs/torch_cnn/best.pt")
        >>> model.eval()
        PretrainedResNet(...)
    """
    payload = load_checkpoint(checkpoint_path, device=device)
    config = payload.get("config", {})
    model = _build_model_from_config(config, arch=arch)
    model.load_state_dict(payload["state_dict"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    LOGGER.info("模型已加载到设备: %s", device)
    return model, config


class _ImageListDataset(Dataset):
    """基于文件路径列表的简单推理数据集。"""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, image_path


def run_inference(model, image_paths, transform, device, batch_size=32, threshold=0.5, show_progress=True, logger=None):
    """执行批量推理并返回结果列表。
    
    参数:
        model: 已加载权重并切换到 eval 模式的 PyTorch 模型。
        image_paths: 图像路径列表（list）。
        transform: 图像预处理变换（torchvision.transforms.Compose）。
        device: 推理设备（str 或 torch.device）。
        batch_size: 推理批次大小（int，默认: 32）。
        threshold: 将概率转换为类别标签的阈值（float，默认: 0.5）。
        show_progress: 是否显示 tqdm 进度条（bool，默认: True）。
        logger: 日志记录器（可选）。若未提供，使用模块日志。
        
    返回:
        list: 结果字典列表，每个元素包含 path、prob、label、label_name。
    """
    dataset = _ImageListDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(device)
    results = []
    sigmoid = torch.nn.Sigmoid()
    tqdm_bar = None
    iterator = loader
    logger = logger or LOGGER
    logger.info("开始推理，总计 %d 张图像，批次大小 %d", len(image_paths), batch_size)
    if show_progress:
        tqdm_bar = tqdm(
            loader,
            desc="🔮 推理中",
            unit="batch",
            ncols=100,
            leave=False,
        )
        iterator = tqdm_bar
    for batch, batch_paths in iterator:
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(batch)
            probs = sigmoid(logits.view(-1))
        for prob, path in zip(probs.tolist(), batch_paths):
            label = 1 if prob >= threshold else 0
            label_name = CLASS_NAMES[label]
            results.append(
                {
                    "path": path,
                    "prob": prob,
                    "label": label,
                    "label_name": label_name,
                }
            )
    if tqdm_bar is not None:
        tqdm_bar.close()
    logger.info("推理完成，生成 %d 条结果。", len(results))
    return results


def summarize_predictions(results):
    """汇总预测结果，统计各类别数量与平均置信度。
    
    参数:
        results: 推理结果列表，由 run_inference 返回。
        
    返回:
        dict: 包含总数、各类别计数、平均置信度等统计信息的字典。
    """
    if not results:
        return {
            "total": 0,
            "cats": {"count": 0, "avg_prob": 0.0},
            "dogs": {"count": 0, "avg_prob": 0.0},
        }
    total = len(results)
    accum = {
        "cats": {"count": 0, "prob_sum": 0.0},
        "dogs": {"count": 0, "prob_sum": 0.0},
    }
    for item in results:
        label_name = item["label_name"]
        prob = item["prob"]
        if label_name == "cats":
            accum["cats"]["count"] += 1
            accum["cats"]["prob_sum"] += 1.0 - prob
        else:
            accum["dogs"]["count"] += 1
            accum["dogs"]["prob_sum"] += prob
    cats_avg = accum["cats"]["prob_sum"] / max(accum["cats"]["count"], 1)
    dogs_avg = accum["dogs"]["prob_sum"] / max(accum["dogs"]["count"], 1)
    return {
        "total": total,
        "cats": {
            "count": accum["cats"]["count"],
            "avg_prob": cats_avg,
        },
        "dogs": {
            "count": accum["dogs"]["count"],
            "avg_prob": dogs_avg,
        },
    }


