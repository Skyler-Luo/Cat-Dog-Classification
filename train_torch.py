"""统一的 PyTorch 训练入口（CNN / ResNet）。

该脚本整合原先的 `train_cnn.py` 与 `train_resnet.py`，
通过 `--arch` 参数选择具体模型架构，复用统一的训练管线。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from src.models.cnn import create_CatDogCNNv1, create_CatDogCNNv2
from src.models.resnet import create_resnet18, create_resnet50, PretrainedResNet
from src.utils import TrainConfig, ARCH_CHOICES
from src.utils.torch_training import train_binary_classification
from src.utils.logger import Logger


def parse_args():
    """解析命令行参数。"""

    description = "统一 PyTorch 训练入口 (CNN / ResNet)"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data_group = parser.add_argument_group("数据集配置")
    data_group.add_argument("--data-dir", type=str, default="dataset", help="数据根目录")
    data_group.add_argument("--train-dirname", type=str, default="train", help="训练集子目录名")
    data_group.add_argument("--val-dirname", type=str, default="val", help="验证集子目录名")
    data_group.add_argument("--test-dirname", type=str, default="test", help="测试集子目录名")

    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument("--arch", type=str, choices=ARCH_CHOICES, default="cnn_v1", help="模型架构 (cnn_v1/cnn_v2/resnet18/resnet34/resnet50)")
    train_group.add_argument("--image-size", type=int, default=224, help="输入图像尺寸")
    train_group.add_argument("--batch-size", type=int, default=512, help="批次大小")
    train_group.add_argument("--epochs", type=int, default=120, help="训练轮数")
    train_group.add_argument("--lr", type=float, default=1e-3, help="学习率")
    train_group.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    train_group.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="优化器类型")   
    train_group.add_argument("--momentum", type=float, default=0.9, help="SGD 动量")
    train_group.add_argument("--scheduler", type=str, choices=["none", "cosine", "step"], default="cosine", help="学习率调度器")
    train_group.add_argument("--step-size", type=int, default=10, help="StepLR 步长")
    train_group.add_argument("--gamma", type=float, default=0.1, help="学习率衰减因子")
    train_group.add_argument("--dropout", type=float, default=0.0, help="分类头 Dropout 概率")
    train_group.add_argument("--normalize-imagenet", action="store_true", help="使用 ImageNet 标准化")
    train_group.add_argument("--no-train-augment", dest="train_augment", action="store_false", help="禁用训练数据增强（默认: 启用）")
    train_group.add_argument("--no-amp", dest="amp", action="store_false", help="禁用混合精度")
    train_group.add_argument("--early-stop-patience", type=int, default=10, help="早停耐心值")
    train_group.add_argument("--early-stop-metric", type=str, default="accuracy", choices=["accuracy", "f1", "precision", "recall", "loss"], help="早停评估指标")
    train_group.add_argument("--seed", type=int, default=42, help="随机种子")

    resnet_group = parser.add_argument_group("ResNet 专属配置")
    resnet_group.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false", help="解冻 ResNet 主干（默认: 冻结）")
    resnet_group.add_argument("--no-resnet-pretrained", dest="resnet_pretrained", action="store_false", help="禁用 ResNet ImageNet 预训练权重（默认: 启用）")

    system_group = parser.add_argument_group("系统配置")
    system_group.add_argument("--num-workers", type=int, default=16, help="DataLoader 工作进程数")
    system_group.add_argument("--run-dir", type=str, default=None, help="运行目录（默认: runs/torch_{arch}_{timestamp}）")
    system_group.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")

    parser.set_defaults(amp=True, train_augment=True, resnet_pretrained=True, freeze_backbone=True)

    return parser.parse_args()


def _build_model_factory(cfg):
    """根据配置构建模型工厂函数。
    
    参数:
        cfg: TrainConfig 配置对象
        
    返回:
        模型构建函数
    """

    arch = cfg.model_version

    def build_model(device):
        if arch == "cnn_v1":
            model = create_CatDogCNNv1(num_classes=1, in_channels=3, dropout_p=cfg.dropout).to(device)
            return model, "CatDogCNNv1"
        if arch == "cnn_v2":
            model = create_CatDogCNNv2(num_classes=1, in_channels=3, dropout_p=cfg.dropout).to(device)
            return model, "CatDogCNNv2"
        if arch == "resnet18":
            model = create_resnet18(
                num_classes=1,
                pretrained=cfg.resnet_pretrained,
                freeze_backbone=cfg.freeze_backbone,
                dropout_p=cfg.dropout,
            ).to(device)
            return model, "ResNet18"
        if arch == "resnet34":
            model = PretrainedResNet(
                model_name="resnet34",
                num_classes=1,
                pretrained=cfg.resnet_pretrained,
                freeze_backbone=cfg.freeze_backbone,
                dropout_p=cfg.dropout,
            ).to(device)
            return model, "ResNet34"
        if arch == "resnet50":
            model = create_resnet50(
                num_classes=1,
                pretrained=cfg.resnet_pretrained,
                freeze_backbone=cfg.freeze_backbone,
                dropout_p=cfg.dropout,
            ).to(device)
            return model, "ResNet50"
        raise ValueError("不支持的 arch: {}".format(arch))

    return build_model


def _default_run_dir(arch, timestamp=None):
    """生成带时间戳的默认运行目录
    
    参数:
        arch: 模型架构名称（如 "cnn_v2", "resnet18"）
        timestamp: 时间戳字符串（格式: YYYYMMDD-HHMMSS），如果为None则自动生成
        
    返回:
        str: 运行目录路径，格式为 runs/torch_{arch}_{timestamp}
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    arch_clean = arch.replace("_", "-")
    base_dir = Path("runs") / "torch_{}_{}".format(arch_clean, timestamp)
    return str(base_dir)


def main():
    """脚本主入口。"""

    args = parse_args()
    args.arch = args.arch.lower()

    start_time = time.time()
    experiment_name = args.arch.replace("_", "-")
    
    # 生成时间戳和运行目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.run_dir:
        args.run_dir = _default_run_dir(args.arch, timestamp)
    
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        name=experiment_name,
        log_dir=str(run_dir),
        filename="{}.log".format(experiment_name),
        level=args.log_level,
    )

    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_dirname=args.train_dirname,
        val_dirname=args.val_dirname,
        test_dirname=args.test_dirname,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        model_version=args.arch,
        dropout=args.dropout,
        normalize_imagenet=args.normalize_imagenet,
        train_augment=args.train_augment,
        amp=args.amp,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        resnet_pretrained=args.resnet_pretrained,
        freeze_backbone=args.freeze_backbone,
        num_workers=args.num_workers,
        seed=args.seed,
        log_dir=str(run_dir),  # TensorBoard 日志目录
        ckpt_dir=str(run_dir),  # 模型保存目录
    )

    config_info = cfg.to_dict()
    config_info["arch"] = args.arch

    logger.block(
        "开始训练",
        [
            "arch: {}".format(args.arch),
            "data_dir: {}".format(args.data_dir),
            "batch_size: {}".format(args.batch_size),
            "epochs: {}".format(args.epochs),
        ],
    )
    logger.info("启动 {} 训练流程".format(experiment_name.upper()))

    build_model = _build_model_factory(cfg)
    title_map = {
        "cnn_v1": "猫狗分类 - CNN v1 训练",
        "cnn_v2": "猫狗分类 - CNN v2 训练",
        "resnet18": "猫狗分类 - ResNet18 训练",
        "resnet34": "猫狗分类 - ResNet34 训练",
        "resnet50": "猫狗分类 - ResNet50 训练",
    }
    title = title_map.get(args.arch, "猫狗分类 - PyTorch 训练")

    results = train_binary_classification(cfg, build_model, title, logger=logger)

    total_time = time.time() - start_time
    best_val = results.get("best_val", {})
    best_acc = best_val.get("accuracy")
    ckpt_dir = results.get("ckpt_dir")

    logger.block(
        "训练完成",
        [
            "耗时: {}".format(Logger.format_duration(total_time)),
            "最佳准确率: {:.4f}".format(best_acc) if best_acc is not None else "最佳准确率: -",
            "运行目录: {}".format(run_dir),
        ],
    )
    logger.info("{} 训练流程完成".format(experiment_name.upper()))

    return results


if __name__ == "__main__":
    main()
