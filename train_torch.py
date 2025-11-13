"""统一的 PyTorch 训练入口（CNN / ResNet）。

该脚本整合原先的 `train_cnn.py` 与 `train_resnet.py`，
通过 `--arch` 参数选择具体模型架构，复用统一的训练管线。
"""

import argparse
import time

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
    train_group.add_argument("--arch", type=str, choices=ARCH_CHOICES, default="cnn_v2", help="模型架构 (cnn_v1/cnn_v2/resnet18/resnet34/resnet50)")
    train_group.add_argument("--image-size", type=int, default=224, help="输入图像尺寸")
    train_group.add_argument("--batch-size", type=int, default=512, help="批次大小")
    train_group.add_argument("--epochs", type=int, default=120, help="训练轮数")
    train_group.add_argument("--lr", type=float, default=1e-3, help="学习率")
    train_group.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    train_group.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="优化器类型")   
    train_group.add_argument("--momentum", type=float, default=0.9, help="SGD 动量")
    train_group.add_argument("--scheduler", type=str, choices=["none", "cosine", "step"], default="none", help="学习率调度器")
    train_group.add_argument("--step-size", type=int, default=10, help="StepLR 步长")
    train_group.add_argument("--gamma", type=float, default=0.1, help="学习率衰减因子")
    train_group.add_argument("--dropout", type=float, default=0.0, help="分类头 Dropout 概率")
    train_group.add_argument("--normalize-imagenet", action="store_true", help="使用 ImageNet 标准化")
    train_group.add_argument("--no-amp", dest="amp", action="store_false", help="禁用混合精度")
    train_group.add_argument("--early-stop-patience", type=int, default=10, help="早停耐心值")
    train_group.add_argument("--seed", type=int, default=42, help="随机种子")

    resnet_group = parser.add_argument_group("ResNet 专属配置")
    resnet_group.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", help="冻结 ResNet 主干 (仅训练分类头)")
    resnet_group.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false", help="解冻 ResNet 主干")
    resnet_group.add_argument("--no-resnet-pretrained", dest="resnet_pretrained", action="store_false", help="禁用 ResNet ImageNet 预训练权重")
    resnet_group.add_argument("--resnet-pretrained", dest="resnet_pretrained", action="store_true", help="启用 ResNet ImageNet 预训练权重")

    system_group = parser.add_argument_group("系统配置")
    system_group.add_argument("--num-workers", type=int, default=16, help="DataLoader 工作进程数")
    system_group.add_argument("--log-dir", type=str, default="runs/torch_cnn", help="TensorBoard 日志目录")
    system_group.add_argument("--ckpt-dir", type=str, default="runs/torch_cnn", help="模型保存目录")
    system_group.add_argument("--log-output-dir", type=str, default="logs/torch", help="文本日志目录")
    system_group.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别 (保留参数，便于兼容)")

    parser.set_defaults(amp=True, resnet_pretrained=True, freeze_backbone=True)

    return parser.parse_args()


def _build_model_factory(args):
    """根据参数构建模型工厂函数。"""

    arch = args.arch

    def build_model(device):
        if arch == "cnn_v1":
            model = create_CatDogCNNv1(num_classes=1, in_channels=3, dropout_p=args.dropout).to(device)
            return model, "CatDogCNNv1"
        if arch == "cnn_v2":
            model = create_CatDogCNNv2(num_classes=1, in_channels=3, dropout_p=args.dropout).to(device)
            return model, "CatDogCNNv2"
        if arch == "resnet18":
            model = create_resnet18(
                num_classes=1,
                pretrained=args.resnet_pretrained,
                freeze_backbone=args.freeze_backbone,
                dropout_p=args.dropout,
            ).to(device)
            return model, "ResNet18"
        if arch == "resnet34":
            model = PretrainedResNet(
                model_name="resnet34",
                num_classes=1,
                pretrained=args.resnet_pretrained,
                freeze_backbone=args.freeze_backbone,
                dropout_p=args.dropout,
            ).to(device)
            return model, "ResNet34"
        if arch == "resnet50":
            model = create_resnet50(
                num_classes=1,
                pretrained=args.resnet_pretrained,
                freeze_backbone=args.freeze_backbone,
                dropout_p=args.dropout,
            ).to(device)
            return model, "ResNet50"
        raise ValueError("不支持的 arch: {}".format(arch))

    return build_model


def main():
    """脚本主入口。"""

    args = parse_args()
    args.arch = args.arch.lower()

    start_time = time.time()
    experiment_name = args.arch.replace("_", "-")

    logger = Logger(
        name=experiment_name,
        log_dir=args.log_output_dir,
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
        amp=args.amp,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
    )

    config_info = cfg.to_dict()
    config_info.update(
        {
            "arch": args.arch,
            "freeze_backbone": args.freeze_backbone,
            "resnet_pretrained": args.resnet_pretrained,
        }
    )

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

    build_model = _build_model_factory(args)
    title_map = {
        "cnn_v1": "猫狗分类 - CNN v1 训练",
        "cnn_v2": "猫狗分类 - CNN v2 训练",
        "resnet18": "猫狗分类 - ResNet18 训练",
        "resnet34": "猫狗分类 - ResNet34 训练",
        "resnet50": "猫狗分类 - ResNet50 训练",
    }
    title = title_map.get(args.arch, "猫狗分类 - PyTorch 训练")

    results = train_binary_classification(cfg, build_model, title)

    total_time = time.time() - start_time
    best_val = results.get("best_val", {})
    best_acc = best_val.get("accuracy")
    ckpt_dir = results.get("ckpt_dir")

    logger.block(
        "训练完成",
        [
            "耗时: {}".format(Logger.format_duration(total_time)),
            "最佳准确率: {:.4f}".format(best_acc) if best_acc is not None else "最佳准确率: -",
            "检查点目录: {}".format(ckpt_dir),
        ],
    )
    logger.info("{} 训练流程完成".format(experiment_name.upper()))

    return results


if __name__ == "__main__":
    main()
