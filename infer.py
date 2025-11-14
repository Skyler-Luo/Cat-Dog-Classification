"""
统一的模型推理脚本。

该脚本支持加载 PyTorch 训练产生的检查点，对单张图片或目录批量执行猫狗分类推理，
并将预测结果导出为 CSV / JSON 文件。
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.utils.inference_utils import (
    collect_image_paths,
    prepare_model,
    run_inference,
    summarize_predictions,
    CLASS_NAMES,
)
from src.data.data_utils import build_transforms
from src.utils.logger import Logger
from tools.reporting import save_predictions_to_csv, save_predictions_to_json


def parse_args():
    """解析命令行参数。
    
    返回:
        argparse.Namespace: 参数对象。
    """
    description = "猫狗分类模型推理脚本"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="输入资源路径，可为图片/目录/TXT/CSV 列表。")
    parser.add_argument("--checkpoint", required=True, help="模型检查点路径，例如 runs/torch_cnn/best.pt。")
    parser.add_argument("--arch", default=None, help="模型架构标识（可选，覆盖检查点配置）。")
    parser.add_argument("--device", default="auto", help="推理设备，例如 cpu / cuda:0，默认自动选择。")
    parser.add_argument("--image-size", type=int, default=None, help="推理图像尺寸，默认读取检查点配置。")
    parser.add_argument("--normalize-imagenet", dest="normalize_imagenet", action="store_true", help="强制使用 ImageNet 归一化。")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批次大小，默认 32。")
    parser.add_argument("--threshold", type=float, default=0.5, help="判断为狗的概率阈值，默认 0.5。")
    parser.add_argument("--output-csv", dest="output_csv", default=None, help="预测结果 CSV 输出路径，默认写入 runs/inference/ 目录。")
    parser.add_argument("--output-json", dest="output_json", default=None, help="额外导出 JSON 结果的路径（可选）。")
    parser.add_argument("--no-recursive", action="store_true", help="处理目录时不递归搜索子目录。")
    parser.add_argument("--quiet", action="store_true", help="关闭进度条与详细日志。")
    return parser.parse_args()


def _resolve_device(device_arg):
    """根据参数解析推理设备标识。"""
    if device_arg and device_arg.lower() != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_csv_path(csv_path):
    """生成 CSV 输出路径。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path("runs") / "inference"
    base_dir.mkdir(parents=True, exist_ok=True)
    if csv_path is None:
        csv_path = base_dir / "predictions_{}.csv".format(timestamp)
    return str(Path(csv_path).expanduser())


def _log_summary(logger, summary):
    """记录推理摘要信息。"""
    total = summary["total"]
    cats = summary["cats"]
    dogs = summary["dogs"]
    logger.block(
        "📊 推理统计",
        [
            "总任务数: {}".format(total),
            "猫预测: {} 张 | 平均置信度 {:.2%}".format(cats["count"], cats["avg_prob"]),
            "狗预测: {} 张 | 平均置信度 {:.2%}".format(dogs["count"], dogs["avg_prob"]),
        ],
    )


def main():
    """脚本主入口。"""
    args = parse_args()
    logger = Logger(name="infer")
    device = _resolve_device(args.device)
    logger.block(
        "🚀 推理启动",
        [
            "设备: {}".format(device.upper()),
            "检查点: {}".format(args.checkpoint),
        ],
    )
    model, config = prepare_model(args.checkpoint, device=device, arch=args.arch)
    if args.image_size is not None:
        image_size = args.image_size
    else:
        image_size = config.get("image_size", 224)
    normalize_imagenet = config.get("normalize_imagenet", False)
    if args.normalize_imagenet is not None:
        normalize_imagenet = args.normalize_imagenet
    transform = build_transforms(
        size=image_size,
        augment=False,
        use_imagenet_norm=normalize_imagenet,
    )
    logger.block(
        "🧮 推理配置",
        [
            "图像尺寸: {}x{}".format(image_size, image_size),
            "标准化: {}".format("ImageNet" if normalize_imagenet else "默认 [0,1]"),
            "批次大小: {}".format(args.batch_size),
            "概率阈值: {:.2f}".format(args.threshold),
        ],
    )
    recursive = not args.no_recursive
    logger.info("开始收集图像资源: %s", args.input)
    image_paths = collect_image_paths(args.input, recursive=recursive)
    logger.info("收集完成，共 %d 张图像。", len(image_paths))
    results = run_inference(
        model,
        image_paths,
        transform,
        device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        show_progress=not args.quiet,
        logger=logger.raw,
    )
    summary = summarize_predictions(results)
    _log_summary(logger, summary)
    csv_path = _prepare_csv_path(args.output_csv)
    saved_csv = save_predictions_to_csv(results, csv_path)
    logger.info("💾 已保存 CSV 结果: %s", saved_csv)
    if args.output_json is not None:
        saved_json = save_predictions_to_json(results, summary, args.output_json, class_names=CLASS_NAMES)
        logger.info("💾 已保存 JSON 结果: %s", saved_json)
    else:
        saved_json = None
    if saved_json is None and args.output_json is None:
        logger.info("ℹ️ 如需 JSON 结果，可使用 --output-json 指定输出路径。")
    logger.info("✨ 推理流程完成。")


if __name__ == "__main__":
    main()
