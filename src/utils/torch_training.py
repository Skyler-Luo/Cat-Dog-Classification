"""PyTorch 训练核心工具与完整训练管线合集。"""

from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from src.data.data_utils import load_data
from .logger import Logger
from .config import TrainConfig


__all__ = [
    "EarlyStopping",
    "build_optimizer",
    "build_scheduler",
    "save_checkpoint",
    "build_bce_with_logits",
    "set_global_seed",
    "compute_binary_metrics",
    "train_one_epoch",
    "evaluate",
    "count_binary_labels",
    "log_epoch_scalars",
    "train_binary_classification",
    "log_tensorboard_images",
    "log_confusion_matrix",
]


class EarlyStopping:
    """早停机制实现类。"""

    def __init__(self, patience=10, mode="min"):
        import math

        self.patience = patience
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad = 0
        self.should_stop = False

    def step(self, value):
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_optimizer(model, cfg):
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError("不支持的优化器: {}".format(cfg.optimizer))


def build_scheduler(optimizer, cfg):
    name = cfg.scheduler.lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )
    raise ValueError("不支持的调度器: {}".format(cfg.scheduler))


def save_checkpoint(model, optimizer, epoch, metrics, cfg, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": cfg.to_dict(),
    }
    torch.save(payload, save_path)


def build_bce_with_logits(neg_count, pos_count, device):
    import torch.nn as nn

    pos_weight = None
    if pos_count > 0 and neg_count > 0:
        pos_weight = torch.tensor(
            [neg_count / max(pos_count, 1)],
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion, pos_weight


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(logits, targets):
    probs = torch.sigmoid(logits)
    preds_flat = (probs >= 0.5).to(torch.long).view(-1)
    t_flat = targets.to(torch.long).view(-1)
    correct = (preds_flat == t_flat).sum().item()
    total = t_flat.numel()
    acc = correct / max(total, 1)
    tp = ((preds_flat == 1) & (t_flat == 1)).sum().item()
    tn = ((preds_flat == 0) & (t_flat == 0)).sum().item()
    fp = ((preds_flat == 1) & (t_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (t_flat == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, epoch=None, epochs=None):
    model.train()
    loss_sum = 0.0
    correct_sum = 0
    total = 0
    tp_sum = 0
    fp_sum = 0
    tn_sum = 0
    fn_sum = 0
    desc = "训练 [{} / {}]".format(epoch, epochs) if (epoch is not None and epochs is not None) else "训练"
    pbar = tqdm(loader, desc=desc, ncols=100, leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            metrics = compute_binary_metrics(logits, targets)
            batch = targets.size(0)
            loss_sum += loss.item() * batch
            correct_sum += metrics["correct"]
            total += batch
            tp_sum += metrics["tp"]
            fp_sum += metrics["fp"]
            tn_sum += metrics["tn"]
            fn_sum += metrics["fn"]
            pbar.set_postfix(
                {
                    "loss": "{:.4f}".format(loss.item()),
                    "acc": "{:.2f}%".format(metrics["accuracy"] * 100),
                    "f1": "{:.3f}".format(metrics["f1"]),
                }
            )
    avg_loss = loss_sum / max(total, 1)
    accuracy = correct_sum / max(total, 1)
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion, desc=None):
    model.eval()
    loss_sum = 0.0
    correct_sum = 0
    total = 0
    tp_sum = 0
    fp_sum = 0
    tn_sum = 0
    fn_sum = 0
    pbar = tqdm(loader, desc=(desc or "评估"), ncols=100, leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().view(-1, 1)
        logits = model(images)
        loss = criterion(logits, targets)
        metrics = compute_binary_metrics(logits, targets)
        batch = targets.size(0)
        loss_sum += loss.item() * batch
        correct_sum += metrics["correct"]
        total += batch
        tp_sum += metrics["tp"]
        fp_sum += metrics["fp"]
        tn_sum += metrics["tn"]
        fn_sum += metrics["fn"]
        pbar.set_postfix(
            {
                "loss": "{:.4f}".format(loss.item()),
                "acc": "{:.2f}%".format(metrics["accuracy"] * 100),
                "f1": "{:.3f}".format(metrics["f1"]),
            }
        )
    avg_loss = loss_sum / max(total, 1)
    accuracy = correct_sum / max(total, 1)
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def count_binary_labels(loader, device):
    num_pos = 0
    num_neg = 0
    for _, targets in loader:
        t = targets.to(device, non_blocking=True).view(-1).float()
        num_pos += (t == 1).sum().item()
        num_neg += (t == 0).sum().item()
    return num_neg, num_pos


def log_epoch_scalars(writer, epoch, train_metrics, val_metrics, current_lr):
    all_metrics = {
        "1_train_loss": train_metrics.get("loss", 0),
        "2_val_loss": val_metrics.get("loss", 0),
        "3_train_accuracy": train_metrics.get("accuracy", 0),
        "4_val_accuracy": val_metrics.get("accuracy", 0),
        "99_learning_rate": current_lr,
    }
    metric_names = ["precision", "recall", "f1"]
    for i, metric_name in enumerate(metric_names, start=5):
        if metric_name in train_metrics:
            all_metrics["{}_train_{}".format(i, metric_name)] = train_metrics[metric_name]
        if metric_name in val_metrics:
            all_metrics["{}_val_{}".format(i + 3, metric_name)] = val_metrics[metric_name]
    writer.add_scalars("Training_Summary", all_metrics, epoch)


def train_binary_classification(cfg, build_model_fn, title):
    set_global_seed(cfg.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tb_dir = Path(cfg.log_dir) / timestamp
    ckpt_run_dir = Path(cfg.ckpt_dir) / timestamp
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(name="catdog", log_dir=str(tb_dir))

    logger.block(title)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = [
        "计算设备: {}".format(device.type.upper()),
    ]
    if device.type == "cuda":
        try:
            device_info.append("GPU型号: {}".format(torch.cuda.get_device_name(0)))
            device_info.append(
                "GPU内存: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3)
            )
        except Exception:
            pass
    device_info.append("混合精度(AMP): {}".format("启用" if (cfg.amp and device.type == "cuda") else "关闭"))
    logger.block("运行环境", device_info)

    logger.info("\n正在加载数据集...")
    if cfg.data_dir is None:
        raise ValueError("数据集目录路径不能为空")
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError("数据集目录不存在: {}".format(data_dir))

    train_loader, val_loader, test_loader, stats = load_data(
        data_dir=str(data_dir),
        train_dir=cfg.train_dirname,
        val_dir=cfg.val_dirname,
        test_dir=cfg.test_dirname,
        size=cfg.image_size,
        batch_size=cfg.batch_size,
        train_augment=True,
        use_imagenet_norm=cfg.normalize_imagenet,
        num_workers=cfg.num_workers,
    )

    n_train = stats["n_train"]
    n_val = stats["n_val"]
    n_test = stats["n_test"]
    steps_per_epoch = len(train_loader)
    dataset_info = [
        "训练样本数: {:,} ({:.1f}%)".format(n_train, n_train / (n_train + n_val) * 100),
        "验证样本数: {:,} ({:.1f}%)".format(n_val, n_val / (n_train + n_val) * 100),
        "测试样本数: {:,}".format(n_test) if n_test > 0 else "测试样本数: 无",
        "类别数量: {} ({})".format(stats["num_classes"], ", ".join(stats["class_names"])),
        "每轮步数: {}".format(steps_per_epoch),
    ]
    logger.block("数据集信息", dataset_info)

    built = build_model_fn(device)
    if isinstance(built, tuple):
        model, arch_name = built[0], built[1]
    else:
        model, arch_name = built, "Model"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = [
        "模型架构: {}".format(arch_name),
        "参数总量: {:,}".format(total_params),
        "可训练参数: {:,}".format(trainable_params),
        "图像尺寸: {}x{}".format(cfg.image_size, cfg.image_size),
        "Dropout: {}".format(cfg.dropout),
        "标准化: {}".format("ImageNet" if cfg.normalize_imagenet else "默认[0,1]"),
    ]
    logger.block("模型配置", model_info)

    logger.info("\n计算数据集类别分布...")
    num_neg, num_pos = count_binary_labels(train_loader, device)
    criterion, pos_weight = build_bce_with_logits(num_neg, num_pos, device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    train_info = [
        "训练轮数: {}".format(cfg.epochs),
        "批次大小: {}".format(cfg.batch_size),
        "优化器: {}".format(cfg.optimizer.upper()),
        "初始学习率: {}".format(cfg.lr),
        "权重衰减: {}".format(cfg.weight_decay),
        "学习率调度: {}".format(cfg.scheduler),
        "早停耐心值: {}".format(cfg.early_stop_patience),
        "工作进程数: {}".format(cfg.num_workers),
        "类别权重: cats={:.2f}, dogs={:.2f}".format(1.0, pos_weight.item() if pos_weight is not None else 1.0),
    ]
    logger.block("训练配置", train_info)

    writer = SummaryWriter(log_dir=str(tb_dir))
    ckpt_best_path = ckpt_run_dir / "best.pt"
    ckpt_last_path = ckpt_run_dir / "last.pt"
    with open(ckpt_run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("\nTensorBoard日志目录: {}".format(tb_dir))
    logger.info("模型保存目录: {}".format(ckpt_run_dir))
    logger.info("训练日志: {}".format(tb_dir / "train.log"))

    try:
        dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size).to(device)
        writer.add_graph(model, dummy_input)
    except Exception as exc:
        logger.warning("无法记录模型结构: {}".format(exc))

    logger.block("开始训练")
    early = EarlyStopping(patience=cfg.early_stop_patience, mode="max")
    best_val = {"loss": float("inf"), "accuracy": 0.0, "epoch": 0}
    train_start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start_time = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            epoch=epoch,
            epochs=cfg.epochs,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            desc="验证 [{:>3d}/{}]".format(epoch, cfg.epochs),
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
            next_lr = optimizer.param_groups[0]["lr"]
        else:
            next_lr = current_lr

        log_epoch_scalars(writer, epoch, train_metrics, val_metrics, current_lr)
        if epoch % 10 == 0 or epoch == 1:
            log_tensorboard_images(writer, model, val_loader, device, epoch, num_images=8)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            dict(train_metrics, **{"val_{}".format(k): v for k, v in val_metrics.items()}),
            cfg,
            ckpt_last_path,
        )

        improved = False
        if val_metrics["accuracy"] > best_val["accuracy"]:
            best_val = {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "epoch": epoch,
            }
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val": val_metrics},
                cfg,
                ckpt_best_path,
            )
            improved = True

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - train_start_time
        avg_epoch_time = elapsed_time / epoch
        eta = avg_epoch_time * (cfg.epochs - epoch)

        logger.info(
            "Epoch {:>3d}/{} | 耗时: {} | ETA: {}".format(
                epoch,
                cfg.epochs,
                Logger.format_duration(epoch_time),
                Logger.format_duration(eta),
            )
        )
        logger.info(
            "   训练 - Loss: {:.4f} | Acc: {:.2f}% | Prec: {:.3f} | Rec: {:.3f} | F1: {:.3f}".format(
                train_metrics["loss"],
                train_metrics["accuracy"] * 100,
                train_metrics.get("precision", 0.0),
                train_metrics.get("recall", 0.0),
                train_metrics.get("f1", 0.0),
            )
        )
        logger.info(
            "   验证 - Loss: {:.4f} | Acc: {:.2f}% | Prec: {:.3f} | Rec: {:.3f} | F1: {:.3f} {}".format(
                val_metrics["loss"],
                val_metrics["accuracy"] * 100,
                val_metrics.get("precision", 0.0),
                val_metrics.get("recall", 0.0),
                val_metrics.get("f1", 0.0),
                "[最佳]" if improved else "",
            )
        )
        logger.info("   学习率: {:.6f} -> {:.6f}".format(current_lr, next_lr))

        if early.step(val_metrics["accuracy"]):
            logger.info("\n触发早停 (连续 {} 轮无改善)".format(cfg.early_stop_patience))
            break

    total_train_time = time.time() - train_start_time
    logger.block("训练完成")
    summary = [
        "总耗时: {}".format(Logger.format_duration(total_train_time)),
        "平均每轮: {}".format(Logger.format_duration(total_train_time / epoch)),
        "完成轮数: {}/{}".format(epoch, cfg.epochs),
        "最佳验证: Epoch {} - Loss: {:.4f}, Acc: {:.2f}%".format(
            best_val["epoch"], best_val["loss"], best_val["accuracy"] * 100
        ),
    ]
    logger.block("训练总结", summary)

    if test_loader is not None:
        logger.info("\n正在测试集上评估最佳模型...")
        try:
            payload = torch.load(ckpt_best_path, map_location=device)
            model.load_state_dict(payload["state_dict"])
            logger.info("   已加载最佳模型 (Epoch {})".format(best_val["epoch"]))
        except Exception as exc:
            logger.warning("   加载失败，使用当前模型: {}".format(exc))

        test_metrics = evaluate(model, test_loader, device, criterion, desc="测试")
        test_info = [
            "测试样本数: {:,}".format(n_test),
            "测试损失: {:.4f}".format(test_metrics["loss"]),
            "测试准确率: {:.2f}%".format(test_metrics["accuracy"] * 100),
            "测试精确率: {:.3f}".format(test_metrics.get("precision", 0.0)),
            "测试召回率: {:.3f}".format(test_metrics.get("recall", 0.0)),
            "测试F1: {:.3f}".format(test_metrics.get("f1", 0.0)),
        ]
        logger.block("测试结果", test_info)

        writer.add_scalar("Test/Loss", test_metrics["loss"], 0)
        writer.add_scalar("Test/Accuracy", test_metrics["accuracy"], 0)
        if "precision" in test_metrics:
            writer.add_scalar("Test/Precision", test_metrics["precision"], 0)
        if "recall" in test_metrics:
            writer.add_scalar("Test/Recall", test_metrics["recall"], 0)
        if "f1" in test_metrics:
            writer.add_scalar("Test/F1", test_metrics["f1"], 0)

    writer.close()
    logger.info("\n所有任务完成!")
    logger.info("   最佳模型: {}".format(ckpt_best_path))
    logger.info("   最后模型: {}".format(ckpt_last_path))
    logger.info("   TensorBoard: tensorboard --logdir={}\n".format(cfg.log_dir))

    return {
        "best_val": best_val,
        "tb_dir": str(tb_dir),
        "ckpt_dir": str(ckpt_run_dir),
    }


def log_tensorboard_images(writer, model, loader, device, epoch, num_images=8):
    model.eval()
    images_logged = 0
    with torch.no_grad():
        for images, targets in loader:
            if images_logged >= num_images:
                break
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            for i in range(min(images.size(0), num_images - images_logged)):
                img = images[i]
                target = int(targets[i].item())
                pred = int(preds[i].item())
                prob = probs[i].item()
                writer.add_image(
                    "预测样本/{}_真实_{}_预测_{}_{}".format(
                        "✓" if pred == target else "✗",
                        target,
                        pred,
                        "{:.2f}".format(prob),
                    ),
                    img,
                    epoch,
                )
                images_logged += 1
            if images_logged >= num_images:
                break


def log_confusion_matrix(writer, model, loader, device, epoch, class_names=("cat", "dog"), normalize=False):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().view(-1)
            y_true.extend(targets.view(-1).long().cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_display = cm.astype(np.float32)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True) + 1e-12
        cm_display = cm_display / row_sums
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    im = ax.imshow(cm_display, cmap=plt.get_cmap("Blues"))
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(2):
        for j in range(2):
            value = cm_display[i, j]
            text = "{:.2f}".format(value) if normalize else "{}".format(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    writer.add_figure("Confusion_Matrix", fig, global_step=epoch)
    plt.close(fig)
