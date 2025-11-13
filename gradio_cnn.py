"""
Gradio 应用 - 猫狗分类 (CNN/ResNet)

功能:
- 自动查找并加载最新的最佳权重 `runs/torch_cnn/*/best.pt`
- 依据权重中的配置与参数字典，自动识别并重建 CNN 或 ResNet18 模型与预处理
- 提供图片上传与实时预测界面，显示类别与置信度

使用:
    python gradio_cnn.py --host 0.0.0.0 --port 7860 --weights runs/torch_cnn/20250101-000000/best.pt
    # 或直接不传 --weights，脚本会自动寻找最新 best.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

import gradio as gr

# 本地模块
from src.models.cnn import create_CatDogCNNv1, create_CatDogCNNv2
from src.models.resnet import create_resnet18


def _auto_find_latest_best(root_dir):
    """在 runs/torch_cnn 下查找最新的 best.pt
    
    参数:
        root_dir: 根目录字符串或 Path，通常为项目根目录
    
    返回:
        Path 或 None: 最新 best.pt 的路径
    """
    root_dir = Path(root_dir)
    runs_dir = root_dir / "runs" / "torch_cnn"
    if not runs_dir.exists():
        return None
    candidates = []
    for sub in runs_dir.iterdir():
        if not sub.is_dir():
            continue
        best_path = sub / "best.pt"
        if best_path.exists():
            candidates.append(best_path)
    if not candidates:
        return None
    # 以目录名的时间戳排序，回退到修改时间
    def _key(p):
        try:
            return p.parent.name
        except Exception:
            return str(p)
    candidates.sort(key=_key, reverse=True)
    return candidates[0]


def _build_transforms(size, use_imagenet_norm):
    """构建与训练一致的测试预处理。
    
    参数:
        size: 图像尺寸（int）
        use_imagenet_norm: 是否使用 ImageNet 标准化（bool）
    
    返回:
        torchvision.transforms.Compose
    """
    tfs = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    if use_imagenet_norm:
        tfs.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(tfs)


def _is_resnet_state(state_dict):
    """基于 state_dict 的键推断是否为 ResNet18 权重。

    参数:
        state_dict: 从 checkpoint 读取的模型参数字典

    返回:
        bool: 是否为 ResNet18 架构
    """
    if not isinstance(state_dict, dict):
        return False
    for k in state_dict.keys():
        if isinstance(k, str) and k.startswith("backbone."):
            return True
    return False


def _create_model(cfg, state_dict, device):
    """依据 checkpoint 信息创建模型（自动识别 CNN 或 ResNet18）。
    
    参数:
        cfg: checkpoint 中保存的配置字典
        state_dict: 模型参数字典（用于推断架构）
        device: 设备
    
    返回:
        已放置到 device 的模型
    """
    dropout = cfg.get("dropout", 0.0)
    use_resnet = bool(cfg.get("use_resnet18", False)) or _is_resnet_state(state_dict)

    if use_resnet:
        # 推理阶段不需要预训练权重，直接构建结构后加载 state_dict
        model = create_resnet18(num_classes=1, pretrained=False, freeze_backbone=True, dropout_p=dropout).to(device)
    else:
        model_version = cfg.get("model_version", "v2")
        if model_version == "v1":
            model = create_CatDogCNNv1(num_classes=1, in_channels=3, dropout_p=dropout).to(device)
        elif model_version == "v2":
            model = create_CatDogCNNv2(num_classes=1, in_channels=3, dropout_p=dropout).to(device)
        else:
            raise ValueError(f"不支持的模型版本: {model_version}")
    model.eval()
    return model


def _remap_resnet_classifier_keys(state_dict):
    """将旧版权重中的 backbone.fc.* 键重映射为 classifier.* 键。

    适配因我们在模型中把 ResNet 的 fc 替换为 Identity，并将分类头命名为 classifier
    导致的历史 checkpoint 键名不一致问题。

    参数:
        state_dict: 原始 state_dict（dict）

    返回:
        新的 state_dict（dict）
    """
    if not isinstance(state_dict, dict):
        return state_dict
    need_remap = any(isinstance(k, str) and k.startswith("backbone.fc.") for k in state_dict.keys())
    if not need_remap:
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("backbone.fc."):
            new_key = k.replace("backbone.fc.", "classifier.")
            new_sd[new_key] = v
        else:
            new_sd[k] = v
    return new_sd


class Predictor:
    """封装模型与预处理的推理器。"""

    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weights_path is None:
            weights_path = _auto_find_latest_best(Path.cwd())
        if weights_path is None:
            raise FileNotFoundError("未找到权重文件。请使用 --weights 指定，或确保 runs/torch_cnn/*/best.pt 存在。")

        weights_path = Path(weights_path)
        payload = torch.load(weights_path, map_location=self.device)

        # 读取训练时配置与参数
        cfg = payload.get("config", {})
        state_dict = payload.get("state_dict", None)
        if state_dict is None:
            raise RuntimeError("checkpoint 不包含 state_dict")
        self.cfg = cfg

        # 构建模型（自动识别 CNN 或 ResNet18）并加载权重
        self.model = _create_model(cfg, state_dict, self.device)
        # 兼容历史 ResNet state_dict 键名
        state_dict = _remap_resnet_classifier_keys(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # 预处理变换
        image_size = int(cfg.get("image_size", 224))
        use_imagenet_norm = bool(cfg.get("normalize_imagenet", False))
        self.transform = _build_transforms(image_size, use_imagenet_norm)

        # 类别信息
        self.class_names = ["cats", "dogs"]

    @torch.inference_mode()
    def predict(self, image):
        """对单张 PIL.Image 进行预测。
        
        返回:
            (label_str, prob_dog, prob_cat)
        """
        img = image.convert("RGB")
        t = self.transform(img)
        # 保障: 若管道未返回张量，则显式转换
        if isinstance(t, Image.Image):
            t = TF.to_tensor(t)
        tensor = t.unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob_dog = torch.sigmoid(logits.squeeze()).item()
        prob_cat = 1.0 - prob_dog
        label = "dogs" if prob_dog >= 0.5 else "cats"
        return label, float(prob_dog), float(prob_cat)


def build_interface(predictor):
    """构建 Gradio 界面。"""

    def _infer(image):
        label, prob_dog, prob_cat = predictor.predict(image)
        conf = prob_dog if label == "dogs" else prob_cat
        probs_map = {"cats": float(prob_cat), "dogs": float(prob_dog)}
        emoji = "🐶" if label == "dogs" else "🐱"
        summary = f"**预测**: {emoji} {('狗' if label=='dogs' else '猫')}  |  **置信度**: {conf:.2%}\n\n" \
                  f"- 狗(dog): {prob_dog:.2%}\n" \
                  f"- 猫(cat): {prob_cat:.2%}"
        return probs_map, summary

    with gr.Blocks(title="猫狗分类 - CNN/ResNet") as demo:
        gr.Markdown("## 🐱🐶 猫狗分类 (PyTorch CNN/ResNet)")
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="上传图片", sources=["upload", "clipboard", "webcam"]) 
                btn = gr.Button("识别")
            with gr.Column():
                probs = gr.Label(label="类别概率", num_top_classes=2)
                summary = gr.Markdown()
        btn.click(_infer, inputs=[image_in], outputs=[probs, summary])
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio 猫狗分类 (CNN)")
    parser.add_argument("--weights", type=str, default=None, help="权重路径 (best.pt)。若不提供，将自动搜索最新 best.pt")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="是否开启 gradio share")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = Predictor(weights_path=args.weights)
    demo = build_interface(predictor)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
