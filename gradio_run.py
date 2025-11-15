"""
Gradio 应用 - 猫狗分类

提供基于 Web 的图像分类推理界面，支持单张图片上传和实时预测。
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr

from src.utils.inference_utils import prepare_model, CLASS_NAMES
from src.data.data_utils import build_transforms


class CatDogClassifier:
    """猫狗分类器封装类，用于 Gradio 应用。"""
    
    def __init__(self, checkpoint_path=None, device=None, arch=None):
        """初始化分类器。
        
        参数:
            checkpoint_path: 模型检查点路径（可选，如果为 None 则延迟加载）。
            device: 推理设备（str，可选），默认自动选择。
            arch: 模型架构名称（str，可选），用于覆盖检查点配置。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.model = None
        self.config = None
        self.checkpoint_path = None
        self.image_size = 224
        self.normalize_imagenet = False
        self.transform = None
        self.class_names = CLASS_NAMES
        
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path):
        """加载或重新加载模型。
        
        参数:
            checkpoint_path: 模型检查点路径。
        """
        self.checkpoint_path = checkpoint_path
        self.model, self.config = prepare_model(checkpoint_path, device=self.device, arch=self.arch)
        
        # 从配置中读取推理参数
        self.image_size = self.config.get("image_size", 224)
        self.normalize_imagenet = self.config.get("normalize_imagenet", False)
        
        # 构建图像变换
        self.transform = build_transforms(
            size=self.image_size,
            augment=False,
            use_imagenet_norm=self.normalize_imagenet,
        )
    
    def predict(self, image, threshold=0.5):
        """对单张图片进行预测。
        
        参数:
            image: PIL.Image 对象或 numpy 数组。
            threshold: 判断为狗的概率阈值（float，默认: 0.5）。
            
        返回:
            dict: 包含预测结果的字典，包含 label、label_name、prob、prob_dog、prob_cat。
        """
        if image is None:
            return None
        
        if self.model is None or self.transform is None:
            return {"error": "模型未加载，请先选择模型"}
        
        # 确保是 PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # 转换为 RGB
        image = image.convert("RGB")
        
        # 预处理
        tensor = self.transform(image)  # 返回 torch.Tensor
        tensor = tensor.unsqueeze(0)  # 添加 batch 维度
        tensor = tensor.to(self.device)
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            prob_dog = torch.sigmoid(logits.view(-1)).item()
        
        prob_cat = 1.0 - prob_dog
        label = 1 if prob_dog >= threshold else 0
        label_name = self.class_names[label]
        
        return {
            "label": label,
            "label_name": label_name,
            "prob": prob_dog,
            "prob_dog": prob_dog,
            "prob_cat": prob_cat,
        }
    
    def predict_with_display(self, image, threshold=0.5):
        """预测并格式化显示结果。
        
        参数:
            image: PIL.Image 对象或 numpy 数组。
            threshold: 判断为狗的概率阈值（float，默认: 0.5）。
            
        返回:
            tuple: (显示文本, 类别名称, 置信度字典, 模型信息)
        """
        result = self.predict(image, threshold)
        if result is None:
            return "❌ 请上传一张图片", None, None, None
        
        if "error" in result:
            return "❌ {}".format(result["error"]), None, None, None
        
        label_name = result["label_name"]
        prob_dog = result["prob_dog"]
        prob_cat = result["prob_cat"]
        
        # 选择 emoji
        emoji = "🐱" if label_name == "cats" else "🐶"
        
        # 模型信息
        model_info = ""
        if self.checkpoint_path:
            model_name = Path(self.checkpoint_path).name
            model_info = f"**当前模型:** `{model_name}`\n"
        
        # 格式化显示文本
        display_text = f"""
{emoji} **预测结果: {label_name.upper()}**

{model_info}📊 **置信度:**
- 🐱 猫: {prob_cat:.2%}
- 🐶 狗: {prob_dog:.2%}

🎯 **阈值:** {threshold:.2f}
"""
        
        # 置信度字典用于可视化（Label 组件格式）
        confidence_dict = {
            "cats": float(prob_cat),
            "dogs": float(prob_dog),
        }
        
        return display_text, label_name, confidence_dict, model_info


def scan_weights_folder(weights_dir):
    """扫描 weights 文件夹，查找所有 .pt 文件。
    
    参数:
        weights_dir: weights 文件夹路径。
        
    返回:
        list: (文件路径, 显示名称) 元组列表，按文件名排序。
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        return []
    
    weight_files = []
    for pt_file in weights_path.glob("*.pt"):
        weight_files.append((str(pt_file.resolve()), pt_file.name))
    
    # 按文件名排序
    weight_files.sort(key=lambda x: x[1])
    return weight_files


def create_gradio_interface(classifier, model_choices, default_model=None, threshold=0.5):
    """创建 Gradio 界面。
    
    参数:
        classifier: CatDogClassifier 实例。
        model_choices: 模型选择列表，格式为 [(路径, 显示名), ...]。
        default_model: 默认选择的模型路径（可选）。
        threshold: 默认概率阈值（float，默认: 0.5）。
        
    返回:
        gr.Blocks: Gradio 界面对象。
    """
    # 初始化默认模型信息显示
    initial_model_info = ""
    if classifier.model is not None and classifier.checkpoint_path:
        # 模型已经加载（在 main 函数中加载的）
        model_name = Path(classifier.checkpoint_path).name
        initial_model_info = f"✅ 模型已加载: `{model_name}`\n\n📐 图像尺寸: {classifier.image_size}x{classifier.image_size}\n🎨 标准化: {'ImageNet' if classifier.normalize_imagenet else '默认 [0,1]'}"
    elif default_model and model_choices:
        # 尝试加载默认模型
        try:
            classifier.load_model(default_model)
            model_name = Path(default_model).name
            initial_model_info = f"✅ 模型已加载: `{model_name}`\n\n📐 图像尺寸: {classifier.image_size}x{classifier.image_size}\n🎨 标准化: {'ImageNet' if classifier.normalize_imagenet else '默认 [0,1]'}"
        except Exception as e:
            initial_model_info = "⚠️ 默认模型加载失败: {}".format(str(e))
    
    def predict_fn(image, threshold_value, model_path):
        """Gradio 预测函数。"""
        # 如果模型路径改变，先加载模型
        if model_path and model_path != classifier.checkpoint_path:
            try:
                classifier.load_model(model_path)
            except Exception as e:
                return "❌ 模型加载失败: {}".format(str(e)), None, None, None
        
        result = classifier.predict(image, threshold_value)
        if result is None:
            return """
            <div style="text-align: center; padding: 40px; color: #f44336;">
                <h3>❌ 请上传一张图片</h3>
                <p>请在上传区域选择或拖拽一张图片</p>
            </div>
            """, None, None, None
        
        if "error" in result:
            return "❌ {}".format(result["error"]), None, None, None
        
        label_name = result["label_name"]
        prob_dog = result["prob_dog"]
        prob_cat = result["prob_cat"]
        
        # 选择 emoji
        emoji = "🐱" if label_name == "cats" else "🐶"
        
        # 模型信息
        model_info = ""
        if classifier.checkpoint_path:
            model_name = Path(classifier.checkpoint_path).name
            model_info = f"**当前模型:** `{model_name}`\n"
        
        # 格式化显示文本（更美观的格式）
        confidence_color_cat = "#4CAF50" if prob_cat > 0.5 else "#757575"
        confidence_color_dog = "#4CAF50" if prob_dog > 0.5 else "#757575"
        
        display_text = f"""
<div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; text-align: center; margin-bottom: 15px;">
    <h2 style="margin: 0; font-size: 32px;">{emoji}</h2>
    <h3 style="margin: 10px 0; font-size: 24px;">预测结果: {label_name.upper()}</h3>
</div>

{model_info}

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
    <h4 style="margin-top: 0;">📊 置信度分析</h4>
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 10px 0;">
        <span style="font-size: 18px;">🐱 猫:</span>
        <span style="font-weight: bold; color: {confidence_color_cat}; font-size: 20px;">{prob_cat:.2%}</span>
    </div>
    <div style="background-color: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden; margin: 5px 0;">
        <div style="background-color: {confidence_color_cat}; height: 100%; width: {prob_cat*100}%;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 10px 0;">
        <span style="font-size: 18px;">🐶 狗:</span>
        <span style="font-weight: bold; color: {confidence_color_dog}; font-size: 20px;">{prob_dog:.2%}</span>
    </div>
    <div style="background-color: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden; margin: 5px 0;">
        <div style="background-color: {confidence_color_dog}; height: 100%; width: {prob_dog*100}%;"></div>
    </div>
</div>

<div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 10px;">
    <strong>🎯 当前阈值:</strong> {threshold_value:.2f}
</div>
"""
        
        # 置信度字典用于可视化
        confidence_dict = {
            "cats": float(prob_cat),
            "dogs": float(prob_dog),
        }
        
        return display_text, label_name, confidence_dict, None
    
    # 准备模型选择列表
    model_options = ["请选择模型"] + [name for _, name in model_choices]
    model_paths = {"请选择模型": None}
    for path, name in model_choices:
        model_paths[name] = path
    
    # 默认选择
    default_choice = "请选择模型"
    if default_model and model_choices:
        for path, name in model_choices:
            if path == default_model:
                default_choice = name
                break
    
    with gr.Blocks(title="🐱🐶 猫狗分类器") as demo:
        # 标题区域
        with gr.Row():
            gr.Markdown(
                """
                <div style="text-align: center;">
                    <h1 style="margin-bottom: 10px;">🐱🐶 猫狗图像分类器</h1>
                    <p style="font-size: 16px; color: #666;">上传图片，AI 自动识别是猫还是狗！</p>
                </div>
                """
            )
        
        gr.Markdown("---")
        
        # 主要内容区域
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1, min_width=400):
                # 模型选择区域（移到左侧上方）
                with gr.Group():
                    gr.Markdown("### 🤖 模型选择")
                    model_dropdown = gr.Dropdown(
                        choices=model_options,
                        value=default_choice,
                        label="选择模型",
                        info="从 weights 文件夹中选择要使用的模型",
                    )
                    model_info_text = gr.Markdown(
                        label="模型信息",
                        value=initial_model_info,
                        elem_classes=["model-info"],
                    )
                
                with gr.Group():
                    gr.Markdown("### 📤 输入区域")
                    image_input = gr.Image(
                        type="pil",
                        label="📷 上传图片",
                        height=350,
                    )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ 参数设置")
                    threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=threshold,
                        step=0.01,
                        label="🎯 概率阈值",
                        info="调整判断为狗的概率阈值（默认: 0.5）",
                    )
                    predict_btn = gr.Button(
                        "🔮 开始预测",
                        variant="primary",
                        size="lg",
                    )
            
            # 右侧：结果区域
            with gr.Column(scale=1, min_width=400):
                with gr.Group():
                    gr.Markdown("### 📊 预测结果")
                    result_text = gr.Markdown(
                        value="""
                        <div style="text-align: center; padding: 40px; color: #999;">
                            <h3>📤 等待上传图片</h3>
                            <p>请在上传区域选择或拖拽一张图片</p>
                        </div>
                        """,
                        elem_classes=["result-text"],
                    )
                
                with gr.Group():
                    gr.Markdown("### 📈 详细分析")
                    result_label = gr.Textbox(
                        label="🏷️ 预测类别",
                        interactive=False,
                    )
                    confidence_plot = gr.Label(
                        label="📊 置信度分布",
                        num_top_classes=2,
                    )
        
        gr.Markdown("---")
        
        # 模型选择事件
        def on_model_change(model_name):
            """模型选择改变时的回调。"""
            if not model_name or model_name == "请选择模型":
                return "⚠️ 请先选择一个模型"
            
            model_path = model_paths.get(model_name)
            if not model_path:
                return "❌ 未找到模型路径"
            
            try:
                classifier.load_model(model_path)
                model_name_display = Path(model_path).name
                info_text = f"✅ 模型加载成功: `{model_name_display}`\n\n📐 图像尺寸: {classifier.image_size}x{classifier.image_size}\n🎨 标准化: {'ImageNet' if classifier.normalize_imagenet else '默认 [0,1]'}"
                return info_text
            except Exception as e:
                error_msg = "❌ 模型加载失败: {}".format(str(e))
                return error_msg
        
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[model_info_text],
        )
        
        # 预测事件
        def predict_with_model(image, threshold_value, model_name):
            """带模型选择的预测函数。"""
            if model_name and model_name != "请选择模型":
                model_path = model_paths.get(model_name)
                if model_path and model_path != classifier.checkpoint_path:
                    try:
                        classifier.load_model(model_path)
                    except Exception as e:
                        return "❌ 模型加载失败: {}".format(str(e)), None, None, None
            
            return predict_fn(image, threshold_value, None)
        
        # 绑定事件
        predict_btn.click(
            fn=predict_with_model,
            inputs=[image_input, threshold_slider, model_dropdown],
            outputs=[result_text, result_label, confidence_plot, model_info_text],
        )
        
        # 自动预测（当图片上传时）
        image_input.change(
            fn=predict_with_model,
            inputs=[image_input, threshold_slider, model_dropdown],
            outputs=[result_text, result_label, confidence_plot, model_info_text],
        )
        
        # 阈值改变时重新预测
        threshold_slider.change(
            fn=predict_with_model,
            inputs=[image_input, threshold_slider, model_dropdown],
            outputs=[result_text, result_label, confidence_plot, model_info_text],
        )
        
        # 底部提示区域
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 4px solid #4a90e2;">
                        <h4 style="margin-top: 0; color: #4a90e2;">💡 使用提示</h4>
                        <ul style="margin-bottom: 0;">
                            <li>支持常见图片格式：<strong>JPG、PNG、BMP、GIF、WEBP</strong></li>
                            <li>建议上传<strong>清晰的猫或狗的照片</strong>以获得最佳效果</li>
                            <li>调整阈值可以改变分类的严格程度</li>
                            <li>切换模型后会自动重新预测</li>
                        </ul>
                    </div>
                    """
                )
    
    return demo


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="猫狗分类 Gradio Web 应用")
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="权重文件夹路径（默认: weights）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="默认模型检查点路径（可选，如果指定则作为初始模型）",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="模型架构标识（可选，覆盖检查点配置）",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备，例如 cpu / cuda:0，默认自动选择",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="默认概率阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="服务器监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="服务器端口（默认: 7860）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公共链接（通过 Gradio 分享）",
    )
    return parser.parse_args()


def _resolve_device(device_arg):
    """根据参数解析推理设备标识。"""
    if device_arg and device_arg.lower() != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """主函数。"""
    args = parse_args()
    device = _resolve_device(args.device)
    
    print("🔍 正在扫描权重文件夹...")
    weights_dir = Path(args.weights_dir)
    
    # 扫描权重文件
    model_choices = scan_weights_folder(weights_dir)
    
    if not model_choices:
        print("⚠️ 警告: 在 '{}' 文件夹中未找到任何 .pt 文件".format(weights_dir))
        print("💡 提示: 请将模型权重文件（.pt）放入 weights 文件夹")
        if args.checkpoint:
            print("📁 使用指定的检查点: {}".format(args.checkpoint))
            model_choices = [(args.checkpoint, Path(args.checkpoint).name)]
        else:
            print("❌ 错误: 没有可用的模型文件")
            return
    else:
        print("✅ 找到 {} 个模型文件:".format(len(model_choices)))
        for path, name in model_choices:
            print("   - {}".format(name))
    
    print("\n💻 设备: {}".format(device.upper()))
    
    # 确定默认模型
    default_model = None
    if args.checkpoint:
        # 检查指定的检查点是否在列表中
        checkpoint_path = Path(args.checkpoint).resolve()
        for path, name in model_choices:
            if Path(path).resolve() == checkpoint_path:
                default_model = path
                break
        if default_model is None:
            # 如果不在列表中，添加到列表
            model_choices.insert(0, (str(checkpoint_path), checkpoint_path.name))
            default_model = str(checkpoint_path)
    elif model_choices:
        # 如果没有指定，使用第一个
        default_model = model_choices[0][0]
    
    # 初始化分类器（延迟加载，不在初始化时加载模型）
    classifier = CatDogClassifier(
        checkpoint_path=None,  # 延迟加载
        device=device,
        arch=args.arch,
    )
    
    # 如果有默认模型，先加载它
    if default_model:
        print("\n🚀 正在加载默认模型...")
        print("📁 检查点: {}".format(default_model))
        try:
            classifier.load_model(default_model)
            print("✅ 模型加载完成！")
            print("📐 图像尺寸: {}x{}".format(classifier.image_size, classifier.image_size))
            print("🎨 标准化: {}".format("ImageNet" if classifier.normalize_imagenet else "默认 [0,1]"))
        except Exception as e:
            print("❌ 默认模型加载失败: {}".format(e))
            print("⚠️ 将在界面中选择模型")
    
    # 创建 Gradio 界面
    demo = create_gradio_interface(
        classifier,
        model_choices=model_choices,
        default_model=default_model,
        threshold=args.threshold,
    )
    
    # 启动服务
    print("\n🌐 正在启动 Web 服务...")
    print("📍 访问地址: http://{}:{}".format(args.server_name, args.server_port))
    if args.share:
        print("🔗 公共链接将在启动后显示")
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

