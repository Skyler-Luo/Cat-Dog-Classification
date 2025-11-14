# 🐱🐶 Cat-Dog Classification

一个基于机器学习和深度学习的猫狗图像分类项目，使用 Python、PyTorch、scikit-learn 等框架实现。

## 📑 目录

- [📋 项目概述](#-项目概述)
- [✨ 主要特性](#-主要特性)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
  - [系统要求](#系统要求)
  - [环境配置](#环境配置)
    - [1. 克隆项目](#1-克隆项目)
    - [2. 创建虚拟环境（推荐）](#2-创建虚拟环境推荐)
    - [3. 安装依赖](#3-安装依赖)
    - [4. 验证安装](#4-验证安装)
  - [数据集准备](#数据集准备)
    - [数据集结构](#数据集结构)
    - [数据集要求](#数据集要求)
    - [数据集统计](#数据集统计)
  - [特征搜索](#特征搜索)
  - [训练模型](#训练模型)
    - [深度学习模型训练（PyTorch）](#深度学习模型训练pytorch)
    - [传统机器学习模型训练（sklearn）](#传统机器学习模型训练sklearn)
    - [集成学习](#集成学习)
  - [推理预测](#推理预测)
  - [Web 界面](#web-界面)
- [📊 模型训练结果](#-模型训练结果)
  - [🎯 深度学习模型（PyTorch）](#-深度学习模型pytorch)
  - [📈 传统机器学习模型（scikit-learn）](#-传统机器学习模型scikit-learn)
  - [🎯 集成学习结果](#-集成学习结果)
- [🔧 技术栈](#-技术栈)
- [📈 性能对比](#-性能对比)
  - [最佳模型排名](#最佳模型排名)
  - [模型特点](#模型特点)
- [🎨 可视化](#-可视化)
  - [自动生成的可视化](#自动生成的可视化)
  - [TensorBoard 可视化](#tensorboard-可视化)
  - [可视化示例](#可视化示例)
- [🏗️ 模型架构详解](#️-模型架构详解)
  - [深度学习模型](#深度学习模型)
  - [传统机器学习模型](#传统机器学习模型)
- [🔄 数据增强技术](#-数据增强技术)
  - [训练时数据增强](#训练时数据增强)
  - [验证/测试时](#验证测试时)
  - [数据增强效果](#数据增强效果)
- [📝 使用示例](#-使用示例)
  - [完整训练流程示例](#完整训练流程示例)
  - [推理示例](#推理示例)
  - [Python API 使用示例](#python-api-使用示例)
- [🔍 实验记录](#-实验记录)
  - [实验结果分析](#实验结果分析)
- [💡 训练技巧和最佳实践](#-训练技巧和最佳实践)
  - [深度学习模型训练技巧](#深度学习模型训练技巧)
  - [传统机器学习模型技巧](#传统机器学习模型技巧)
- [⚡ 性能优化建议](#-性能优化建议)
  - [训练速度优化](#训练速度优化)
  - [模型性能优化](#模型性能优化)
- [❓ 常见问题解答（FAQ）](#-常见问题解答faq)
  - [安装问题](#安装问题)
  - [训练问题](#训练问题)
  - [推理问题](#推理问题)
  - [其他问题](#其他问题)
- [🐛 故障排除](#-故障排除)
  - [常见错误及解决方案](#常见错误及解决方案)
- [📚 扩展阅读](#-扩展阅读)
  - [相关论文](#相关论文)
  - [相关资源](#相关资源)
- [🔮 未来计划](#-未来计划)
- [🤝 贡献](#-贡献)
  - [贡献指南](#贡献指南)
  - [代码规范](#代码规范)
- [📄 许可证](#-许可证)
- [🙏 致谢](#-致谢)

---

## 📋 项目概述

本项目实现了多种机器学习方法对猫狗图像进行分类，包括：

- 🔥 **深度学习模型**：CNN、ResNet18、ResNet34
- 📊 **传统机器学习模型**：SVM、逻辑回归、随机森林
- 🎯 **集成学习**：投票集成、堆叠集成、AdaBoost

## ✨ 主要特性

- 🚀 支持多种深度学习架构（CNN、ResNet）
- 📈 完整的训练流程（数据增强、早停、学习率调度）
- 📊 丰富的可视化（训练曲线、混淆矩阵、ROC曲线）
- 🎨 基于 Gradio 的 Web 界面
- 💾 模型检查点保存与加载
- 📝 TensorBoard 日志记录
- 🔍 特征工程与特征选择
- 🎯 集成学习提升性能

## 📁 项目结构

```
Cat_Dog_Classification/
├── dataset/                    # 数据集目录
│   ├── train/                 # 训练集
│   │   ├── cats/             # 猫的图片
│   │   └── dogs/             # 狗的图片
│   ├── val/                   # 验证集
│   └── test/                  # 测试集
│
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   │   ├── data_utils.py     # 数据加载和预处理
│   │   ├── feature_extraction.py  # 特征提取
│   │   └── feature_search.py # 特征搜索
│   ├── features/             # 特征提取器
│   │   ├── hog.py            # HOG 特征
│   │   ├── color_hist.py     # 颜色直方图
│   │   ├── lbp.py            # LBP 特征
│   │   ├── glcm.py           # GLCM 纹理特征
│   │   ├── gabor.py          # Gabor 滤波器
│   │   └── ...               # 其他特征提取器
│   ├── models/               # 模型定义
│   │   ├── cnn.py            # CNN 模型
│   │   ├── resnet.py         # ResNet 模型
│   │   ├── svm.py            # SVM 模型
│   │   ├── logistic_regression.py  # 逻辑回归
│   │   └── random_forest.py  # 随机森林
│   └── utils/                # 工具函数
│       ├── config.py         # 配置管理
│       ├── torch_training.py # PyTorch 训练工具
│       ├── ml_training.py    # 传统 ML 训练工具
│       ├── inference_utils.py # 推理工具
│       └── logger.py         # 日志工具
│
├── tools/                     # 工具脚本
│   ├── visualization.py      # 可视化工具
│   ├── reporting.py          # 报告生成
│   └── check_gpu_cuda.py     # GPU 检查
│
├── runs/                      # 训练结果（自动生成）
│   ├── torch_resnet34_*/     # ResNet34 训练结果
│   ├── torch_resnet18_*/     # ResNet18 训练结果
│   ├── torch_cnn-*/          # CNN 训练结果
│   ├── sklearn_*/            # sklearn 模型结果
│   └── ensemble_*/           # 集成学习结果
│
├── weights/                   # 预训练权重（可选）
│
├── train_torch.py             # PyTorch 训练入口
├── train_sklearn.py           # sklearn 训练入口
├── ensemble.py                # 集成学习脚本
├── infer.py                   # 推理脚本
├── gradio_run.py              # Gradio Web 界面
├── requirements.txt           # Python 依赖
├── LICENSE                    # 许可证
└── README.md                  # 项目说明
```

## 🚀 快速开始

### 系统要求

- **Python**: 3.8+
- **操作系统**: Linux / Windows / macOS
- **GPU**: 推荐 NVIDIA GPU（支持 CUDA），至少 4GB 显存
- **内存**: 建议 8GB+
- **磁盘空间**: 至少 5GB（用于数据集和模型）

### 环境配置

#### 1. 克隆项目

```bash
git clone <repository-url>
cd Cat_Dog_Classification
```

#### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n catdog python=3.9
conda activate catdog

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

#### 3. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果使用 GPU，确保安装对应版本的 PyTorch
# 访问 https://pytorch.org/ 获取正确的安装命令
# 例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. 验证安装

```bash
# 检查 GPU 是否可用
python tools/check_gpu_cuda.py

# 验证主要依赖
python -c "import torch; import sklearn; print('✅ 安装成功')"
```

### 数据集准备

#### 数据集结构

将数据集按以下结构组织：

```
dataset/
├── train/              # 训练集（建议 8000+ 张）
│   ├── cats/          # 猫的图片
│   └── dogs/          # 狗的图片
├── val/                # 验证集（建议 1000+ 张）
│   ├── cats/
│   └── dogs/
└── test/               # 测试集（建议 1000+ 张）
    ├── cats/
    └── dogs/
```

#### 数据集要求

- **格式**: JPG, PNG 等常见图像格式
- **大小**: 建议至少 224×224 像素
- **类别平衡**: 建议每个类别的样本数量大致相等
- **数据划分**: 建议训练集:验证集:测试集 = 8:1:1

#### 数据集统计

本项目使用的数据集统计：
- **训练集**: 8,000 张（cats: 4,000, dogs: 4,000）
- **验证集**: 1,000 张（cats: 500, dogs: 500）
- **测试集**: 1,000 张（cats: 500, dogs: 500）
- **总计**: 10,000 张图像

### 特征搜索

特征搜索功能使用**束搜索（Beam Search）**算法自动寻找最优的特征组合，帮助提升传统机器学习模型的性能。

> 💡 **提示**：特征搜索主要用于传统机器学习模型，建议在训练传统 ML 模型之前先运行特征搜索，找到最佳特征组合。

#### 功能特点

- 🔍 **束搜索算法**：高效搜索最优特征组合
- 📊 **交叉验证评估**：使用 5 折交叉验证评估特征组合性能
- 🎯 **自动特征选择**：从 10 种特征中自动选择最佳组合
- 📉 **PCA 降维**：自动处理高维特征，使用 PCA 降维到 512 维
- 💾 **结果保存**：自动保存搜索历史和最佳结果

#### 使用方法

```bash
# 基础使用（使用默认配置）
python src/data/feature_search.py

# 自定义配置
python src/data/feature_search.py \
    --model svm \
    --beam-width 5 \
    --cv 5 \
    --pca-components 512 \
    --out-dir runs/feature_search
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 评估模型（svm/rf） | svm |
| `--beam-width` | 束搜索的束宽 | 5 |
| `--cv` | 交叉验证折数 | 5 |
| `--pca-components` | PCA 降维后的维度 | 512 |
| `--image-size` | 图像尺寸 | 128 |
| `--out-dir` | 结果输出目录 | runs/feature_search |

#### 搜索结果

根据实验，最佳特征组合为：

**最佳特征组合**（准确率：**80.14%**）：
- color_hist（颜色直方图）
- corner_edge（角点边缘密度）
- edge_hist（边缘直方图）
- gabor（Gabor 滤波器）
- glcm（灰度共生矩阵）
- hog（方向梯度直方图）
- lbp（局部二值模式）
- sift（尺度不变特征变换）

**特征性能排名**（单特征）：
1. 🥇 **HOG** - 77.07% ± 1.36%
2. 🥈 **SIFT** - 70.82% ± 1.47%
3. 🥉 **Edge Histogram** - 64.47% ± 1.19%
4. **LBP** - 63.98% ± 0.93%
5. **GLCM** - 63.46% ± 1.27%

**搜索过程**：
- 第 1 步：评估 10 个单特征，保留 top 5
- 第 2 步：扩展为 2 特征组合，评估 35 个组合
- 第 3-8 步：逐步增加特征数量，最终找到 8 特征最优组合
- 总评估次数：约 200+ 个特征组合

#### 结果文件

特征搜索完成后，会在输出目录生成：

- `search_results.json` - 搜索结果（最佳组合、搜索历史、特征信息）
- `feature_search.log` - 详细搜索日志

#### 使用建议

1. **束宽选择**：
   - 小数据集：束宽 3-5
   - 大数据集：束宽 5-10
   - 计算资源充足：可增大束宽获得更好结果

2. **特征池**：
   - 默认使用全部 10 种特征
   - 可根据需求自定义特征列表

3. **PCA 降维**：
   - 高维特征（>512 维）自动使用 PCA 降维
   - 低维特征（<512 维）直接使用

4. **评估模型**：
   - SVM：快速，适合初步搜索
   - 随机森林：更准确，但速度较慢

### 训练模型

## 🔥 深度学习模型训练（PyTorch）

深度学习模型使用端到端的学习方式，直接从原始图像学习特征表示，无需手工特征工程。

#### 训练 PyTorch 模型

**基础训练命令：**

```bash
# 训练 ResNet50（性能最佳，推荐）
python train_torch.py --arch resnet50 --batch-size 512 --epochs 120

# 训练 ResNet34（平衡性能和速度）
python train_torch.py --arch resnet34 --batch-size 512 --epochs 120

# 训练 ResNet18（速度快，性能优秀）
python train_torch.py --arch resnet18 --batch-size 512 --epochs 120

# 训练 CNN-v2（轻量级模型）
python train_torch.py --arch cnn_v2 --batch-size 512 --epochs 120

# 训练 CNN-v1（基础 CNN）
python train_torch.py --arch cnn_v1 --batch-size 512 --epochs 120
```

**高级训练选项：**

```bash
# 自定义学习率和调度器
python train_torch.py \
    --arch resnet34 \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --early-stop-patience 15

# 不使用预训练权重（从头训练）
python train_torch.py \
    --arch resnet34 \
    --no-resnet-pretrained \
    --batch-size 128 \
    --epochs 200

# 解冻 backbone（微调整个网络）
python train_torch.py \
    --arch resnet34 \
    --no-freeze-backbone \
    --lr 0.00001 \
    --batch-size 128
```

**训练参数说明：**

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--arch` | 模型架构 | cnn_v1 | resnet34, resnet18 |
| `--batch-size` | 批次大小 | 512 | 256-512（GPU 内存允许） |
| `--epochs` | 训练轮数 | 120 | 100-200 |
| `--lr` | 学习率 | 0.001 | 0.0001-0.001 |
| `--weight-decay` | 权重衰减 | 0.0001 | 0.0001-0.001 |
| `--scheduler` | 学习率调度器 | cosine | cosine, step, none |
| `--early-stop-patience` | 早停耐心值 | 10 | 10-20 |
| `--dropout` | Dropout 概率 | 0.0 | 0.0-0.5 |

#### 深度学习模型特点

- **端到端学习**：直接从图像学习特征，无需手工特征工程
- **迁移学习**：使用 ImageNet 预训练权重，快速收敛
- **高性能**：ResNet50 达到 98.40% 测试准确率（验证准确率 99.20%）
- **GPU 加速**：支持 CUDA 加速训练和推理
- **混合精度**：使用 AMP 加速训练并节省显存

#### 模型选择建议

| 模型 | 适用场景 | 优势 |
|------|---------|------|
| **ResNet50** | 追求极致性能 | 准确率最高（98.40%），验证准确率 99.20% |
| **ResNet34** | 平衡性能和速度 | 准确率高（98.30%），训练较快 |
| **ResNet18** | 快速训练 | 训练最快，性能优秀（98.20%） |
| **CNN-v2** | 轻量级需求 | 模型小，从头训练 |
| **CNN-v1** | 基础学习 | 简单架构，易于理解 |

---

## 📊 传统机器学习模型训练（sklearn）

传统机器学习模型基于手工提取的特征进行分类，训练速度快，适合快速原型开发。

#### 训练 sklearn 模型

**基础训练：**

```bash
# 训练 SVM（支持向量机）
python train_sklearn.py --model svm

# 训练逻辑回归
python train_sklearn.py --model logreg

# 训练随机森林
python train_sklearn.py --model random_forest
```

**高级选项：**

```bash
# 自定义特征提取方法
python train_sklearn.py \
    --model svm \
    --features hog,color_hist,lbp

# 自定义交叉验证折数
python train_sklearn.py \
    --model svm \
    --cv 10
```

#### 传统机器学习模型特点

- **快速训练**：训练时间短，通常几分钟内完成
- **特征工程**：基于手工特征（HOG、颜色直方图等）
- **可解释性**：模型和特征易于理解和分析
- **资源需求低**：不需要 GPU，CPU 即可训练
- **适合小数据集**：在数据量较少时也能取得不错效果

#### 模型选择建议

| 模型 | 适用场景 | 优势 |
|------|---------|------|
| **SVM** | 追求最佳性能 | 准确率最高（80.70%），泛化能力强 |
| **逻辑回归** | 快速原型 | 训练快，模型简单，可解释性强 |
| **随机森林** | 特征重要性分析 | 可分析特征重要性，鲁棒性好 |

#### 特征工程

传统机器学习模型依赖特征工程，本项目实现了多种特征提取方法：

| 特征类型 | 说明 | 维度 | 用途 |
|---------|------|------|------|
| **HOG** | 方向梯度直方图，捕获边缘和形状信息 | 3780 | 主要特征 |
| **颜色直方图** | RGB/HSV 颜色分布 | 768 | 颜色特征 |
| **LBP** | 局部二值模式，纹理特征 | 256 | 纹理分析 |
| **GLCM** | 灰度共生矩阵，纹理统计 | 20 | 纹理特征 |
| **Gabor** | Gabor 滤波器响应 | 40 | 多尺度纹理 |
| **SIFT** | 尺度不变特征变换 | 变长 | 关键点特征 |
| **边缘直方图** | 边缘方向分布 | 64 | 边缘特征 |
| **角点密度** | 角点和边缘密度 | 2 | 结构特征 |
| **矩特征** | 图像矩统计 | 7 | 形状特征 |

**特征组合策略**：
- 主要使用 HOG + 颜色直方图组合（效果最好）
- 使用特征搜索找到最优特征组合
- 支持多特征融合和特征选择
- 使用 PCA 进行降维（可选）

---

#### 集成学习

```bash
# 软投票集成（推荐）
python ensemble.py --method voting --voting soft

# 硬投票集成
python ensemble.py --method voting --voting hard

# 堆叠集成
python ensemble.py --method stacking

# AdaBoost 集成
python ensemble.py --method adaboost
```

### 推理预测

**单张图片推理：**

```bash
# 基础推理
python infer.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --image path/to/image.jpg

# 保存预测结果
python infer.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --image path/to/image.jpg \
    --output prediction.json
```

**批量推理：**

```bash
# 推理整个目录
python infer.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --image-dir dataset/test/ \
    --output predictions.csv

# 导出为 JSON 格式
python infer.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --image-dir dataset/test/ \
    --output predictions.json \
    --format json
```

**推理选项：**

- `--checkpoint`: 模型检查点路径（必需）
- `--image`: 单张图片路径
- `--image-dir`: 图片目录路径
- `--output`: 输出文件路径
- `--format`: 输出格式（csv/json）
- `--batch-size`: 批量推理批次大小
- `--device`: 推理设备（cuda/cpu）

### Web 界面

**启动 Gradio 界面：**

```bash
# 基础启动
python gradio_run.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt

# 自定义端口和主机
python gradio_run.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --port 7860 \
    --host 0.0.0.0

# 启用分享链接（公网访问）
python gradio_run.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --share
```

启动后，在浏览器中访问显示的 URL（通常是 `http://localhost:7860`）即可使用 Web 界面进行图像分类。

## 📊 模型训练结果

### 🎯 深度学习模型（PyTorch）

| 模型 | 架构 | 验证准确率 | 测试准确率 | 训练轮数 | 批次大小 | 学习率 | 优化器 | 调度器 | 预训练 | 冻结backbone |
|------|------|-----------|-----------|---------|---------|--------|--------|--------|--------|-------------|
| 🥇 ResNet50 | ResNet50 | **99.20%** | **98.40%** | 23/120 | 512 | 0.001 | Adam | Cosine | ✅ | ✅ |
| 🥈 ResNet34 | ResNet34 | **98.60%** | **98.30%** | 29/120 | 512 | 0.001 | Adam | Cosine | ✅ | ✅ |
| 🥉 ResNet18 | ResNet18 | **98.10%** | **98.20%** | 13/120 | 512 | 0.001 | Adam | Cosine | ✅ | ✅ |
| CNN-v2 | CNN | 91.90% | 93.40% | 40/120 | 512 | 0.001 | Adam | Cosine | ❌ | - |
| CNN-v1 | CNN | 91.50% | 92.60% | 79/120 | 512 | 0.001 | Adam | Cosine | ❌ | - |

**训练配置（深度学习模型）：**
- 图像尺寸：224×224
- 权重衰减：0.0001
- 早停耐心值：10
- 混合精度训练（AMP）：✅
- 数据增强：✅
- 工作进程数：16

**详细训练信息：**

| 模型 | 参数量 | 可训练参数 | 训练时间 | 最佳轮次 | 测试损失 | 测试 F1 |
|------|--------|-----------|---------|---------|---------|---------|
| ResNet50 | 24.5M | 1.05M | 7.6 分钟 | 23/33 | 0.0451 | 0.984 |
| ResNet34 | 21.5M | 263K | 4.9 分钟 | 19/29 | 0.0544 | 0.983 |
| ResNet18 | 11.4M | 263K | 3.4 分钟 | 13/23 | 0.0574 | 0.982 |
| CNN-v2 | - | - | 9.2 分钟 | 40/50 | 0.1805 | 0.933 |
| CNN-v1 | - | - | 16.0 分钟 | 79/89 | 0.2134 | 0.926 |

**训练技巧：**
- ResNet 模型使用 ImageNet 预训练权重，冻结 backbone，只训练分类头
- 使用 Cosine 学习率调度器，平滑降低学习率
- 早停机制防止过拟合，节省训练时间
- 混合精度训练（AMP）加速训练并节省显存

### 📈 传统机器学习模型（scikit-learn）

| 模型 | 训练准确率 | 验证准确率 | 测试准确率 | 特征提取方法 | 交叉验证 |
|------|-----------|-----------|-----------|------------|---------|
| 🥇 SVM | 100.00% | **81.50%** | **80.70%** | HOG + 颜色直方图 | 5折 |
| 🥈 逻辑回归 | 81.53% | 77.00% | 77.80% | HOG + 颜色直方图 | 5折 |
| 🥉 随机森林 | 99.88% | 74.80% | 74.30% | HOG + 颜色直方图 | 5折 |

**特征工程：**

本项目实现了多种传统图像特征提取方法：

| 特征类型 | 说明 | 维度 | 用途 |
|---------|------|------|------|
| **HOG** | 方向梯度直方图，捕获边缘和形状信息 | 3780 | 主要特征 |
| **颜色直方图** | RGB/HSV 颜色分布 | 768 | 颜色特征 |
| **LBP** | 局部二值模式，纹理特征 | 256 | 纹理分析 |
| **GLCM** | 灰度共生矩阵，纹理统计 | 20 | 纹理特征 |
| **Gabor** | Gabor 滤波器响应 | 40 | 多尺度纹理 |
| **SIFT** | 尺度不变特征变换 | 变长 | 关键点特征 |
| **边缘直方图** | 边缘方向分布 | 64 | 边缘特征 |
| **角点密度** | 角点和边缘密度 | 2 | 结构特征 |
| **矩特征** | 图像矩统计 | 7 | 形状特征 |

**特征组合策略：**
- 主要使用 HOG + 颜色直方图组合（效果最好）
- 支持多特征融合和特征选择
- 使用 PCA 进行降维（可选）

### 🎯 集成学习结果

| 集成方法 | 准确率 | 说明 |
|---------|--------|------|
| 多数投票 | 79.10% | 三个基模型（SVM、逻辑回归、随机森林）的硬投票 |
| 最佳单模型 | 81.50% | SVM（最佳单模型） |
| 软投票 | - | 使用概率加权平均 |

**模型一致性分析：**
- 模型一致预测：754 个样本（75.4%）
- 模型不一致：246 个样本（24.6%）
- 最佳单模型（SVM）错误率：18.5%

## 🔧 技术栈

- **深度学习框架**：PyTorch 2.0+
- **机器学习库**：scikit-learn 1.3+
- **图像处理**：OpenCV, PIL, scikit-image
- **可视化**：Matplotlib, Seaborn, TensorBoard
- **Web界面**：Gradio 4.0+
- **数据处理**：NumPy, Pandas

## 📈 性能对比

### 最佳模型排名

1. 🥇 **ResNet50** - 测试准确率 **98.40%** ⭐⭐
2. 🥈 **ResNet34** - 测试准确率 **98.30%** ⭐
3. 🥉 **ResNet18** - 测试准确率 **98.20%** ⭐
4. **CNN-v2** - 测试准确率 **93.40%**
5. **CNN-v1** - 测试准确率 **92.60%**
6. **SVM** - 测试准确率 **80.70%**
7. **逻辑回归** - 测试准确率 **77.80%**
8. **随机森林** - 测试准确率 **74.30%**

### 模型特点

- **ResNet 系列**：使用 ImageNet 预训练权重，冻结 backbone，只训练分类头，收敛快、性能优秀
- **CNN 系列**：从头训练，需要更多轮次，但模型更轻量
- **传统 ML 模型**：基于手工特征，训练快速，但性能有限

## 🎨 可视化

### 自动生成的可视化

项目在训练过程中自动生成以下可视化结果（保存在 `runs/*/figures/` 目录）：

- 📊 **训练/验证损失曲线** - 监控训练过程，检测过拟合
- 📈 **训练/验证准确率曲线** - 评估模型性能提升
- 🎯 **混淆矩阵** - 分析分类错误模式
- 📉 **ROC 曲线** - 评估分类器性能
- 🔍 **特征重要性分析** - 了解哪些特征最重要
- 📋 **分类报告** - 详细的性能指标（精确率、召回率、F1）

### TensorBoard 可视化

使用 TensorBoard 实时查看训练过程：

```bash
# 查看单个实验
tensorboard --logdir=runs/torch_resnet34_20251114-105541/events

# 对比多个实验
tensorboard --logdir=runs/

# 指定端口
tensorboard --logdir=runs/ --port 6006
```

TensorBoard 显示内容：
- 实时损失和准确率曲线
- 学习率变化
- 模型计算图
- 图像样本（如果启用）

### 可视化示例

```python
# 在 Python 中查看训练曲线
from tools.visualization import plot_training_curves
import json

# 加载训练历史
with open('runs/torch_resnet34_20251114-105541/training_history.json') as f:
    history = json.load(f)

# 绘制曲线
plot_training_curves(history)
```

## 🏗️ 模型架构详解

### 深度学习模型

#### ResNet 系列

**ResNet50 架构：**
- **Backbone**: ImageNet 预训练的 ResNet50（冻结）
- **分类头**: 全连接层 (2048 → 128 → 1)
- **参数量**: 24.5M（总），1.05M（可训练）
- **特点**: 最深的 ResNet 模型，性能最佳，验证准确率可达 99.20%

**ResNet34 架构：**
- **Backbone**: ImageNet 预训练的 ResNet34（冻结）
- **分类头**: 全连接层 (512 → 128 → 1)
- **参数量**: 21.5M（总），263K（可训练）
- **特点**: 使用残差连接，解决深层网络退化问题，平衡性能和速度

**ResNet18 架构：**
- **Backbone**: ImageNet 预训练的 ResNet18（冻结）
- **分类头**: 全连接层 (512 → 128 → 1)
- **参数量**: 11.4M（总），263K（可训练）
- **特点**: 更轻量，训练最快，适合快速迭代

#### CNN 系列

**CNN-v2 架构：**
```
输入 (3, 224, 224)
  ↓
Conv Block 1: 32 通道
  ↓
Conv Block 2: 64 通道
  ↓
Conv Block 3: 128 通道
  ↓
Conv Block 4: 256 通道
  ↓
全局平均池化
  ↓
全连接层: 256 → 128 → 1
```

**CNN-v1 架构：**
- 3 个卷积块（32 → 64 → 128 通道）
- 全局平均池化
- 全连接层（128 → 128 → 1）

### 传统机器学习模型

#### SVM（支持向量机）
- **核函数**: RBF（径向基函数）
- **特征**: HOG + 颜色直方图
- **超参数**: C, gamma（通过网格搜索优化）

#### 逻辑回归
- **优化器**: liblinear, lbfgs, saga
- **正则化**: L2
- **特征**: HOG + 颜色直方图

#### 随机森林
- **树数量**: 200-500
- **最大深度**: 10-30
- **特征**: HOG + 颜色直方图

## 🔄 数据增强技术

### 训练时数据增强

项目实现了以下数据增强技术（仅在训练时使用）：

| 增强方法 | 参数 | 说明 |
|---------|------|------|
| **RandomResizedCrop** | scale=(0.9, 1.0), ratio=(0.9, 1.1) | 随机裁剪和缩放，提高泛化能力 |
| **RandomHorizontalFlip** | p=0.5 | 随机水平翻转，增加数据多样性 |
| **RandomRotation** | degrees=10 | 随机旋转 ±10 度，增强鲁棒性 |
| **ColorJitter** | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 | 颜色抖动，适应不同光照条件 |
| **GaussianBlur** | kernel_size=3, sigma=(0.1, 1.0) | 高斯模糊，模拟图像模糊 |

### 验证/测试时

- 仅使用基础变换：Resize → ToTensor
- 不使用任何随机增强，保证结果可复现

### 数据增强效果

数据增强显著提升了模型性能：
- **无增强**: 测试准确率约 85-90%
- **有增强**: 测试准确率 92-98%

## 📝 使用示例

### 完整训练流程示例

```bash
# 1. 检查 GPU
python tools/check_gpu_cuda.py

# 2. 训练最佳模型（ResNet50）
python train_torch.py \
    --arch resnet50 \
    --batch-size 512 \
    --epochs 120 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --early-stop-patience 10

# 3. 查看训练结果
tensorboard --logdir=runs/torch_resnet34_*/

# 4. 在测试集上评估
python infer.py \
    --checkpoint runs/torch_resnet34_*/best.pt \
    --image-dir dataset/test/ \
    --output test_predictions.csv

# 5. 启动 Web 界面
python gradio_run.py \
    --checkpoint runs/torch_resnet34_*/best.pt \
    --share
```

### 推理示例

```bash
# 单张图片推理（显示详细信息）
python infer.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --image dataset/test/cats/cat.1.jpg \
    --output prediction.json \
    --verbose

# 批量推理（CSV 格式）
python infer.py \
    --checkpoint runs/torch_resnet34_20251114-105541/best.pt \
    --image-dir dataset/test/ \
    --output predictions.csv \
    --batch-size 32
```

### Python API 使用示例

```python
# 加载模型进行推理
from src.utils.inference_utils import prepare_model
from src.data.data_utils import build_transforms
from PIL import Image
import torch

# 准备模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = prepare_model('runs/torch_resnet34_20251114-105541/best.pt', device)

# 准备图像
transform = build_transforms(224, augment=False)
image = Image.open('path/to/image.jpg')
image_tensor = transform(image).unsqueeze(0).to(device)

# 推理
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    probability = torch.sigmoid(output).item()
    prediction = 'dog' if probability > 0.5 else 'cat'
    
print(f'预测: {prediction}, 置信度: {probability:.2%}')
```

## 🔍 实验记录

所有训练实验都保存在 `runs/` 目录下，包含：

- ✅ 模型检查点（`best.pt`, `last.pt`）
- 📄 训练配置文件（`config.json`）
- 📊 训练日志（`*.log`）
- 📈 可视化图表（`figures/`）
- 📉 TensorBoard 事件文件（`events/`）

### 实验结果分析

**深度学习模型表现：**
- ResNet 系列表现最佳，得益于 ImageNet 预训练权重
- ResNet50 性能最优（测试准确率 98.40%，验证准确率 99.20%）
- ResNet34 和 ResNet18 性能接近，但 ResNet18 训练更快
- CNN 模型从头训练，需要更多轮次才能收敛
- 所有模型都通过早停机制避免过拟合

**传统机器学习模型表现：**
- SVM 表现最好，但存在过拟合（训练准确率 100%）
- 逻辑回归表现稳定，泛化能力较好
- 随机森林训练准确率高但验证准确率低，存在过拟合

**集成学习分析：**
- 三个基模型的错误重叠率约 40-50%
- 多数投票未能显著提升性能
- 建议使用更强大的基模型或调整权重

## 💡 训练技巧和最佳实践

### 深度学习模型训练技巧

1. **使用预训练权重**
   - ResNet 模型使用 ImageNet 预训练权重可显著提升性能
   - 冻结 backbone 只训练分类头，训练更快且不易过拟合

2. **学习率设置**
   - 预训练模型：0.001（分类头），0.00001（微调 backbone）
   - 从头训练：0.001-0.01
   - 使用 Cosine 调度器平滑降低学习率

3. **批次大小选择**
   - GPU 内存充足：512（训练快）
   - GPU 内存有限：128-256
   - 批次大小影响训练稳定性和速度

4. **数据增强**
   - 训练时启用数据增强提升泛化能力
   - 验证/测试时不使用增强保证结果可复现

5. **早停机制**
   - 耐心值设置为 10-20 轮
   - 监控验证准确率而非训练准确率

### 传统机器学习模型技巧

1. **特征选择**
   - HOG + 颜色直方图组合效果最好
   - 可以尝试添加 LBP、GLCM 等纹理特征
   - 使用特征选择算法去除冗余特征

2. **超参数调优**
   - 使用网格搜索或随机搜索
   - 交叉验证评估模型性能
   - 避免在测试集上直接调参

3. **处理过拟合**
   - 增加正则化强度
   - 减少模型复杂度
   - 使用更多训练数据

## ⚡ 性能优化建议

### 训练速度优化

1. **使用 GPU**
   ```bash
   # 检查 GPU 是否可用
   python tools/check_gpu_cuda.py
   ```

2. **混合精度训练**
   - 默认启用 AMP，可加速 1.5-2 倍
   - 显存占用减少约 50%

3. **多进程数据加载**
   - 设置 `num_workers=16`（根据 CPU 核心数调整）
   - 使用 `pin_memory=True` 加速数据传输

4. **批次大小优化**
   - 在 GPU 内存允许的情况下使用更大的批次
   - 批次大小影响训练速度和稳定性

### 模型性能优化

1. **模型选择**
   - 追求极致性能：ResNet50（98.40%）
   - 平衡性能和速度：ResNet34（98.30%）
   - 快速训练：ResNet18（98.20%）
   - 轻量级需求：CNN-v2（93.40%）

2. **超参数调优**
   - 学习率：0.0001-0.001
   - 权重衰减：0.0001-0.001
   - Dropout：0.0-0.5（根据过拟合情况调整）

3. **数据增强**
   - 适度增强提升泛化能力
   - 过度增强可能降低性能

## ❓ 常见问题解答（FAQ）

### 安装问题

**Q: 安装 PyTorch 时出错？**
```bash
# 访问 https://pytorch.org/ 获取正确的安装命令
# 根据你的 CUDA 版本选择
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: GPU 不可用？**
```bash
# 检查 CUDA 是否安装
nvidia-smi

# 检查 PyTorch 是否支持 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 训练问题

**Q: 训练时显存不足（OOM）？**
- 减小批次大小（如 256 → 128）
- 减小图像尺寸（如 224 → 160）
- 使用梯度累积
- 启用混合精度训练（AMP）

**Q: 训练准确率很高但验证准确率低？**
- 这是过拟合现象
- 增加数据增强
- 增加 Dropout
- 增加权重衰减
- 使用早停机制

**Q: 训练很慢？**
- 检查是否使用 GPU
- 增加 `num_workers`（数据加载进程数）
- 使用更大的批次大小
- 启用混合精度训练

**Q: 模型不收敛？**
- 检查学习率是否过大或过小
- 尝试不同的优化器（Adam/SGD）
- 检查数据标签是否正确
- 检查数据预处理是否正确

### 推理问题

**Q: 推理结果不准确？**
- 检查输入图像预处理是否与训练时一致
- 确认使用的是最佳模型检查点（`best.pt`）
- 检查图像质量和分辨率

**Q: 批量推理很慢？**
- 增加 `--batch-size` 参数
- 使用 GPU 进行推理
- 使用 TensorRT 等推理优化工具

### 其他问题

**Q: 如何复现实验结果？**
- 设置随机种子：`--seed 42`
- 使用相同的训练配置
- 使用相同的数据集划分

**Q: 如何选择最佳模型？**
- 查看验证集准确率（主要指标）
- 查看测试集准确率（最终评估）
- 查看训练曲线判断是否过拟合

## 🐛 故障排除

### 常见错误及解决方案

1. **ModuleNotFoundError**
   ```bash
   # 确保安装了所有依赖
   pip install -r requirements.txt
   ```

2. **CUDA out of memory**
   ```bash
   # 减小批次大小
   python train_torch.py --batch-size 128 ...
   ```

3. **数据加载错误**
   - 检查数据集路径是否正确
   - 检查数据集结构是否符合要求
   - 检查图像文件是否损坏

4. **模型加载失败**
   - 检查检查点路径是否正确
   - 确认模型架构匹配
   - 检查 PyTorch 版本兼容性

## 📚 扩展阅读

### 相关论文

- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **HOG**: [Histograms of Oriented Gradients for Human Detection](https://ieeexplore.ieee.org/document/1467360)
- **集成学习**: [Ensemble Methods in Machine Learning](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)

### 相关资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [scikit-learn 用户指南](https://scikit-learn.org/stable/user_guide.html)
- [ImageNet 数据集](https://www.image-net.org/)

## 🔮 未来计划

- [ ] 支持更多模型架构（EfficientNet, Vision Transformer）
- [ ] 实现模型量化（INT8 推理）
- [ ] 添加模型蒸馏功能
- [ ] 支持多类别分类扩展
- [ ] 实现自动超参数搜索
- [ ] 添加模型解释性分析（Grad-CAM）
- [ ] 支持在线学习
- [ ] 添加模型部署工具（ONNX, TensorRT）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本项目
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的注释和文档字符串
- 编写单元测试（如果可能）
- 更新 README.md（如果添加新功能）

## 📄 许可证

详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- PyTorch 团队
- scikit-learn 社区
- 所有开源贡献者

---

**最后更新**：2025-11-14

**最佳模型**：ResNet50 (测试准确率 98.40%, 验证准确率 99.20%) 🏆
