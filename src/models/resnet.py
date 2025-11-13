"""
猫狗分类 - ResNet预训练模型

该模块实现了基于PyTorch预训练ResNet模型的迁移学习。
利用在ImageNet上预训练的权重，通过微调来适应猫狗分类任务。

主要功能:
    - 支持多种ResNet架构（ResNet18, ResNet34, ResNet50等）
    - 可选的特征提取模式和微调模式
    - 自适应分类头，支持不同的输出维度
    - 渐进式解冻训练策略
"""
import torch
import torch.nn as nn
from torchvision import models


class PretrainedResNet(nn.Module):
    """预训练ResNet模型类
    
    基于torchvision预训练模型，通过替换分类头来适应猫狗分类任务。
    支持特征提取和微调两种训练模式。
    
    参数:
        model_name: ResNet模型名称 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: 输出类别数（二分类通常为1）
        pretrained: 是否使用ImageNet预训练权重
        freeze_backbone: 是否冻结主干网络（仅训练分类头）
        dropout_p: 分类头中的Dropout概率
    """
    
    def __init__(self, model_name='resnet18', num_classes=1, pretrained=True, freeze_backbone=False, dropout_p=0.5):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # 验证模型名称
        available_models = ['resnet18', 'resnet34', 'resnet50']
        if model_name not in available_models:
            raise ValueError(f"模型名称必须是以下之一: {available_models}")
        
        # 加载预训练模型
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        
        # 获取原始分类头的输入维度
        num_features = self.backbone.fc.in_features
        
        # 将ResNet的fc替换为Identity，并单独定义分类头
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(512, num_classes)
        )
        
        # 可选地冻结主干网络
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结主干网络参数，只训练分类头"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🧊 已冻结主干网络，仅训练分类头")
    
    def unfreeze_backbone(self):
        """解冻主干网络，允许微调整个网络"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("🔥 已解冻主干网络，开启微调模式")
    
    def unfreeze_last_n_layers(self, n=1):
        """解冻最后n个ResNet块
        
        参数:
            n: 要解冻的ResNet块数量
        """
        # ResNet的主要块：layer1, layer2, layer3, layer4
        layers = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
        
        # 先冻结所有层
        self._freeze_backbone()
        
        # 解冻最后n个层
        for i in range(min(n, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        print(f"🔓 已解冻最后 {min(n, len(layers))} 个ResNet层")
    
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入图像张量，形状为 (N, C, H, W)
               N: batch size
               C: 通道数（RGB为3）
               H, W: 图像高度和宽度
        
        返回:
            输出logits，形状为 (N, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_trainable_params(self):
        """获取可训练参数的信息
        
        返回:
            包含参数统计信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }


class ResNetTrainer:
    """ResNet训练器类
    
    封装了预训练ResNet模型的训练流程，支持渐进式解冻策略。
    
    参数:
        model_name: ResNet架构名称
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
        dropout_p: Dropout概率
        device: 训练设备 ('cuda' 或 'cpu')
    """
    
    def __init__(self, model_name='resnet18', num_classes=1, pretrained=True, dropout_p=0.5, device='cuda'):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_p = dropout_p
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def build_model(self, freeze_backbone=True):
        """构建ResNet模型
        
        参数:
            freeze_backbone: 是否冻结主干网络
            
        返回:
            构建好的模型
        """
        self.model = PretrainedResNet(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            freeze_backbone=freeze_backbone,
            dropout_p=self.dropout_p
        )
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        params_info = self.model.get_trainable_params()
        print(f"🏗️  构建 {self.model_name} 模型:")
        print(f"   总参数: {params_info['total_params']:,}")
        print(f"   可训练参数: {params_info['trainable_params']:,}")
        print(f"   可训练比例: {params_info['trainable_ratio']:.2%}")
        
        return self.model
    
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4):
        """设置训练组件（优化器和损失函数）
        
        参数:
            learning_rate: 学习率
            weight_decay: 权重衰减（L2正则化）
        """
        if self.model is None:
            raise RuntimeError("请先调用build_model()构建模型")
        
        # 为冻结和未冻结的参数设置不同的学习率
        if self.model.freeze_backbone:
            # 特征提取模式：只训练分类头
            params = [{'params': self.model.classifier.parameters(), 'lr': learning_rate}]
        else:
            # 微调模式：主干网络使用较小学习率，分类头使用较大学习率
            backbone_params = []
            for name, param in self.model.backbone.named_parameters():
                if param.requires_grad:
                    backbone_params.append(param)
            
            params = [
                {'params': backbone_params, 'lr': learning_rate * 0.1},  # 主干网络用较小学习率
                {'params': self.model.classifier.parameters(), 'lr': learning_rate}  # 分类头用正常学习率
            ]
        
        self.optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        
        print(f"⚙️  训练配置:")
        print(f"   优化器: Adam")
        print(f"   学习率: {learning_rate}")
        print(f"   权重衰减: {weight_decay}")
        print(f"   损失函数: {'BCEWithLogitsLoss' if self.num_classes == 1 else 'CrossEntropyLoss'}")
    
    def progressive_unfreeze(self, stage):
        """渐进式解冻策略
        
        参数:
            stage: 解冻阶段
                0: 只训练分类头
                1: 解冻最后1个ResNet块
                2: 解冻最后2个ResNet块  
                3: 解冻整个网络
        """
        if self.model is None:
            raise RuntimeError("请先构建模型")
        
        if stage == 0:
            self.model._freeze_backbone()
        elif stage == 1:
            self.model.unfreeze_last_n_layers(1)
        elif stage == 2:
            self.model.unfreeze_last_n_layers(2)
        elif stage >= 3:
            self.model.unfreeze_backbone()
        
        # 重新设置优化器以包含新的可训练参数
        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            weight_decay = self.optimizer.param_groups[0]['weight_decay']
            self.setup_training(learning_rate=lr, weight_decay=weight_decay)


def create_resnet18(num_classes=1, pretrained=True, freeze_backbone=True, dropout_p=0.5):
    """创建ResNet18模型的便捷函数
    
    参数:
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结主干网络
        dropout_p: 分类头的Dropout概率（float，默认: 0.5）
        
    返回:
        PretrainedResNet实例
    """
    return PretrainedResNet('resnet18', num_classes, pretrained, freeze_backbone, dropout_p=dropout_p)


def create_resnet50(num_classes=1, pretrained=True, freeze_backbone=True, dropout_p=0.5):
    """创建ResNet50模型的便捷函数
    
    参数:
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结主干网络
        dropout_p: 分类头的Dropout概率（float，默认: 0.5）
        
    返回:
        PretrainedResNet实例
    """
    return PretrainedResNet('resnet50', num_classes, pretrained, freeze_backbone, dropout_p=dropout_p)


def create_resnet_trainer(model_name='resnet18', **kwargs):
    """创建ResNet训练器的工厂函数
    
    参数:
        model_name: ResNet架构名称
        **kwargs: ResNetTrainer的其他参数
        
    返回:
        ResNetTrainer实例
    """
    return ResNetTrainer(model_name=model_name, **kwargs)
