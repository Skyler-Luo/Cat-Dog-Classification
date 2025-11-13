"""
猫狗分类 - 卷积神经网络模型

模型变体:
    - CatDogCNNv1
    - CatDogCNNv2

所有模型均采用自适应池化设计，能够处理任意尺寸的输入图像。
"""
import torch
import torch.nn as nn


class CatDogCNNv1(nn.Module):
    """
    网络结构:
        - 3个卷积块，每个包含Conv2D + ReLU + BatchNorm + MaxPool
        - 通道数递增: in_channels -> 32 -> 64 -> 128
        - 自适应全局平均池化
        - 分类器: 128 -> 128 -> num_classes, 带Dropout正则化
    
    参数:
        num_classes: 输出单元数。对于二分类，使用1（配合BCEWithLogitsLoss）。
                     对于两个logit输出，设置为2。
        in_channels: 输入通道数（RGB图像为3）
        dropout_p: 分类器中的Dropout概率，用于防止过拟合
    """
    
    def __init__(self, num_classes=1, in_channels=3, dropout_p=0.5):
        super().__init__()
        
        # 特征提取层：3个卷积块
        self.features = nn.Sequential(
            # 第1个卷积块: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第2个卷积块: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第3个卷积块: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自适应全局平均池化：将任意大小的特征图池化为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器：全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平为一维向量
            nn.Linear(128, 128),  # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),  # Dropout正则化
            nn.Linear(128, num_classes),  # 输出层
        )
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入图像张量，形状为 (N, C, H, W)
               N: batch size
               C: 通道数（通常为3，表示RGB）
               H, W: 图像高度和宽度
        
        返回:
            输出logits，形状为 (N, num_classes)
        """
        x = self.features(x)  # 特征提取
        x = self.avgpool(x)  # 全局池化
        x = self.classifier(x)  # 分类
        return x


class CatDogCNNv2(nn.Module):
    """
    网络结构:
        - 4个卷积块，具有残差风格的跳跃连接
        - 通道数递增: in_channels -> 32 -> 64 -> 128 -> 128
        - 投影层用于匹配跳跃连接的维度
        - 自适应全局平均池化
        - 分类器: 128 -> 128 -> num_classes, 带Dropout正则化
    
    参数:
        num_classes: 输出单元数。对于二分类，使用1（配合BCEWithLogitsLoss）。
                     对于两个logit输出，设置为2。
        in_channels: 输入通道数（RGB图像为3）
        dropout_p: 分类器中的Dropout概率，用于防止过拟合
    """
    
    def __init__(self, num_classes=1, in_channels=3, dropout_p=0.5):
        super().__init__()
        
        # 第1个卷积块: in_channels -> 32
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 第2个卷积块: 32 -> 64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 第3个卷积块: 64 -> 128 (参与残差连接)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 投影层用于跳跃连接（匹配尺寸和通道数）
        self.projection = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(128),
        )
        
        # 第4个卷积块: 128 -> 128
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        # 自适应全局平均池化：将任意大小的特征图池化为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器：全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平为一维向量
            nn.Linear(128, 128),  # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),  # Dropout正则化
            nn.Linear(128, num_classes),  # 输出层
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入图像张量，形状为 (N, C, H, W)
               N: batch size
               C: 通道数（通常为3，表示RGB）
               H, W: 图像高度和宽度
        
        返回:
            输出logits，形状为 (N, num_classes)
        """
        # 前两个卷积块
        x = self.conv_block1(x)
        x1 = self.conv_block2(x)  # 保存用于跳跃连接
        
        # 第3个卷积块
        x2 = self.conv_block3(x1)
        
        # 投影跳跃连接（匹配尺寸和通道）
        x1_proj = self.projection(x1)
        
        # 残差连接
        x = x2 + x1_proj
        x = self.relu(x)
        
        # 第4个卷积块
        x = self.conv_block4(x)
        
        # 全局池化和分类
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def create_CatDogCNNv1(num_classes=1, in_channels=3, dropout_p=0.5):
    """工厂函数：创建CatDogCNNv1版本的CNN模型
    
    参数:
        num_classes: 输出类别数（二分类通常为1）
        in_channels: 输入图像通道数（RGB为3）
        dropout_p: Dropout比例，用于防止过拟合
        
    返回:
        CatDogCNNv1实例
    """
    return CatDogCNNv1(num_classes=num_classes, in_channels=in_channels, dropout_p=dropout_p)


def create_CatDogCNNv2(num_classes=1, in_channels=3, dropout_p=0.5):
    """工厂函数：创建CatDogCNNv2版本的CNN模型
    
    参数:
        num_classes: 输出类别数（二分类通常为1）
        in_channels: 输入图像通道数（RGB为3）
        dropout_p: Dropout比例，用于防止过拟合
        
    返回:
        CatDogCNNv2实例
    """
    return CatDogCNNv2(num_classes=num_classes, in_channels=in_channels, dropout_p=dropout_p)
