"""
猫狗分类 - BP神经网络模型

该模块实现了经典的BP（Back Propagation）神经网络。
BP神经网络是最基础的多层感知机，通过反向传播算法训练。

主要功能:
    - 支持多隐藏层的全连接网络
    - 可配置的激活函数（ReLU, Sigmoid, Tanh）
    - 多种优化器支持（SGD, Adam, RMSprop）
    - Dropout正则化和批归一化
    - 学习率衰减策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


class BPNeuralNetwork(nn.Module):
    """BP神经网络分类器
    
    经典的多层感知机（MLP），使用反向传播算法训练。
    适合处理扁平化的图像特征或其他结构化数据。
    
    参数:
        input_size: 输入特征维度
        hidden_sizes: 隐藏层大小列表，如[512, 256, 128]表示3个隐藏层
        num_classes: 输出类别数（二分类通常为1）
        activation: 激活函数类型 ('relu', 'sigmoid', 'tanh', 'leaky_relu')
        dropout_p: Dropout概率，用于防止过拟合
        use_batch_norm: 是否使用批归一化
        bias: 是否使用偏置项
    """
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=1, 
                 activation='relu', dropout_p=0.5, use_batch_norm=True, bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.activation_name = activation
        self.dropout_p = dropout_p
        self.use_batch_norm = use_batch_norm
        
        # 选择激活函数
        self.activation = self._get_activation_function(activation)
        
        # 构建网络层
        self.layers = self._build_layers(input_size, hidden_sizes, num_classes, bias)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_activation_function(self, activation):
        """获取激活函数
        
        参数:
            activation: 激活函数名称
            
        返回:
            PyTorch激活函数
        """
        activations = {
            'relu': nn.ReLU(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU()
        }
        
        if activation not in activations:
            raise ValueError(f"不支持的激活函数: {activation}。可选: {list(activations.keys())}")
        
        return activations[activation]
    
    def _build_layers(self, input_size, hidden_sizes, num_classes, bias):
        """构建网络层
        
        参数:
            input_size: 输入维度
            hidden_sizes: 隐藏层大小列表
            num_classes: 输出类别数
            bias: 是否使用偏置
            
        返回:
            网络层的ModuleList
        """
        layers = nn.ModuleList()
        
        # 所有层的大小
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        
        # 构建隐藏层
        for i in range(len(all_sizes) - 1):
            in_features = all_sizes[i]
            out_features = all_sizes[i + 1]
            is_output_layer = (i == len(all_sizes) - 2)
            
            # 线性层
            linear = nn.Linear(in_features, out_features, bias=bias)
            layers.append(linear)
            
            # 非输出层添加激活函数、批归一化和Dropout
            if not is_output_layer:
                # 批归一化（在激活函数之前）
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_features))
                
                # 激活函数
                layers.append(self.activation)
                
                # Dropout
                if self.dropout_p > 0:
                    layers.append(nn.Dropout(p=self.dropout_p))
        
        return layers
    
    def _initialize_weights(self):
        """初始化网络权重
        
        使用Xavier/Glorot初始化来保持梯度稳定
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Xavier初始化
                if self.activation_name == 'relu':
                    # He初始化更适合ReLU
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(layer.weight)
                
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量，形状为 (N, input_size)
               N: batch size
               input_size: 输入特征维度
        
        返回:
            输出logits，形状为 (N, num_classes)
        """
        # 确保输入是正确的形状
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # 展平为二维
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_model_info(self):
        """获取模型信息
        
        返回:
            包含模型统计信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BP Neural Network',
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'activation': self.activation_name,
            'dropout_p': self.dropout_p,
            'use_batch_norm': self.use_batch_norm,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_layers': len(self.hidden_sizes) + 1
        }


class BPNeuralNetworkTrainer:
    """BP神经网络训练器
    
    封装了BP神经网络的训练、评估和保存功能。
    支持多种优化策略和训练技巧。
    
    参数:
        input_size: 输入特征维度
        hidden_sizes: 隐藏层配置
        num_classes: 输出类别数
        activation: 激活函数
        dropout_p: Dropout概率
        use_batch_norm: 是否使用批归一化
        device: 训练设备
    """
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=1,
                 activation='relu', dropout_p=0.5, use_batch_norm=True, device='cuda'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.activation = activation
        self.dropout_p = dropout_p
        self.use_batch_norm = use_batch_norm
        self.device = device
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.training_history = []
    
    def build_model(self):
        """构建BP神经网络模型
        
        返回:
            构建好的模型
        """
        self.model = BPNeuralNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=self.num_classes,
            activation=self.activation,
            dropout_p=self.dropout_p,
            use_batch_norm=self.use_batch_norm
        )
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        info = self.model.get_model_info()
        print(f"🧠 构建BP神经网络:")
        print(f"   架构: {self.input_size} → {' → '.join(map(str, self.hidden_sizes))} → {self.num_classes}")
        print(f"   激活函数: {info['activation']}")
        print(f"   总参数: {info['total_params']:,}")
        print(f"   层数: {info['num_layers']}")
        print(f"   批归一化: {'是' if self.use_batch_norm else '否'}")
        print(f"   Dropout: {self.dropout_p}")
        
        return self.model
    
    def setup_training(self, optimizer='adam', learning_rate=1e-3, weight_decay=1e-4,
                      scheduler_type='step', scheduler_params=None):
        """设置训练组件
        
        参数:
            optimizer: 优化器类型 ('adam', 'sgd', 'rmsprop', 'adamw')
            learning_rate: 初始学习率
            weight_decay: 权重衰减（L2正则化）
            scheduler_type: 学习率调度器类型 ('step', 'cosine', 'plateau', None)
            scheduler_params: 调度器参数字典
        """
        if self.model is None:
            raise RuntimeError("请先调用build_model()构建模型")
        
        # 设置优化器
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        if optimizer not in optimizers:
            raise ValueError(f"不支持的优化器: {optimizer}")
        
        optimizer_class = optimizers[optimizer]
        
        if optimizer == 'sgd':
            self.optimizer = optimizer_class(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                momentum=0.9  # SGD通常需要动量
            )
        else:
            self.optimizer = optimizer_class(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 设置损失函数
        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 设置学习率调度器
        if scheduler_type is not None:
            scheduler_params = scheduler_params or {}
            
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_params.get('step_size', 10),
                    gamma=scheduler_params.get('gamma', 0.1)
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_params.get('T_max', 50)
                )
            elif scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',  # 监控准确率
                    factor=scheduler_params.get('factor', 0.5),
                    patience=scheduler_params.get('patience', 5),
                    verbose=True
                )
        
        print(f"⚙️  训练配置:")
        print(f"   优化器: {optimizer}")
        print(f"   学习率: {learning_rate}")
        print(f"   权重衰减: {weight_decay}")
        print(f"   调度器: {scheduler_type}")
        print(f"   损失函数: {'BCEWithLogitsLoss' if self.num_classes == 1 else 'CrossEntropyLoss'}")
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            epoch: 当前epoch数
            total_epochs: 总epoch数
            
        返回:
            epoch训练损失和准确率
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("请先调用build_model()和setup_training()")
            
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # 展平输入数据（如果是图像）
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            
            # 确保目标格式正确
            if self.num_classes == 1:
                targets = targets.float().view(-1, 1)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            if self.num_classes == 1:
                # 二分类
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == targets).sum().item()
            else:
                # 多分类
                predicted = outputs.argmax(dim=1)
                correct += (predicted == targets).sum().item()
            
            total += targets.size(0)
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader, dataset_name="Test"):
        """评估模型
        
        参数:
            test_loader: 测试数据加载器
            dataset_name: 数据集名称
            
        返回:
            测试损失和准确率
        """
        if self.model is None or self.criterion is None:
            raise RuntimeError("请先调用build_model()和setup_training()")
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # 展平输入数据
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                # 确保目标格式正确
                if self.num_classes == 1:
                    targets = targets.float().view(-1, 1)
                
                # 前向传播
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                
                if self.num_classes == 1:
                    # 二分类
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (predicted == targets).sum().item()
                else:
                    # 多分类
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == targets).sum().item()
                
                total += targets.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        print(f"\n📊 {dataset_name} 结果:")
        print(f"   损失: {avg_loss:.4f}")
        print(f"   准确率: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader=None, epochs=100, early_stopping_patience=10, save_path=None):
        """训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_path: 模型保存路径
            
        返回:
            训练历史
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("请先调用build_model()和setup_training()")
        
        print(f"🚀 开始训练BP神经网络 ({epochs} epochs)")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs)
            
            # 验证
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, "Validation")
            
            # 记录历史
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc if val_loader else train_acc)
                else:
                    self.scheduler.step()
            
            # 早停检查
            if val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_path:
                        self.save_model(save_path, epoch_history)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 早停触发 (patience={early_stopping_patience})")
                    break
            
            print(f"Epoch {epoch+1:3d}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        print(f"✅ 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        return self.training_history
    
    def save_model(self, save_path, training_info=None):
        """保存模型
        
        参数:
            save_path: 保存路径
            training_info: 训练信息
        """
        if self.model is None:
            raise RuntimeError("没有模型可保存")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重和配置
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_info(),
            'training_history': self.training_history,
            'training_info': training_info or {}
        }
        
        torch.save(model_data, save_path)
        print(f"💾 模型已保存至: {save_path}")


def create_bp_network(input_size, hidden_sizes=[512, 256, 128], num_classes=1, activation='relu'):
    """创建BP神经网络的便捷函数
    
    参数:
        input_size: 输入特征维度
        hidden_sizes: 隐藏层配置
        num_classes: 输出类别数
        activation: 激活函数
        
    返回:
        BPNeuralNetwork实例
    """
    return BPNeuralNetwork(input_size, hidden_sizes, num_classes, activation)


def create_bp_trainer(input_size, **kwargs):
    """创建BP神经网络训练器的便捷函数
    
    参数:
        input_size: 输入特征维度
        **kwargs: BPNeuralNetworkTrainer的其他参数
        
    返回:
        BPNeuralNetworkTrainer实例
        
    示例:
        >>> # 为224x224的RGB图像创建BP网络
        >>> trainer = create_bp_trainer(
        ...     input_size=224*224*3,  # 展平后的图像尺寸
        ...     hidden_sizes=[1024, 512, 256],
        ...     activation='relu',
        ...     dropout_p=0.5
        ... )
        >>> 
        >>> # 构建和设置训练
        >>> model = trainer.build_model()
        >>> trainer.setup_training(optimizer='adam', learning_rate=1e-3)
        >>> 
        >>> # 开始训练
        >>> history = trainer.train(train_loader, val_loader, epochs=50)
    """
    return BPNeuralNetworkTrainer(input_size=input_size, **kwargs)


def load_bp_model(model_path, input_size):
    """加载保存的BP神经网络模型
    
    参数:
        model_path: 模型文件路径
        input_size: 输入特征维度
        
    返回:
        加载好的模型
    """
    model_data = torch.load(model_path)
    model_config = model_data['model_config']
    
    # 重建模型
    model = BPNeuralNetwork(
        input_size=input_size,
        hidden_sizes=model_config['hidden_sizes'],
        num_classes=model_config['num_classes'],
        activation=model_config['activation'],
        dropout_p=model_config['dropout_p'],
        use_batch_norm=model_config['use_batch_norm']
    )
    
    # 加载权重
    model.load_state_dict(model_data['model_state_dict'])
    
    print(f"📂 已加载BP神经网络模型: {model_path}")
    return model, model_data
