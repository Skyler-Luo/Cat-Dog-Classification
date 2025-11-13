"""
猫狗分类 - CNN数据加载器

为CNN提供PyTorch DataLoader：
- 创建训练、验证、测试的DataLoader
- 自动处理数据增强和预处理
- 直接使用 torchvision.datasets.ImageFolder
"""
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def build_transforms(size, augment=False, use_imagenet_norm=False):
    """构建图像变换
    
    参数:
        size: 图像尺寸
        augment: 是否使用数据增强（训练时可选择开启）
        use_imagenet_norm: 是否使用ImageNet标准化
        
    返回:
        torchvision.transforms.Compose对象
    """
    # 构建变换列表
    transform_list = []
    
    if augment:
        # 随机裁剪和缩放 (更好的泛化能力)
        transform_list.append(transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.9, 1.1)))
        # 随机水平翻转
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # 随机旋转 (±10度)
        transform_list.append(transforms.RandomRotation(degrees=10))
        # 颜色增强
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # 高斯模糊
        transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
    else:
        # 不使用数据增强时只进行基础变换
        transform_list.append(transforms.Resize((size, size)))
    
    # 转为张量
    transform_list.append(transforms.ToTensor())
    
    # 可选的ImageNet标准化
    if use_imagenet_norm:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return transforms.Compose(transform_list)


def create_dataloader(data_dir, size, batch_size, augment=False, shuffle=True,
                      use_imagenet_norm=False, num_workers=4, drop_last=False):
    """创建单个DataLoader
    
    参数:
        data_dir: 数据目录路径（包含cats/dogs子目录）
        size: 图像尺寸
        batch_size: 批次大小
        augment: 是否使用数据增强
        shuffle: 是否打乱数据
        use_imagenet_norm: 是否使用ImageNet标准化
        num_workers: 工作进程数
        drop_last: 是否丢弃最后不完整的批次
        
    返回:
        DataLoader对象
    """
    transform = build_transforms(size, augment, use_imagenet_norm)
    
    # 使用ImageFolder自动加载数据
    # cats目录 -> 标签0, dogs目录 -> 标签1
    dataset = datasets.ImageFolder(root=str(data_dir),transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    
    return loader


def load_data(data_dir, train_dir="train", val_dir="val", test_dir="test",
                  size=160, batch_size=64, train_augment=True, use_imagenet_norm=False, num_workers=4):
    """加载CNN训练所需的DataLoader
    
    使用 torchvision.datasets.ImageFolder 直接从目录结构加载数据。
    期望的目录结构:
        data_dir/
        ├── train/
        │   ├── cats/
        │   └── dogs/
        ├── val/
        │   ├── cats/
        │   └── dogs/
        └── test/ (可选)
            ├── cats/
            └── dogs/
    
    参数:
        data_dir: 数据集根目录
        train_dir: 训练集目录名
        val_dir: 验证集目录名  
        test_dir: 测试集目录名
        size: 图像尺寸
        batch_size: 批次大小
        train_augment: 训练集是否使用数据增强（默认True）
        use_imagenet_norm: 是否使用ImageNet标准化
        num_workers: 工作进程数
        
    返回:
        (train_loader, val_loader, test_loader, stats) 元组
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器（不使用数据增强）
        - test_loader: 测试数据加载器（可能为None，不使用数据增强）
        - stats: 数据集统计信息
    """
    data_dir = Path(data_dir)
    
    # 加载训练集
    train_path = data_dir / train_dir
    if not train_path.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_path}")
    
    train_loader = create_dataloader(
        train_path, size, batch_size,
        augment=train_augment, shuffle=True, drop_last=True,
        use_imagenet_norm=use_imagenet_norm, num_workers=num_workers
    )
    
    # 加载验证集
    val_path = data_dir / val_dir
    if not val_path.exists():
        raise FileNotFoundError(f"验证集目录不存在: {val_path}")
    
    val_loader = create_dataloader(
        val_path, size, batch_size,
        augment=False, shuffle=False, drop_last=False,
        use_imagenet_norm=use_imagenet_norm, num_workers=num_workers
    )
    
    # 加载测试集（可选）
    test_loader = None
    test_path = data_dir / test_dir
    if test_path.exists():
        try:
            test_loader = create_dataloader(
                test_path, size, batch_size,
                augment=False, shuffle=False, drop_last=False,
                use_imagenet_norm=use_imagenet_norm, num_workers=num_workers
            )
        except (FileNotFoundError, RuntimeError):
            test_loader = None
    
    # 生成统计信息
    stats = {
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'n_test': len(test_loader.dataset) if test_loader is not None else 0,
        'num_classes': 2,
        'class_names': ['cats', 'dogs'],
    }
    
    return train_loader, val_loader, test_loader, stats
