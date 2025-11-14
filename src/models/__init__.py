# 深度学习模型
from .cnn import (
    CatDogCNNv1, 
    CatDogCNNv2,
    create_CatDogCNNv1,
    create_CatDogCNNv2
)

from .resnet import (
    PretrainedResNet,
    ResNetTrainer,
    create_resnet18,
    create_resnet50,
    create_resnet_trainer
)

# 传统机器学习模型
from .svm import SVMTrainer

from .random_forest import RandomForestTrainer


from .logistic_regression import LogisticRegressionTrainer


__all__ = [
    # CNN模型
    'CatDogCNNv1', 'CatDogCNNv2', 'create_CatDogCNNv1', 'create_CatDogCNNv2',
    'PretrainedResNet', 'ResNetTrainer', 'create_resnet18', 'create_resnet50', 'create_resnet_trainer',
    'SVMTrainer', 'LogisticRegressionTrainer', 'RandomForestTrainer',
]
