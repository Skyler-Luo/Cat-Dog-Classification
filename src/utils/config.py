"""
统一配置模块

集中管理训练与通用配置，供各处导入使用。
"""

# 模型名称映射
MODEL_TYPE_NAMES = {
    "svm": "SVM",
    "logreg": "逻辑回归",
    "random_forest": "随机森林",
    "bp": "BP神经网络",
    "cnn_v1": "CNN v1",
    "cnn_v2": "CNN v2",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
}


# 可选的 PyTorch 架构标识
ARCH_CHOICES = [
    "cnn_v1",
    "cnn_v2",
    "resnet18",
    "resnet34",
    "resnet50",
]

# 统一的模型默认保存路径
DEFAULT_SAVE_PATHS = {
    "svm": "runs/sklearn_svm/best.joblib",
    "logreg": "runs/sklearn_lr/best.joblib",
    "random_forest": "runs/sklearn_rf/best.joblib",
    "default": "runs/sklearn/best.joblib",
}

class TrainConfig:
    """训练配置参数类，包含所有训练相关的超参数和配置。"""

    def __init__(
        self,
        # 数据集相关配置
        data_dir=None,
        train_dirname="train",
        val_dirname="val",
        test_dirname="test",
        # 图像和训练基础参数
        image_size=150,
        batch_size=64,
        epochs=40,
        lr=1e-3,
        weight_decay=1e-4,
        # 优化器相关
        optimizer="adam",
        momentum=0.9,
        # 学习率调度器相关
        scheduler="none",
        step_size=10,
        gamma=0.1,
        # 模型与训练技巧
        model_version="v1",
        dropout=0.0,
        normalize_imagenet=False,
        train_augment=True,
        amp=True,
        early_stop_patience=10,
        early_stop_metric="accuracy",
        # ResNet 专属配置
        resnet_pretrained=True,
        freeze_backbone=True,
        # 系统配置
        num_workers=4,
        seed=42,
        log_dir="runs/torch_cnn",
        ckpt_dir="runs/torch_cnn",
    ):
        self.data_dir = data_dir
        self.train_dirname = train_dirname
        self.val_dirname = val_dirname
        self.test_dirname = test_dirname

        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.optimizer = optimizer
        self.momentum = momentum

        self.scheduler = scheduler
        self.step_size = step_size
        self.gamma = gamma

        self.model_version = model_version
        self.dropout = dropout
        self.normalize_imagenet = normalize_imagenet
        self.train_augment = train_augment
        self.amp = amp
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric

        self.resnet_pretrained = resnet_pretrained
        self.freeze_backbone = freeze_backbone

        self.num_workers = num_workers
        self.seed = seed
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir

    def to_dict(self):
        return {
            "data_dir": self.data_dir,
            "train_dirname": self.train_dirname,
            "val_dirname": self.val_dirname,
            "test_dirname": self.test_dirname,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "scheduler": self.scheduler,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "model_version": self.model_version,
            "dropout": self.dropout,
            "normalize_imagenet": self.normalize_imagenet,
            "train_augment": self.train_augment,
            "amp": self.amp,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_metric": self.early_stop_metric,
            "resnet_pretrained": self.resnet_pretrained,
            "freeze_backbone": self.freeze_backbone,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "log_dir": self.log_dir,
            "ckpt_dir": self.ckpt_dir,
        }


# 传统机器学习配置（类形式）
class SklearnCommonConfig:
    def __init__(self, scoring="accuracy", n_jobs=16, seed=42, cv_folds=5):
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.seed = seed
        self.cv_folds = cv_folds

    def to_dict(self):
        return {
            "scoring": self.scoring,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "cv_folds": self.cv_folds,
        }


class SVMConfig:
    def __init__(self, c_values=None, gamma_values=None, svm_probability=False):
        self.c_values = c_values or ["0.1", "1", "10", "100"]
        self.gamma_values = gamma_values or ["scale", "0.001", "0.0001"]
        self.svm_probability = svm_probability

    def to_dict(self):
        return {
            "c_values": self.c_values,
            "gamma_values": self.gamma_values,
            "svm_probability": self.svm_probability,
        }


class LogisticRegressionConfig:
    def __init__(self, solver="liblinear"):
        self.solver = solver

    def to_dict(self):
        return {
            "solver": self.solver,
        }


class RandomForestConfig:
    def __init__(self):
        pass

    def to_dict(self):
        return {}


class SklearnConfig:
    """聚合的传统机器学习配置。"""

    def __init__(
        self,
        common=None,
        svm=None,
        logistic_regression=None,
        random_forest=None,
    ):
        self.common = common or SklearnCommonConfig()
        self.svm = svm or SVMConfig()
        self.logistic_regression = logistic_regression or LogisticRegressionConfig()
        self.random_forest = random_forest or RandomForestConfig()

    def to_dict(self):
        return {
            "common": self.common.to_dict(),
            "svm": self.svm.to_dict(),
            "logistic_regression": self.logistic_regression.to_dict(),
            "random_forest": self.random_forest.to_dict(),
        }


__all__ = [
    "MODEL_TYPE_NAMES",
    "ARCH_CHOICES",
    "TrainConfig",
    "SklearnCommonConfig",
    "SVMConfig",
    "LogisticRegressionConfig",
    "RandomForestConfig",
    "SklearnConfig",
    "DEFAULT_SAVE_PATHS",
]
