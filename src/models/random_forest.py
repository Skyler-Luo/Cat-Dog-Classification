"""
猫狗分类 - 随机森林模型

该模块实现了基于scikit-learn的随机森林分类器。
随机森林是一种集成学习方法，通过组合多个决策树来提高预测性能。

主要功能:
    - 自动超参数搜索（网格搜索）
    - 特征重要性分析
    - 完整的模型评估和保存功能
"""
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from src.utils.ml_training import compute_classification_metrics, save_sklearn_model
from src.utils.logger import Logger


class RandomForestTrainer:
    """随机森林训练器类
    
    封装了随机森林的训练、超参数搜索、评估和保存功能。
    随机森林通过Bootstrap聚合和特征随机选择来减少过拟合。
    
    参数:
        n_estimators_values: 决策树数量的候选值列表
        max_depth_values: 树最大深度的候选值列表
        min_samples_split_values: 分裂内部节点所需的最小样本数候选值
        min_samples_leaf_values: 叶子节点所需的最小样本数候选值
        max_features_values: 寻找最佳分割时考虑的特征数量（'sqrt', 'log2', None等）
        cv_folds: 交叉验证的折数
        n_jobs: 并行训练的作业数（-1表示使用所有CPU核心）
        random_state: 随机种子，保证实验可复现
        max_samples: Bootstrap采样时使用的最大样本数（float表示比例，int表示绝对数量，None表示使用全部）
    """
    
    def __init__(
        self,
        n_estimators_values=[100, 200, 300],
        max_depth_values=[10, 20, 30, None],
        min_samples_split_values=[2, 5, 10],
        min_samples_leaf_values=[1, 2, 4],
        max_features_values=['sqrt', 'log2'],
        cv_folds=5,
        n_jobs=4,
        random_state=42,
        scoring='accuracy',
        do_search=True,
        default_params=None,
        max_samples=None,
    ):
        self.n_estimators_values = n_estimators_values
        self.max_depth_values = max_depth_values
        self.min_samples_split_values = min_samples_split_values
        self.min_samples_leaf_values = min_samples_leaf_values
        self.max_features_values = max_features_values
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model = None
        self.feature_importance_ = None
        self.scoring = scoring
        self.do_search = do_search
        self.logger = None
        self.default_params = default_params or {}
        self.cv_results_ = None
        self.max_samples = max_samples
    
    def _get_logger(self):
        """获取logger，如果不存在则创建一个默认的"""
        if self.logger is None:
            self.logger = Logger(name="random_forest_trainer")
        return self.logger

    def _build_model(self, **params):
        """构建随机森林分类器
        
        参数:
            **params: 随机森林的超参数
        
        返回:
            RandomForestClassifier 对象
        """
        model_params = {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        # 如果设置了 max_samples，添加到模型参数中
        if self.max_samples is not None:
            model_params["max_samples"] = self.max_samples
        return RandomForestClassifier(**model_params)

    def _param_grid(self):
        """返回用于搜索的参数网格"""
        param_grid = {
            "n_estimators": self.n_estimators_values,
            "max_depth": self.max_depth_values,
            "min_samples_split": self.min_samples_split_values,
            "min_samples_leaf": self.min_samples_leaf_values,
        }
        # 添加 max_features 到参数网格（如果提供了该参数）
        if self.max_features_values:
            param_grid["max_features"] = self.max_features_values
        # 如果设置了 max_samples，也添加到参数网格中
        if self.max_samples is not None:
            param_grid["max_samples"] = [self.max_samples]
        return param_grid

    def build_model(self):
        """构建默认的随机森林分类器"""
        return self._build_model()

    def build_model_with_params(self, **params):
        """构建指定超参数的随机森林分类器
        
        参数:
            **params: 随机森林的超参数
            
        返回:
            配置好的 RandomForestClassifier 对象
        """
        model = self._build_model()
        try:
            model.set_params(**params)
        except Exception:
            pass
        return model

    def fit(self, X_train, y_train, show_progress=False):
        """训练随机森林模型并进行超参数搜索
        
        使用网格搜索执行超参数优化，找到最佳参数组合。
        训练完成后，最佳模型保存在self.best_model中。
        
        参数:
            X_train: 训练特征矩阵，形状为(n_samples, n_features)（已预处理）
            y_train: 训练标签数组，形状为(n_samples,)
            show_progress: 是否显示详细的训练进度
        """
        logger = self._get_logger()
        if self.do_search:
            logger.info("🌲 开始随机森林训练，使用网格搜索...")
        else:
            logger.info("⏭️ 跳过超参数搜索，使用默认随机森林配置进行训练")
        
        model = self._build_model()
        if not self.do_search:
            try:
                model.set_params(**{k: v for k, v in self.default_params.items() if v is not None})
            except Exception:
                pass
            model.fit(X_train, y_train)
            self.best_model = model
            self.cv_results_ = None
        else:
            param_grid = self._param_grid()
            search = GridSearchCV(
                model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                n_jobs=self.n_jobs,
                verbose=1 if show_progress else 0,
            )
            search.fit(X_train, y_train)
            self.best_model = search.best_estimator_
            self.cv_results_ = search.cv_results_
        
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance_ = self.best_model.feature_importances_
        
        if self.do_search:
            logger.log_cv_results(search)
            logger.info(f"✅ 训练完成！最佳CV准确率: {search.best_score_:.4f}")
            logger.info(f"📊 最佳参数: {search.best_params_}")
        else:
            logger.info("✅ 训练完成！(未进行搜索)")

    def evaluate(self, X, y, name):
        """在给定数据集上评估模型性能
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 真实标签数组，形状为(n_samples,)
            name: 数据集名称（如"Validation", "Test"），用于打印信息
            
        返回:
            dict: 主要分类指标（accuracy/precision/recall/f1）
        """
        if self.best_model is None:
            raise RuntimeError("模型尚未训练。请先调用fit()方法。")
        
        # 进行预测
        y_pred = self.best_model.predict(X)
        y_proba = None
        if hasattr(self.best_model, 'predict_proba'):
            try:
                y_proba = self.best_model.predict_proba(X)
            except Exception:
                y_proba = None
        logger = self._get_logger()
        metrics_dict = compute_classification_metrics(y, y_pred, y_proba=y_proba, positive_label=1)
        msg = "{} | acc={:.4f}, prec={:.4f}, rec={:.4f}, f1={:.4f}".format(
            name, metrics_dict['accuracy'], metrics_dict['precision'], metrics_dict['recall'], metrics_dict['f1']
        )
        logger.info(msg)
        return metrics_dict

    def get_feature_importance(self, top_k=10):
        """获取特征重要性排序
        
        参数:
            top_k: 返回前k个最重要的特征
            
        返回:
            特征重要性数组（按重要性降序排列）
        """
        if self.feature_importance_ is None:
            raise RuntimeError("特征重要性不可用。请先训练模型。")
        
        # 获取重要性索引排序
        importance_indices = np.argsort(self.feature_importance_)[::-1][:top_k]
        importance_values = self.feature_importance_[importance_indices]
        
        logger = self._get_logger()
        logger.info(f"\n🔍 前{top_k}个最重要特征:")
        for i, (idx, importance) in enumerate(zip(importance_indices, importance_values)):
            logger.info(f"{i+1:2d}. 特征{idx:4d}: {importance:.4f}")
        
        return importance_indices, importance_values

    def save(self, save_path, save_results=None, config=None):
        """保存训练好的模型到磁盘
        
        使用joblib保存整个Pipeline（包括预处理步骤和分类器），并可选保存训练结果与配置。
        
        参数:
            save_path: 模型保存路径（.joblib或.pkl文件）
            save_results: 训练/评估结果字典（可选）
            config: 训练配置字典（可选）
        """
        if self.best_model is None:
            raise RuntimeError("没有模型可保存。请先训练模型。")
        
        logger = self._get_logger()
        save_sklearn_model(
            self.best_model,
            save_path,
            model_type='RandomForest',
            save_results=save_results,
            config=config,
            extra_model_info={
                'preprocessing': '特征已预处理（在特征提取阶段完成）'
            },
            logger=logger
        )
