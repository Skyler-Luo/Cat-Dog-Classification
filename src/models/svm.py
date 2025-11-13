"""
猫狗分类 - SVM训练器

该模块实现了基于scikit-learn的RBF核支持向量机(SVM)训练器。
支持超参数搜索。

主要功能:
    - 基于GridSearchCV的超参数搜索（支持并行）
    - 完整的模型评估和保存功能
"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.utils.ml_training import compute_classification_metrics, save_sklearn_model
from src.utils.logger import Logger


class SVMTrainer:
    """SVM训练器类
    
    封装了RBF核SVM的训练、超参数搜索、评估和保存功能。
    
    参数:
        c_values: C参数的候选值列表
        gamma_values: gamma参数的候选值列表
        cv_folds: 交叉验证的折数（默认: 5）
        n_jobs: 并行训练的作业数（默认: 4，-1表示使用所有CPU核心）
        random_state: 随机种子（默认: 42）
        svm_probability: 是否启用概率估计（默认: False）
    """
    
    def __init__(self, c_values, gamma_values, cv_folds=5, n_jobs=4, 
                 random_state=42, svm_probability=False, scoring="accuracy", do_search=True, default_params=None):
        self.c_values = c_values
        self.gamma_values = gamma_values
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.svm_probability = svm_probability
        self.scoring = scoring
        self.do_search = do_search
        self.default_params = default_params or {}
        self.best_model = None
        self.logger = None
        self.cv_results_ = None
    
    def _get_logger(self):
        """获取logger，如果不存在则创建一个默认的"""
        if self.logger is None:
            self.logger = Logger(name="svm_trainer")
        return self.logger

    def _build_model(self):
        """构建SVM分类器
        
        返回:
            SVC分类器对象
        """
        return SVC(
            kernel="rbf",
            probability=self.svm_probability,
            random_state=self.random_state
        )


    def fit(self, X_train, y_train):
        """训练SVM模型并进行超参数搜索
        
        参数:
            X_train: 训练特征矩阵，形状为(n_samples, n_features)
            y_train: 训练标签数组，形状为(n_samples,)

        返回:
            best_model: 最佳模型
            best_score: 最佳分数
            best_params: 最佳参数
        """
        logger = self._get_logger()
        model = self._build_model()
        if not self.do_search:
            logger.info("⏭️ 跳过超参数搜索，使用默认SVM配置进行训练")
            try:
                model.set_params(**self.default_params)
            except Exception:
                pass
            model.fit(X_train, y_train)
            self.best_model = model
            self.cv_results_ = None
            return {
                "best_model": self.best_model,
                "best_score": None,
                "best_params": None
            }
        param_grid = {"C": self.c_values, "gamma": self.gamma_values}
        search = GridSearchCV(
            model, param_grid, scoring=self.scoring, cv=self.cv_folds,
            n_jobs=self.n_jobs, verbose=2
        )
        search.fit(X_train, y_train)
        self.best_model = search.best_estimator_
        self.cv_results_ = search.cv_results_
        logger.log_cv_results(search)
        return {
            "best_model": self.best_model,
            "best_score": search.best_score_,
            "best_params": search.best_params_
        }
    
    def evaluate(self, X, y, name):
        """在给定数据集上评估模型性能
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 真实标签数组，形状为(n_samples,)
            name: 数据集名称（用于打印信息）
            
        返回:
            dict: 主要分类指标（accuracy/precision/recall/f1）
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        y_pred = self.best_model.predict(X)
        y_scores = None
        if hasattr(self.best_model, 'predict_proba'):
            try:
                proba = self.best_model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_scores = proba[:, 1]
                else:
                    y_scores = proba
            except Exception:
                y_scores = None
        if y_scores is None and hasattr(self.best_model, 'decision_function'):
            try:
                y_scores = self.best_model.decision_function(X)
            except Exception:
                y_scores = None
        logger = self._get_logger()
        metrics_dict = compute_classification_metrics(y, y_pred, y_proba=y_scores, positive_label=1)
        msg = "{} | acc={:.4f}, prec={:.4f}, rec={:.4f}, f1={:.4f}".format(
            name, metrics_dict['accuracy'], metrics_dict['precision'], metrics_dict['recall'], metrics_dict['f1']
        )
        logger.info(msg)
        return metrics_dict

    def save(self, save_path, save_results=None, config=None):
        """保存训练好的模型到磁盘
        
        参数:
            save_path: 模型保存路径
            save_results: 训练结果字典（可选）
            config: 训练配置字典（可选）
        """
        if self.best_model is None:
            raise RuntimeError("No model to save. Train first.")
        
        logger = self._get_logger()
        save_sklearn_model(
            self.best_model,
            save_path,
            model_type='SVM',
            save_results=save_results,
            config=config,
            extra_model_info={
                'kernel': 'rbf',
                'preprocessing': '特征已预处理（在特征提取阶段完成）'
            },
            logger=logger
        )
