"""
猫狗分类 - 逻辑回归模型

该模块实现了基于scikit-learn的逻辑回归分类器。
逻辑回归是一种简单但高效的线性分类算法，适合作为基线模型。

主要功能:
    - 支持多种正则化方法（L1, L2, ElasticNet）
    - 自动超参数搜索（网格搜索）
    - 模型系数分析
"""
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.utils.ml_training import compute_classification_metrics, save_sklearn_model
from src.utils.logger import Logger


class LogisticRegressionTrainer:
    """逻辑回归训练器类
    
    封装了逻辑回归的训练、超参数搜索、评估和保存功能。
    支持多种正则化策略。
    
    参数:
        C_values: 正则化强度的候选值列表（越小正则化越强）
        penalty_types: 正则化类型 ('l1', 'l2', 'elasticnet', 'none')
        l1_ratios: ElasticNet的L1比例（仅当penalty='elasticnet'时使用）
        max_iter: 最大迭代次数
        cv_folds: 交叉验证的折数
        n_jobs: 并行训练的作业数
        random_state: 随机种子，保证实验可复现
        solvers: 求解器候选列表 ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga')
    """
    
    def __init__(
        self,
        C_values=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        penalty_types=['l1', 'l2'],
        l1_ratios=[0.1, 0.5, 0.7, 0.9],
        max_iter=1000,
        cv_folds=5,
        n_jobs=4,
        random_state=42,
        solvers=None,
        scoring='accuracy',
        do_search=True,
        default_params=None,
    ):
        self.C_values = C_values
        self.penalty_types = penalty_types
        self.l1_ratios = l1_ratios
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_model = None
        self.feature_names = None
        self.scoring = scoring
        self.do_search = do_search
        self.logger = None
        self.default_params = default_params or {}
        self.solvers = solvers or ['liblinear']
        if self.default_params.get('solver') and self.default_params['solver'] not in self.solvers:
            self.solvers.append(self.default_params['solver'])
        self.solvers = list(dict.fromkeys(self.solvers))
        self.valid_solver_penalty_pairs = []
        self.default_solver = None
        self.cv_results_ = None
        
        # 验证solver和penalty的兼容性
        self._validate_solver_penalty()
        if not self.default_solver:
            candidate_solver = self.default_params.get('solver')
            if candidate_solver and any(solver == candidate_solver for solver, _ in self.valid_solver_penalty_pairs):
                self.default_solver = candidate_solver
        if not self.default_solver and self.valid_solver_penalty_pairs:
            self.default_solver = self.valid_solver_penalty_pairs[0][0]
            self.default_params['solver'] = self.default_solver
    
    def _get_logger(self):
        """获取logger，如果不存在则创建一个默认的"""
        if self.logger is None:
            self.logger = Logger(name="logistic_regression_trainer")
        return self.logger

    def _validate_solver_penalty(self):
        """验证求解器和正则化类型的兼容性"""
        incompatible = []
        valid = []
        
        for solver in self.solvers:
            for penalty in self.penalty_types:
                if penalty == 'elasticnet' and solver != 'saga':
                    incompatible.append((solver, penalty))
                    continue
                if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                    incompatible.append((solver, penalty))
                    continue
                if penalty == 'none' and solver not in ['newton-cg', 'lbfgs', 'sag', 'saga']:
                    incompatible.append((solver, penalty))
                    continue
                if solver == 'liblinear' and penalty == 'none':
                    incompatible.append((solver, penalty))
                    continue
                valid.append((solver, penalty))
        
        if not valid:
            raise ValueError("未找到合法的 solver-penalty 组合，请调整参数。")
        
        seen = set()
        for solver, penalty in valid:
            if (solver, penalty) not in seen:
                self.valid_solver_penalty_pairs.append((solver, penalty))
                seen.add((solver, penalty))
        if incompatible:
            logger = self._get_logger()
            msg = "⚠️  检测到不兼容的solver-penalty组合:"
            logger.info(msg)
            for solver, penalty in incompatible:
                logger.info(f"   {solver} + {penalty}")
        if self.default_params.get('penalty') and self.valid_solver_penalty_pairs:
            for solver, penalty in self.valid_solver_penalty_pairs:
                if self.default_params['penalty'] == penalty:
                    self.default_solver = solver
                    break
        if not self.default_solver and self.default_params.get('solver'):
            candidate_solver = self.default_params['solver']
            if any(solver == candidate_solver for solver, _ in self.valid_solver_penalty_pairs):
                self.default_solver = candidate_solver

    def _build_model(self, **params):
        """构建逻辑回归分类器
        
        参数:
            **params: 逻辑回归的超参数
        
        返回:
            LogisticRegression 对象
        """
        lr_params = {
            'max_iter': self.max_iter,
            'random_state': self.random_state,
        }
        solver = params.pop('solver', None)
        if solver is None:
            solver = self.default_params.get('solver')
        if solver is None and self.default_solver:
            solver = self.default_solver
        if solver is None and self.valid_solver_penalty_pairs:
            solver = self.valid_solver_penalty_pairs[0][0]
        if solver is not None:
            lr_params['solver'] = solver
        lr_params.update(params)
        clean_params = {k: v for k, v in lr_params.items() if v is not None}
        model = LogisticRegression()
        if clean_params:
            model.set_params(**clean_params)
        return model

    def _param_grid(self):
        """返回用于搜索的参数网格"""
        param_grid = []
        
        for solver, penalty in self.valid_solver_penalty_pairs:
            if penalty == 'elasticnet':
                grid = {
                    "C": self.C_values,
                    "penalty": [penalty],
                    "solver": [solver],
                    "l1_ratio": self.l1_ratios
                }
            else:
                grid = {
                    "C": self.C_values,
                    "penalty": [penalty],
                    "solver": [solver]
                }
            param_grid.append(grid)
        
        return param_grid

    def build_model(self):
        """构建默认的逻辑回归分类器"""
        return self._build_model()

    def build_model_with_params(self, **params):
        """构建指定超参数的逻辑回归分类器
        
        参数:
            **params: 逻辑回归的超参数
            
        返回:
            配置好的 LogisticRegression 对象
        """
        return self._build_model(**params)

    def fit(self, X_train, y_train, feature_names=None, show_progress=False):
        """训练逻辑回归模型并进行超参数搜索
        
        根据选择的搜索方法执行超参数优化，找到最佳参数组合。
        训练完成后，最佳模型保存在self.best_model中。
        
        参数:
            X_train: 训练特征矩阵，形状为(n_samples, n_features)（已预处理）
            y_train: 训练标签数组，形状为(n_samples,)
            feature_names: 特征名称列表（用于系数分析）
            show_progress: 是否显示详细的训练进度
        """
        logger = self._get_logger()
        if self.do_search:
            logger.info("📊 开始逻辑回归训练，使用网格搜索...")
        else:
            logger.info("⏭️ 跳过超参数搜索，使用默认逻辑回归配置进行训练")
        
        self.feature_names = feature_names
        model = self._build_model()
        if not self.do_search:
            try:
                model.set_params(**{k: v for k, v in self.default_params.items() if v is not None})
            except Exception:
                pass
            model.fit(X_train, y_train)
            self.best_model = model
            self.cv_results_ = None
            logger.info("✅ 训练完成！(未进行搜索)")
        else:
            param_grid = self._param_grid()
            search = GridSearchCV(
                model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                n_jobs=self.n_jobs,
                verbose=1 if show_progress else 0,
                error_score='raise',  # type: ignore[arg-type]
            )
            search.fit(X_train, y_train)
            self.best_model = search.best_estimator_
            self.cv_results_ = search.cv_results_
            logger.log_cv_results(search)
            logger.info(f"✅ 训练完成！最佳CV准确率: {search.best_score_:.4f}")
            logger.info(f"📋 最佳参数: {search.best_params_}")
        
        # 分析特征系数
        self._analyze_coefficients()

    def _analyze_coefficients(self):
        """分析特征系数"""
        if self.best_model is None:
            return
        
        # 获取逻辑回归分类器
        coefficients = self.best_model.coef_[0]
        
        logger = self._get_logger()
        logger.info(f"\n📈 模型系数分析:")
        intercept_value = np.ravel(getattr(self.best_model, "intercept_", np.array([0.0])))[0]
        logger.info(f"   截距项: {intercept_value:.4f}")
        logger.info(f"   特征系数范围: [{coefficients.min():.4f}, {coefficients.max():.4f}]")
        logger.info(f"   非零系数数量: {np.count_nonzero(coefficients)}/{len(coefficients)}")
        
        # 显示最重要的特征
        if self.feature_names is not None and len(self.feature_names) == len(coefficients):
            abs_coef = np.abs(coefficients)
            top_indices = np.argsort(abs_coef)[-10:][::-1]
            
            logger.info(f"\n🔍 前10个最重要特征:")
            for i, idx in enumerate(top_indices):
                logger.info(f"   {i+1:2d}. {self.feature_names[idx]}: {coefficients[idx]:.4f}")

    def evaluate(self, X, y, name):
        """在给定数据集上评估模型性能
        
        参数:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 真实标签数组，形状为(n_samples,)
            name: 数据集名称（如"Validation", "Test"），用于打印信息
            
        返回:
            dict: 主要分类指标（accuracy/precision/recall/f1/auc）
        """
        if self.best_model is None:
            raise RuntimeError("模型尚未训练。请先调用fit()方法。")
        
        # 进行预测
        y_pred = self.best_model.predict(X)
        y_pred_proba = self.best_model.predict_proba(X)[:, 1]
        logger = self._get_logger()
        metrics_dict = compute_classification_metrics(y, y_pred, y_proba=y_pred_proba, positive_label=1)
        parts = [
            "acc={:.4f}".format(metrics_dict['accuracy']),
            "prec={:.4f}".format(metrics_dict['precision']),
            "rec={:.4f}".format(metrics_dict['recall']),
            "f1={:.4f}".format(metrics_dict['f1']),
        ]
        if 'auc' in metrics_dict:
            parts.append("auc={:.4f}".format(metrics_dict['auc']))
        msg = "{} | {}".format(name, ", ".join(parts))
        logger.info(msg)
        return metrics_dict

    def get_feature_importance(self, top_k=10):
        """获取特征重要性（基于系数绝对值）
        
        参数:
            top_k: 返回前k个最重要的特征
            
        返回:
            特征重要性信息
        """
        if self.best_model is None:
            raise RuntimeError("模型尚未训练。请先调用fit()方法。")
        
        coefficients = self.best_model.coef_[0]
        abs_coef = np.abs(coefficients)
        
        # 获取重要性索引排序
        importance_indices = np.argsort(abs_coef)[::-1][:top_k]
        importance_values = abs_coef[importance_indices]
        
        logger = self._get_logger()
        logger.info(f"\n🔍 前{top_k}个最重要特征（按系数绝对值）:")
        for i, (idx, importance) in enumerate(zip(importance_indices, importance_values)):
            coef_sign = "+" if coefficients[idx] >= 0 else "-"
            feature_name = self.feature_names[idx] if self.feature_names else f"特征{idx}"
            logger.info(f"{i+1:2d}. {feature_name}: {coef_sign}{importance:.4f}")
        
        return importance_indices, importance_values, coefficients[importance_indices]

    def predict_with_confidence(self, X, confidence_threshold=0.7):
        """带置信度的预测
        
        参数:
            X: 输入特征
            confidence_threshold: 置信度阈值
            
        返回:
            预测结果、概率和置信度标记
        """
        if self.best_model is None:
            raise RuntimeError("模型尚未训练。请先调用fit()方法。")
        
        y_pred = self.best_model.predict(X)
        y_pred_proba = self.best_model.predict_proba(X)
        
        # 计算置信度（最大概率）
        max_proba = np.max(y_pred_proba, axis=1)
        high_confidence = max_proba >= confidence_threshold
        
        results = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confidence': max_proba,
            'high_confidence': high_confidence,
            'high_confidence_ratio': np.mean(high_confidence)
        }
        
        logger = self._get_logger()
        logger.info(f"🔮 预测完成:")
        logger.info(f"   高置信度样本比例: {results['high_confidence_ratio']:.2%}")
        logger.info(f"   平均置信度: {np.mean(max_proba):.4f}")
        
        return results

    def save(self, save_path, save_results=None, config=None):
        """保存训练好的模型到磁盘
        
        使用joblib保存Pipeline（包括预处理步骤和分类器），并可选保存训练结果与配置到JSON。
        
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
            model_type='LogisticRegression',
            save_results=save_results,
            config=config,
            extra_model_info={
                'solver': getattr(self.best_model, 'solver', self.default_solver),
                'preprocessing': '特征已预处理（在特征提取阶段完成）'
            },
            logger=logger
        )
