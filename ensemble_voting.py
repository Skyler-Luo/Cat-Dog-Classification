"""
猫狗分类 - 集成学习投票预测脚本

本脚本加载已训练的 SVM、逻辑回归、随机森林 三个模型，
使用投票机制进行集成预测，并生成可视化结果。

支持两种投票方式：
    - 硬投票（Hard Voting）：基于预测类别进行多数投票
    - 软投票（Soft Voting）：基于预测概率进行加权平均
"""

import argparse
import time
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.ml_training import (
    load_train_val_test,
    compute_classification_metrics,
)
from src.utils.logger import Logger
from tools.visualization import (
    plot_split_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


class VotingEnsemble:
    """投票集成分类器
    
    封装了多个模型的投票预测功能，支持硬投票和软投票两种方式。
    
    参数:
        models: 模型列表，每个模型必须实现 predict() 和 predict_proba() 方法
        model_names: 模型名称列表（可选）
        voting: 投票方式，'hard' 或 'soft'（默认: 'soft'）
        weights: 模型权重列表（可选，默认等权重）
    """
    
    def __init__(self, models, model_names=None, voting='soft', weights=None):
        if not models:
            raise ValueError("模型列表不能为空")
        
        self.models = models
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(models))]
        if len(self.model_names) != len(models):
            raise ValueError("模型名称数量必须与模型数量一致")
        
        self.voting = voting.lower()
        if self.voting not in ['hard', 'soft']:
            raise ValueError("voting 必须是 'hard' 或 'soft'")
        
        if weights is None:
            self.weights = [1.0] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("权重数量必须与模型数量一致")
            self.weights = np.array(weights, dtype=float)
            self.weights = self.weights / self.weights.sum()  # 归一化
        
        self.logger = None
    
    def _get_logger(self):
        """获取logger，如果不存在则创建一个默认的"""
        if self.logger is None:
            self.logger = Logger(name="voting_ensemble")
        return self.logger
    
    def predict(self, X):
        """进行硬投票预测
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        返回:
            预测标签数组，形状为 (n_samples,)
        """
        if self.voting == 'hard':
            predictions = np.array([model.predict(X) for model in self.models])
            # 对每个样本进行加权投票
            weighted_votes = np.zeros((X.shape[0], 2))
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                for j, label in enumerate(pred):
                    weighted_votes[j, int(label)] += weight
            return np.argmax(weighted_votes, axis=1)
        else:
            # 软投票：使用概率的平均值
            return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X):
        """进行软投票预测（返回概率）
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            
        返回:
            预测概率数组，形状为 (n_samples, n_classes)
        """
        if self.voting == 'hard':
            # 硬投票模式下，将预测转换为概率
            predictions = np.array([model.predict(X) for model in self.models])
            proba = np.zeros((X.shape[0], 2))
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                for j, label in enumerate(pred):
                    proba[j, int(label)] += weight
            return proba / proba.sum(axis=1, keepdims=True)
        else:
            # 软投票：加权平均概率
            probas = []
            for model, weight in zip(self.models, self.weights):
                try:
                    proba = model.predict_proba(X)
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        probas.append(proba * weight)
                    else:
                        # 如果只有一维，转换为二维
                        if proba.ndim == 1:
                            proba_2d = np.column_stack([1 - proba, proba])
                        else:
                            proba_2d = proba
                        probas.append(proba_2d * weight)
                except AttributeError:
                    # 如果没有 predict_proba，使用 decision_function
                    try:
                        scores = model.decision_function(X)
                        # 将决策函数转换为概率（简单 sigmoid）
                        proba = 1 / (1 + np.exp(-scores))
                        proba_2d = np.column_stack([1 - proba, proba])
                        probas.append(proba_2d * weight)
                    except AttributeError:
                        # 最后尝试：使用硬预测
                        pred = model.predict(X)
                        proba = np.zeros((X.shape[0], 2))
                        for j, label in enumerate(pred):
                            proba[j, int(label)] = 1.0
                        probas.append(proba * weight)
            
            if not probas:
                raise RuntimeError("无法从任何模型获取概率预测")
            
            ensemble_proba = np.sum(probas, axis=0)
            # 归一化
            row_sums = ensemble_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            ensemble_proba = ensemble_proba / row_sums
            return ensemble_proba
    
    def evaluate(self, X, y, name):
        """在给定数据集上评估集成模型性能
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 真实标签数组，形状为 (n_samples,)
            name: 数据集名称（如 "Validation", "Test"），用于打印信息
            
        返回:
            dict: 主要分类指标（accuracy/precision/recall/f1/auc）
        """
        logger = self._get_logger()
        
        # 进行预测
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        metrics_dict = compute_classification_metrics(
            y, y_pred, y_proba=y_pred_proba, positive_label=1
        )
        
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


class StackingEnsemble:
    """Stacking 集成分类器
    
    使用元学习器学习如何组合基学习器的预测。
    通过交叉验证生成 out-of-fold 预测，避免数据泄露。
    
    参数:
        base_models: 基模型列表
        base_model_names: 基模型名称列表（可选）
        meta_model: 元学习器（默认: LogisticRegression）
        cv_folds: 交叉验证折数（默认: 5）
        use_proba: 是否使用概率作为特征（默认: True）
        random_state: 随机种子
    """
    
    def __init__(self, base_models, base_model_names=None, meta_model=None, 
                 cv_folds=5, use_proba=True, random_state=42):
        if not base_models:
            raise ValueError("基模型列表不能为空")
        
        self.base_models = base_models
        self.base_model_names = base_model_names or [f"Base_{i+1}" for i in range(len(base_models))]
        if len(self.base_model_names) != len(base_models):
            raise ValueError("基模型名称数量必须与模型数量一致")
        
        self.cv_folds = cv_folds
        self.use_proba = use_proba
        self.random_state = random_state
        
        # 默认使用逻辑回归作为元学习器
        if meta_model is None:
            self.meta_model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver='liblinear'
            )
        else:
            self.meta_model = meta_model
        
        self.trained_base_models = None
        self.logger = None
    
    def _get_logger(self):
        """获取logger，如果不存在则创建一个默认的"""
        if self.logger is None:
            self.logger = Logger(name="stacking_ensemble")
        return self.logger
    
    def _get_base_predictions(self, model, X, use_proba=True):
        """获取基模型的预测
        
        参数:
            model: 基模型
            X: 特征矩阵
            use_proba: 是否使用概率
            
        返回:
            预测特征（概率或类别）
        """
        if use_proba:
            try:
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba[:, 1]  # 只返回正类概率
                else:
                    return proba.flatten()
            except AttributeError:
                # 如果没有 predict_proba，使用 decision_function
                try:
                    scores = model.decision_function(X)
                    # 转换为概率（sigmoid）
                    proba = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
                    return proba
                except AttributeError:
                    # 最后使用硬预测
                    pred = model.predict(X)
                    return pred.astype(float)
        else:
            return model.predict(X).astype(float)
    
    def fit(self, X_train, y_train):
        """训练 Stacking 集成模型
        
        参数:
            X_train: 训练特征矩阵
            y_train: 训练标签数组
        """
        logger = self._get_logger()
        logger.info("🔧 开始训练 Stacking 集成模型...")
        logger.info(f"   基模型数量: {len(self.base_models)}")
        logger.info(f"   交叉验证折数: {self.cv_folds}")
        logger.info(f"   使用概率特征: {self.use_proba}")
        
        n_samples = X_train.shape[0]
        n_base_models = len(self.base_models)
        
        # 初始化 out-of-fold 预测矩阵
        if self.use_proba:
            oof_predictions = np.zeros((n_samples, n_base_models))
        else:
            oof_predictions = np.zeros((n_samples, n_base_models))
        
        # 存储训练好的基模型（用于测试时预测）
        self.trained_base_models = []
        
        # 交叉验证生成 out-of-fold 预测
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = y_train[train_idx]
            
            logger.info(f"   训练 Fold {fold_idx + 1}/{self.cv_folds}...")
            
            # 在每个 fold 上训练基模型
            for model_idx, (base_model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
                # 克隆模型（避免修改原始模型）
                from sklearn.base import clone
                model_clone = clone(base_model)
                
                # 训练模型
                model_clone.fit(X_fold_train, y_fold_train)
                
                # 在验证集上预测
                val_pred = self._get_base_predictions(model_clone, X_fold_val, self.use_proba)
                oof_predictions[val_idx, model_idx] = val_pred
        
        logger.info("✅ 基模型交叉验证完成")
        
        # 在完整训练集上重新训练所有基模型（用于测试时预测）
        logger.info("🔄 在完整训练集上重新训练基模型...")
        for model_idx, (base_model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
            from sklearn.base import clone
            model_clone = clone(base_model)
            model_clone.fit(X_train, y_train)
            self.trained_base_models.append(model_clone)
            logger.info(f"   ✅ {model_name} 训练完成")
        
        # 训练元学习器
        logger.info("🎯 训练元学习器...")
        self.meta_model.fit(oof_predictions, y_train)
        logger.info("✅ Stacking 集成模型训练完成")
        
        # 显示元学习器系数（如果可用）
        if hasattr(self.meta_model, 'coef_'):
            coef = self.meta_model.coef_[0]
            logger.info("📊 元学习器系数:")
            for name, c in zip(self.base_model_names, coef):
                logger.info(f"   {name}: {c:.4f}")
    
    def predict(self, X):
        """进行预测
        
        参数:
            X: 特征矩阵
            
        返回:
            预测标签数组
        """
        if self.trained_base_models is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")
        
        # 获取所有基模型的预测
        base_preds = []
        for model in self.trained_base_models:
            pred = self._get_base_predictions(model, X, self.use_proba)
            base_preds.append(pred)
        
        # 组合成特征矩阵
        meta_features = np.column_stack(base_preds)
        
        # 使用元学习器预测
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """返回预测概率
        
        参数:
            X: 特征矩阵
            
        返回:
            预测概率数组，形状为 (n_samples, n_classes)
        """
        if self.trained_base_models is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")
        
        # 获取所有基模型的预测
        base_preds = []
        for model in self.trained_base_models:
            pred = self._get_base_predictions(model, X, self.use_proba)
            base_preds.append(pred)
        
        # 组合成特征矩阵
        meta_features = np.column_stack(base_preds)
        
        # 使用元学习器预测概率
        return self.meta_model.predict_proba(meta_features)
    
    def evaluate(self, X, y, name):
        """在给定数据集上评估模型性能
        
        参数:
            X: 特征矩阵
            y: 真实标签数组
            name: 数据集名称
            
        返回:
            dict: 主要分类指标
        """
        logger = self._get_logger()
        
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        metrics_dict = compute_classification_metrics(
            y, y_pred, y_proba=y_pred_proba, positive_label=1
        )
        
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


def load_model(model_path, model_name="Model"):
    """加载已训练的模型
    
    参数:
        model_path: 模型文件路径（.joblib 文件）
        model_name: 模型名称（用于日志）
        
    返回:
        加载的模型对象
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        model = joblib.load(model_path)
        Logger(name="ensemble").info(f"✅ 成功加载 {model_name}: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败 {model_path}: {e}")


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="集成学习投票预测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 模型路径参数
    parser.add_argument(
        '--lr-model', type=str,
        default='runs/sklearn_lr_20251112-195315/best.joblib',
        help='逻辑回归模型路径'
    )
    parser.add_argument(
        '--rf-model', type=str,
        default='runs/sklearn_rf_20251113-085457/best.joblib',
        help='随机森林模型路径'
    )
    parser.add_argument(
        '--svm-model', type=str,
        default='runs/sklearn_svm_20251112-195558/best.joblib',
        help='SVM 模型路径'
    )
    
    # 数据参数
    parser.add_argument(
        '--data-dir', type=str, default='features',
        help='特征文件目录'
    )
    parser.add_argument(
        '--train-dirname', type=str, default='train',
        help='训练集子目录名'
    )
    parser.add_argument(
        '--val-dirname', type=str, default='val',
        help='验证集子目录名'
    )
    parser.add_argument(
        '--test-dirname', type=str, default='test',
        help='测试集子目录名'
    )
    
    # 投票参数
    parser.add_argument(
        '--method', type=str, choices=['voting', 'stacking'],
        default='voting', help='集成方法：voting（投票）或 stacking（堆叠）'
    )
    parser.add_argument(
        '--voting', type=str, choices=['hard', 'soft'],
        default='soft', help='投票方式：hard（硬投票）或 soft（软投票）（仅当 method=voting 时有效）'
    )
    parser.add_argument(
        '--weights', type=float, nargs='+', default=None,
        help='模型权重列表（顺序：LR, RF, SVM），默认等权重'
    )
    parser.add_argument(
        '--auto-weights', action='store_true',
        help='根据验证集性能自动计算权重（基于准确率）'
    )
    parser.add_argument(
        '--weight-power', type=float, default=2.0,
        help='权重计算的幂次（默认: 2.0，越大则好模型权重越高）'
    )
    parser.add_argument(
        '--top-k', type=int, default=None,
        help='只使用表现最好的K个模型（默认: 使用所有模型）'
    )
    parser.add_argument(
        '--min-accuracy', type=float, default=None,
        help='最小准确率阈值，低于此值的模型将被排除（默认: 不排除）'
    )
    parser.add_argument(
        '--analyze', action='store_true',
        help='进行详细的模型分析（预测一致性、错误分析等）'
    )
    parser.add_argument(
        '--stacking-cv', type=int, default=5,
        help='Stacking 交叉验证折数（默认: 5）'
    )
    parser.add_argument(
        '--stacking-use-proba', type=lambda x: (str(x).lower() == 'true'),
        default=True, metavar='BOOL',
        help='Stacking 是否使用概率作为特征（默认: True）'
    )
    
    # 输出参数
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='结果保存目录（默认：runs/ensemble_{timestamp}）'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    
    return parser


def analyze_models(models, model_names, datasets, save_dir, logger):
    """分析模型之间的预测一致性、错误模式等
    
    参数:
        models: 模型列表
        model_names: 模型名称列表
        datasets: 数据集字典
        save_dir: 保存目录
        logger: Logger 对象
    """
    X_val, y_val = datasets.get('val', (None, None))
    if X_val is None or y_val is None:
        logger.warning("⚠️  验证集不存在，跳过详细分析")
        return
    
    logger.info("\n" + "="*60)
    logger.info("📊 模型诊断分析报告")
    logger.info("="*60)
    
    # 1. 获取所有模型的预测
    all_predictions = []
    all_probas = []
    
    for model, name in zip(models, model_names):
        try:
            y_pred = model.predict(X_val)
            all_predictions.append(y_pred)
            
            # 获取概率
            try:
                proba = model.predict_proba(X_val)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    all_probas.append(proba[:, 1])
                else:
                    all_probas.append(proba)
            except:
                all_probas.append(None)
        except Exception as e:
            logger.warning(f"   {name}: 预测失败 - {e}")
            all_predictions.append(None)
            all_probas.append(None)
    
    all_predictions = np.array([p for p in all_predictions if p is not None])
    if len(all_predictions) == 0:
        logger.warning("⚠️  无法获取任何模型的预测，跳过分析")
        return
    
    # 2. 计算模型间的一致性
    logger.info("\n📈 模型预测一致性分析:")
    n_models = len(all_predictions)
    n_samples = len(X_val)
    
    # 计算每对模型之间的一致性
    from sklearn.metrics import cohen_kappa_score
    agreement_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i+1, n_models):
            kappa = cohen_kappa_score(all_predictions[i], all_predictions[j])
            agreement_matrix[i, j] = kappa
            agreement_matrix[j, i] = kappa
            logger.info(f"   {model_names[i]} vs {model_names[j]}: Kappa = {kappa:.4f}")
    
    # 3. 分析完全一致的样本
    all_agree = np.all(all_predictions == all_predictions[0], axis=0)
    n_agree = np.sum(all_agree)
    logger.info(f"\n✅ 所有模型预测一致的样本: {n_agree}/{n_samples} ({n_agree/n_samples*100:.2f}%)")
    
    # 4. 分析分歧样本
    disagreements = ~all_agree
    n_disagree = np.sum(disagreements)
    logger.info(f"❌ 模型预测存在分歧的样本: {n_disagree}/{n_samples} ({n_disagree/n_samples*100:.2f}%)")
    
    # 5. 分析每个模型的错误
    logger.info("\n🔍 各模型错误分析:")
    from sklearn.metrics import accuracy_score, confusion_matrix
    model_errors = {}
    for i, (pred, name) in enumerate(zip(all_predictions, model_names)):
        acc = accuracy_score(y_val, pred)
        errors = (pred != y_val)
        model_errors[name] = {
            'accuracy': acc,
            'error_mask': errors,
            'n_errors': np.sum(errors)
        }
        logger.info(f"   {name}: 准确率 = {acc:.4f}, 错误数 = {np.sum(errors)}")
    
    # 6. 分析错误样本的重叠
    logger.info("\n🔗 模型错误重叠分析:")
    error_overlap = {}
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                errors1 = model_errors[name1]['error_mask']
                errors2 = model_errors[name2]['error_mask']
                both_wrong = np.sum(errors1 & errors2)
                only1_wrong = np.sum(errors1 & ~errors2)
                only2_wrong = np.sum(~errors1 & errors2)
                neither_wrong = np.sum(~errors1 & ~errors2)
                
                overlap_ratio = both_wrong / (both_wrong + only1_wrong + only2_wrong) if (both_wrong + only1_wrong + only2_wrong) > 0 else 0
                logger.info(f"   {name1} vs {name2}:")
                logger.info(f"      共同错误: {both_wrong}, 仅{name1}错误: {only1_wrong}, 仅{name2}错误: {only2_wrong}")
                logger.info(f"      错误重叠率: {overlap_ratio:.4f}")
                error_overlap[(name1, name2)] = {
                    'both_wrong': both_wrong,
                    'only1': only1_wrong,
                    'only2': only2_wrong,
                    'overlap_ratio': overlap_ratio
                }
    
    # 7. 分析互补性（一个模型错但另一个对的情况）
    logger.info("\n🔄 模型互补性分析:")
    best_model_idx = np.argmax([model_errors[name]['accuracy'] for name in model_names])
    best_model_name = model_names[best_model_idx]
    best_errors = model_errors[best_model_name]['error_mask']
    
    for i, name in enumerate(model_names):
        if i != best_model_idx:
            other_errors = model_errors[name]['error_mask']
            # 最佳模型错但其他模型对的样本
            best_wrong_other_right = np.sum(best_errors & ~other_errors)
            # 最佳模型对但其他模型错的样本
            best_right_other_wrong = np.sum(~best_errors & other_errors)
            # 互补性：其他模型能纠正最佳模型的错误
            complementarity = best_wrong_other_right / np.sum(best_errors) if np.sum(best_errors) > 0 else 0
            logger.info(f"   {name} 对 {best_model_name} 的互补性:")
            logger.info(f"      {best_model_name}错但{name}对: {best_wrong_other_right}")
            logger.info(f"      {best_model_name}对但{name}错: {best_right_other_wrong}")
            logger.info(f"      互补性比率: {complementarity:.4f}")
    
    # 8. 分析集成可能失败的原因
    logger.info("\n💡 集成效果分析:")
    logger.info(f"   最佳单模型: {best_model_name} (准确率: {model_errors[best_model_name]['accuracy']:.4f})")
    
    # 计算如果使用多数投票的结果
    from scipy import stats
    majority_vote = stats.mode(all_predictions, axis=0)[0].flatten()
    majority_acc = accuracy_score(y_val, majority_vote)
    logger.info(f"   多数投票准确率: {majority_acc:.4f}")
    
    if majority_acc < model_errors[best_model_name]['accuracy']:
        logger.info(f"   ⚠️  多数投票不如最佳单模型，下降: {model_errors[best_model_name]['accuracy'] - majority_acc:.4f}")
        logger.info("\n   可能原因:")
        
        # 原因1: 弱模型拖累
        weak_models = [name for name in model_names if model_errors[name]['accuracy'] < model_errors[best_model_name]['accuracy'] - 0.05]
        if weak_models:
            logger.info(f"   1. 弱模型拖累: {', '.join(weak_models)} 表现明显较差")
        
        # 原因2: 错误高度重叠
        avg_overlap = np.mean([v['overlap_ratio'] for v in error_overlap.values()])
        if avg_overlap > 0.7:
            logger.info(f"   2. 模型错误高度重叠 (平均重叠率: {avg_overlap:.4f})，缺乏多样性")
        
        # 原因3: 最佳模型已经很好，其他模型无法提供有效补充
        if model_errors[best_model_name]['accuracy'] > 0.8:
            logger.info(f"   3. 最佳模型已经表现很好 ({model_errors[best_model_name]['accuracy']:.4f})，集成收益有限")
        
        # 原因4: 分歧样本中，弱模型经常占多数
        if n_disagree > 0:
            disagree_predictions = all_predictions[:, disagreements]
            disagree_labels = y_val[disagreements]
            disagree_majority = stats.mode(disagree_predictions, axis=0)[0].flatten()
            disagree_majority_acc = accuracy_score(disagree_labels, disagree_majority)
            logger.info(f"   4. 在分歧样本中，多数投票准确率: {disagree_majority_acc:.4f}")
            if disagree_majority_acc < 0.5:
                logger.info(f"      在分歧样本中，多数投票表现很差，说明弱模型在分歧时占主导")
    
    logger.info("\n" + "="*60)
    
    # 保存分析结果
    analysis_data = {
        'agreement_matrix': agreement_matrix.tolist(),
        'model_names': model_names,
        'n_agree': int(n_agree),
        'n_disagree': int(n_disagree),
        'model_errors': {name: {
            'accuracy': float(model_errors[name]['accuracy']),
            'n_errors': int(model_errors[name]['n_errors'])
        } for name in model_names},
        'error_overlap': {f"{k[0]}_vs_{k[1]}": {
            'both_wrong': int(v['both_wrong']),
            'only1': int(v['only1']),
            'only2': int(v['only2']),
            'overlap_ratio': float(v['overlap_ratio'])
        } for k, v in error_overlap.items()},
        'majority_vote_accuracy': float(majority_acc),
        'best_single_accuracy': float(model_errors[best_model_name]['accuracy'])
    }
    
    analysis_path = save_dir / 'model_analysis.json'
    import json
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 分析结果已保存: {analysis_path}")


def generate_visualizations(ensemble, datasets, results, save_dir, logger):
    """生成集成模型的可视化结果
    
    参数:
        ensemble: VotingEnsemble 对象
        datasets: 数据集字典，键为 'train'/'val'/'test'，值为 (X, y) 元组
        results: 评估结果字典
        save_dir: 保存目录
        logger: Logger 对象
    """
    figures_dir = Path(save_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PR 曲线
    try:
        has_pr_data = any(
            isinstance(results.get(split), dict) and (
                results[split].get('precision_curve') is not None or
                isinstance(results[split].get('pr_curve'), dict)
            )
            for split in ['train', 'val', 'test']
        )
        if has_pr_data:
            method_name = "Stacking" if isinstance(ensemble, StackingEnsemble) else "投票（{}）".format("硬" if getattr(ensemble, 'voting', 'soft') == 'hard' else "软")
            plot_split_metrics(
                results,
                metric='pr',
                title="集成模型 PR 曲线（{}）".format(method_name),
                save_path=figures_dir / "ensemble_pr_curve.png"
            )
            logger.info("📊 PR 曲线已保存")
    except Exception as exc:
        logger.debug(f"生成 PR 曲线失败: {exc}")
    
    # 2. 混淆矩阵
    for split in ['train', 'val', 'test']:
        X, y = datasets.get(split, (None, None))
        if X is None or y is None:
            continue
        
        try:
            y_pred = ensemble.predict(X)
            class_names = ['Cat', 'Dog']
            from sklearn import metrics
            cm = metrics.confusion_matrix(y, y_pred)
            method_name = "Stacking" if isinstance(ensemble, StackingEnsemble) else "投票（{}）".format("硬" if getattr(ensemble, 'voting', 'soft') == 'hard' else "软")
            plot_confusion_matrix(
                cm,
                class_names=class_names,
                title="集成模型 {} 混淆矩阵（{}）".format(
                    split.capitalize(),
                    method_name
                ),
                save_path=figures_dir / "ensemble_{}_confusion.png".format(split)
            )
            logger.info(f"📊 {split} 混淆矩阵已保存")
        except Exception as exc:
            logger.debug(f"生成 {split} 混淆矩阵失败: {exc}")
        
        # 3. ROC 曲线
        try:
            y_scores = ensemble.predict_proba(X)[:, 1]
            if len(np.unique(y)) == 2:
                plot_roc_curve(
                    y,
                    y_scores,
                    title="集成模型 {} ROC 曲线".format(split.capitalize()),
                    save_path=figures_dir / "ensemble_{}_roc.png".format(split)
                )
                logger.info(f"📊 {split} ROC 曲线已保存")
        except Exception as exc:
            logger.debug(f"生成 {split} ROC 曲线失败: {exc}")


def main():
    """主函数：执行集成学习投票预测"""
    parser = build_parser()
    args = parser.parse_args()
    
    # 生成时间戳和保存目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.save_dir is None:
        args.save_dir = f"runs/ensemble_{timestamp}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(
        name="ensemble_voting",
        log_dir=str(save_dir),
        filename="ensemble_voting.log",
        level=args.log_level,
    )
    
    start_time = time.time()
    
    try:
        # 日志配置信息
        cfg_lines = [
            "集成方法: {}".format(args.method),
            "数据目录: {}".format(args.data_dir),
            "保存目录: {}".format(save_dir),
        ]
        if args.method == 'voting':
            cfg_lines.append("投票方式: {}".format(args.voting))
        elif args.method == 'stacking':
            cfg_lines.append("交叉验证折数: {}".format(args.stacking_cv))
            cfg_lines.append("使用概率特征: {}".format(args.stacking_use_proba))
        if args.auto_weights:
            cfg_lines.append("权重策略: 自动计算（基于验证集性能，幂次={}）".format(args.weight_power))
        elif args.weights:
            cfg_lines.append("模型权重: {}".format(args.weights))
        else:
            cfg_lines.append("权重策略: 等权重")
        if args.top_k is not None:
            cfg_lines.append("模型筛选: 只使用表现最好的 {} 个模型".format(args.top_k))
        if args.min_accuracy is not None:
            cfg_lines.append("准确率阈值: 排除准确率 < {} 的模型".format(args.min_accuracy))
        logger.block("开始集成学习投票预测", cfg_lines)
        
        # 1. 加载模型
        logger.info("📥 加载已训练的模型...")
        lr_model = load_model(args.lr_model, "逻辑回归")
        rf_model = load_model(args.rf_model, "随机森林")
        svm_model = load_model(args.svm_model, "SVM")
        
        models = [lr_model, rf_model, svm_model]
        model_names = ["逻辑回归", "随机森林", "SVM"]
        
        # 2. 加载数据
        logger.info("📊 加载数据集...")
        datasets = load_train_val_test(
            args.data_dir,
            args.train_dirname,
            args.val_dirname,
            args.test_dirname,
            logger=logger
        )
        
        # 3. 评估单个模型性能（用于自动权重计算和模型筛选）
        weights = args.weights
        model_scores = None
        
        # 如果需要自动权重或模型筛选，先评估所有模型
        if args.auto_weights or args.top_k is not None or args.min_accuracy is not None:
            logger.info("📊 评估单个模型性能...")
            X_val, y_val = datasets.get('val', (None, None))
            if X_val is None or y_val is None:
                logger.warning("⚠️  验证集不存在，无法评估模型性能")
                if args.auto_weights:
                    logger.warning("   无法自动计算权重，使用等权重")
                    weights = None
            else:
                model_scores = []
                for model, name in zip(models, model_names):
                    try:
                        y_pred = model.predict(X_val)
                        from sklearn.metrics import accuracy_score
                        acc = accuracy_score(y_val, y_pred)
                        model_scores.append(acc)
                        logger.info(f"   {name}: 验证集准确率 = {acc:.4f}")
                    except Exception as e:
                        logger.warning(f"   {name}: 评估失败 ({e})，使用默认分数")
                        model_scores.append(0.0)  # 默认分数
                
                model_scores = np.array(model_scores)
                
                # 根据准确率筛选模型
                if args.min_accuracy is not None:
                    keep_mask = model_scores >= args.min_accuracy
                    if not np.any(keep_mask):
                        logger.warning("⚠️  所有模型都被过滤，使用所有模型")
                        keep_mask = np.ones(len(models), dtype=bool)
                    else:
                        filtered_count = np.sum(~keep_mask)
                        if filtered_count > 0:
                            logger.info(f"🔍 过滤掉 {filtered_count} 个低性能模型（准确率 < {args.min_accuracy:.4f}）")
                            models = [m for m, keep in zip(models, keep_mask) if keep]
                            model_names = [n for n, keep in zip(model_names, keep_mask) if keep]
                            model_scores = model_scores[keep_mask]
                
                # 根据 top_k 筛选模型
                if args.top_k is not None and args.top_k < len(models):
                    top_indices = np.argsort(model_scores)[-args.top_k:][::-1]
                    logger.info(f"🔍 只使用表现最好的 {args.top_k} 个模型")
                    models = [models[i] for i in top_indices]
                    model_names = [model_names[i] for i in top_indices]
                    model_scores = model_scores[top_indices]
                    logger.info(f"   选中的模型: {', '.join(model_names)}")
                
                # 自动计算权重
                if args.auto_weights and model_scores is not None:
                    if np.all(model_scores > 0):
                        # 使用指定的幂次来增强好模型的权重
                        weights = (model_scores ** args.weight_power).tolist()
                        logger.info(f"✅ 自动计算权重（幂次={args.weight_power}）: {weights}")
                    else:
                        weights = model_scores.tolist()
                        logger.info(f"✅ 自动计算权重: {weights}")
        
        # 检查模型数量
        if len(models) == 0:
            raise ValueError("没有可用的模型进行集成")
        if len(models) == 1:
            logger.warning("⚠️  只有一个模型，集成效果等同于单个模型")
        
        # 4. 创建集成模型
        if args.method == 'voting':
            logger.info("🔧 创建投票集成模型...")
            ensemble = VotingEnsemble(
                models=models,
                model_names=model_names,
                voting=args.voting,
                weights=weights,
            )
            ensemble.logger = logger
            
            voting_type = "硬投票" if args.voting == 'hard' else "软投票"
            logger.info(f"✅ 集成模型创建完成（{voting_type}）")
            if weights:
                weight_info = ", ".join(
                    f"{name}={w:.3f}" for name, w in zip(model_names, ensemble.weights)
                )
                logger.info(f"   权重: {weight_info}")
            
            # 5. 评估模型
            logger.info("📈 评估集成模型...")
            results = {}
            for split_name in ['train', 'val', 'test']:
                X, y = datasets.get(split_name, (None, None))
                if X is not None and y is not None:
                    results[split_name] = ensemble.evaluate(X, y, name=split_name.capitalize())
                else:
                    results[split_name] = None
                    
        elif args.method == 'stacking':
            logger.info("🔧 创建 Stacking 集成模型...")
            ensemble = StackingEnsemble(
                base_models=models,
                base_model_names=model_names,
                cv_folds=args.stacking_cv,
                use_proba=args.stacking_use_proba,
                random_state=42,
            )
            ensemble.logger = logger
            
            # 训练 Stacking 模型
            X_train, y_train = datasets.get('train', (None, None))
            if X_train is None or y_train is None:
                raise ValueError("训练集不存在，无法训练 Stacking 模型")
            
            ensemble.fit(X_train, y_train)
            logger.info("✅ Stacking 集成模型训练完成")
            
            # 5. 评估模型
            logger.info("📈 评估集成模型...")
            results = {}
            for split_name in ['train', 'val', 'test']:
                X, y = datasets.get(split_name, (None, None))
                if X is not None and y is not None:
                    results[split_name] = ensemble.evaluate(X, y, name=split_name.capitalize())
                else:
                    results[split_name] = None
        else:
            raise ValueError(f"未知的集成方法: {args.method}")
        
        # 6. 模型诊断分析
        if args.analyze:
            logger.info("🔍 进行模型诊断分析...")
            analyze_models(models, model_names, datasets, save_dir, logger)
        
        # 7. 生成可视化
        logger.info("🎨 生成可视化结果...")
        generate_visualizations(ensemble, datasets, results, save_dir, logger)
        
        # 8. 保存结果
        import json
        weights_list = None
        if args.method == 'voting' and (args.weights or args.auto_weights):
            if isinstance(ensemble, VotingEnsemble):
                if isinstance(ensemble.weights, np.ndarray):
                    weights_list = ensemble.weights.tolist()
                else:
                    weights_list = list(ensemble.weights)
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'method': args.method,
            'voting': args.voting if args.method == 'voting' else None,
            'weights': weights_list,
            'auto_weights': args.auto_weights,
            'stacking_cv': args.stacking_cv if args.method == 'stacking' else None,
            'stacking_use_proba': args.stacking_use_proba if args.method == 'stacking' else None,
            'model_paths': {
                'lr': str(args.lr_model),
                'rf': str(args.rf_model),
                'svm': str(args.svm_model),
            },
            'results': results,
        }
        results_path = save_dir / 'ensemble_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 结果已保存: {results_path}")
        
        # 9. 总结和性能对比
        total_time = time.time() - start_time
        best_acc = None
        if results.get("val") and results["val"].get("accuracy") is not None:
            best_acc = results["val"]["accuracy"]
        elif results.get("test") and results["test"].get("accuracy") is not None:
            best_acc = results["test"]["accuracy"]
        
        summary_lines = [
            "耗时: {}".format(Logger.format_duration(total_time)),
            "验证集准确率: {:.4f}".format(results["val"]["accuracy"]) if results.get("val") and results["val"].get("accuracy") is not None else "验证集准确率: -",
            "测试集准确率: {:.4f}".format(results["test"]["accuracy"]) if results.get("test") and results["test"].get("accuracy") is not None else "测试集准确率: -",
            "结果目录: {}".format(save_dir),
        ]
        
        # 如果评估了单个模型，显示对比
        if model_scores is not None and len(model_scores) > 0:
            best_single_acc = np.max(model_scores)
            ensemble_val_acc = results.get("val", {}).get("accuracy")
            if ensemble_val_acc is not None:
                improvement = ensemble_val_acc - best_single_acc
                if improvement > 0.001:  # 提升超过0.1%
                    summary_lines.append("📈 相比最佳单模型（验证集）提升: +{:.4f}".format(improvement))
                elif improvement < -0.001:  # 下降超过0.1%
                    summary_lines.append("📉 相比最佳单模型（验证集）下降: {:.4f}".format(improvement))
                else:
                    summary_lines.append("➡️  与最佳单模型（验证集）持平")
        
        logger.block("集成学习完成", summary_lines)
        
        return ensemble, results
        
    except Exception as e:
        logger.exception("集成学习过程中发生错误: %s", e)
        raise


if __name__ == "__main__":
    main()

