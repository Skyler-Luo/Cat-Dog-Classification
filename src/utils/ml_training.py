"""传统机器学习训练通用工具函数。"""

import random
import time
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn import metrics

from .config import MODEL_TYPE_NAMES
from .logger import Logger
from tools.visualization import (
    plot_split_metrics,
    plot_confusion_matrix,
    plot_metric_curves,
    plot_roc_curve,
    plot_cv_results,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_features(features_dir, split, verbose=True, logger=None):
    features_path = Path(features_dir) / '{}_features.joblib'.format(split)
    if not features_path.exists():
        raise FileNotFoundError(
            '特征文件不存在: {}\n💡 提示：请先运行 \'python tools/extract_best_features.py\' 提取特征'.format(features_path)
        )
    Logger.maybe_print(verbose, logger, '📥 加载特征: {}'.format(features_path))
    data = joblib.load(features_path)
    if isinstance(data, dict):
        X = data.get('features', data.get('X'))
        y = data.get('labels', data.get('y'))
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        X, y = data[0], data[1]
    else:
        raise ValueError('无法解析特征文件格式: {}\n期望格式: dict 或 (X, y) 元组'.format(features_path))
    if X is None:
        raise ValueError('特征矩阵为 None: {}'.format(features_path))
    y = np.array(y) if y is not None else None
    if len(X) == 0:
        raise ValueError('特征矩阵为空: {}'.format(features_path))
    if X.ndim != 2:
        raise ValueError('特征矩阵维度错误: 期望 2D，实际 {}D，形状: {}'.format(X.ndim, X.shape))
    if y is not None and len(X) != len(y):
        raise ValueError('特征和标签数量不匹配: X={}, y={}'.format(len(X), len(y)))
    Logger.maybe_print(verbose, logger, '   • 特征形状: {}'.format(X.shape))
    if y is not None:
        Logger.maybe_print(verbose, logger, '   • 标签数量: {}'.format(len(y)))
        Logger.maybe_print(verbose, logger, '   • 类别: {}'.format(np.unique(y)))
    return X, y


def load_train_val_test(features_dir, train_split='train', val_split='val', test_split='test', verbose=True, logger=None):
    Logger.maybe_print(verbose, logger, '📊 加载数据集...')
    datasets = {}
    for split_name, split_key in [(train_split, 'train'), (val_split, 'val'), (test_split, 'test')]:
        try:
            X, y = load_features(features_dir, split_name, verbose=verbose, logger=logger)
            datasets[split_key] = (X, y)
        except FileNotFoundError:
            Logger.maybe_print(verbose, logger, '⚠️  {} 集不存在，跳过'.format(split_key.upper()))
            datasets[split_key] = (None, None)
    Logger.maybe_print(verbose, logger, '📊 数据集规模:')
    for split_key in ['train', 'val', 'test']:
        X, y = datasets[split_key]
        if X is not None:
            msg = '   • {}: {} 样本'.format(split_key.upper().ljust(5), len(X))
        else:
            msg = '   • {}: -'.format(split_key.upper().ljust(5))
        Logger.maybe_print(verbose, logger, msg)
    return datasets


def evaluate_all_splits(trainer, datasets, verbose=True, logger=None):
    Logger.maybe_print(verbose, logger, '📊 评估模型...')
    results = {}
    for split_name in ['train', 'val', 'test']:
        X, y = datasets.get(split_name, (None, None))
        if X is not None and y is not None:
            results[split_name] = trainer.evaluate(X, y, name=split_name.capitalize())
        else:
            results[split_name] = None
    return results


def build_config(args, model_type, model_specific_config):
    return {
        'data': {
            'features_dir': getattr(args, 'data_dir', 'features/best_features'),
            'train_dirname': getattr(args, 'train_dirname', 'train'),
            'val_dirname': getattr(args, 'val_dirname', 'val'),
            'test_dirname': getattr(args, 'test_dirname', 'test'),
        },
        model_type: model_specific_config,
        'runtime': {
            'n_jobs': getattr(args, 'n_jobs', 8),
            'seed': getattr(args, 'seed', 42),
            'timestamp': datetime.now().isoformat(),
        },
        'preprocessing': '特征已预处理（StandardScaler + PCA 已在特征提取阶段完成）',
    }


def parse_numeric_or_str(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def parse_param_list(values):
    return [parse_numeric_or_str(v) for v in values]


def run_sklearn_training(args, trainer_class, trainer_kwargs, model_type, hyperparams_info=None, logger=None):
    set_seed(args.seed)
    start_time = time.time()
    model_name = MODEL_TYPE_NAMES.get(model_type, model_type.upper())
    logger = logger or Logger(name="sklearn_training")
    logger.header(model_name, args)
    if hyperparams_info:
        logger.info('🧪 超参数: {}'.format(hyperparams_info))
    datasets = load_train_val_test(
        args.data_dir,
        args.train_dirname,
        args.val_dirname,
        args.test_dirname,
        logger=logger
    )
    X_train, y_train = datasets['train']
    if X_train is None or len(X_train) == 0:
        raise RuntimeError('训练集为空或加载失败')
    logger.info('🔧 创建并训练 {}...'.format(model_name))
    trainer = trainer_class(**trainer_kwargs)
    try:
        setattr(trainer, 'logger', logger)
    except Exception:
        pass
    trainer.fit(X_train, y_train)
    results = evaluate_all_splits(trainer, datasets, logger=logger)
    run_dir = Path(args.save_path).parent
    model_slug = str(model_type).lower().replace(' ', '_')
    _generate_visualizations(trainer, datasets, results, run_dir, model_name, model_slug, logger)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config = build_config(args, model_type=model_type, model_specific_config=trainer_kwargs)
    try:
        trainer.save(save_path, save_results=results, config=config, logger=logger)
    except TypeError:
        trainer.save(save_path, save_results=results, config=config)
    elapsed = time.time() - start_time
    logger.summary(save_path, elapsed)
    return trainer, results


def _generate_visualizations(trainer, datasets, results, run_dir, model_name, model_slug, logger):
    figures_dir = Path(run_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        slug = model_slug or model_name.lower()
        has_pr_data = any(
            isinstance(results.get(split), dict) and (
                results[split].get('precision_curve') is not None or
                isinstance(results[split].get('pr_curve'), dict)
            )
            for split in ['train', 'val', 'test']
        )
        if has_pr_data:
            plot_split_metrics(
                results,
                metric='pr',
                title="{} PR 曲线".format(model_name),
                save_path=figures_dir / "{}_pr_curve.png".format(slug)
            )
    except Exception as exc:
        if logger:
            logger.debug("生成 PR 曲线失败: {}".format(exc))
    history = results.get('history')
    if history:
        try:
            plot_metric_curves(
                history,
                title="{} Metric Curves".format(model_name),
                save_path=figures_dir / "metric_curves.png"
            )
        except Exception as exc:
            if logger:
                logger.debug("生成训练曲线失败: {}".format(exc))
    for split in ['train', 'val', 'test']:
        X, y = datasets.get(split, (None, None))
        if X is None or y is None:
            continue
        try:
            y_pred = trainer.best_model.predict(X)
            class_names = [str(cls) for cls in sorted(np.unique(y))]
            cm = metrics.confusion_matrix(y, y_pred)
            plot_confusion_matrix(
                cm,
                class_names=class_names,
                title="{} {} 混淆矩阵".format(model_name, split.capitalize()),
                save_path=figures_dir / "{}_{}_confusion.png".format(slug, split)
            )
        except Exception as exc:
            if logger:
                logger.debug("生成 {} 混淆矩阵失败: {}".format(split, exc))
        try:
            y_scores = None
            if hasattr(trainer.best_model, 'predict_proba'):
                proba = trainer.best_model.predict_proba(X)
                if proba.ndim == 2:
                    y_scores = proba[:, 1]
                else:
                    y_scores = proba
            elif hasattr(trainer.best_model, 'decision_function'):
                y_scores = trainer.best_model.decision_function(X)
            if y_scores is not None and len(np.unique(y)) == 2:
                plot_roc_curve(
                    y,
                    y_scores,
                    title="{} {} ROC".format(model_name, split.capitalize()),
                    save_path=figures_dir / "{}_{}_roc.png".format(slug, split)
                )
        except Exception as exc:
            if logger:
                logger.debug("生成 {} ROC 曲线失败: {}".format(split, exc))
    cv_results = getattr(trainer, 'cv_results_', None)
    if cv_results:
        try:
            plot_cv_results(
                cv_results,
                title="{} 参数搜索结果".format(model_name),
                save_path=figures_dir / "cv_results.png"
            )
        except Exception as exc:
            if logger:
                logger.debug("生成参数搜索可视化失败: {}".format(exc))


def compute_classification_metrics(y_true, y_pred, y_proba=None, positive_label=1, zero_division='warn'):
    """计算分类任务的常用指标。
    
    参数:
        y_true: 真实标签数组，形状为(n_samples,)
        y_pred: 预测标签数组，形状为(n_samples,)
        y_proba: 预测为正类的概率或决策得分，形状为(n_samples,) 或 (n_samples, 2)（可选）
        positive_label: 正类标签（默认: 1）
        zero_division: 当出现除零时的处理方式（'warn'/'0'/'1'，默认: 'warn'）
        
    返回:
        dict: 指标字典，包含 accuracy、precision、recall、f1。当提供概率或得分时，还包含:
            - auc: ROC 曲线下面积
            - average_precision: 平均精确率（AP）
            - pr_curve: 包含 precision、recall、thresholds 列表的字典
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, pos_label=positive_label, zero_division=zero_division)
    recall = metrics.recall_score(y_true, y_pred, pos_label=positive_label, zero_division=zero_division)
    f1 = metrics.f1_score(y_true, y_pred, pos_label=positive_label, zero_division=zero_division)
    result = {}
    result['accuracy'] = float(acc)
    result['precision'] = float(precision)
    result['recall'] = float(recall)
    result['f1'] = float(f1)
    # AUC 仅在提供概率时计算
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                pos_scores = y_proba[:, 1]
            else:
                pos_scores = y_proba
            auc = metrics.roc_auc_score(y_true, pos_scores)
            result['auc'] = float(auc)
            precision_curve, recall_curve, thresholds = metrics.precision_recall_curve(
                y_true,
                pos_scores,
                pos_label=positive_label
            )
            average_precision = metrics.average_precision_score(y_true, pos_scores, pos_label=positive_label)
            result['average_precision'] = float(average_precision)
            pr_curve_data = {
                'precision': [float(value) for value in precision_curve],
                'recall': [float(value) for value in recall_curve],
                'thresholds': [float(value) for value in thresholds],
            }
            result['pr_curve'] = pr_curve_data
        except Exception:
            pass
    return result


def save_sklearn_model(model, save_path, model_type, save_results=None, config=None, extra_model_info=None, logger=None):
    """统一的 sklearn 模型保存函数（含结果与配置）。
    
    参数:
        model: 已训练的 sklearn 模型对象
        save_path: 保存路径（str 或 Path）
        model_type: 模型类型名称（str），例如 'SVM'、'LogisticRegression'、'RandomForest'
        save_results: 训练/评估结果字典（可选）
        config: 训练配置字典（可选）
        extra_model_info: 额外的模型信息字典（可选）
    """
    import json
    import joblib
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    (logger.info('💾 模型已保存至: {}'.format(save_path)) if logger else print('💾 模型已保存至: {}'.format(save_path)))
    if save_results is not None or config is not None:
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': model_type,
            }
        }
        if extra_model_info:
            results_data['model_info'].update(extra_model_info)
        if config is not None:
            results_data['config'] = config
        if save_results is not None:
            results_data['results'] = save_results
        results_path = save_path.parent / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        (logger.info('📝 训练配置与结果已保存: {}'.format(results_path)) if logger else print('📝 训练配置与结果已保存: {}'.format(results_path)))
