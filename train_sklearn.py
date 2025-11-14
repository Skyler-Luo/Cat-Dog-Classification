"""
猫狗分类 - 统一传统机器学习训练脚本

本脚本统一管理 SVM、逻辑回归、随机森林 三种传统机器学习模型的训练流程。
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

from src.utils.ml_training import (
    parse_param_list,
    run_sklearn_training,
)
from src.utils.logger import Logger
from src.models.svm import SVMTrainer
from src.models.logistic_regression import LogisticRegressionTrainer
from src.models.random_forest import RandomForestTrainer


def _default_save_path(model, timestamp=None):
    """生成带时间戳的默认保存路径
    
    参数:
        model: 模型类型（"svm", "logreg", "random_forest"）
        timestamp: 时间戳字符串（格式: YYYYMMDD-HHMMSS），如果为None则自动生成
        
    返回:
        str: 保存路径，格式为 runs/sklearn_{model}_{timestamp}/best.joblib
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 模型名称映射
    model_name_map = {
        "svm": "svm",
        "logreg": "lr",
        "random_forest": "rf",
    }
    
    model_name = model_name_map.get(model, "sklearn")
    base_dir = Path("runs") / "sklearn_{}_{}".format(model_name, timestamp)
    return str(base_dir / "best.joblib")


def build_parser(default_model=None):
    """构建统一训练参数解析器。"""
    parser = argparse.ArgumentParser(
        description="传统机器学习统一训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 通用训练参数
    parser.add_argument('--data-dir', type=str, default='features', help='特征文件目录')
    parser.add_argument('--train-dirname', type=str, default='train', help='训练集名称')
    parser.add_argument('--val-dirname', type=str, default='val', help='验证集名称')
    parser.add_argument('--test-dirname', type=str, default='test', help='测试集名称')
    parser.add_argument('--n-jobs', type=int, default=16, help='并行线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cv-folds', type=int, default=5, help='交叉验证折数')

    # 通用控制参数
    parser.add_argument("--model", type=str, choices=["svm", "logreg", "random_forest"],
                        default=default_model or "svm", help="选择训练模型")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--save-path", type=str, default=None, help="模型保存路径（默认按模型自动选择）")
    parser.add_argument("--no-search", action="store_true", help="禁用超参数搜索，直接训练默认配置模型")

    # SVM 特定参数
    parser.add_argument("--svm-c-values", type=str, nargs="+", default=["0.1", "1", "10", "100"],
                        help="SVM 的 C 候选值")
    parser.add_argument("--svm-gamma-values", type=str, nargs="+", default=["scale", "0.001", "0.0001"],
                        help="SVM 的 gamma 候选值")
    parser.add_argument("--svm-probability", action="store_true", help="SVM 启用概率估计")
    # SVM 默认参数（在禁用搜索时生效）
    parser.add_argument("--svm-default-c", type=float, default=10.0, help="SVM 默认 C（no-search 时使用）")
    parser.add_argument("--svm-default-gamma", type=str, default="scale", help="SVM 默认 gamma（no-search 时使用）")

    # 逻辑回归 特定参数
    parser.add_argument("--lr-solvers", type=str, nargs="+", default=["liblinear", "lbfgs", "saga"], help="逻辑回归 求解器候选列表")
    parser.add_argument("--lr-c-values", type=str, nargs="+", default=["0.001", "0.01", "0.1", "1.0", "10.0", "100.0"],
                        help="逻辑回归 的 C 候选值")
    parser.add_argument("--lr-penalty-types", type=str, nargs="+", default=["l1", "l2"],
                        help="逻辑回归 的 penalty 候选值")
    parser.add_argument("--lr-l1-ratios", type=str, nargs="+", default=["0.1", "0.5", "0.7", "0.9"],
                        help="逻辑回归 的 l1 候选值")
    parser.add_argument("--lr-max-iter", type=int, default=1000, help="逻辑回归 的最大迭代次数")
    # 逻辑回归 默认参数（在禁用搜索时生效）
    parser.add_argument("--lr-default-c", type=float, default=1.0, help="逻辑回归 默认 C（no-search 时使用）")
    parser.add_argument("--lr-default-penalty", type=str, default="l2", help="逻辑回归 默认 penalty（no-search 时使用）")
    parser.add_argument("--lr-default-l1-ratio", type=float, default=None, help="逻辑回归 默认 l1_ratio（仅 elasticnet 有效）")
    parser.add_argument("--lr-default-solver", type=str, default="liblinear", help="逻辑回归 默认 solver（no-search 时使用）")

    # 随机森林 特定参数
    parser.add_argument("--rf-n-estimators", type=str, nargs="+", default=["200", "300", "500"],
                        help="随机森林 的 n_estimators 候选值列表")
    parser.add_argument("--rf-max-depth", type=str, nargs="+", default=["10", "15", "20", "30"],
                        help="随机森林 的 max_depth 候选值列表（None表示不限制深度）")
    parser.add_argument("--rf-min-samples-split", type=str, nargs="+", default=["5", "10", "20"],
                        help="随机森林 的 min_samples_split 候选值列表")
    parser.add_argument("--rf-min-samples-leaf", type=str, nargs="+", default=["2", "4", "8"],
                        help="随机森林 的 min_samples_leaf 候选值列表")
    parser.add_argument("--rf-max-features", type=str, nargs="+", default=["sqrt", "log2"],
                        help="随机森林 的 max_features 候选值列表")
    parser.add_argument("--rf-max-samples", type=float, default=None,
                        help="随机森林 Bootstrap 采样比例（0.0-1.0，例如0.8表示使用80%%样本，None表示使用全部）")
    # 随机森林 默认参数（在禁用搜索时生效）
    parser.add_argument("--rf-default-n-estimators", type=int, default=500, help="随机森林 默认 n_estimators（no-search 时使用）")
    parser.add_argument("--rf-default-max-depth", type=int, default=15, help="随机森林 默认 max_depth（no-search 时使用）")
    parser.add_argument("--rf-default-min-samples-split", type=int, default=10, help="随机森林 默认 min_samples_split（no-search 时使用）")
    parser.add_argument("--rf-default-min-samples-leaf", type=int, default=2, help="随机森林 默认 min_samples_leaf（no-search 时使用）")
    parser.add_argument("--rf-default-max-features", type=str, default="sqrt", help="随机森林 默认 max_features（no-search 时使用）")
    parser.add_argument("--rf-default-max-samples", type=float, default=None, help="随机森林 默认 max_samples（no-search 时使用，0.8表示使用80%样本）")

    # 通用评估指标
    parser.add_argument("--scoring", type=str, default="accuracy", choices=["accuracy", "precision", "recall", "f1"], help="评分指标")

    return parser


def _select_trainer_and_kwargs(args):
    """根据 args 返回 trainer_class、trainer_kwargs、模型标识、超参描述。"""
    if args.model == "svm":
        trainer_class = SVMTrainer
        c_values = parse_param_list(args.svm_c_values)
        gamma_values = parse_param_list(args.svm_gamma_values)
        trainer_kwargs = {
            "c_values": c_values,
            "gamma_values": gamma_values,
            "cv_folds": args.cv_folds,
            "n_jobs": args.n_jobs,
            "random_state": args.seed,
            "svm_probability": args.svm_probability,
            "scoring": args.scoring,
            "do_search": not args.no_search,
            "default_params": {
                "C": args.svm_default_c,
                "gamma": args.svm_default_gamma,
                "probability": args.svm_probability,
                "kernel": "rbf",
                "random_state": args.seed,
            },
        }
        model_type = "svm"
        hyper = "C={}, gamma={}, prob={}".format(c_values, gamma_values, args.svm_probability)
        return trainer_class, trainer_kwargs, model_type, hyper

    if args.model == "logreg":
        trainer_class = LogisticRegressionTrainer
        trainer_kwargs = {
            "cv_folds": args.cv_folds,
            "n_jobs": args.n_jobs,
            "random_state": args.seed,
            "solvers": args.lr_solvers,
            "scoring": args.scoring,
            "do_search": not args.no_search,
            "default_params": {
                "C": args.lr_default_c,
                "penalty": args.lr_default_penalty,
                "l1_ratio": args.lr_default_l1_ratio,
                "max_iter": args.lr_max_iter,
                "solver": args.lr_default_solver,
                "random_state": args.seed,
            },
        }
        model_type = "logistic_regression"
        hyper = "cv={}, solvers={}, scoring={}".format(
            args.cv_folds, args.lr_solvers, args.scoring
        )
        return trainer_class, trainer_kwargs, model_type, hyper

    if args.model == "random_forest":
        trainer_class = RandomForestTrainer
        rf_n_estimators = [int(float(v)) for v in args.rf_n_estimators]
        rf_max_depth = []
        for v in args.rf_max_depth:
            if str(v).lower() == "none":
                rf_max_depth.append(None)
            else:
                rf_max_depth.append(int(float(v)))
        rf_min_samples_split = [int(float(v)) for v in args.rf_min_samples_split]
        rf_min_samples_leaf = [int(float(v)) for v in args.rf_min_samples_leaf]
        rf_max_features = []
        for v in args.rf_max_features:
            v_lower = str(v).lower()
            if v_lower == "none":
                rf_max_features.append(None)
            elif v_lower in ["sqrt", "log2"]:
                rf_max_features.append(v_lower)
            else:
                try:
                    rf_max_features.append(float(v))
                except (TypeError, ValueError):
                    rf_max_features.append(v)
        # 处理 max_samples 参数
        rf_max_samples = args.rf_max_samples
        if rf_max_samples is not None and (rf_max_samples <= 0 or rf_max_samples > 1):
            raise ValueError("--rf-max-samples 必须在 (0, 1] 范围内")
        
        # 确定实际使用的 max_samples
        # 在搜索模式下，使用 --rf-max-samples（如果设置了）
        # 在 no-search 模式下，优先使用 --rf-default-max-samples，否则使用 --rf-max-samples
        actual_max_samples = rf_max_samples
        if args.no_search:
            # no-search 模式下，优先使用 default_max_samples
            if args.rf_default_max_samples is not None:
                actual_max_samples = args.rf_default_max_samples
        
        trainer_kwargs = {
            "n_estimators_values": rf_n_estimators,
            "max_depth_values": rf_max_depth,
            "min_samples_split_values": rf_min_samples_split,
            "min_samples_leaf_values": rf_min_samples_leaf,
            "max_features_values": rf_max_features,
            "cv_folds": args.cv_folds,
            "n_jobs": args.n_jobs,
            "random_state": args.seed,
            "scoring": args.scoring,
            "do_search": not args.no_search,
            "max_samples": actual_max_samples if not args.no_search else None,  # 搜索模式使用
            "default_params": {
                "n_estimators": args.rf_default_n_estimators,
                "max_depth": args.rf_default_max_depth,
                "min_samples_split": args.rf_default_min_samples_split,
                "min_samples_leaf": args.rf_default_min_samples_leaf,
                "max_features": args.rf_default_max_features,
                "random_state": args.seed,
                "n_jobs": args.n_jobs,
            },
        }
        # 在 no-search 模式下，将 max_samples 添加到 default_params
        if args.no_search and actual_max_samples is not None:
            trainer_kwargs["default_params"]["max_samples"] = actual_max_samples
        model_type = "random_forest"
        max_samples_str = "None" if actual_max_samples is None else str(actual_max_samples)
        hyper = "cv={}, scoring={}, n_estimators={}, max_depth={}, min_split={}, min_leaf={}, max_features={}, max_samples={}".format(
            args.cv_folds, args.scoring,
            rf_n_estimators, rf_max_depth,
            rf_min_samples_split, rf_min_samples_leaf,
            rf_max_features, max_samples_str
        )
        return trainer_class, trainer_kwargs, model_type, hyper

    raise ValueError("未知模型: {}".format(args.model))


def main():
    """标准入口：从命令行运行并执行统一训练逻辑。"""
    parser = build_parser()
    args = parser.parse_args()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 选择 trainer（需要在生成保存路径之前）
    trainer_class, trainer_kwargs, model_type, hyper = _select_trainer_and_kwargs(args)
    
    # 生成带时间戳的保存路径
    if not args.save_path:
        args.save_path = _default_save_path(args.model, timestamp)
    
    # 从保存路径中提取运行目录（用于日志）
    save_path_obj = Path(args.save_path)
    run_dir = save_path_obj.parent  # runs/sklearn_svm_20250105-123456
    
    # 创建运行目录
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 将日志保存到运行目录
    experiment_name = "sklearn_{}".format(args.model)
    logger = Logger(
        name=experiment_name,
        log_dir=str(run_dir),
        filename="{}.log".format(experiment_name),
        level=args.log_level,
    )

    start_time = time.time()
    try:
        cfg_lines = [
            "model: {}".format(args.model),
            "data: {}".format(args.data_dir),
            "seed: {}".format(args.seed),
            "cv_folds: {}".format(args.cv_folds),
            "n_jobs: {}".format(args.n_jobs),
            "run_dir: {}".format(run_dir),
        ]
        logger.block("开始训练", cfg_lines)

        # 执行统一训练流程
        trainer, results = run_sklearn_training(
            args=args,
            trainer_class=trainer_class,
            trainer_kwargs=trainer_kwargs,
            model_type=model_type,
            hyperparams_info=hyper,
            logger=logger,
        )

        # 日志：训练完成
        total_time = time.time() - start_time
        best_acc = None
        if results and results.get("val") and results["val"].get("accuracy") is not None:
            best_acc = results["val"]["accuracy"]
        elif results and results.get("test") and results["test"].get("accuracy") is not None:
            best_acc = results["test"]["accuracy"]
        summary_lines = [
            "耗时: {}".format(Logger.format_duration(total_time)),
            "最佳准确率: {:.4f}".format(best_acc) if best_acc is not None else "最佳准确率: -",
            "模型保存: {}".format(args.save_path),
        ]
        logger.block("训练完成", summary_lines)
        return trainer, results
    except Exception as e:
        # 确保异常也写入日志
        logger.exception("训练过程中发生错误: %s", e)
        raise


if __name__ == "__main__":
    main()
