"""
项目统一轻量日志工具。
"""

import logging
from pathlib import Path

__all__ = ["Logger"]


class Logger:
    """
    统一的轻量日志类封装：
    - 控制台 + 可选文件输出
    - 提供 block/dict 等便捷方法
    """

    def __init__(self, name="catdog", log_dir=None, filename="train.log", level="INFO"):
        self._logger = self._build_logger(name, log_dir, filename, level)

    @staticmethod
    def _build_logger(name, log_dir, filename, level):
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        numeric_level = getattr(logging, str(level).upper(), logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path / filename, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(numeric_level)
        logger.propagate = False
        return logger

    # 基础方法
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self._logger.log(level, msg, *args, **kwargs)

    @property
    def raw(self):
        return self._logger

    # 便捷方法
    def block(self, title, lines=None, level=logging.INFO):
        self._logger.log(level, title)
        if not lines:
            return
        for line in lines:
            self._logger.log(level, f"  {line}")

    def dict(self, title, data, level=logging.INFO, indent="  "):
        items = [(str(k), data[k]) for k in data]
        if not items:
            self._logger.log(level, title)
            return
        width = max(len(k) for k, _ in items)
        self._logger.log(level, title)
        for key, value in items:
            self._logger.log(level, f"{indent}{key.ljust(width)} : {value}")

    @staticmethod
    def format_duration(seconds):
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return f"{int(seconds)}秒"
        if seconds < 3600:
            return f"{seconds / 60:.1f}分钟"
        return f"{seconds / 3600:.1f}小时"

    @staticmethod
    def maybe_print(verbose, logger, message):
        if not verbose:
            return
        if logger is None:
            logger = Logger()
        logger.info(message)

    def header(self, model_name, args):
        msg = "🚀 {} | data_dir={} | seed={} | cv_folds={} | n_jobs={}".format(
            model_name, args.data_dir, args.seed, getattr(args, 'cv_folds', '-'), args.n_jobs
        )
        self.info(msg)

    def summary(self, save_path, elapsed_seconds):
        msg = "✅ 完成 | {} | 模型: {}".format(
            self.format_duration(elapsed_seconds), save_path
        )
        self.info(msg)
    
    def log_cv_results(self, search):
        """记录所有配置的交叉验证结果
        
        参数:
            search: GridSearchCV 对象（必须包含 cv_results_ 属性）
        """
        if search is None or not hasattr(search, 'cv_results_'):
            return
        
        cv_results = search.cv_results_
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        params_list = cv_results['params']
        
        # 确保输出到控制台和文件
        self.info("")
        self.info("📊 所有配置的交叉验证结果:")
        self.info("-" * 80)
        
        # 按分数排序（从高到低）
        results = list(zip(mean_scores, std_scores, params_list))
        results.sort(key=lambda x: x[0], reverse=True)
        
        for idx, (mean_score, std_score, params) in enumerate(results, 1):
            # 格式化参数显示
            param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())])
            self.info("  [{:2d}] {} | score={:.4f} ± {:.4f}".format(
                idx, param_str, mean_score, std_score
            ))
        
        self.info("-" * 80)
        self.info("✅ 最佳配置: {} | score={:.4f} ± {:.4f}".format(
            ", ".join([f"{k}={v}" for k, v in sorted(search.best_params_.items())]),
            search.best_score_,
            std_scores[search.best_index_]
        ))
        self.info("")
        
        # 强制刷新所有处理器
        for handler in self._logger.handlers:
            handler.flush()