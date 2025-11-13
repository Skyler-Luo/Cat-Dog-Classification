"""特征预处理组件。

负责在特征工程阶段执行常见的数据处理步骤：
    - 缩放 (Standard/MinMax/Robust)
    - 单变量特征选择 (F-score 或互信息)
    - PCA 降维

通过 ``FeaturePreprocessor`` 类统一封装，便于在特征提取与模型搜索流程中复用。
"""

import logging

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler


LOGGER = logging.getLogger('feature_preprocessor')


class FeaturePreprocessor:
    """特征预处理器，负责缩放、特征选择与降维。

    参数:
        config: dict，预处理配置
        verbose: bool，是否输出详细日志

    属性:
        scaler: ``sklearn`` 缩放器实例或 ``None``
        feature_selector: ``SelectKBest`` 实例或 ``None``
        pca: ``PCA`` 实例或 ``None``

    示例:
        >>> preprocessor = FeaturePreprocessor({'scaler': 'standard', 'pca': 128})
        >>> X_proc = preprocessor.process(X, y)
    """

    def __init__(self, config=None, verbose=True):
        self.config = config or {
            'scaler': 'standard',
            'pca': 512,
            'feature_selection': None,
            'selection_method': 'f_classif'
        }
        self.verbose = verbose
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.is_fitted = False

    def process(self, features, labels=None):
        if not self.is_fitted:
            processed = self._fit(features, labels)
            self.is_fitted = True
            return processed
        return self._transform(features)

    def _fit(self, features, labels):
        features = self._apply_scaler(features, fit=True)
        features = self._apply_selector(features, labels, fit=True)
        features = self._apply_pca(features, fit=True)
        return features

    def _transform(self, features):
        features = self._apply_scaler(features, fit=False)
        features = self._apply_selector(features, None, fit=False)
        features = self._apply_pca(features, fit=False)
        return features

    def _apply_scaler(self, features, fit):
        scaler_type = self.config.get('scaler')
        if not scaler_type:
            return features
        if fit or self.scaler is None:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                from sklearn.preprocessing import RobustScaler

                self.scaler = RobustScaler()
            else:
                raise ValueError('未知的缩放器: {}'.format(scaler_type))
            result = self.scaler.fit_transform(features)
            if self.verbose:
                LOGGER.info('   ✓ 应用{}缩放'.format(scaler_type))
            return result
        return self.scaler.transform(features)

    def _apply_selector(self, features, labels, fit):
        select_k = self.config.get('feature_selection')
        if not select_k:
            return features
        if fit:
            if labels is None:
                LOGGER.warning('特征选择已启用但缺少标签，跳过本次拟合')
                return features
            method = self.config.get('selection_method', 'f_classif')
            if method == 'f_classif':
                self.feature_selector = SelectKBest(f_classif, k=select_k)
            elif method == 'mutual_info':
                self.feature_selector = SelectKBest(mutual_info_classif, k=select_k)
            else:
                raise ValueError('未知的特征选择方法: {}'.format(method))
            transformed = self.feature_selector.fit_transform(features, labels)
            if self.verbose:
                LOGGER.info('   ✓ 特征选择: {} 维'.format(transformed.shape[1]))
            return transformed
        if self.feature_selector is None:
            return features
        return self.feature_selector.transform(features)

    def _apply_pca(self, features, fit):
        n_components = self.config.get('pca')
        if not n_components:
            return features
        if fit or self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            transformed = self.pca.fit_transform(features)
            if self.verbose:
                variance = self.pca.explained_variance_ratio_.sum()
                LOGGER.info('   ✓ PCA降维: {} 维 (解释方差 {:.3f})'.format(transformed.shape[1], variance))
            return transformed
        return self.pca.transform(features)

    def get_info(self):
        info = {}
        if self.scaler is not None:
            info['scaler'] = type(self.scaler).__name__
        if self.feature_selector is not None:
            info['selected_features'] = self.feature_selector.k
        if self.pca is not None:
            info['pca_components'] = self.pca.n_components_
            info['pca_explained_variance'] = float(self.pca.explained_variance_ratio_.sum())
        return info


