import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import joblib
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.color_hist import extract_color_hist_from_path
from src.features.hog import extract_hog_from_path
from src.features.sift import extract_sift_from_path
from src.features.lbp import extract_lbp_from_path
from src.features.moments import extract_color_moments_from_path, extract_hu_moments_from_path
from src.features.glcm import extract_glcm_from_path
from src.features.gabor import extract_gabor_from_path
from src.features.edge_hist import extract_edge_hist_from_path
from src.features.corner_edge_density import extract_corner_edge_density_from_path


LOGGER = logging.getLogger('feature_extractor')

FEATURE_POOL = [
    'color_hist',
    'hog',
    'sift',
    'lbp',
    'color_moments',
    'hu_moments',
    'glcm',
    'gabor',
    'edge_hist',
    'corner_edge',
]

FEATURE_DEFAULT_CONFIG = {
    'color_hist': {'color_space': 'HSV', 'hist_size': (8, 8, 8)},
    'hog': {
        'orientations': 9,
        'pixels_per_cell': (16, 16),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
    },
    'sift': {
        'n_features': 0,
        'n_octave_layers': 3,
        'contrast_threshold': 0.04,
        'edge_threshold': 10,
        'sigma': 1.6,
        'num_scale_bins': 8,
        'num_response_bins': 8,
        'num_angle_bins': 18,
    },
    'lbp': {'P': 8, 'R': 1, 'method': 'uniform'},
    'color_moments': {'color_space': 'HSV'},
    'hu_moments': {'use_edges': False, 'canny_threshold1': 100, 'canny_threshold2': 200},
    'glcm': {
        'distances': (1, 2, 4),
        'angles': (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        'levels': 256,
        'symmetric': True,
        'normed': True,
    },
    'gabor': {
        'ksizes': (15,),
        'sigmas': (4.0,),
        'thetas': (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        'lambdas': (10.0, 20.0),
        'gammas': (0.5, 0.8),
        'psis': (0, np.pi / 2),
    },
    'edge_hist': {
        'num_orientation_bins': 9,
        'num_magnitude_bins': 32,
        'canny_threshold1': 100,
        'canny_threshold2': 200,
    },
    'corner_edge': {
        'harris_block_size': 2,
        'harris_ksize': 3,
        'harris_k': 0.04,
        'shi_max_corners': 500,
        'shi_quality_level': 0.01,
        'shi_min_distance': 5,
        'canny_threshold1': 100,
        'canny_threshold2': 200,
    },
}

FEATURE_EXTRACTOR_MAP = {
    'color_hist': extract_color_hist_from_path,
    'hog': extract_hog_from_path,
    'sift': extract_sift_from_path,
    'lbp': extract_lbp_from_path,
    'color_moments': extract_color_moments_from_path,
    'hu_moments': extract_hu_moments_from_path,
    'glcm': extract_glcm_from_path,
    'gabor': extract_gabor_from_path,
    'edge_hist': extract_edge_hist_from_path,
    'corner_edge': extract_corner_edge_density_from_path,
}

DEFAULT_SPLITS = ('train', 'val', 'test')


def collect_image_paths_and_labels(dataset_dir, split='train'):
    """收集数据集分割中的图像路径与标签。

    参数:
        dataset_dir: 数据集根目录
        split: 分割名称 (train/val/test)

    返回:
        tuple: ``(paths, labels)`` 列表

    异常:
        FileNotFoundError: 当分割目录不存在时抛出
    """
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise FileNotFoundError('数据集分割目录不存在: {}'.format(split_dir))

    paths = []
    labels = []
    class_map = (('cats', 0), ('dogs', 1))

    for class_name, label in class_map:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for pattern in ('*.jpg', '*.jpeg', '*.png'):
            for image_path in sorted(class_dir.glob(pattern)):
                paths.append(str(image_path))
                labels.append(label)

    return paths, labels


def save_features_to_file(features, labels, valid_indices, feature_info, save_path):
    """保存特征矩阵与元数据。

    参数:
        features: numpy.ndarray 特征矩阵
        labels: numpy.ndarray 或 None，特征对应标签
        valid_indices: list，成功提取的原始索引
        feature_info: dict，额外记录的特征信息
        save_path: joblib 文件保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'features': features,
        'labels': labels,
        'valid_indices': list(valid_indices),
        'feature_info': feature_info,
        'feature_shape': features.shape,
        'n_samples': int(features.shape[0]),
        'n_features': int(features.shape[1]) if len(features.shape) == 2 else 0,
    }
    joblib.dump(payload, save_path)
    LOGGER.info('💾 特征已保存: %s', save_path)


class UnifiedFeatureExtractor:
    """根据配置提取并拼接多种经典图像特征。"""
    
    _single_extractor_cache = {}  # 类级别的单特征提取器缓存

    @staticmethod
    def normalize_feature_config(feature_config):
        """校验并整理特征配置。

        参数:
            feature_config: dict，自定义特征配置

        返回:
            dict: 清洗后的配置字典

        异常:
            ValueError: 当配置包含未知特征时抛出
        """
        if feature_config is None:
            return dict(FEATURE_DEFAULT_CONFIG)
        
        if not isinstance(feature_config, dict):
            raise ValueError('feature_config 必须是字典')

        normalized = {}
        for name, params in feature_config.items():
            if name not in FEATURE_EXTRACTOR_MAP:
                raise ValueError('未知的特征类型: {}'.format(name))
            normalized[name] = params or {}
        return normalized

    def __init__(self, feature_config=None, image_size=128, n_jobs=None, verbose=True):
        """初始化特征提取器。

        参数:
            feature_config: dict，特征类型到参数的映射
            image_size: int，图像缩放尺寸
            n_jobs: int 或 None，线程池并发数 (None 表示自动选择，1 表示禁用并发)
            verbose: bool，是否输出进度信息
        """
        self.image_size = int(image_size)
        self.verbose = bool(verbose)
        if n_jobs is None:
            cpu_count = os.cpu_count() or 1
            self.n_jobs = max(1, min(4, cpu_count))
        else:
            self.n_jobs = max(1, int(n_jobs))
        self.feature_config = self.normalize_feature_config(feature_config)

    def _extract_feature(self, image_path, feature_type, params):
        extractor = FEATURE_EXTRACTOR_MAP.get(feature_type)
        if extractor is None:
            raise ValueError('未知的特征类型: {}'.format(feature_type))
        kwargs = {'image_size': self.image_size}
        kwargs.update(params or {})
        return extractor(image_path, **kwargs)

    def extract_features_from_image(self, image_path):
        """提取单张图像的特征向量。

        参数:
            image_path: str，图像路径

        返回:
            numpy.ndarray 或 None: 拼接后的特征向量
        """
        vectors = []
        for feature_type, params in self.feature_config.items():
            try:
                feature_vector = self._extract_feature(image_path, feature_type, params)
            except Exception as exc:
                if self.verbose:
                    LOGGER.warning('提取失败 %s (%s): %s', feature_type, image_path, exc)
                feature_vector = None
            if feature_vector is not None:
                vectors.append(feature_vector)
        if not vectors:
            return None
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def extract_features_batch(self, image_paths, labels=None):
        """批量提取图像特征。

        参数:
            image_paths: list，图像路径列表
            labels: list 或 None，对应标签

        返回:
            tuple: ``(features, label_array, valid_indices)``

        异常:
            RuntimeError: 当全部图像提取失败时抛出
        """
        if self.verbose:
            LOGGER.info('🔍 开始批量特征提取...')
            LOGGER.info('   图像数量: {}'.format(len(image_paths)))
            LOGGER.info('   特征类型: {}'.format(list(self.feature_config.keys())))
            LOGGER.info('   并行线程: {}'.format(self.n_jobs))

        features = []
        selected_labels = []
        valid_indices = []
        use_parallel = self.n_jobs > 1

        # 创建进度条
        progress = None
        if self.verbose:
            progress = tqdm(total=len(image_paths), desc='提取特征', leave=False)

        try:
            if use_parallel:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = [
                        (idx, executor.submit(self.extract_features_from_image, image_path))
                        for idx, image_path in enumerate(image_paths)
                    ]
                    for idx, future in futures:
                        vector = future.result()
                        if vector is not None:
                            features.append(vector)
                            valid_indices.append(idx)
                            if labels is not None:
                                selected_labels.append(labels[idx])
                        if progress is not None:
                            progress.update(1)
            else:
                for idx, image_path in enumerate(image_paths):
                    vector = self.extract_features_from_image(image_path)
                    if vector is not None:
                        features.append(vector)
                        valid_indices.append(idx)
                        if labels is not None:
                            selected_labels.append(labels[idx])
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        if not features:
            raise RuntimeError('没有成功提取任何特征')

        feature_matrix = np.vstack(features).astype(np.float32)
        label_array = None
        if labels is not None and selected_labels:
            label_array = np.array(selected_labels)

        if self.verbose:
            LOGGER.info('✅ 特征提取完成: {}/{} 样本'.format(len(valid_indices), len(image_paths)))
            LOGGER.info('   特征矩阵形状: {}'.format(feature_matrix.shape))

        return feature_matrix, label_array, valid_indices

    @classmethod
    def extract_single_feature_matrix(cls, image_paths, labels, feature_type, image_size=128, n_jobs=None, show_progress=True):
        """提取单一特征类型的特征矩阵。

        参数:
            image_paths: list，图像路径
            labels: list，标签列表
            feature_type: str，要评估的特征类型
            image_size: int，图像缩放尺寸
            n_jobs: int 或 None，并行线程数
            show_progress: bool，是否显示进度条

        返回:
            dict: 包含特征矩阵与元数据的字典
        """
        if feature_type not in FEATURE_EXTRACTOR_MAP:
            raise ValueError('未知的特征类型: {}'.format(feature_type))

        cache_key = (feature_type, image_size, n_jobs)
        extractor = cls._single_extractor_cache.get(cache_key)
        if extractor is None:
            extractor = cls(
                feature_config={feature_type: FEATURE_DEFAULT_CONFIG.get(feature_type, {})},
                image_size=image_size,
                n_jobs=n_jobs,
                verbose=show_progress,
            )
            cls._single_extractor_cache[cache_key] = extractor
        features, label_array, valid_indices = extractor.extract_features_batch(image_paths, labels)

        payload = {
            'X': features.astype(np.float32),
            'y': label_array.astype(np.int64) if label_array is not None else None,
            'valid_indices': list(valid_indices),
            'dim': int(features.shape[1]) if len(features.shape) == 2 else 0,
        }

        return payload

    @classmethod
    def extract_and_save_dataset_features(cls, dataset_dir='dataset', save_dir='features', feature_config=None,
                                          image_size=128, n_jobs=None):
        """为标准数据集分割提取特征并保存。

        参数:
            dataset_dir: 数据集根目录
            save_dir: 特征保存目录
            feature_config: dict，自定义特征配置
            image_size: int，图像缩放尺寸
            n_jobs: int 或 None，并行线程数

        返回:
            dict: 各分割的提取结果统计
        """
        extractor = cls(
            feature_config=feature_config,
            image_size=image_size,
            n_jobs=n_jobs,
            verbose=True,
        )

        results = {}
        for split in DEFAULT_SPLITS:
            LOGGER.info('🔄 开始处理 %s 分割', split)
            try:
                image_paths, labels = collect_image_paths_and_labels(dataset_dir, split)
            except FileNotFoundError as exc:
                LOGGER.warning('跳过 %s: %s', split, exc)
                continue

            if not image_paths:
                LOGGER.warning('%s 分割没有图像，跳过', split)
                continue

            features, label_array, valid_indices = extractor.extract_features_batch(image_paths, labels)
            info = {
                'feature_config': extractor.feature_config,
                'image_size': image_size,
                'total_dim': int(features.shape[1]) if len(features.shape) == 2 else 0,
            }
            save_path = Path(save_dir) / split / '{}_features.joblib'.format(split)
            save_features_to_file(features, label_array, valid_indices, info, save_path)

            results[split] = {
                'n_total': len(image_paths),
                'n_valid': len(valid_indices),
                'feature_shape': features.shape,
                'save_path': str(save_path),
            }
            LOGGER.info('✅ %s 分割完成: %d/%d 样本', split, len(valid_indices), len(image_paths))

        return results


if __name__ == '__main__':
    # 读取束搜索产生的最佳特征组合
    results_json = Path('runs/feature_search/search_results.json')
    try:
        with open(results_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        best_subset = results.get('best_result', {}).get('subset', [])
        config_block = results.get('config', {}) or {}
        image_size = int(config_block.get('image_size', 128))
        n_jobs = config_block.get('n_jobs', None)
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        LOGGER.info('🟢 使用 search_results.json 中的最佳配置: %s', ' + '.join(best_subset) if best_subset else '(空)')
    except Exception as exc:
        LOGGER.warning('无法读取最佳配置 (%s)，将使用默认配置: %s', exc, str(results_json))
        best_subset = []
        image_size = 128
        n_jobs = None

    # 基于最佳子集构建 feature_config；若为空则回退到默认配置（使用 FEATURE_DEFAULT_CONFIG）
    if best_subset:
        feature_config = {name: FEATURE_DEFAULT_CONFIG.get(name, {}) for name in best_subset if name in FEATURE_EXTRACTOR_MAP}
    else:
        feature_config = dict(FEATURE_DEFAULT_CONFIG)

    # 基础日志配置（仅在未配置过时生效）
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )

    LOGGER.info('📦 开始根据最佳配置提取数据集特征')
    LOGGER.info('   数据集目录: %s', 'dataset')
    LOGGER.info('   保存目录: %s', 'features')
    LOGGER.info('   图像尺寸: %d', image_size)
    LOGGER.info('   并行线程: %s', str(n_jobs) if n_jobs else '(auto)')
    LOGGER.info('   特征类型: %s', ', '.join(sorted(feature_config.keys())))

    try:
        # 使用统一提取器提取原始特征（不做 PCA），随后基于训练集拟合并一致地应用到 val/test
        extractor = UnifiedFeatureExtractor(
            feature_config=feature_config,
            image_size=image_size,
            n_jobs=n_jobs,
            verbose=True,
        )

        raw = {}
        for split in DEFAULT_SPLITS:
            try:
                image_paths, labels = collect_image_paths_and_labels('dataset', split)
            except FileNotFoundError as exc:
                LOGGER.warning('跳过 %s: %s', split, exc)
                continue
            if not image_paths:
                LOGGER.warning('%s 分割没有图像，跳过', split)
                continue
            X, y, idx = extractor.extract_features_batch(image_paths, labels)
            raw[split] = {'X': X, 'y': y, 'idx': idx}
            LOGGER.info('✅ 原始特征完成 %s: %d/%d | 形状 %s', split, len(idx), len(image_paths), X.shape)

        # 与搜索结果对齐：始终标准化；PCA 维度严格依据 search_results.json 的 best_result.pca
        if 'train' not in raw:
            raise RuntimeError('缺少训练集，无法拟合标准化与PCA')

        # 从 search_results.json 读取 pca 维度（>0 启用；否则不启用）
        try:
            pca_components = int((results or {}).get('best_result', {}).get('pca', 0))
        except Exception:
            pca_components = 0
        apply_pca = pca_components is not None and pca_components > 0

        LOGGER.info('🧪 拟合 StandardScaler（基于训练集）')
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(raw['train']['X'])

        transformed = {}
        if apply_pca:
            LOGGER.info('🧪 根据 JSON 指定执行 PCA(%d)', pca_components)
            pca = PCA(n_components=pca_components, random_state=42)
            transformed['train'] = pca.fit_transform(X_train_scaled)
        else:
            LOGGER.info('🧪 JSON 未指定有效 PCA 维度（或为 0），不执行 PCA')
            pca = None
            transformed['train'] = X_train_scaled

        # 应用到各分割
        for split in ('val', 'test'):
            if split in raw:
                X_scaled = scaler.transform(raw[split]['X'])
                transformed[split] = pca.transform(X_scaled) if pca is not None else X_scaled

        # 保存到文件（携带 scaler 与 pca）
        for split in ('train', 'val', 'test'):
            if split not in raw:
                continue
            X_out = transformed[split]
            y_out = raw[split]['y']
            idx_out = raw[split]['idx']
            info = {
                'feature_config': extractor.feature_config,
                'image_size': image_size,
                'original_dim': int(raw[split]['X'].shape[1]),
                'pca': pca_components,
                'scaler': scaler,
                'pca_model': pca,
                'applied_standardization': True,
                'applied_pca': apply_pca,
            }
            save_path = Path('features') /  f'{split}_features.joblib'
            save_features_to_file(X_out, y_out, idx_out, info, save_path)
            LOGGER.info('💾 已保存: %s -> %s | 形状 %s', split, save_path, X_out.shape)
    except Exception as exc:
        LOGGER.error('提取过程失败: %s', exc)