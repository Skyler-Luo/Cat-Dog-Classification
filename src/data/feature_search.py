import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.feature_extraction import (  # noqa: E402
    FEATURE_POOL,
    UnifiedFeatureExtractor,
    collect_image_paths_and_labels,
    save_features_to_file,
)


LOGGER = logging.getLogger('feature_search')


def align_and_concatenate_features(feature_payloads):
    """对齐多个特征矩阵并水平拼接。
    
    参数:
        feature_payloads: list，单特征提取结果字典，每个字典包含 'X', 'y', 'valid_indices'
        
    返回:
        tuple: (X_concat, y_aligned, common_indices)
            - X_concat: numpy.ndarray，对齐并拼接后的特征矩阵
            - y_aligned: numpy.ndarray 或 None，对齐后的标签
            - common_indices: list，共同样本的原始索引
            
    异常:
        RuntimeError: 当特征没有共同样本时抛出
        
    示例:
        >>> payload1 = {'X': X1, 'y': y1, 'valid_indices': [0, 1, 2]}
        >>> payload2 = {'X': X2, 'y': y2, 'valid_indices': [1, 2, 3]}
        >>> X, y, indices = align_and_concatenate_features([payload1, payload2])
        >>> # X 形状: (2, dim1 + dim2)，indices: [1, 2]
    """
    if len(feature_payloads) == 1:
        single = feature_payloads[0]
        return single['X'], single.get('y'), list(single['valid_indices'])
    
    # 找到所有特征的共同样本索引
    index_sets = [set(payload['valid_indices']) for payload in feature_payloads]
    common_indices = sorted(set.intersection(*index_sets))
    if not common_indices:
        raise RuntimeError('各特征没有共同样本，无法对齐')
    
    # 构建索引映射并对齐特征矩阵
    aligned_blocks = []
    for payload in feature_payloads:
        position_map = {idx: pos for pos, idx in enumerate(payload['valid_indices'])}
        rows = [position_map[idx] for idx in common_indices]
        aligned_blocks.append(payload['X'][rows])
    
    X_concat = np.hstack(aligned_blocks).astype(np.float32)
    
    # 对齐标签（使用第一个 payload 的标签）
    ref_payload = feature_payloads[0]
    ref_position_map = {idx: pos for pos, idx in enumerate(ref_payload['valid_indices'])}
    ref_rows = [ref_position_map[idx] for idx in common_indices]
    y_aligned = None
    if ref_payload.get('y') is not None:
        y_aligned = ref_payload['y'][ref_rows]
    
    return X_concat, y_aligned, common_indices


class FeatureSearch:
    """特征组合搜索器，使用束搜索方法寻找最优特征组合。
    
    封装特征提取、评估和搜索流程，支持配置管理和状态维护。
    """
    
    def __init__(self, model='svm', cv=5, scoring='accuracy', beam_width=5, 
                 enable_cache=True, image_size=128, n_jobs=None, pca_components=512):
        """初始化特征搜索器。
        
        参数:
            model: str，评估模型 (svm/rf)
            cv: int，交叉验证折数
            scoring: str，评估指标
            beam_width: int，束搜索的束宽（建议 3-10，默认 5）
            enable_cache: bool，是否启用评估结果缓存
            image_size: int，图像缩放尺寸
            n_jobs: int 或 None，特征提取并行线程数
            pca_components: int 或 None，PCA降维后的维度（None表示不降维，默认512）
        """
        self.model_name = model
        self.cv = cv
        self.scoring = scoring
        self.beam_width = beam_width
        self.enable_cache = enable_cache
        self.image_size = image_size
        self.n_jobs = n_jobs
        self.pca_components = pca_components
        
        # 状态变量
        self.feature_cache = {}
        self.evaluation_cache = {} if enable_cache else None
        self.search_history = []
        self.best_result = None
        self.dataset_dir = None
        self.split = None
        self.sample_ratio = None
        self.image_paths = None
        self.labels = None
    
    def _evaluate_subset_cv(self, X, y, subset_names):
        """使用交叉验证评估特征子集（内部方法）。
        
        参数:
            X: numpy.ndarray，特征矩阵
            y: numpy.ndarray，标签
            subset_names: list，特征子集名称（用于缓存键）
            
        返回:
            dict: 包含得分、方差与降维信息的字典
        """
        # 使用特征名称和形状生成缓存键
        if self.evaluation_cache is not None and subset_names is not None:
            cache_key = (tuple(sorted(subset_names)), X.shape, self.model_name, self.cv, self.scoring)
            if cache_key in self.evaluation_cache:
                self.evaluation_cache['_hits'] = self.evaluation_cache.get('_hits', 0) + 1
                return self.evaluation_cache[cache_key]
        else:
            cache_key = None
        
        n_features = X.shape[1] if len(X.shape) == 2 else 0
        
        # 构建 Pipeline
        steps = [('scaler', StandardScaler())]
        pca_components = 0
        if self.pca_components is not None and n_features > self.pca_components:
            pca_components = self.pca_components
            steps.append(('pca', PCA(n_components=pca_components, random_state=42)))  # type: ignore
        
        if self.model_name == 'svm':
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
        elif self.model_name == 'rf':
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        else:
            raise ValueError('未知模型: {}'.format(self.model_name))
        
        steps.append(('model', model))  # type: ignore
        pipeline = Pipeline(steps)
        
        # 交叉验证评估
        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        start_time = time.time()
        scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=self.scoring, n_jobs=-1)
        elapsed = time.time() - start_time
        
        result = {
            'score': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'pca': int(pca_components),
            'original_dim': int(n_features),
            'time': elapsed,
        }
        
        if self.evaluation_cache is not None and cache_key is not None:
            self.evaluation_cache[cache_key] = result
        
        return result
    
    def _evaluate_subset(self, subset_names):
        """评估特征子集并返回结果（内部方法）。
        
        参数:
            subset_names: list，特征子集名称列表
            
        返回:
            dict: 评估结果字典，包含 score, std, pca, original_dim
        """
        payloads = [self.feature_cache[name] for name in subset_names]
        X_sub, y_sub, _ = align_and_concatenate_features(payloads)
        result = self._evaluate_subset_cv(X_sub, y_sub, subset_names)
        return {
            'subset': subset_names,
            'score': result['score'],
            'std': result['std'],
            'pca': result['pca'],
            'original_dim': result['original_dim'],
        }
    
    @staticmethod
    def _sample_data(image_paths, labels, sample_ratio, seed=42):
        """对数据进行采样（静态方法）。
        
        参数:
            image_paths: list，图像路径列表
            labels: list，标签列表
            sample_ratio: float，采样比例 (0-1]
            seed: int，随机种子
            
        返回:
            tuple: (采样后的 image_paths, labels)
        """
        if sample_ratio >= 1.0:
            return image_paths, labels
        
        total = len(image_paths)
        size = max(1, int(total * sample_ratio))
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(total, size=size, replace=False)
        sampled_paths = [image_paths[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
        LOGGER.info('📦 采样 %.0f%% 数据: %d -> %d', sample_ratio * 100, total, len(sampled_paths))
        return sampled_paths, sampled_labels
    
    def load_data(self, dataset_dir='dataset', split='train', sample_ratio=1.0):
        """加载数据集。
        
        参数:
            dataset_dir: str，数据集根目录
            split: str，数据集分割 (train/val/test)
            sample_ratio: float，采样比例 (0-1]
            
        返回:
            tuple: (image_paths, labels)
        """
        # 保存数据集信息供后续使用
        self.dataset_dir = dataset_dir
        self.split = split
        self.sample_ratio = sample_ratio
        
        LOGGER.info('🚀 启动特征搜索')
        image_paths, labels = collect_image_paths_and_labels(dataset_dir, split)
        if not image_paths:
            raise RuntimeError('训练集为空，无法运行搜索')
        
        # 保存完整数据集路径（用于后续完整数据集提取）
        self.image_paths = image_paths
        self.labels = labels
        
        image_paths, labels = self._sample_data(image_paths, labels, sample_ratio)
        if sample_ratio >= 1.0:
            LOGGER.info('📦 使用全部 %d 张图像', len(image_paths))
        
        return image_paths, labels
    
    def extract_features(self, image_paths, labels, feature_names=None):
        """提取特征并缓存。
        
        参数:
            image_paths: list，图像路径列表
            labels: list，标签列表
            feature_names: list 或 None，要提取的特征名称列表（None 表示使用全部）
            
        返回:
            dict: 特征缓存字典
        """
        if feature_names is None:
            feature_names = FEATURE_POOL
        
        self.feature_cache = {}
        for feature_name in feature_names:
            if feature_name not in FEATURE_POOL:
                LOGGER.warning('   跳过未知特征: %s', feature_name)
                continue
                
            LOGGER.info('🔧 提取特征: %s', feature_name)
            try:
                payload = UnifiedFeatureExtractor.extract_single_feature_matrix(
                    image_paths,
                    labels,
                    feature_name,
                    image_size=self.image_size,
                    n_jobs=self.n_jobs,
                    show_progress=True,
                )
                self.feature_cache[feature_name] = payload
                LOGGER.info('   ✓ 维度 %d | 样本 %d', payload['dim'], len(payload['valid_indices']))
            except Exception as exc:
                LOGGER.warning('   ✗ 跳过 %s: %s', feature_name, exc)
        
        if not self.feature_cache:
            raise RuntimeError('没有成功提取的特征')
        
        return self.feature_cache
    
    def search(self, max_features=5):
        """执行束搜索。
        
        参数:
            max_features: int，最大特征组合数量
            
        返回:
            dict: 搜索结果，包含 best, history, cache_hits
        """
        if not self.feature_cache:
            raise RuntimeError('请先调用 extract_features() 提取特征')
        
        feature_names = list(self.feature_cache.keys())
        
        LOGGER.info('')
        LOGGER.info('=' * 60)
        LOGGER.info('开始特征组合搜索（束搜索）')
        LOGGER.info('=' * 60)
        LOGGER.info('🔍 束搜索 (Beam Search)')
        LOGGER.info('   特征池大小: %d', len(feature_names))
        LOGGER.info('   最大组合数: %d', max_features)
        LOGGER.info('   束宽: %d', self.beam_width)
        
        # 初始化：评估所有单特征
        LOGGER.info('📊 第 1 步: 评估单特征组合')
        candidates = []
        for name in feature_names:
            try:
                result = self._evaluate_subset([name])
                candidates.append(result)
                LOGGER.info('   • %s -> %.4f ± %.4f', name, result['score'], result['std'])
            except Exception as exc:
                LOGGER.warning('   评估失败 %s: %s', name, exc)
        
        # 按得分排序，保留 top beam_width
        candidates.sort(key=lambda x: x['score'], reverse=True)
        beam = candidates[:self.beam_width]
        best_overall = beam[0].copy() if beam else {'score': -1.0, 'subset': []}
        history = [{'step': 1, 'beam_size': len(beam), 'best_score': beam[0]['score'] if beam else -1.0}]
        
        LOGGER.info('   ✅ 保留 top %d 个候选', len(beam))
        for i, cand in enumerate(beam, 1):
            LOGGER.info('      %d. %.4f | %s', i, cand['score'], ' + '.join(cand['subset']))
        
        # 逐步扩展束
        for step in range(2, max_features + 1):
            if not beam:
                break
            
            LOGGER.info('')
            LOGGER.info('📊 第 %d 步: 扩展束 (当前束大小: %d)', step, len(beam))
            
            # 从当前束中的每个候选生成新候选
            new_candidates = []
            used_combinations = set()
            
            for cand in beam:
                current_subset = set(cand['subset'])
                remaining = [name for name in feature_names if name not in current_subset]
                
                for new_feature in remaining:
                    new_subset = sorted(list(current_subset) + [new_feature])
                    subset_key = tuple(new_subset)
                    
                    # 避免重复评估
                    if subset_key in used_combinations:
                        continue
                    used_combinations.add(subset_key)
                    
                    try:
                        result = self._evaluate_subset(new_subset)
                        new_candidates.append(result)
                        
                        LOGGER.info(
                            '   • %s -> %.4f ± %.4f (%d 维%s)',
                            ' + '.join(new_subset),
                            result['score'],
                            result['std'],
                            result['original_dim'],
                            ' + PCA({})'.format(result['pca']) if result['pca'] else '',
                        )
                    except Exception as exc:
                        LOGGER.warning('   评估失败 %s: %s', ' + '.join(new_subset), exc)
            
            if not new_candidates:
                LOGGER.warning('   没有新候选，提前结束')
                break
            
            # 按得分排序，保留 top beam_width
            new_candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = new_candidates[:self.beam_width]
            
            # 更新全局最佳
            if beam and beam[0]['score'] > best_overall['score']:
                best_overall = beam[0].copy()
            
            history.append({
                'step': step,
                'beam_size': len(beam),
                'best_score': beam[0]['score'] if beam else -1.0,
                'evaluated': len(new_candidates),
            })
            
            LOGGER.info('   ✅ 保留 top %d 个候选', len(beam))
            for i, cand in enumerate(beam, 1):
                LOGGER.info('      %d. %.4f ± %.4f | %s', i, cand['score'], cand['std'], ' + '.join(cand['subset']))
        
        cache_hits = 0
        if self.evaluation_cache is not None:
            cache_hits = self.evaluation_cache.get('_hits', 0)
        
        LOGGER.info('')
        LOGGER.info('⭐ 束搜索最佳组合: %s', ' + '.join(best_overall['subset']))
        LOGGER.info('   得分: %.4f ± %.4f', best_overall['score'], best_overall.get('std', 0.0))
        LOGGER.info('   维度: %d%s', 
                    best_overall.get('original_dim', 0),
                    ' (PCA -> {})'.format(best_overall.get('pca', 0)) if best_overall.get('pca', 0) else '')
        if cache_hits > 0:
            LOGGER.info('   缓存命中: %d 次', cache_hits)
        
        # 保存结果到实例变量
        self.best_result = best_overall
        self.search_history = history
        
        return {
            'best': best_overall,
            'history': history,
            'cache_hits': cache_hits,
            'method': 'beam_search',
        }
    
    def save_best_features(self, project_root='.', output_dir='features'):
        """保存最佳特征组合到文件（分别对 train/val/test 提取并保存）。
        
        参数:
            output_dir: str 或 Path，输出目录
            
        返回:
            dict 或 None: {split: Path} 字典；若无最佳组合或缺少数据集信息则返回 None
            
        异常:
            RuntimeError: 当缺少数据集信息时抛出
        """
        if self.best_result is None:
            LOGGER.warning('未找到最佳特征组合，请先执行 search()')
            return None
        
        subset = self.best_result.get('subset', [])
        if not subset:
            LOGGER.warning('最佳特征组合为空')
            return None
        
        # 检查数据集信息
        if self.dataset_dir is None:
            LOGGER.warning('无法保存特征：缺少数据集信息')
            return None
        
        # 统一保存到项目根目录下的 features 目录
        output_dir = Path(project_root) / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 重置PCA相关状态（如果存在）
        if hasattr(self, '_scaler'):
            delattr(self, '_scaler')
        if hasattr(self, '_pca'):
            delattr(self, '_pca')
        
        LOGGER.info('✨ 最佳组合: %s', ' + '.join(subset))
        LOGGER.info('🔄 分别在 train/val/test 上提取并保存最佳特征组合...')
        
        saved_paths = {}
        for split_name in ['train', 'val', 'test']:
            # 加载对应 split 的完整数据
            image_paths, labels = collect_image_paths_and_labels(self.dataset_dir, split_name)
            LOGGER.info('   [%s] 数据集大小: %d 张图像', split_name, len(image_paths))
            if not image_paths:
                LOGGER.warning('   [%s] 无数据，跳过保存', split_name)
                continue
            
            # 提取该 split 的最佳特征组合
            payloads = []
            for feature_name in subset:
                LOGGER.info('   [%s] 🔧 提取特征: %s', split_name, feature_name)
                try:
                    payload = UnifiedFeatureExtractor.extract_single_feature_matrix(
                        image_paths,
                        labels,
                        feature_name,
                        image_size=self.image_size,
                        n_jobs=self.n_jobs,
                        show_progress=True,
                    )
                    payloads.append(payload)
                    LOGGER.info('      [%s] ✓ 维度 %d | 样本 %d', split_name, payload['dim'], len(payload['valid_indices']))
                except Exception as exc:
                    LOGGER.error('      [%s] ✗ 提取失败 %s: %s', split_name, feature_name, exc)
                    raise RuntimeError('无法在 {} 提取特征: {}'.format(split_name, exc))
            
            X_split, y_split, indices = align_and_concatenate_features(payloads)
            LOGGER.info('   [%s] ✅ 特征提取完成: %d 样本', split_name, len(indices))
            
            # 如果评估时使用了PCA，需要在保存时也应用PCA
            original_dim = X_split.shape[1]
            pca_components = 0
            if self.pca_components is not None and original_dim > self.pca_components:
                pca_components = self.pca_components
            
            if pca_components > 0:
                # 需要在train集上拟合PCA，然后应用到所有split
                if split_name == 'train':
                    # 在train集上拟合标准化器和PCA
                    scaler = StandardScaler()
                    X_split_scaled = scaler.fit_transform(X_split)
                    pca = PCA(n_components=pca_components, random_state=42)
                    X_split = pca.fit_transform(X_split_scaled)
                    # 保存scaler和pca供后续split使用
                    self._scaler = scaler
                    self._pca = pca
                    LOGGER.info('   [%s] 🔧 应用PCA: %d -> %d 维', split_name, original_dim, pca_components)
                else:
                    # 在val/test集上使用train集拟合的标准化器和PCA
                    if not hasattr(self, '_scaler') or not hasattr(self, '_pca'):
                        raise RuntimeError('PCA未在train集上拟合，无法应用到{}集'.format(split_name))
                    X_split_scaled = self._scaler.transform(X_split)
                    X_split = self._pca.transform(X_split_scaled)
                    LOGGER.info('   [%s] 🔧 应用PCA: %d -> %d 维', split_name, original_dim, pca_components)
            
            feature_info = {
                'selected_features': subset,
                'original_dim': self.best_result.get('original_dim'),
                'pca': pca_components,
                'score': self.best_result.get('score'),
                'std': self.best_result.get('std'),
                'split': split_name,
                'use_full_dataset': True,
                'n_samples': len(indices),
            }
            split_features_path = output_dir / f'{split_name}_features.joblib'
            save_features_to_file(X_split, y_split, indices, feature_info, split_features_path)
            LOGGER.info('💾 [%s] 特征已保存: %s', split_name, split_features_path)
            saved_paths[split_name] = split_features_path
        
        if not saved_paths:
            return None
        return saved_paths
    
    def save_search_results(self, output_dir='runs/feature_search'):
        """保存搜索结果到 JSON 文件。
        
        参数:
            output_dir: str 或 Path，输出目录
            
        返回:
            Path: 保存的 JSON 文件路径，如果未找到结果则返回 None
        """
        if self.best_result is None:
            LOGGER.warning('未找到搜索结果，请先执行 search()')
            return None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.model_name,
                'cv': self.cv,
                'scoring': self.scoring,
                'beam_width': self.beam_width,
                'enable_cache': self.enable_cache,
                'image_size': self.image_size,
                'n_jobs': self.n_jobs,
                'pca_components': self.pca_components,
                'dataset_dir': str(self.dataset_dir) if self.dataset_dir else None,
                'split': self.split,
                'sample_ratio': self.sample_ratio,
            },
            'best_result': {
                'subset': self.best_result.get('subset', []),
                'score': self.best_result.get('score', 0.0),
                'std': self.best_result.get('std', 0.0),
                'original_dim': self.best_result.get('original_dim', 0),
                'pca': self.best_result.get('pca', 0),
            },
            'search_history': self.search_history,
            'feature_cache_info': {
                name: {
                    'dim': payload.get('dim', 0),
                    'n_samples': len(payload.get('valid_indices', [])),
                }
                for name, payload in self.feature_cache.items()
            },
        }
        
        if self.evaluation_cache is not None:
            cache_hits = self.evaluation_cache.get('_hits', 0)
            results_data['cache_info'] = {
                'hits': cache_hits,
                'size': len([k for k in self.evaluation_cache.keys() if k != '_hits']),
            }
        
        results_path = output_dir / 'search_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        LOGGER.info('📝 搜索结果已保存: %s', results_path)
        return results_path
    
    def run(self, dataset_dir='dataset', split='train', sample_ratio=1.0, 
            max_features=5, output_dir='runs/feature_search', feature_names=None,
            save_features=True, save_log=True):
        """运行完整的特征搜索流程。
        
        参数:
            dataset_dir: str，数据集根目录
            split: str，数据集分割 (train/val/test)
            sample_ratio: float，采样比例 (0-1]
            max_features: int，最大特征组合数量
            output_dir: str 或 Path，输出目录
            feature_names: list 或 None，要提取的特征名称列表
            save_features: bool，是否保存最佳特征（使用完整数据集提取）
            save_log: bool，是否保存日志文件
            
        返回:
            dict: 搜索摘要信息
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件
        if save_log:
            log_file = output_dir / 'feature_search.log'
            # 将文件日志器添加到主日志器
            if not any(isinstance(h, logging.FileHandler) for h in LOGGER.handlers):
                file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                LOGGER.addHandler(file_handler)
            LOGGER.info('📝 日志将保存到: %s', log_file)
        
        # 加载数据
        image_paths, labels = self.load_data(dataset_dir, split, sample_ratio)
        
        # 提取特征
        self.extract_features(image_paths, labels, feature_names)
        
        # 执行搜索
        search_result = self.search(max_features=max_features)
        
        # 保存搜索结果到 JSON
        results_path = self.save_search_results(str(output_dir))
        
        # 保存最佳特征（使用完整数据集，如果启用）
        best_features_path = None
        if save_features:
            saved = self.save_best_features(project_root=str(project_root), output_dir='features')
            if isinstance(saved, dict):
                best_features_path = {k: str(v) for k, v in saved.items()}
            else:
                best_features_path = None
        else:
            LOGGER.info('⏭️  跳过特征保存（使用 --no_save_features 时）')
        
        best = search_result.get('best', {})
        subset = best.get('subset', [])
        
        if not subset:
            LOGGER.info('未找到满足条件的特征组合')
        
        return {
            'best_subset': subset,
            'best_score': best.get('score', 0.0),
            'best_std': best.get('std', 0.0),
            'output_dir': str(output_dir),
            'best_features_path': best_features_path if best_features_path else None,
            'results_path': str(results_path) if results_path else None,
            'search_method': 'beam_search',
            'search_history': search_result.get('history', []),
            'cache_hits': search_result.get('cache_hits', 0),
        }


def build_arg_parser():
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='特征组合搜索（束搜索）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset_dir', default='dataset', help='数据集根目录')
    parser.add_argument('--image_size', type=int, default=128, help='图像缩放尺寸')
    parser.add_argument('--model', default='svm', choices=['svm', 'rf'], help='评估模型')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='数据采样比例 (0-1]')
    parser.add_argument('--max_features', type=int, default=8, help='最大选择特征数量')
    parser.add_argument('--scoring', default='accuracy', help='评分指标')
    parser.add_argument('--out_dir', default='runs/feature_search', help='结果输出目录')
    parser.add_argument('--n_jobs', type=int, default=16, help='特征提取并行线程数')
    parser.add_argument('--beam_width', type=int, default=5, help='束搜索的束宽')
    parser.add_argument('--pca_components', type=int, default=512, help='PCA降维后的维度（设置为0或负数表示不降维）')
    parser.add_argument('--enable_cache', action='store_true', help='启用评估结果缓存')
    parser.add_argument('--no_save_features', action='store_true', help='不保存最佳特征文件')
    parser.add_argument('--no_save_log', action='store_true', help='不保存日志文件')
    parser.add_argument('--log_level', default='INFO', help='日志级别')
    return parser


def parse_args(args=None):
    """解析命令行参数。"""
    parser = build_arg_parser()
    return parser.parse_args(args=args)


def main():
    """命令行入口。"""
    args = parse_args()
    
    # 配置日志
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    )
    
    # 显示搜索配置
    LOGGER.info('=' * 60)
    LOGGER.info('特征搜索配置（束搜索）')
    LOGGER.info('=' * 60)
    LOGGER.info('数据集: %s', args.dataset_dir)
    LOGGER.info('采样比例: %.0f%%', args.sample_ratio * 100)
    LOGGER.info('最大特征数: %d', args.max_features)
    LOGGER.info('束宽: %d', args.beam_width)
    LOGGER.info('评估模型: %s', args.model)
    LOGGER.info('交叉验证: %d 折', args.cv)
    pca_components = args.pca_components if args.pca_components > 0 else None
    if pca_components is not None:
        LOGGER.info('PCA降维: %d 维', pca_components)
    else:
        LOGGER.info('PCA降维: 禁用')
    LOGGER.info('=' * 60)
    LOGGER.info('')
    
    # 创建搜索器并运行
    searcher = FeatureSearch(
        model=args.model,
        cv=args.cv,
        scoring=args.scoring,
        beam_width=args.beam_width,
        enable_cache=args.enable_cache,
        image_size=args.image_size,
        n_jobs=args.n_jobs,
        pca_components=pca_components,
    )
    
    summary = searcher.run(
        dataset_dir=args.dataset_dir,
        split='train',
        sample_ratio=args.sample_ratio,
        max_features=args.max_features,
        output_dir=args.out_dir,
        save_features=not args.no_save_features,
        save_log=not args.no_save_log,
    )
    
    LOGGER.info('')
    LOGGER.info('=' * 60)
    LOGGER.info('搜索完成')
    LOGGER.info('=' * 60)
    LOGGER.info('最佳组合: %s', ' + '.join(summary['best_subset']))
    LOGGER.info('最佳得分: %.4f ± %.4f', summary['best_score'], summary['best_std'])
    if summary.get('cache_hits', 0) > 0:
        LOGGER.info('缓存命中: %d 次', summary['cache_hits'])
    LOGGER.info('结果目录: %s', summary['output_dir'])
    if summary.get('results_path'):
        LOGGER.info('搜索结果: %s', summary['results_path'])
    if summary.get('best_features_path'):
        LOGGER.info('最佳特征: %s', summary['best_features_path'])
    LOGGER.info('=' * 60)


if __name__ == '__main__':
    main()
