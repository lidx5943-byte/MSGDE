"""
特征提取模块
============

从轨迹数据中提取动力学特征。

使用示例
--------
>>> from src.dynamics.features import extract_trajectory_features
>>> 
>>> features = extract_trajectory_features(trajectories)
"""

import warnings
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from typing import Tuple, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils.logger import get_logger, create_progress, console

logger = get_logger(__name__)

# 尝试导入nolds库（抑制警告）
NOLDS_AVAILABLE = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    pass


def extract_trajectory_features(
    trajectories: np.ndarray,
    center_node_idx: int = 0,
    verbose: bool = False,
    enable_chaos: bool = True,
    enable_sync: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    从轨迹数据中提取特征（优化版本：使用向量化操作和并行计算）
    
    提取的特征包括：
    - 时域统计量（均值、方差、最大值、最小值）× 3个变量 = 12个特征（索引0-11）
    - 混沌特征（Lyapunov指数）= 1个特征（索引12，可选）
    - 同步特征（与中心节点的相关系数）= 1个特征（索引13，可选）
    共12-14个特征（取决于enable_chaos和enable_sync）
    
    注意：特征顺序与data_processed_code保持一致
    
    参数
    ----
    trajectories : np.ndarray
        轨迹数据，形状为 (n_nodes, n_times, 3)
    center_node_idx : int
        中心节点索引，用于计算同步特征
    verbose : bool
        是否显示进度条
    enable_chaos : bool
        是否提取混沌特征
    enable_sync : bool
        是否提取同步特征
    n_jobs : int
        并行作业数（用于混沌特征计算），1表示串行
        
    返回
    ----
    np.ndarray
        特征矩阵，形状为 (n_nodes, n_features)
    """
    n_nodes, n_times, n_vars = trajectories.shape
    
    # 计算特征数量
    n_features = 12  # 基础时域统计特征
    if enable_chaos:
        n_features += 1
    if enable_sync:
        n_features += 1
    
    # 初始化特征矩阵
    features = np.zeros((n_nodes, n_features))
    
    # ============================================
    # 优化1：向量化时域统计量（批量计算）
    # ============================================
    # 一次性计算所有节点的统计量，避免循环
    for j in range(3):
        var_data = trajectories[:, :, j]  # (n_nodes, n_times)
        
        # 批量计算：均值、方差、最大值、最小值
        features[:, j*4] = np.mean(var_data, axis=1)      # 均值
        features[:, j*4+1] = np.var(var_data, axis=1)    # 方差
        features[:, j*4+2] = np.max(var_data, axis=1)     # 最大值
        features[:, j*4+3] = np.min(var_data, axis=1)     # 最小值
    
    feature_idx = 12
    
    # ============================================
    # 优化2：混沌特征计算（索引12）- 与data_processed_code保持一致
    # ============================================
    if enable_chaos:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if NOLDS_AVAILABLE and n_jobs > 1 and n_nodes > 10:
                # 并行计算Lyapunov指数（仅当节点数较多时使用并行）
                def compute_lyap(x_data):
                    """计算单个节点的Lyapunov指数（用于并行）"""
                    try:
                        return nolds.lyap_r(x_data, min_tsep=10, tau=1, min_neighbors=20)
                    except:
                        return 0.0
                
                x_data_list = [trajectories[i, :, 0] for i in range(n_nodes)]
                
                # 使用进程池并行计算
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = {executor.submit(compute_lyap, x_data): i 
                              for i, x_data in enumerate(x_data_list)}
                    
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            features[idx, feature_idx] = future.result()
                        except:
                            features[idx, feature_idx] = 0.0
            else:
                # 串行计算（如果nolds不可用、n_jobs=1或节点数较少）
                if NOLDS_AVAILABLE:
                    for i in range(n_nodes):
                        try:
                            x_data = trajectories[i, :, 0]
                            features[i, feature_idx] = nolds.lyap_r(
                                x_data, min_tsep=10, tau=1, min_neighbors=20
                            )
                        except:
                            features[i, feature_idx] = 0.0
                else:
                    # 如果没有 nolds，为了和旧版保持一致，返回 0.0
                    # 旧版逻辑：features[i, 12] = 0.0
                    features[:, feature_idx] = 0.0
        
        feature_idx += 1
    
    # ============================================
    # 优化3：同步特征计算（索引13）- 与data_processed_code保持一致
    # ============================================
    if enable_sync:
        center_trajectory = trajectories[center_node_idx]  # (n_times, 3)
        
        # 使用np.corrcoef()计算相关系数，与data_processed_code保持一致
        for i in range(n_nodes):
            try:
                node_trajectory = trajectories[i]  # (n_times, 3)
                correlation = np.corrcoef(
                    node_trajectory.flatten(),
                    center_trajectory.flatten()
                )[0, 1]
                features[i, feature_idx] = correlation if not np.isnan(correlation) else 0.0
            except:
                features[i, feature_idx] = 0.0
        
        feature_idx += 1
    
    return features


def _extract_node_features(
    features: np.ndarray,
    node_idx: int,
    node_trajectory: np.ndarray,
    center_trajectory: np.ndarray,
    enable_chaos: bool = True,
    enable_sync: bool = True,
) -> None:
    """
    提取单个节点的特征（内部函数）
    
    参数
    ----
    features : np.ndarray
        特征矩阵，会原地修改
    node_idx : int
        节点索引
    node_trajectory : np.ndarray
        节点轨迹，形状为 (n_times, 3)
    center_trajectory : np.ndarray
        中心节点轨迹
    enable_chaos : bool
        是否提取混沌特征
    enable_sync : bool
        是否提取同步特征
    """
    # 1. 时域统计量（12个特征）
    for j in range(3):
        var_data = node_trajectory[:, j]
        features[node_idx, j*4] = np.mean(var_data)
        features[node_idx, j*4+1] = np.var(var_data)
        features[node_idx, j*4+2] = np.max(var_data)
        features[node_idx, j*4+3] = np.min(var_data)
    
    # 特征索引偏移
    feature_idx = 12
    
    # 2. 混沌特征（1个特征，索引12）- Lyapunov指数，与data_processed_code保持一致
    if enable_chaos:
        if NOLDS_AVAILABLE:
            try:
                x_data = node_trajectory[:, 0]
                lyap_exp = nolds.lyap_r(x_data, min_tsep=10, tau=1, min_neighbors=20)
                features[node_idx, feature_idx] = lyap_exp
            except:
                features[node_idx, feature_idx] = 0.0
        else:
            # 如果没有 nolds，为了和旧版保持一致，返回 0.0
            features[node_idx, feature_idx] = 0.0
        feature_idx += 1
    
    # 3. 同步特征（1个特征，索引13）- 与中心节点的相关系数，与data_processed_code保持一致
    if enable_sync:
        try:
            correlation = np.corrcoef(
                node_trajectory.flatten(),
                center_trajectory.flatten()
            )[0, 1]
            features[node_idx, feature_idx] = correlation if not np.isnan(correlation) else 0.0
        except:
            features[node_idx, feature_idx] = 0.0


def extract_features_at_step(
    trajectories: np.ndarray,
    step: int,
    window_size: Optional[int] = None,
    center_node_idx: int = 0,
    enable_chaos: bool = True,
    enable_sync: bool = True,
) -> np.ndarray:
    """
    在指定演化步骤提取特征
    
    参数
    ----
    trajectories : np.ndarray
        轨迹数据，形状为 (n_nodes, n_times, 3)
    step : int
        演化步骤
    window_size : int, optional
        窗口大小
    center_node_idx : int
        中心节点索引
    enable_chaos : bool
        是否提取混沌特征
    enable_sync : bool
        是否提取同步特征
        
    返回
    ----
    np.ndarray
        特征矩阵
    """
    n_nodes, n_times, n_vars = trajectories.shape
    
    # 确定数据范围
    if window_size is not None and window_size > 0:
        start_step = max(0, step - window_size)
    else:
        start_step = 0
    
    # 提取窗口内的轨迹数据
    trajectories_window = trajectories[:, start_step:step+1, :]
    
    if trajectories_window.shape[1] == 0:
        # 计算特征数量
        n_features = 12
        if enable_chaos:
            n_features += 1
        if enable_sync:
            n_features += 1
        return np.zeros((n_nodes, n_features))
    
    return extract_trajectory_features(
        trajectories_window, center_node_idx, verbose=False,
        enable_chaos=enable_chaos, enable_sync=enable_sync,
        n_jobs=1  # 差异性计算时使用串行
    )


def compute_feature_diversity(
    features: np.ndarray,
    method: str = "combined",
) -> float:
    """
    计算特征差异性指标
    
    参数
    ----
    features : np.ndarray
        特征矩阵，形状为 (n_nodes, n_features)
    method : str
        差异性计算方法：
        - 'std': 标准差
        - 'frobenius': Frobenius范数
        - 'mean_distance': 平均欧氏距离
        - 'variance': 方差
        - 'entropy': 熵
        - 'combined': 组合多个指标
        
    返回
    ----
    float
        差异性指标值
    """
    if features.shape[0] == 0:
        return 0.0
    
    if method == 'std':
        feature_stds = np.std(features, axis=0)
        return np.mean(feature_stds)
    
    elif method == 'frobenius':
        return np.linalg.norm(features, 'fro')
    
    elif method == 'mean_distance':
        if features.shape[0] == 1:
            return 0.0
        distances = pdist(features, metric='euclidean')
        return np.mean(distances)
    
    elif method == 'variance':
        return np.var(features)
    
    elif method == 'entropy':
        n_bins = 20
        flattened = features.flatten()
        hist, _ = np.histogram(flattened, bins=n_bins)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        return entropy(hist)
    
    elif method == 'combined':
        std_score = compute_feature_diversity(features, 'std')
        dist_score = compute_feature_diversity(features, 'mean_distance')
        var_score = compute_feature_diversity(features, 'variance')
        
        scores = np.array([std_score, dist_score, var_score])
        if np.std(scores) > 0:
            scores = (scores - np.mean(scores)) / np.std(scores)
        return np.mean(scores)
    
    else:
        raise ValueError(f"未知的差异性方法: {method}")


def compute_diversity_over_steps(
    trajectories: np.ndarray,
    step_interval: int = 100,
    window_size: Optional[int] = None,
    method: str = "combined",
    center_node_idx: int = 0,
    enable_chaos: bool = True,
    enable_sync: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算特征差异性随演化步骤的变化
    
    参数
    ----
    trajectories : np.ndarray
        轨迹数据
    step_interval : int
        计算间隔
    window_size : int, optional
        窗口大小
    method : str
        差异性计算方法
    center_node_idx : int
        中心节点索引
    enable_chaos : bool
        是否提取混沌特征
    enable_sync : bool
        是否提取同步特征
        
    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        (步骤数组, 差异性数组)
    """
    n_nodes, n_times, n_vars = trajectories.shape
    
    # 确定计算步骤
    if window_size is not None and window_size > 0:
        start_step = window_size
    else:
        start_step = step_interval
    
    steps = np.arange(start_step, n_times, step_interval)
    if n_times - 1 not in steps:
        steps = np.append(steps, n_times - 1)
    
    diversity_scores = []
    
    with create_progress() as progress:
        task = progress.add_task("计算差异性", total=len(steps))
        
        for step in steps:
            features = extract_features_at_step(
                trajectories, step, window_size, center_node_idx,
                enable_chaos=enable_chaos, enable_sync=enable_sync
            )
            diversity = compute_feature_diversity(features, method)
            diversity_scores.append(diversity)
            
            progress.update(task, advance=1)
    
    return steps, np.array(diversity_scores)
