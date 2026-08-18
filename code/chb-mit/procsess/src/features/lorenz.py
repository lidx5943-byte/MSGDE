# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
Lorenz动力学特征提取
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from pathlib import Path
from scipy.sparse import csr_matrix, issparse, diags
import scipy.sparse as sparse
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from ..utils.console import console, print_info as logger, print_warning, print_error


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class LorenzOscillator:
    """Lorenz振子参数（参考com文件）"""
    delta: float = 10.0      # δ参数
    gamma: float = 60.0      # γ参数
    beta: float = 8.0 / 3.0  # β参数
    rk: float = 7.0          # 环形耦合强度


@dataclass
class RosslerOscillator:
    """Rossler振子参数"""
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7
    rk: float = 0.0          # 环形耦合强度 (可选)


@dataclass
class LorenzConfig:
    """Lorenz求解器配置"""
    coupling_strength: float = 0.42
    coupling_mode: str = "xyz_all"  # 耦合模式: "x_only", "xyz_all", "nearest_neighbor"
    dt: float = 0.01
    total_steps: int = 3000
    steady_steps: int = 1000
    initial_range: float = 1.0
    sparsity_threshold: float = 1e-6
    use_periodic_boundary: bool = True  # 是否使用周期性边界条件
    # 数值稳定性参数
    state_clip_min: float = -200.0
    state_clip_max: float = 200.0
    dynamics_clip_min: float = -5000.0
    dynamics_clip_max: float = 5000.0


@dataclass
class RosslerConfig:
    """Rossler求解器配置"""
    coupling_strength: float = 0.1
    coupling_mode: str = "x_only"  # Rossler 通常只耦合 x 或 y
    dt: float = 0.01
    total_steps: int = 3000
    steady_steps: int = 1000
    initial_range: float = 1.0
    sparsity_threshold: float = 1e-6
    use_periodic_boundary: bool = False
    # 数值稳定性参数
    state_clip_min: float = -100.0
    state_clip_max: float = 100.0
    dynamics_clip_min: float = -2000.0
    dynamics_clip_max: float = 2000.0


# ============================================================
# 轨迹保存辅助函数（供 step_6_transformer 训练使用）
# ============================================================

def _trajectory_save_enabled(config_dict: Dict[str, Any]) -> bool:
    lorenz_cfg = (config_dict or {}).get("lorenz", {}) or {}
    ts_cfg = lorenz_cfg.get("trajectory_save", {}) or {}
    return bool(ts_cfg.get("enabled", False))


def _trajectory_save_settings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    lorenz_cfg = (config_dict or {}).get("lorenz", {}) or {}
    ts_cfg = lorenz_cfg.get("trajectory_save", {}) or {}
    # 默认：保存分量跟 lorenz.components 一致
    save_components = ts_cfg.get("components", None)
    if save_components is None:
        save_components = lorenz_cfg.get("components", ["x", "y", "z"])
    stride = int(ts_cfg.get("stride", 10))
    if stride <= 0:
        stride = 1
    dtype_str = str(ts_cfg.get("dtype", "float16")).lower()
    if dtype_str not in ("float16", "float32"):
        dtype_str = "float16"
    scales_to_save = ts_cfg.get("scales_to_save", None)
    return {
        "components": list(save_components),
        "stride": stride,
        "dtype": np.float16 if dtype_str == "float16" else np.float32,
        "dtype_str": dtype_str,
        "scales_to_save": scales_to_save,
    }


def _select_components_trajectory(
    steady_trajectory: np.ndarray,
    components: List[str],
    stride: int,
    out_dtype: np.dtype,
) -> np.ndarray:
    """
    steady_trajectory: (n_nodes, steady_steps, 3)
    return: (n_nodes, T, C_sel)
    """
    comp_indices = {"x": 0, "y": 1, "z": 2}
    sel = [comp_indices[c] for c in components if c in comp_indices]
    if len(sel) == 0:
        sel = [0]  # 至少保存 x

    # 下采样 + 分量选择
    traj = steady_trajectory[:, ::stride, :]
    traj = traj[:, :, sel]
    return traj.astype(out_dtype, copy=False)


def _save_scale_trajectory_npy(
    output_dir: str,
    scale_idx: int,
    steady_trajectory: np.ndarray,
    config_dict: Dict[str, Any],
) -> Optional[str]:
    """
    将单尺度轨迹保存为 npy：
      {output_dir}/scale_{scale_idx:02d}.npy
    返回保存路径；若未保存则返回 None。
    """
    if not output_dir or not _trajectory_save_enabled(config_dict):
        return None

    settings = _trajectory_save_settings(config_dict)
    scales_to_save = settings.get("scales_to_save", None)
    if scales_to_save is not None:
        try:
            if int(scale_idx) not in set(int(s) for s in scales_to_save):
                return None
        except Exception:
            pass

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_path = out_p / f"scale_{int(scale_idx):02d}.npy"

    traj = _select_components_trajectory(
        steady_trajectory=steady_trajectory,
        components=settings["components"],
        stride=int(settings["stride"]),
        out_dtype=settings["dtype"],
    )
    np.save(str(save_path), traj)
    return str(save_path)


# ============================================================
# 辅助函数
# ============================================================

def _laplacian_to_adjacency(L: np.ndarray) -> np.ndarray:
    """
    从拉普拉斯矩阵计算邻接矩阵
    
    参数
    ----
    L : np.ndarray
        拉普拉斯矩阵 L = D - A
        
    返回
    ----
    np.ndarray
        邻接矩阵 A（对角线元素为0）
        
    数学原理
    --------
    - 拉普拉斯矩阵：L = D - A，其中 D 是度矩阵
    - 对于非对角元素：L_ij = -A_ij (i≠j)
    - 对于对角元素：L_ii = D_ii = Σ_j A_ij
    - 因此：A_ij = -L_ij (i≠j)，A_ii = 0
    """
    if issparse(L):
        L = L.toarray()
    
    A = -L.copy()
    np.fill_diagonal(A, 0.0)
    
    # 确保非对角元素非负
    A = np.maximum(A, 0.0)
    
    return A


def _compute_stats(data: np.ndarray) -> np.ndarray:
    """
    计算统计特征
    
    参数
    ----
    data : np.ndarray
        输入数据，形状为 (n_steps,) 或 (n_nodes, n_steps)
        
    返回
    ----
    np.ndarray
        统计特征 [mean, max, min, median, var, std]
    """
    return np.array([
        np.mean(data),
        np.max(data),
        np.min(data),
        np.median(data),
        np.var(data),
        np.std(data)
    ], dtype=np.float32)


# ============================================================
# 稀疏矩阵优化的Lorenz求解器
# ============================================================

def _coupled_dynamics_sparse(
    state_matrix: np.ndarray,
    A_sparse: csr_matrix,
    oscillator: LorenzOscillator,
    config: LorenzConfig,
    h: float = 1.0e-3
) -> np.ndarray:
    """
    计算耦合Lorenz系统的动力学（支持三种耦合模式）
    
    支持的耦合模式：
    1. "x_only": 全局网络x方向耦合，只对x分量耦合，使用拉普拉斯矩阵
    2. "xyz_all": 3个分量的直积单位矩阵耦合，使用邻接矩阵 A 的直积形式（默认）
    3. "nearest_neighbor": 最近邻扩散耦合，使用周期性边界条件
    
    参数
    ----
    state_matrix : np.ndarray
        状态矩阵，形状为 (n_nodes, 3) 表示 [x, y, z]
    A_sparse : csr_matrix
        稀疏邻接矩阵（二值化，来自Cutoff_0_1.npy）
    oscillator : LorenzOscillator
        振子参数
    config : LorenzConfig
        求解器配置（包含耦合模式和强度）
    h : float
        时间步长
        
    返回
    ----
    np.ndarray
        导数矩阵，形状为 (n_nodes, 3)
    """
    n = state_matrix.shape[0]
    x = state_matrix[:, 0]
    y = state_matrix[:, 1]
    z = state_matrix[:, 2]
    
    # 限制状态值
    np.clip(
        state_matrix,
        config.state_clip_min,
        config.state_clip_max,
        out=state_matrix
    )
    
    # 计算局部动力学（向量化）
    local_dynamics = np.empty_like(state_matrix, dtype=np.float32)
    local_dynamics[:, 0] = oscillator.delta * (y - x)
    local_dynamics[:, 1] = oscillator.gamma * x - y - x * z
    local_dynamics[:, 2] = x * y - oscillator.beta * z
    
    # 检查局部动力学是否包含 NaN 或 Inf
    if np.any(~np.isfinite(local_dynamics)):
        local_dynamics = np.nan_to_num(
            local_dynamics,
            nan=0.0,
            posinf=config.dynamics_clip_max,
            neginf=config.dynamics_clip_min
        )
    
    # 限制局部动力学
    np.clip(
        local_dynamics,
        config.dynamics_clip_min,
        config.dynamics_clip_max,
        out=local_dynamics
    )
    
    # 计算耦合项
    epsilon = config.coupling_strength
    
    if config.coupling_mode == "x_only":
        # 方式1：全局网络x方向耦合
        # 只对x分量耦合，使用拉普拉斯矩阵 L = D - A
        # 计算度矩阵
        if issparse(A_sparse):
            degrees = np.array(A_sparse.sum(axis=1)).flatten()
            L = sparse.diags(degrees, format='csr') - A_sparse
            coupling_x = -L.dot(x)
        else:
            degrees = np.sum(A_sparse, axis=1)
            L = np.diag(degrees) - A_sparse
            coupling_x = -np.dot(L, x)
        
        # 将耦合项加到 dy 上
        local_dynamics[:, 1] += epsilon * coupling_x
        
        # 如果使用周期性边界，添加环形耦合
        if config.use_periodic_boundary:
            x_next = np.roll(x, -1)  # x[i+1]
            x_prev = np.roll(x, 1)   # x[i-1]
            cc1 = oscillator.rk * (x_next - x_prev)
            local_dynamics[:, 1] += cc1
            
    elif config.coupling_mode == "xyz_all":
        # 方式2：3个分量的直积单位矩阵耦合（默认）
        # 使用邻接矩阵 A 的直积形式：ε * (A ⊗ I_3) @ u
        # 即：ε * Σ_j A[i,j] * u[j]
        if issparse(A_sparse):
            coupling = A_sparse.dot(state_matrix)
        else:
            coupling = np.dot(A_sparse, state_matrix)
        
        # 检查耦合项是否包含 NaN 或 Inf
        if np.any(~np.isfinite(coupling)):
            coupling = np.nan_to_num(
                coupling,
                nan=0.0,
                posinf=config.dynamics_clip_max,
                neginf=config.dynamics_clip_min
            )
        
        # 限制耦合项范围
        np.clip(
            coupling,
            config.dynamics_clip_min,
            config.dynamics_clip_max,
            out=coupling
        )
        
        local_dynamics += epsilon * coupling
        
        if config.use_periodic_boundary:
            x_next = np.roll(x, -1)  # x[i+1]
            x_prev = np.roll(x, 1)   # x[i-1]
            cc1 = oscillator.rk * (x_next - x_prev)
            local_dynamics[:, 1] += cc1
            
    elif config.coupling_mode == "nearest_neighbor":
        # 方式3：最近邻扩散耦合
        # 公式：ε * (u_{i-1} - 2u_i + u_{i+1})
        # 使用周期性边界条件
        u_prev = np.roll(state_matrix, 1, axis=0)   # u_{i-1}
        u_next = np.roll(state_matrix, -1, axis=0)   # u_{i+1}
        
        # 计算耦合项：u_{i-1} - 2u_i + u_{i+1}
        coupling = u_prev - 2.0 * state_matrix + u_next
        
        # 检查耦合项是否包含 NaN 或 Inf
        if np.any(~np.isfinite(coupling)):
            coupling = np.nan_to_num(
                coupling,
                nan=0.0,
                posinf=config.dynamics_clip_max,
                neginf=config.dynamics_clip_min
            )
        
        # 限制耦合项范围
        np.clip(
            coupling,
            config.dynamics_clip_min,
            config.dynamics_clip_max,
            out=coupling
        )
        
        local_dynamics += epsilon * coupling
        
    else:
        # 默认：使用拉普拉斯矩阵（向后兼容）
        if issparse(A_sparse):
            degrees = np.array(A_sparse.sum(axis=1)).flatten()
            L = sparse.diags(degrees, format='csr') - A_sparse
            coupling = L.dot(state_matrix)
        else:
            degrees = np.sum(A_sparse, axis=1)
            L = np.diag(degrees) - A_sparse
            coupling = np.dot(L, state_matrix)
        
        # 检查耦合项是否包含 NaN 或 Inf
        if np.any(~np.isfinite(coupling)):
            coupling = np.nan_to_num(
                coupling,
                nan=0.0,
                posinf=config.dynamics_clip_max,
                neginf=config.dynamics_clip_min
            )
        
        # 限制耦合项范围
        np.clip(
            coupling,
            config.dynamics_clip_min,
            config.dynamics_clip_max,
            out=coupling
        )
        
        local_dynamics += epsilon * coupling
    
    # 最终检查并限制范围
    if np.any(~np.isfinite(local_dynamics)):
        local_dynamics = np.nan_to_num(
            local_dynamics,
            nan=0.0,
            posinf=config.dynamics_clip_max,
            neginf=config.dynamics_clip_min
        )
    
    np.clip(
        local_dynamics,
        config.dynamics_clip_min,
        config.dynamics_clip_max,
        out=local_dynamics
    )
    
    return local_dynamics


def _rk4_step_sparse(
    state_matrix: np.ndarray,
    A_sparse: csr_matrix,
    oscillator: LorenzOscillator,
    config: LorenzConfig,
    h: float
) -> np.ndarray:
    """
    RK4单步积分
    
    参数
    ----
    state_matrix : np.ndarray
        当前状态，形状为 (n_nodes, 3)
    A_sparse : csr_matrix
        稀疏邻接矩阵
    oscillator : LorenzOscillator
        振子参数
    config : LorenzConfig
        求解器配置
    h : float
        时间步长
        
    返回
    ----
    np.ndarray
        更新后的状态，形状为 (n_nodes, 3)
    """
    k1 = _coupled_dynamics_sparse(state_matrix, A_sparse, oscillator, config, h)
    k2 = _coupled_dynamics_sparse(state_matrix + 0.5 * h * k1, A_sparse, oscillator, config, h)
    k3 = _coupled_dynamics_sparse(state_matrix + 0.5 * h * k2, A_sparse, oscillator, config, h)
    k4 = _coupled_dynamics_sparse(state_matrix + h * k3, A_sparse, oscillator, config, h)
    
    new_state = state_matrix + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # 裁剪状态值
    np.clip(new_state, config.state_clip_min, config.state_clip_max, out=new_state)
    
    return new_state


def _coupled_rossler_dynamics_sparse(
    state_matrix: np.ndarray,
    A_sparse: csr_matrix,
    oscillator: RosslerOscillator,
    config: RosslerConfig,
    h: float = 1.0e-3
) -> np.ndarray:
    """
    计算耦合Rossler系统的动力学
    """
    n = state_matrix.shape[0]
    x = state_matrix[:, 0]
    y = state_matrix[:, 1]
    z = state_matrix[:, 2]
    
    # 限制状态值
    np.clip(
        state_matrix,
        config.state_clip_min,
        config.state_clip_max,
        out=state_matrix
    )
    
    # 计算局部动力学
    local_dynamics = np.empty_like(state_matrix, dtype=np.float32)
    local_dynamics[:, 0] = -y - z
    local_dynamics[:, 1] = x + oscillator.a * y
    local_dynamics[:, 2] = oscillator.b + z * (x - oscillator.c)
    
    # 限制局部动力学
    np.clip(
        local_dynamics,
        config.dynamics_clip_min,
        config.dynamics_clip_max,
        out=local_dynamics
    )
    
    # 计算耦合项
    epsilon = config.coupling_strength
    
    if issparse(A_sparse):
        degrees = np.array(A_sparse.sum(axis=1)).flatten()
        L = sparse.diags(degrees, format='csr') - A_sparse
        coupling_x = -L.dot(x)
    else:
        degrees = np.sum(A_sparse, axis=1)
        L = np.diag(degrees) - A_sparse
        coupling_x = -np.dot(L, x)
        
    local_dynamics[:, 0] += epsilon * coupling_x
    
    if config.use_periodic_boundary and oscillator.rk != 0:
        x_next = np.roll(x, -1)
        x_prev = np.roll(x, 1)
        local_dynamics[:, 0] += oscillator.rk * (x_next - 2*x + x_prev)

    return local_dynamics


def _rk4_step_rossler_sparse(
    state_matrix: np.ndarray,
    A_sparse: csr_matrix,
    oscillator: RosslerOscillator,
    config: RosslerConfig,
    h: float
) -> np.ndarray:
    """Rossler RK4单步积分"""
    k1 = _coupled_rossler_dynamics_sparse(state_matrix, A_sparse, oscillator, config, h)
    k2 = _coupled_rossler_dynamics_sparse(state_matrix + 0.5 * h * k1, A_sparse, oscillator, config, h)
    k3 = _coupled_rossler_dynamics_sparse(state_matrix + 0.5 * h * k2, A_sparse, oscillator, config, h)
    k4 = _coupled_rossler_dynamics_sparse(state_matrix + h * k3, A_sparse, oscillator, config, h)
    return state_matrix + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_lorenz_sparse(
    adj_matrix: np.ndarray,
    config: LorenzConfig,
    oscillator: LorenzOscillator,
    random_seed: int = 42
) -> Dict:
    """
    稀疏矩阵优化的Lorenz动力学模拟
    
    参数
    ----
    adj_matrix : np.ndarray
        邻接矩阵或拉普拉斯矩阵
    config : LorenzConfig
        求解器配置
    oscillator : LorenzOscillator
        振子参数
    random_seed : int
        随机种子
        
    返回
    ----
    Dict
        包含:
        - 'steady_trajectory': 稳态轨迹 (n_nodes, steady_steps, 3)
        - 'stats': 统计特征字典
        - 'sparsity': 稀疏度
    """
    np.random.seed(random_seed)
    N = adj_matrix.shape[0]
    
    # 处理输入矩阵
    threshold = float(config.sparsity_threshold)
    A = np.abs(adj_matrix).astype(np.float32)
    
    nnz_before = np.count_nonzero(A)
    
    A[A < threshold] = 0
    np.fill_diagonal(A, 0)  # 确保对角线为0
    
    # 归一化
    adj_max = np.max(A)
    if adj_max > 1e-6:
        A /= adj_max
        # 归一化后，如果值变得很小，可能再次被阈值过滤
        # 但这里不再应用阈值，因为已经归一化了
    
    # 检查稀疏度
    nnz = np.count_nonzero(A)
    sparsity = 1.0 - nnz / (N * N)
    
    if nnz_before > 0 and nnz == 0:
        print_warning(
            f"警告: 输入矩阵有 {nnz_before} 个非零元素，但经过阈值过滤 "
            f"(threshold={threshold}) 和归一化后变为0。这可能是因为所有边的权重都小于阈值。"
        )
    
    if nnz == 0:
        # 完全没有连接，返回默认特征
        # 创建空的稳态轨迹
        steady_trajectory = np.zeros((N, config.steady_steps, 3), dtype=np.float32)
        
        # 计算 per-node 统计特征（全为0）
        per_node_stats = []
        for comp_idx, comp_name in enumerate(['x', 'y', 'z']):
            # 每个节点在该分量上的轨迹: (n_nodes, steady_steps)
            comp_traj = steady_trajectory[:, :, comp_idx]
            
            # 对每个节点计算统计量 → (n_nodes,)
            node_mean = np.mean(comp_traj, axis=1)
            node_max = np.max(comp_traj, axis=1)
            node_min = np.min(comp_traj, axis=1)
            node_median = np.median(comp_traj, axis=1)
            node_var = np.var(comp_traj, axis=1)
            node_std = np.std(comp_traj, axis=1)
            
            # 堆叠: (n_nodes, 6)
            comp_stats = np.stack([node_mean, node_max, node_min, node_median, node_var, node_std], axis=1)
            per_node_stats.append(comp_stats)
        
        # 合并所有分量: (n_nodes, 18)
        per_node_features = np.hstack(per_node_stats).astype(np.float32)
        
        return {
            'steady_trajectory': steady_trajectory,
            'per_node_features': per_node_features,  # 添加缺失的键
            'stats': {
                'x': {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'median': 0.0, 'var': 0.0, 'std': 0.0},
                'y': {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'median': 0.0, 'var': 0.0, 'std': 0.0},
                'z': {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'median': 0.0, 'var': 0.0, 'std': 0.0},
            },
            'sparsity': 1.0
        }
    
    # 转换为稀疏矩阵
    A_sparse = csr_matrix(A, dtype=np.float32)
    
    # 初始化状态（随机小扰动）
    state = (np.random.rand(N, 3).astype(np.float32) - 0.5) * 2 * config.initial_range
    
    # 预分配稳态轨迹存储
    transient_steps = config.total_steps - config.steady_steps
    steady_trajectory = np.zeros((N, config.steady_steps, 3), dtype=np.float32)
    
    # RK4积分
    h = config.dt 
    
    # 瞬态演化（丢弃）
    for _ in range(transient_steps):
        state = _rk4_step_sparse(
            state, A_sparse, oscillator, config, h
        )
    
    for step in range(config.steady_steps):
        state = _rk4_step_sparse(
            state, A_sparse, oscillator, config, h
        )
        steady_trajectory[:, step, :] = state
    
    
    per_node_stats = []
    for comp_idx, comp_name in enumerate(['x', 'y', 'z']):
        comp_traj = steady_trajectory[:, :, comp_idx]
        
        node_mean = np.mean(comp_traj, axis=1)
        node_max = np.max(comp_traj, axis=1)
        node_min = np.min(comp_traj, axis=1)
        node_median = np.median(comp_traj, axis=1)
        node_var = np.var(comp_traj, axis=1)
        node_std = np.std(comp_traj, axis=1)
        
        # 堆叠: (n_nodes, 6)
        comp_stats = np.stack([node_mean, node_max, node_min, node_median, node_var, node_std], axis=1)
        per_node_stats.append(comp_stats)
    
    per_node_features = np.hstack(per_node_stats).astype(np.float32)
    
    # 同时保留全局统计量用于可视化
    x_traj = steady_trajectory[:, :, 0].flatten()
    y_traj = steady_trajectory[:, :, 1].flatten()
    z_traj = steady_trajectory[:, :, 2].flatten()
    
    global_stats = {
        'x': {
            'mean': float(np.mean(x_traj)), 'max': float(np.max(x_traj)),
            'min': float(np.min(x_traj)), 'median': float(np.median(x_traj)),
            'var': float(np.var(x_traj)), 'std': float(np.std(x_traj))
        },
        'y': {
            'mean': float(np.mean(y_traj)), 'max': float(np.max(y_traj)),
            'min': float(np.min(y_traj)), 'median': float(np.median(y_traj)),
            'var': float(np.var(y_traj)), 'std': float(np.std(y_traj))
        },
        'z': {
            'mean': float(np.mean(z_traj)), 'max': float(np.max(z_traj)),
            'min': float(np.min(z_traj)), 'median': float(np.median(z_traj)),
            'var': float(np.var(z_traj)), 'std': float(np.std(z_traj))
        },
    }
    
    return {
        'steady_trajectory': steady_trajectory,
        'per_node_features': per_node_features,  #
        'stats': global_stats,
        'sparsity': sparsity
    }


def simulate_rossler_sparse(
    adj_matrix: np.ndarray,
    config: RosslerConfig,
    oscillator: RosslerOscillator,
    random_seed: int = 42
) -> Dict:
    """
    稀疏矩阵优化的Rossler动力学模拟
    """
    np.random.seed(random_seed)
    N = adj_matrix.shape[0]
    
    # 处理输入矩阵
    threshold = float(config.sparsity_threshold)
    A = np.abs(adj_matrix).astype(np.float32)
    A[A < threshold] = 0
    np.fill_diagonal(A, 0)
    
    # 归一化
    adj_max = np.max(A)
    if adj_max > 1e-6:
        A /= adj_max
    
    nnz = np.count_nonzero(A)
    sparsity = 1.0 - nnz / (N * N)
    
    if nnz == 0:
        steady_trajectory = np.zeros((N, config.steady_steps, 3), dtype=np.float32)
        return {
            'steady_trajectory': steady_trajectory,
            'per_node_features': np.zeros((N, 18), dtype=np.float32),
            'stats': {
                'x': {k: 0.0 for k in ['mean', 'max', 'min', 'median', 'var', 'std']},
                'y': {k: 0.0 for k in ['mean', 'max', 'min', 'median', 'var', 'std']},
                'z': {k: 0.0 for k in ['mean', 'max', 'min', 'median', 'var', 'std']},
            },
            'sparsity': 1.0
        }
    
    A_sparse = csr_matrix(A, dtype=np.float32)
    state = (np.random.rand(N, 3).astype(np.float32) - 0.5) * 2 * config.initial_range
    
    h = config.dt
    transient_steps = config.total_steps - config.steady_steps
    for _ in range(transient_steps):
        state = _rk4_step_rossler_sparse(state, A_sparse, oscillator, config, h)
        
    steady_trajectory = np.zeros((N, config.steady_steps, 3), dtype=np.float32)
    for step in range(config.steady_steps):
        state = _rk4_step_rossler_sparse(state, A_sparse, oscillator, config, h)
        steady_trajectory[:, step, :] = state
        
    # 计算统计指标
    per_node_stats = []
    for comp_idx in range(3):
        comp_traj = steady_trajectory[:, :, comp_idx]
        node_stats = np.stack([
            np.mean(comp_traj, axis=1), np.max(comp_traj, axis=1),
            np.min(comp_traj, axis=1), np.median(comp_traj, axis=1),
            np.var(comp_traj, axis=1), np.std(comp_traj, axis=1)
        ], axis=1)
        per_node_stats.append(node_stats)
    
    per_node_features = np.hstack(per_node_stats).astype(np.float32)
    
    x_stats = {
        'mean': float(np.mean(steady_trajectory[:,:,0])),
        'max': float(np.max(steady_trajectory[:,:,0])),
        'min': float(np.min(steady_trajectory[:,:,0])),
        'median': float(np.median(steady_trajectory[:,:,0])),
        'var': float(np.var(steady_trajectory[:,:,0])),
        'std': float(np.std(steady_trajectory[:,:,0]))
    }
    
    return {
        'steady_trajectory': steady_trajectory,
        'per_node_features': per_node_features,
        'stats': {'x': x_stats},
        'sparsity': sparsity
    }


# ============================================================
# 特征提取接口
# ============================================================

def extract_lorenz_features_single_scale(
    scale_matrix: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict]:
    """
    提取单个尺度的 Lorenz per-node 特征
    
    参数
    ----
    scale_matrix : np.ndarray
        单个尺度的子图矩阵，形状 (n_samples, n_samples)
    config : Dict[str, Any]
        配置字典
        
    返回
    ----
    Tuple[np.ndarray, Dict]
        - per_node_features: 每个节点的特征 (n_samples, n_features)
          维度取决于 components 配置: ["x"] -> 6维, ["x","y"] -> 12维, ["x","y","z"] -> 18维
        - result_dict: 包含轨迹等信息的字典
    """
    lorenz_cfg = config.get('lorenz', {})
    seed = config.get('experiment', {}).get('random_seed', 42)
    
    # 构建配置对象
    lorenz_config = LorenzConfig(
        coupling_strength=lorenz_cfg.get('coupling_strength', 0.42),
        coupling_mode=lorenz_cfg.get('coupling_mode', 'xyz_all'),
        dt=lorenz_cfg.get('dt', 0.01),
        total_steps=lorenz_cfg.get('total_steps', 3000),
        steady_steps=lorenz_cfg.get('steady_steps', 1000),
        initial_range=lorenz_cfg.get('initial_range', 1.0),
        sparsity_threshold=lorenz_cfg.get('sparsity_threshold', 1e-6),
        use_periodic_boundary=lorenz_cfg.get('use_periodic_boundary', True),
        state_clip_min=lorenz_cfg.get('state_clip_min', -200.0),
        state_clip_max=lorenz_cfg.get('state_clip_max', 200.0),
        dynamics_clip_min=lorenz_cfg.get('dynamics_clip_min', -5000.0),
        dynamics_clip_max=lorenz_cfg.get('dynamics_clip_max', 5000.0),
    )
    oscillator = LorenzOscillator(
        delta=lorenz_cfg.get('delta', 10.0),
        gamma=lorenz_cfg.get('gamma', 60.0),
        beta=lorenz_cfg.get('beta', 8.0/3.0),
        rk=lorenz_cfg.get('rk', 7.0),
    )
    
    # 运行模拟
    result = simulate_lorenz_sparse(scale_matrix, lorenz_config, oscillator, seed)
    
    x_stats = result['stats']['x']
    global_x_features = np.array([
        x_stats['mean'], x_stats['max'], x_stats['min'], 
        x_stats['median'], x_stats['var'], x_stats['std']
    ], dtype=np.float32)
    
    n_nodes = scale_matrix.shape[0]
    per_node_features = np.tile(global_x_features, (n_nodes, 1))
    # --------------------------------------------------------------------------
    
    return per_node_features, result


def _process_scale_lorenz(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    """处理单个尺度的Lorenz特征（用于并行化）
    
    返回
    ----
    Tuple[int, np.ndarray, np.ndarray]
        - scale_idx: 尺度索引
        - per_node_features: 每个节点的特征 (n_nodes, n_features)
        - traj_summary: 轨迹摘要 (3,)
    """
    scale_idx, scale_matrix, config_dict, seed, trajectory_output_dir = args
    
    # 重建配置对象
    lorenz_cfg = config_dict.get('lorenz', {})
    config = LorenzConfig(
        coupling_strength=lorenz_cfg.get('coupling_strength', 0.42),
        coupling_mode=lorenz_cfg.get('coupling_mode', 'xyz_all'),
        dt=lorenz_cfg.get('dt', 0.01),
        total_steps=lorenz_cfg.get('total_steps', 3000),
        steady_steps=lorenz_cfg.get('steady_steps', 1000),
        initial_range=lorenz_cfg.get('initial_range', 1.0),
        sparsity_threshold=lorenz_cfg.get('sparsity_threshold', 1e-6),
        use_periodic_boundary=lorenz_cfg.get('use_periodic_boundary', True),
        state_clip_min=lorenz_cfg.get('state_clip_min', -200.0),
        state_clip_max=lorenz_cfg.get('state_clip_max', 200.0),
        dynamics_clip_min=lorenz_cfg.get('dynamics_clip_min', -5000.0),
        dynamics_clip_max=lorenz_cfg.get('dynamics_clip_max', 5000.0),
    )
    oscillator = LorenzOscillator(
        delta=lorenz_cfg.get('delta', 10.0),
        gamma=lorenz_cfg.get('gamma', 60.0),
        beta=lorenz_cfg.get('beta', 8.0/3.0),
        rk=lorenz_cfg.get('rk', 7.0),
    )
    
    # 获取使用的分量配置
    components = lorenz_cfg.get('components', ['x', 'y', 'z'])
    comp_indices = {'x': 0, 'y': 1, 'z': 2}
    
    try:
        result = simulate_lorenz_sparse(scale_matrix, config, oscillator, seed + scale_idx)
        
        # 检查返回的字典是否包含必需的键
        if 'per_node_features' not in result:
            print_error(f"尺度 {scale_idx}: simulate_lorenz_sparse 返回的字典缺少 'per_node_features' 键。"
                        f"返回的键: {list(result.keys())}")
            n_nodes = scale_matrix.shape[0]
            components = lorenz_cfg.get('components', ['x', 'y', 'z'])
            features_per_comp = 6
            n_features = len(components) * features_per_comp
            result['per_node_features'] = np.zeros((n_nodes, 18), dtype=np.float32)
        
        all_per_node = result['per_node_features'] 
    except Exception as e:
        print_error(f"尺度 {scale_idx} 在 simulate_lorenz_sparse 中出错: {e}")
        import traceback
        import traceback
        print_error(traceback.format_exc())
        # 创建默认返回值
        n_nodes = scale_matrix.shape[0]
        components = lorenz_cfg.get('components', ['x', 'y', 'z'])
        n_features = len(components) * 6
        all_per_node = np.zeros((n_nodes, 18), dtype=np.float32)
        result = {
            'per_node_features': all_per_node,
            'steady_trajectory': np.zeros((n_nodes, config.steady_steps, 3), dtype=np.float32)
        }
    
    try:
        _save_scale_trajectory_npy(
            output_dir=trajectory_output_dir,
            scale_idx=scale_idx,
            steady_trajectory=result.get("steady_trajectory"),
            config_dict=config_dict,
        )
    except Exception as e:
        print_warning(f"尺度 {scale_idx} 轨迹保存失败: {e}")

    x_stats = result['stats']['x']
    global_x_features = np.array([
        x_stats['mean'], x_stats['max'], x_stats['min'], 
        x_stats['median'], x_stats['var'], x_stats['std']
    ], dtype=np.float32)
    
    n_nodes = scale_matrix.shape[0]
    per_node_features = np.tile(global_x_features, (n_nodes, 1))
    # ----------------------------------------------------------------------
    
    traj_summary = np.mean(result['steady_trajectory'], axis=(0, 1))  # 形状 (3,)
    
    return scale_idx, per_node_features, traj_summary


def extract_lorenz_features(
    subgraphs: np.ndarray,
    config: Dict,
    num_workers: int = 4,
    verbose: bool = True,
    trajectory_output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    提取Lorenz动力学特征（并行优化版）
    
    参数
    ----
    subgraphs : np.ndarray
        多尺度子图数据，形状为 (num_scales, num_samples, num_samples)
    config : Dict
        配置字典
    num_workers : int
        并行工作进程数
    verbose : bool
        是否显示详细进度
        
    返回
    ----
    Tuple[np.ndarray, List[np.ndarray]]
        - 特征数组，形状为 (num_samples, num_scales * 18)
        - 轨迹摘要列表
    额外说明
    --------
    - 若配置 `lorenz.trajectory_save.enabled: true` 且提供 `trajectory_output_dir`，
      则会按尺度保存 per-sample 稳态轨迹：`scale_XX.npy`，形状为 (num_samples, T, C)。
    """
    num_scales, num_samples, _ = subgraphs.shape
    
    lorenz_cfg = config.get('lorenz', {})
    total_steps = lorenz_cfg.get('total_steps', 3000)
    steady_steps = lorenz_cfg.get('steady_steps', 1000)
    coupling = lorenz_cfg.get('coupling_strength', 0.42)
    seed = config.get('experiment', {}).get('random_seed', 42)
    
    features_per_scale = 6
    components = ['x']
    all_features = np.zeros((num_samples, num_scales * features_per_scale), dtype=np.float32)
    all_trajectories = [None] * num_scales
    
    if verbose:
        comp_str = ','.join(components)
        console.print(f"[cyan]提取Lorenz特征: {num_scales}个尺度, ε={coupling}, "
                      f"steps={total_steps} (稳态: {steady_steps}), 分量: [{comp_str}][/cyan]")
        if _trajectory_save_enabled(config) and trajectory_output_dir:
            settings = _trajectory_save_settings(config)
            console.print(
                f"[dim]轨迹保存: dir={trajectory_output_dir}, stride={settings['stride']}, "
                f"dtype={settings['dtype_str']}, components={settings['components']}[/dim]"
            )
    
    start_time = time.time()
    completed_count = 0
    
    # 进度条配置
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]Lorenz"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} 尺度"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    
    if num_workers > 1 and num_scales > 1:
        # 准备任务参数
        tasks = [
            (scale_idx, subgraphs[scale_idx], config, seed, trajectory_output_dir)
            for scale_idx in range(num_scales)
        ]
        
        with Progress(*progress_columns, console=console) as progress:
            task_id = progress.add_task("处理中", total=num_scales)
            
            with ProcessPoolExecutor(max_workers=min(num_workers, num_scales)) as executor:
                future_to_scale = {
                    executor.submit(_process_scale_lorenz, task): task[0]
                    for task in tasks
                }
                
                for future in as_completed(future_to_scale):
                    scale_idx = future_to_scale[future]
                    try:
                        scale_idx_result, features, traj_summary = future.result()
                        
                        start_idx = scale_idx_result * features_per_scale
                        all_features[:, start_idx:start_idx+features_per_scale] = features
                        all_trajectories[scale_idx_result] = traj_summary
                        
                        completed_count += 1
                        progress.update(task_id, completed=completed_count)
                    except Exception as e:
                        print_error(f"尺度 {scale_idx} 处理失败: {e}")
                        start_idx = scale_idx * features_per_scale
                        all_features[:, start_idx:start_idx+features_per_scale] = 0.0
                        all_trajectories[scale_idx] = np.zeros(3, dtype=np.float32)
                        completed_count += 1
                        progress.update(task_id, completed=completed_count)
    else:
        lorenz_config = LorenzConfig(
            coupling_strength=lorenz_cfg.get('coupling_strength', 0.42),
            dt=lorenz_cfg.get('dt', 0.01),
            total_steps=lorenz_cfg.get('total_steps', 3000),
            steady_steps=lorenz_cfg.get('steady_steps', 1000),
            initial_range=lorenz_cfg.get('initial_range', 1.0),
            sparsity_threshold=lorenz_cfg.get('sparsity_threshold', 1e-6),
        )
        oscillator = LorenzOscillator(
            delta=lorenz_cfg.get('delta', 10.0),
            gamma=lorenz_cfg.get('gamma', 60.0),
            beta=lorenz_cfg.get('beta', 8.0/3.0),
            rk=lorenz_cfg.get('rk', 7.0),
        )
        
        with Progress(*progress_columns, console=console) as progress:
            task_id = progress.add_task("处理中", total=num_scales)
            
            for scale_idx in range(num_scales):
                result = simulate_lorenz_sparse(
                    subgraphs[scale_idx], lorenz_config, oscillator, seed + scale_idx
                )

                try:
                    _save_scale_trajectory_npy(
                        output_dir=trajectory_output_dir,
                        scale_idx=scale_idx,
                        steady_trajectory=result.get("steady_trajectory"),
                        config_dict=config,
                    )
                except Exception as e:
                    print_warning(f"尺度 {scale_idx} 轨迹保存失败: {e}")
                
                x_stats = result['stats']['x']
                global_x_features = np.array([
                    x_stats['mean'], x_stats['max'], x_stats['min'], 
                    x_stats['median'], x_stats['var'], x_stats['std']
                ], dtype=np.float32)
                
                start_idx = scale_idx * features_per_scale
                all_features[:, start_idx:start_idx+features_per_scale] = global_x_features
                all_trajectories[scale_idx] = np.mean(result['steady_trajectory'], axis=(0, 1))
                
                progress.update(task_id, completed=scale_idx + 1)
    
    total_elapsed = time.time() - start_time
    if verbose:
        console.print(f"[green]✓ Lorenz特征提取完成，总耗时: {total_elapsed:.2f}s[/green]")
    logger(f"✓ Lorenz特征提取完成: {all_features.shape}, 总耗时: {total_elapsed:.2f}s")

    if _trajectory_save_enabled(config) and trajectory_output_dir:
        try:
            out_p = Path(trajectory_output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            settings = _trajectory_save_settings(config)
            stride = int(settings["stride"])
            t_len = int((steady_steps + stride - 1) // stride)
            c_len = len([c for c in settings["components"] if c in ("x", "y", "z")]) or 1

            files = []
            for s in range(num_scales):
                p = out_p / f"scale_{s:02d}.npy"
                if p.exists():
                    files.append({"scale_idx": int(s), "path": str(p)})
            meta = {
                "format": "lorenz_steady_trajectory_v1",
                "trajectory_dir": str(out_p),
                "num_scales": int(num_scales),
                "num_samples": int(num_samples),
                "steady_steps": int(steady_steps),
                "stride": int(stride),
                "T": int(t_len),
                "C": int(c_len),
                "dtype": str(settings["dtype_str"]),
                "components": settings["components"],
                "files": files,
            }
            with open(out_p / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger(f"写入轨迹 meta.json 失败: {e}")
    
    return all_features, all_trajectories


# ============================================================
# 可视化函数
# ============================================================

def visualize_lorenz_trajectory(
    subgraph: np.ndarray,
    config: Dict,
    output_path: str,
    scale_idx: int = 0,
    random_seed: int = 42
) -> str:
    """
    生成Lorenz x-z相图可视化（蝴蝶振荡）
    
    参数
    ----
    subgraph : np.ndarray
        单个尺度的子图矩阵
    config : Dict
        配置字典
    output_path : str
        输出路径
    scale_idx : int
        尺度索引
    random_seed : int
        随机种子
        
    返回
    ----
    str
        保存的图像路径
    """
    import matplotlib.pyplot as plt
    
    lorenz_cfg = config.get('lorenz', {})
    vis_cfg = lorenz_cfg.get('visualization', {})
    
    # 获取可视化参数
    sample_interval = vis_cfg.get('sample_interval', 10)
    figsize = tuple(vis_cfg.get('figsize', [12, 10]))
    
    # 创建配置
    lorenz_config = LorenzConfig(
        coupling_strength=lorenz_cfg.get('coupling_strength', 0.42),
        coupling_mode=lorenz_cfg.get('coupling_mode', 'xyz_all'),
        dt=lorenz_cfg.get('dt', 0.01),
        total_steps=lorenz_cfg.get('total_steps', 3000),
        steady_steps=lorenz_cfg.get('steady_steps', 1000),
        initial_range=lorenz_cfg.get('initial_range', 1.0),
        sparsity_threshold=lorenz_cfg.get('sparsity_threshold', 1e-6),
        use_periodic_boundary=lorenz_cfg.get('use_periodic_boundary', True),
        state_clip_min=lorenz_cfg.get('state_clip_min', -200.0),
        state_clip_max=lorenz_cfg.get('state_clip_max', 200.0),
        dynamics_clip_min=lorenz_cfg.get('dynamics_clip_min', -5000.0),
        dynamics_clip_max=lorenz_cfg.get('dynamics_clip_max', 5000.0),
    )
    oscillator = LorenzOscillator(
        delta=lorenz_cfg.get('delta', 10.0),
        gamma=lorenz_cfg.get('gamma', 60.0),
        beta=lorenz_cfg.get('beta', 8.0/3.0),
        rk=lorenz_cfg.get('rk', 7.0),
    )
    
    # 运行模拟
    result = simulate_lorenz_sparse(subgraph, lorenz_config, oscillator, random_seed)
    trajectory = result['steady_trajectory']  
    
    mean_traj = np.mean(trajectory, axis=0)  
    x = mean_traj[::sample_interval, 0]
    z = mean_traj[::sample_interval, 2]
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用渐变色显示时间演化
    colors = np.linspace(0, 1, len(x))
    scatter = ax.scatter(x, z, c=colors, cmap='viridis', s=5, alpha=0.7)
    ax.plot(x, z, 'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title(f'Lorenz Attractor (Scale {scale_idx}) - x-z Phase Plot\n'
                 f'ε={lorenz_config.coupling_strength}, γ={oscillator.gamma}', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (normalized)', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def visualize_lorenz_multi_scale(
    subgraphs: np.ndarray,
    config: Dict,
    output_path: str,
    random_seed: int = 42
) -> str:
    """
    生成多尺度Lorenz x-z相图可视化
    
    参数
    ----
    subgraphs : np.ndarray
        多尺度子图数据，形状为 (num_scales, num_samples, num_samples)
    config : Dict
        配置字典
    output_path : str
        输出路径
    random_seed : int
        随机种子
        
    返回
    ----
    str
        保存的图像路径
    """
    import matplotlib.pyplot as plt
    
    lorenz_cfg = config.get('lorenz', {})
    vis_cfg = lorenz_cfg.get('visualization', {})
    
    if not vis_cfg.get('enabled', True):
        return None
    
    # 获取可视化参数
    sample_interval = vis_cfg.get('sample_interval', 10)
    scales_to_show = vis_cfg.get('scales_to_show', None)
    figsize = tuple(vis_cfg.get('figsize', [12, 10]))
    
    num_scales = subgraphs.shape[0]
    
    # 确定要显示的尺度
    if scales_to_show is None:
        scales_to_show = list(range(min(num_scales, 6)))  
    else:
        scales_to_show = [s for s in scales_to_show if s < num_scales]
    
    n_plots = len(scales_to_show)
    if n_plots == 0:
        return None
    
    # 计算子图布局
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows / 2))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 创建配置
    lorenz_config = LorenzConfig(
        coupling_strength=lorenz_cfg.get('coupling_strength', 0.42),
        coupling_mode=lorenz_cfg.get('coupling_mode', 'xyz_all'),
        dt=lorenz_cfg.get('dt', 0.01),
        total_steps=lorenz_cfg.get('total_steps', 3000),
        steady_steps=lorenz_cfg.get('steady_steps', 1000),
        initial_range=lorenz_cfg.get('initial_range', 1.0),
        sparsity_threshold=lorenz_cfg.get('sparsity_threshold', 1e-6),
        use_periodic_boundary=lorenz_cfg.get('use_periodic_boundary', True),
        state_clip_min=lorenz_cfg.get('state_clip_min', -200.0),
        state_clip_max=lorenz_cfg.get('state_clip_max', 200.0),
        dynamics_clip_min=lorenz_cfg.get('dynamics_clip_min', -5000.0),
        dynamics_clip_max=lorenz_cfg.get('dynamics_clip_max', 5000.0),
    )
    oscillator = LorenzOscillator(
        delta=lorenz_cfg.get('delta', 10.0),
        gamma=lorenz_cfg.get('gamma', 60.0),
        beta=lorenz_cfg.get('beta', 8.0/3.0),
        rk=lorenz_cfg.get('rk', 7.0),
    )
    
    for idx, scale_idx in enumerate(scales_to_show):
        ax = axes[idx]
        
        # 运行模拟
        result = simulate_lorenz_sparse(
            subgraphs[scale_idx], lorenz_config, oscillator, random_seed + scale_idx
        )
        trajectory = result['steady_trajectory'] 
        
        mean_traj = np.mean(trajectory, axis=0)  
        x = mean_traj[::sample_interval, 0]
        z = mean_traj[::sample_interval, 2]
        
        # 使用渐变色显示时间演化
        colors = np.linspace(0, 1, len(x))
        scatter = ax.scatter(x, z, c=colors, cmap='viridis', s=3, alpha=0.7)
        ax.plot(x, z, 'b-', alpha=0.2, linewidth=0.3)
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('z', fontsize=10)
        ax.set_title(f'Scale {scale_idx}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Lorenz Attractor x-z Phase Plots\n'
                 f'ε={lorenz_config.coupling_strength}, γ={oscillator.gamma}, '
                 f'sample_interval={sample_interval}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger(f"Lorenz x-z phase plots saved: {output_path}")
    return output_path


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    console.print("[yellow]运行Lorenz测试...[/yellow]")
    
    np.random.seed(42)
    # 创建测试数据：3个尺度，50个节点的稀疏对称矩阵
    test_data = np.zeros((3, 50, 50), dtype=np.float32)
    for i in range(3):
        n_edges = 100  # 稀疏度 ~4%
        rows = np.random.randint(0, 50, n_edges)
        cols = np.random.randint(0, 50, n_edges)
        test_data[i, rows, cols] = np.random.rand(n_edges)
        test_data[i] = (test_data[i] + test_data[i].T) / 2
    
    config = {
        'experiment': {'random_seed': 42},
        'lorenz': {
            'coupling_strength': 0.42,
            'dt': 0.01,
            'total_steps': 300,  # 快速测试
            'steady_steps': 100,
            'alpha': 10.0,
            'beta': 8.0/3.0,
            'gamma': 28.0,
            'components': ['x'],
            'visualization': {
                'enabled': True,
                'sample_interval': 5,
                'scales_to_show': [0, 1, 2],
            }
        }
    }
    
    features, trajs = extract_lorenz_features(test_data, config, num_workers=1, verbose=True)
    console.print(f"[green]✓ 完成! 特征形状: {features.shape}[/green]")
    
    # 测试可视化
    vis_path = visualize_lorenz_multi_scale(test_data, config, '/tmp/lorenz_test.png')
    if vis_path:
        console.print(f"[green]✓ 可视化保存: {vis_path}[/green]")

