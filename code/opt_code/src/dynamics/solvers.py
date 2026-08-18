"""
数值求解器模块
==============

提供耦合振荡器系统的数值积分方法。

使用示例
--------
>>> from src.dynamics.solvers import EulerSolver, RK4Solver
>>> 
>>> solver = EulerSolver(config)
>>> trajectories = solver.solve(L, oscillator, initial_state)
"""

import numpy as np
from scipy import sparse
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .oscillators import LorenzOscillator
from ..utils.logger import get_logger, console


logger = get_logger(__name__)


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
    if sparse.issparse(L):
        L = L.toarray()
    
    A = -L.copy()
    np.fill_diagonal(A, 0.0)
    
    # 确保非对角元素非负（邻接矩阵权重应为非负）
    A = np.maximum(A, 0.0)
    
    return A


@dataclass
class SolverConfig:
    """求解器配置"""
    time_step: float = 1e-3
    coupling_strength: float = 0.42
    coupling_mode: str = "x_only"
    use_periodic_boundary: bool = True
    rk: float = 7.0
    state_clip_min: float = -200.0
    state_clip_max: float = 200.0
    dynamics_clip_min: float = -5000.0
    dynamics_clip_max: float = 5000.0


class BaseSolver:
    """
    求解器基类
    
    定义耦合振荡器系统的通用接口。
    """
    
    def __init__(self, config: SolverConfig = None):
        """
        初始化求解器
        
        参数
        ----
        config : SolverConfig
            求解器配置
        """
        self.config = config or SolverConfig()
    
    def coupled_dynamics(
        self,
        t: float,
        state: np.ndarray,
        L: np.ndarray,
        oscillator: LorenzOscillator,
    ) -> np.ndarray:
        """
        计算耦合振荡器系统的动力学
        
        参数
        ----
        t : float
            当前时间
        state : np.ndarray
            状态向量，形状为 (n_nodes * 3,)
        L : np.ndarray
            拉普拉斯矩阵
        oscillator : LorenzOscillator
            振荡器对象
            
        返回
        ----
        np.ndarray
            导数向量
        """
        n_nodes = L.shape[0]
        state_matrix = state.reshape(n_nodes, 3)
        
        # 限制状态值
        np.clip(
            state_matrix,
            self.config.state_clip_min,
            self.config.state_clip_max,
            out=state_matrix
        )
        
        # 计算局部动力学（向量化）
        x = state_matrix[:, 0]
        y = state_matrix[:, 1]
        z = state_matrix[:, 2]
        
        local_dynamics = np.empty_like(state_matrix)
        local_dynamics[:, 0] = oscillator.alpha * (y - x)
        local_dynamics[:, 1] = x * (oscillator.gamma - z) - y
        local_dynamics[:, 2] = x * y - oscillator.beta * z
        
        # 检查局部动力学是否包含 NaN 或 Inf（与data_processed_code保持一致）
        if np.any(np.isnan(local_dynamics)) or np.any(np.isinf(local_dynamics)):
            logger.warning(f"警告: 局部动力学包含 NaN 或 Inf，时间 t={t}")
            # 第一次检查时使用硬编码值（与老版保持一致）
            local_dynamics = np.nan_to_num(
                local_dynamics, 
                nan=0.0, 
                posinf=1000, 
                neginf=-1000
            )
        
        # 限制局部动力学
        np.clip(
            local_dynamics,
            self.config.dynamics_clip_min,
            self.config.dynamics_clip_max,
            out=local_dynamics
        )
        
        # 计算耦合项
        epsilon = self.config.coupling_strength
        
        if self.config.coupling_mode == "x_only":
            # 方式1：全局网络x方向耦合
            # 只对x分量耦合，使用拉普拉斯矩阵
            coupling_x = -L.dot(x) if sparse.issparse(L) else -np.dot(L, x)
            local_dynamics[:, 1] += epsilon * coupling_x
            
        elif self.config.coupling_mode == "xyz_all":
            # 方式2：3个分量的直积单位矩阵耦合
            # 使用邻接矩阵 A 的直积形式：ε * (A ⊗ I_3) @ u
            # 即：ε * Σ_j A[i,j] * uu[j][k]
            A = _laplacian_to_adjacency(L)
            if sparse.issparse(L):
                # 如果 L 是稀疏矩阵，A 也是稀疏的
                A_sparse = sparse.csr_matrix(A) if not sparse.issparse(A) else A
                coupling = A_sparse.dot(state_matrix)
            else:
                coupling = np.dot(A, state_matrix)
            
            # 检查耦合项是否包含 NaN 或 Inf（与data_processed_code保持一致）
            if np.any(np.isnan(coupling)) or np.any(np.isinf(coupling)):
                logger.warning(f"警告: 耦合项包含 NaN 或 Inf，时间 t={t}")
                coupling = np.nan_to_num(
                    coupling,
                    nan=0.0,
                    posinf=self.config.dynamics_clip_max,
                    neginf=self.config.dynamics_clip_min
                )
            
            # 限制耦合项范围
            np.clip(
                coupling,
                self.config.dynamics_clip_min,
                self.config.dynamics_clip_max,
                out=coupling
            )
            local_dynamics += epsilon * coupling
            
        elif self.config.coupling_mode == "nearest_neighbor":
            # 方式3：最近邻扩散耦合
            # 公式：ε * (u_{i-1} - 2u_i + u_{i+1})
            # 使用周期性边界条件
            n_nodes = state_matrix.shape[0]
            
            # 使用 np.roll 实现周期性边界
            u_prev = np.roll(state_matrix, 1, axis=0)  # u_{i-1}
            u_next = np.roll(state_matrix, -1, axis=0)  # u_{i+1}
            
            # 计算耦合项：u_{i-1} - 2u_i + u_{i+1}
            coupling = u_prev - 2.0 * state_matrix + u_next
            
            # 检查耦合项是否包含 NaN 或 Inf
            if np.any(np.isnan(coupling)) or np.any(np.isinf(coupling)):
                logger.warning(f"警告: 耦合项包含 NaN 或 Inf，时间 t={t}")
                coupling = np.nan_to_num(
                    coupling,
                    nan=0.0,
                    posinf=self.config.dynamics_clip_max,
                    neginf=self.config.dynamics_clip_min
                )
            
            # 限制耦合项范围
            np.clip(
                coupling,
                self.config.dynamics_clip_min,
                self.config.dynamics_clip_max,
                out=coupling
            )
            local_dynamics += epsilon * coupling
            
        else:
            # 默认：使用拉普拉斯矩阵（向后兼容）
            coupling = L.dot(state_matrix) if sparse.issparse(L) else np.dot(L, state_matrix)
            
            # 检查耦合项是否包含 NaN 或 Inf（与data_processed_code保持一致）
            if np.any(np.isnan(coupling)) or np.any(np.isinf(coupling)):
                logger.warning(f"警告: 耦合项包含 NaN 或 Inf，时间 t={t}")
                coupling = np.nan_to_num(
                    coupling,
                    nan=0.0,
                    posinf=self.config.dynamics_clip_max,
                    neginf=self.config.dynamics_clip_min
                )
            
            # 限制耦合项范围
            np.clip(
                coupling,
                self.config.dynamics_clip_min,
                self.config.dynamics_clip_max,
                out=coupling
            )
            local_dynamics += epsilon * coupling
        
        # 最终检查并限制范围（与data_processed_code保持一致）
        if np.any(np.isnan(local_dynamics)) or np.any(np.isinf(local_dynamics)):
            local_dynamics = np.nan_to_num(
                local_dynamics,
                nan=0.0,
                posinf=self.config.dynamics_clip_max,
                neginf=self.config.dynamics_clip_min
            )
        
        np.clip(
            local_dynamics,
            self.config.dynamics_clip_min,
            self.config.dynamics_clip_max,
            out=local_dynamics
        )
        
        return local_dynamics.flatten()
    
    def solve(
        self,
        L: np.ndarray,
        oscillator: LorenzOscillator,
        initial_state: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        求解耦合系统
        
        参数
        ----
        L : np.ndarray
            拉普拉斯矩阵
        oscillator : LorenzOscillator
            振荡器对象
        initial_state : np.ndarray
            初始状态
        n_steps : int
            时间步数
            
        返回
        ----
        np.ndarray
            轨迹数据
        """
        raise NotImplementedError


class EulerSolver(BaseSolver):
    """
    Euler方法求解器
    
    使用前向Euler方法求解耦合振荡器系统。
    """
    
    def solve(
        self,
        L: np.ndarray,
        oscillator: LorenzOscillator,
        initial_state: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        使用Euler方法求解
        
        返回
        ----
        np.ndarray
            轨迹数据，形状为 (n_nodes, n_steps+1, 3)
        """
        n_nodes = L.shape[0]
        h = self.config.time_step
        
        # 转换为稀疏矩阵
        if not sparse.issparse(L):
            L = sparse.csr_matrix(L)
        
        # 初始化
        n_vars = initial_state.size
        y = initial_state.flatten().copy()
        
        # 存储轨迹
        trajectory = np.zeros((n_steps + 1, n_vars))
        trajectory[0] = y
        
        # 周期性边界条件
        use_periodic = self.config.use_periodic_boundary
        
        # Euler迭代
        for step in range(n_steps):
            if use_periodic:
                y = self._euler_step_periodic(y, h, L, oscillator, n_nodes)
            else:
                f = self.coupled_dynamics(step * h, y, L, oscillator)
                y = y + h * f
            
            trajectory[step + 1] = y
        
        # 重塑为 (n_nodes, n_steps+1, 3)
        return trajectory.reshape(n_steps + 1, n_nodes, 3).transpose(1, 0, 2)
    
    def _euler_step_periodic(
        self,
        y: np.ndarray,
        h: float,
        L: np.ndarray,
        oscillator: LorenzOscillator,
        n_nodes: int,
    ) -> np.ndarray:
        """带周期性边界条件的Euler步"""
        state_matrix = y.reshape(n_nodes, 3)
        x = state_matrix[:, 0]
        y_state = state_matrix[:, 1]
        z = state_matrix[:, 2]
        
        # 耦合项
        epsilon = self.config.coupling_strength
        couple_y = -epsilon * L.dot(x)
        
        # 周期性边界
        x_next = np.roll(x, -1)
        x_prev = np.roll(x, 1)
        cc1 = self.config.rk * (x_next - x_prev)
        
        # Euler更新
        dx = x + h * (oscillator.alpha * (y_state - x))
        dy = y_state + h * (oscillator.gamma * x - y_state - x * z + couple_y + cc1)
        dz = z + h * (x * y_state - oscillator.beta * z)
        
        return np.column_stack([dx, dy, dz]).flatten()


class RK4Solver(BaseSolver):
    """
    4阶Runge-Kutta方法求解器
    
    使用经典RK4方法求解耦合振荡器系统。
    """
    
    def solve(
        self,
        L: np.ndarray,
        oscillator: LorenzOscillator,
        initial_state: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        使用RK4方法求解
        
        返回
        ----
        np.ndarray
            轨迹数据，形状为 (n_nodes, n_steps+1, 3)
        """
        n_nodes = L.shape[0]
        h = self.config.time_step
        
        # 转换为稀疏矩阵
        if not sparse.issparse(L):
            L = sparse.csr_matrix(L)
        
        # 初始化
        n_vars = initial_state.size
        y = initial_state.flatten().copy()
        
        # 存储轨迹
        trajectory = np.zeros((n_steps + 1, n_vars))
        trajectory[0] = y
        
        # RK4迭代
        for step in range(n_steps):
            t = step * h
            y = self._rk4_step(t, y, h, L, oscillator)
            trajectory[step + 1] = y
        
        # 重塑为 (n_nodes, n_steps+1, 3)
        return trajectory.reshape(n_steps + 1, n_nodes, 3).transpose(1, 0, 2)
    
    def _rk4_step(
        self,
        t: float,
        y: np.ndarray,
        h: float,
        L: np.ndarray,
        oscillator: LorenzOscillator,
    ) -> np.ndarray:
        """RK4单步"""
        k1 = self.coupled_dynamics(t, y, L, oscillator)
        k2 = self.coupled_dynamics(t + h/2, y + h*k1/2, L, oscillator)
        k3 = self.coupled_dynamics(t + h/2, y + h*k2/2, L, oscillator)
        k4 = self.coupled_dynamics(t + h, y + h*k3, L, oscillator)
        
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def create_solver(config=None) -> BaseSolver:
    """
    创建求解器实例
    
    参数
    ----
    config : Config or SolverConfig
        配置对象
        
    返回
    ----
    BaseSolver
        求解器实例
    """
    # 提取求解器配置
    if hasattr(config, 'dynamics'):
        dynamics = config.dynamics
        solver_config = SolverConfig(
            time_step=dynamics.numerical.time_step,
            coupling_strength=dynamics.numerical.coupling_strength,
            coupling_mode=dynamics.numerical.coupling_mode,
            use_periodic_boundary=dynamics.numerical.use_periodic_boundary,
            rk=dynamics.oscillator.rk,
            state_clip_min=dynamics.stability.state_clip_min,
            state_clip_max=dynamics.stability.state_clip_max,
            dynamics_clip_min=dynamics.stability.dynamics_clip_min,
            dynamics_clip_max=dynamics.stability.dynamics_clip_max,
        )
        method = dynamics.numerical.method
    elif isinstance(config, SolverConfig):
        solver_config = config
        method = "Euler"
    else:
        solver_config = SolverConfig()
        method = "Euler"
    
    # 创建求解器
    if method.lower() == "euler":
        return EulerSolver(solver_config)
    elif method.lower() == "rk4":
        return RK4Solver(solver_config)
    else:
        raise ValueError(f"未知的求解方法: {method}")


# ============================================================================
# 与老版完全一致的求解函数（用于保持结果一致性）
# ============================================================================

def _rk4_step_legacy(t, y, h, L, oscillator, epsilon, coupling_mode="xyz_all"):
    """
    固定步长 RK4 方法的一步（与老版完全一致）
    """
    # 创建临时求解器来调用coupled_dynamics
    temp_solver = BaseSolver(SolverConfig(
        coupling_strength=epsilon,
        coupling_mode=coupling_mode
    ))
    k1 = temp_solver.coupled_dynamics(t, y, L, oscillator)
    k2 = temp_solver.coupled_dynamics(t + h/2, y + h*k1/2, L, oscillator)
    k3 = temp_solver.coupled_dynamics(t + h/2, y + h*k2/2, L, oscillator)
    k4 = temp_solver.coupled_dynamics(t + h, y + h*k3, L, oscillator)
    
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def _rk4_solve_legacy(L, oscillator, epsilon, t_span, y0, t_eval, h, coupling_mode="xyz_all",
                      state_clip_min=-200.0, state_clip_max=200.0,
                      dynamics_clip_min=-5000.0, dynamics_clip_max=5000.0):
    """
    固定步长 RK4 求解器（与老版完全一致）
    """
    t0 = t_span[0]
    n_eval = len(t_eval)
    n_vars = len(y0)
    
    # 初始化解数组
    y_sol = np.zeros((n_eval, n_vars))
    y_sol[0] = y0
    
    # 当前时间和状态
    y_current = y0.copy()
    
    # 使用固定步长 h 进行 RK4 迭代
    t_current = t0
    eval_idx = 1  # 下一个需要保存结果的时间点索引
    
    # 创建临时求解器
    temp_solver = BaseSolver(SolverConfig(
        coupling_strength=epsilon,
        coupling_mode=coupling_mode
    ))
    
    # RK4 迭代求解，使用固定步长 h
    while eval_idx < n_eval:
        # 计算从当前时间到下一个评估点的时间步长
        dt = min(h, t_eval[eval_idx] - t_current)
        
        # 如果 dt 太小（几乎为0），说明已经到达评估点
        if dt < 1e-10:
            y_sol[eval_idx] = y_current
            eval_idx += 1
            continue
        
        # RK4 一步
        k1 = temp_solver.coupled_dynamics(t_current, y_current, L, oscillator)
        k2 = temp_solver.coupled_dynamics(t_current + dt/2, y_current + dt*k1/2, L, oscillator)
        k3 = temp_solver.coupled_dynamics(t_current + dt/2, y_current + dt*k2/2, L, oscillator)
        k4 = temp_solver.coupled_dynamics(t_current + dt, y_current + dt*k3, L, oscillator)
        y_current = y_current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        t_current += dt
        
        # 如果已经到达或超过下一个评估点，保存结果
        if t_current >= t_eval[eval_idx] or abs(t_current - t_eval[eval_idx]) < h/100:
            y_sol[eval_idx] = y_current
            eval_idx += 1
    
    # 转换为与 solve_ivp 相同的格式（转置）
    return {'t': t_eval, 'y': y_sol.T, 'success': True}


def _euler_step_with_periodic_legacy(t, y, h, L, oscillator, epsilon, config):
    """
    带周期性边界条件的 Euler 一步（与老版完全一致）
    """
    n_nodes = L.shape[0]
    state_matrix = y.reshape(n_nodes, 3)
    x = state_matrix[:, 0]
    y_state = state_matrix[:, 1]  # 重命名避免与 y 冲突
    z = state_matrix[:, 2]
    
    # 1. 计算耦合项 (向量化)
    couple_y = - epsilon * L.dot(x)
    
    # 2. 周期性边界 (向量化)
    x_next = np.roll(x, -1)  # x[i+1]
    x_prev = np.roll(x, 1)   # x[i-1]
    cc1 = config.rk * (x_next - x_prev)
    
    # 3. Euler 更新 (向量化)
    dx = x + h * (oscillator.alpha * (y_state - x))
    dy = y_state + h * (oscillator.gamma * x - y_state - x * z + couple_y + cc1)
    dz = z + h * (x * y_state - oscillator.beta * z)
    
    # 重组状态
    new_state = np.column_stack((dx, dy, dz))
    
    return new_state.flatten()


def _euler_solve_legacy(L, oscillator, epsilon, t_span, y0, t_eval, h, config,
                       state_clip_min=-200.0, state_clip_max=200.0,
                       dynamics_clip_min=-5000.0, dynamics_clip_max=5000.0):
    """
    固定步长 Euler 求解器（与老版完全一致）
    """
    t0 = t_span[0]
    n_eval = len(t_eval)
    n_vars = len(y0)
    
    # 初始化解数组
    y_sol = np.zeros((n_eval, n_vars))
    y_sol[0] = y0
    
    # 当前时间和状态
    y_current = y0.copy()
    
    # 使用固定步长 h 进行 Euler 迭代
    t_current = t0
    eval_idx = 1
    
    # 创建临时求解器（使用传入的clip值）
    temp_solver = BaseSolver(SolverConfig(
        coupling_strength=epsilon,
        coupling_mode=getattr(config, 'coupling_mode', 'xyz_all'),
        state_clip_min=state_clip_min,
        state_clip_max=state_clip_max,
        dynamics_clip_min=dynamics_clip_min,
        dynamics_clip_max=dynamics_clip_max
    ))
    
    # Euler 迭代求解
    while eval_idx < n_eval:
        # 计算从当前时间到下一个评估点的时间步长
        dt = min(h, t_eval[eval_idx] - t_current)
        
        if dt < 1e-10:
            y_sol[eval_idx] = y_current
            eval_idx += 1
            continue
        
        # Euler 一步：y_{n+1} = y_n + h * f(t_n, y_n)
        coupling_mode = getattr(config, 'coupling_mode', 'xyz_all')
        if hasattr(config, 'use_periodic_boundary') and config.use_periodic_boundary:
            # 使用周期性边界条件
            y_next = _euler_step_with_periodic_legacy(t_current, y_current, dt, L, oscillator, epsilon, config)
        else:
            # 标准 Euler 方法
            f = temp_solver.coupled_dynamics(t_current, y_current, L, oscillator)
            y_next = y_current + dt * f
        
        y_current = y_next
        t_current += dt
        
        # 保存结果
        if t_current >= t_eval[eval_idx] or abs(t_current - t_eval[eval_idx]) < h/100:
            y_sol[eval_idx] = y_current
            eval_idx += 1
    
    return {'t': t_eval, 'y': y_sol.T, 'success': True}

