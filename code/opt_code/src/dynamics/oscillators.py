"""
振荡器模块
==========

定义混沌振荡器系统。

使用示例
--------
>>> from src.dynamics.oscillators import LorenzOscillator
>>> 
>>> oscillator = LorenzOscillator(alpha=10, beta=8/3, gamma=60)
>>> state = [1, 1, 1]
>>> derivatives = oscillator.equations(0, state)
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class LorenzOscillator:
    """
    Lorenz混沌振荡器
    
    实现经典的Lorenz系统：
    dx/dt = α(y - x)
    dy/dt = x(γ - z) - y
    dz/dt = xy - βz
    
    属性
    ----
    alpha : float
        σ参数，默认10
    beta : float
        β参数，默认8/3
    gamma : float
        ρ参数，默认60
    rk : float
        周期性边界耦合系数
        
    使用示例
    --------
    >>> oscillator = LorenzOscillator(alpha=10, beta=8/3, gamma=60)
    >>> state = np.array([1.0, 1.0, 1.0])
    >>> dstate = oscillator.equations(0, state)
    """
    
    alpha: float = 10.0
    beta: float = 8.0/3.0  # 与老版保持一致，使用浮点数除法而不是近似值
    gamma: float = 60  # 固定参数60
    rk: float = 7.0
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        计算Lorenz方程的导数
        
        参数
        ----
        t : float
            时间（未使用，保留用于ODE求解器接口）
        state : np.ndarray
            状态向量 [x, y, z]
            
        返回
        ----
        np.ndarray
            导数向量 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state[0], state[1], state[2]
        
        dxdt = self.alpha * (y - x)
        dydt = x * (self.gamma - z) - y
        dzdt = x * y - self.beta * z
        
        return np.array([dxdt, dydt, dzdt])
    
    def equations_vectorized(
        self,
        t: float,
        state_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        向量化的Lorenz方程
        
        参数
        ----
        t : float
            时间
        state_matrix : np.ndarray
            状态矩阵，形状为 (n_nodes, 3)
            
        返回
        ----
        np.ndarray
            导数矩阵，形状为 (n_nodes, 3)
        """
        x = state_matrix[:, 0]
        y = state_matrix[:, 1]
        z = state_matrix[:, 2]
        
        dxdt = self.alpha * (y - x)
        dydt = x * (self.gamma - z) - y
        dzdt = x * y - self.beta * z
        
        return np.column_stack([dxdt, dydt, dzdt])


@dataclass
class RosslerOscillator:
    """
    Rössler混沌振荡器
    
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)
    """
    
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """计算Rössler方程的导数"""
        x, y, z = state[0], state[1], state[2]
        
        dxdt = -y - z
        dydt = x + self.a * y
        dzdt = self.b + z * (x - self.c)
        
        return np.array([dxdt, dydt, dzdt])
    
    def equations_vectorized(
        self,
        t: float,
        state_matrix: np.ndarray,
    ) -> np.ndarray:
        """向量化的Rössler方程"""
        x = state_matrix[:, 0]
        y = state_matrix[:, 1]
        z = state_matrix[:, 2]
        
        dxdt = -y - z
        dydt = x + self.a * y
        dzdt = self.b + z * (x - self.c)
        
        return np.column_stack([dxdt, dydt, dzdt])


def generate_initial_conditions(
    n_nodes: int,
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Tuple[float, float] = (-10, 10),
    z_range: Tuple[float, float] = (0, 50),
    seed: int = None,
) -> np.ndarray:
    """
    生成随机初始条件
    
    参数
    ----
    n_nodes : int
        节点数量
    x_range : Tuple[float, float]
        x坐标范围
    y_range : Tuple[float, float]
        y坐标范围
    z_range : Tuple[float, float]
        z坐标范围
    seed : int, optional
        随机种子
        
    返回
    ----
    np.ndarray
        初始条件，形状为 (n_nodes, 3)
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        # 为了和旧版保持一致，显式调用无参数的 seed()
        # 这会重置随机状态，这在多进程环境中可能导致不同的行为
        # 但旧版确实这样做了
        np.random.seed()
    
    x0 = np.random.uniform(x_range[0], x_range[1], n_nodes)
    y0 = np.random.uniform(y_range[0], y_range[1], n_nodes)
    z0 = np.random.uniform(z_range[0], z_range[1], n_nodes)
    
    return np.column_stack([x0, y0, z0])

