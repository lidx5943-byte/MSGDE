"""
多尺度拉普拉斯模块
==================

提供基于图拉普拉斯矩阵的多尺度特征提取。

使用示例
--------
>>> from src.laplacian.multiscale import LaplacianPipeline
>>> 
>>> pipeline = LaplacianPipeline(config)
>>> laplacian_family = pipeline.run(correlation_matrix)
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from ..utils.logger import (
    get_logger, console, create_progress,
    print_header, print_step, print_success, print_table
)
from ..utils.timer import Timer
from ..utils.io import save_numpy, ensure_dir


logger = get_logger(__name__)


def compute_adjacency_matrix(
    correlation_matrix: np.ndarray,
    sigma: float = 3.0,
    k: int = 1,
) -> np.ndarray:
    """
    从相关性矩阵计算邻接矩阵
    
    基于空间距离构建加权邻接矩阵。
    
    参数
    ----
    correlation_matrix : np.ndarray
        相关性矩阵，形状为 (n_samples, n_samples)
    sigma : float
        尺度参数，控制衰减速度
    k : int
        指数参数，默认为1
        
    返回
    ----
    np.ndarray
        邻接矩阵，形状为 (n_samples, n_samples)
        
    数学原理
    --------
    1. 计算相似度矩阵：C = max(0, cor_mat)（负相关设为0）
    2. 直接使用相似度矩阵：d_ij = C_ij
    3. 计算耦合强度（当 i ≠ j 时）：
       A_ij = 1 - exp(-d_ij^k / sigma^k)
       注意：由于 d_ij = C_ij，相关系数越大，链接强度越大
    4. 对角线元素：A_ii = - Σ_(j≠i) A_ij
    """
    # 计算绝对值相关矩阵
    # C = np.abs(correlation_matrix)
    # 负相关为0
    # 将负相关设为0
    C = np.where(correlation_matrix < 0, 0, correlation_matrix)
    
    # 计算空间距离矩阵
    # d_ij = 1.0 - C
    d_ij = C
    
    # 计算 sigma^k
    sigma_k = sigma ** k
    
    # 计算非对角线元素：A_ij = 1 - exp(-d_ij^k / sigma^k)
    A = 1.0 - np.exp(-(d_ij ** k) / sigma_k)
    
    # 设置对角线元素为0（将在拉普拉斯矩阵计算中处理）
    np.fill_diagonal(A, 0.0)
    
    # 计算对角线元素：A_ii = - Σ_(j≠i) A_ij
    row_sums = np.sum(A, axis=1)
    np.fill_diagonal(A, -row_sums)
    
    return A


def compute_laplacian_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    计算图拉普拉斯矩阵
    
    参数
    ----
    adjacency_matrix : np.ndarray
        邻接矩阵，形状为 (n, n)
        注意：对角线元素应为 A_ii = - Σ_(j≠i) A_ij
        
    返回
    ----
    np.ndarray
        拉普拉斯矩阵 L = D - A
        
    数学原理
    --------
    - 度矩阵 D：对角矩阵，D_ii = Σ_(j≠i) A_ij（只计算非对角线元素）
    - 拉普拉斯矩阵：L = D - A
    - 由于 A_ii = - Σ_(j≠i) A_ij，所以 D_ii = -A_ii
    """
    # 计算度矩阵：只考虑非对角线元素
    # 由于对角线元素 A_ii = - Σ_(j≠i) A_ij，所以度矩阵 D_ii = -A_ii
    A_no_diag = adjacency_matrix.copy()
    np.fill_diagonal(A_no_diag, 0)
    degrees = np.sum(A_no_diag, axis=1)
    D = np.diag(degrees)
    
    # 计算拉普拉斯矩阵
    L = D - adjacency_matrix
    
    return L


def compute_multiscale_laplacians(
    adjacency_matrix: np.ndarray,
    n_scales: int = 5,
    partition_method: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算多尺度拉普拉斯矩阵
    
    通过不同的阈值生成不同密度的子图，从而得到多尺度的拉普拉斯矩阵。
    
    参数
    ----
    adjacency_matrix : np.ndarray
        邻接矩阵，形状为 (n, n)
    n_scales : int
        尺度数量
    partition_method : str
        划分阈值的方法：
        - "uniform": 在 [l_min, l_max] 范围内均匀选择阈值（默认）
        - "quantile": 基于边权重的分位数选择阈值，使每个尺度的边数分布更均匀
        
    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        (拉普拉斯矩阵族, 阈值数组)
        拉普拉斯矩阵族形状为 (n_scales, n, n)
        
    数学原理
    --------
    1. 从邻接矩阵的非对角元素中找到所有非零权重
    2. 根据 partition_method 生成K个阈值α
    3. 对于每个阈值α，保留权重 >= α 的边，形成子图
    4. 对每个子图计算拉普拉斯矩阵
    """
    n = adjacency_matrix.shape[0]
    
    # 移除对角线元素
    A_no_diag = adjacency_matrix.copy()
    np.fill_diagonal(A_no_diag, 0)
    
    # 找到非零元素的范围
    nonzero = A_no_diag[A_no_diag > 0]
    
    if len(nonzero) == 0:
        raise ValueError("邻接矩阵中所有非对角元素都为0")
    
    # 生成阈值
    if partition_method == "quantile":
        # 使用分位数，使每个尺度保留的边数更均匀
        # linspace(0, 100, n_scales) 生成从 0% 到 100% 的分位数
        alphas = np.percentile(nonzero, np.linspace(0, 100, n_scales))
        logger.info(f"使用分位数划分阈值: {alphas}")
    else:
        # 默认使用均匀间隔划分
        l_min = nonzero.min()
        l_max = nonzero.max()
        alphas = np.linspace(l_min, l_max, n_scales)
        logger.info(f"使用均匀间隔划分阈值: {alphas}")
    
    # 计算多尺度拉普拉斯矩阵
    laplacians = np.zeros((n_scales, n, n))
    
    with create_progress() as progress:
        task = progress.add_task("计算多尺度拉普拉斯", total=n_scales)
        
        for k, alpha in enumerate(alphas):
            # 生成子图邻接矩阵
            A_alpha = np.where(A_no_diag >= alpha, A_no_diag, 0)
            np.fill_diagonal(A_alpha, 0)
            
            # 计算拉普拉斯矩阵
            laplacians[k] = compute_laplacian_matrix(A_alpha)
            
            progress.update(task, advance=1)
    
    return laplacians, alphas


def plot_weight_distribution(
    weights: np.ndarray,
    alphas: np.ndarray,
    save_path: Path,
    title: str = "Edge Weight Distribution",
) -> None:
    """
    绘制边权重分布图，并标出选择的阈值
    
    参数
    ----
    weights : np.ndarray
        所有非零边的权重
    alphas : np.ndarray
        选择的阈值数组
    save_path : Path
        保存路径
    title : str
        图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='Edge Weight Distribution')
    
    # 标出阈值线
    for i, alpha in enumerate(alphas):
        label = f"Scale {i+1} (α={alpha:.4f})"
        plt.axvline(alpha, color='red', linestyle='--', alpha=0.6, label=label)
    
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # 设置中文字体（如果需要，这里使用默认）
    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # plt.rcParams['axes.unicode_minus'] = False
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"权重分布图已保存至: {save_path}")


class LaplacianPipeline:
    """
    拉普拉斯特征提取流水线
    
    属性
    ----
    config : Config
        配置对象
    output_dir : Path
        输出目录
    """
    
    def __init__(self, config=None, output_dir: str = None):
        """
        初始化流水线
        
        参数
        ----
        config : Config, optional
            配置对象
        output_dir : str, optional
            输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = get_logger("laplacian")
    
    def run(
        self,
        correlation_matrix: np.ndarray,
        save_results: bool = True,
        prefix: str = "",
    ) -> np.ndarray:
        """
        执行拉普拉斯特征提取
        
        参数
        ----
        correlation_matrix : np.ndarray
            相关性矩阵
        save_results : bool
            是否保存结果
        prefix : str
            文件名前缀
            
        返回
        ----
        np.ndarray
            拉普拉斯矩阵族，形状为 (n_scales, n, n)
        """
        print_header("多尺度拉普拉斯特征提取")
        
        # 获取参数
        if self.config is not None:
            sigma = self.config.laplacian.sigma
            n_scales = self.config.laplacian.n_scales
            k = self.config.laplacian.k
            partition_method = getattr(self.config.laplacian, "partition_method", "uniform")
            plot_distribution = getattr(self.config.laplacian, "plot_distribution", False)
        else:
            sigma = 3.0
            n_scales = 5
            k = 1
            partition_method = "uniform"
            plot_distribution = False
        
        console.print(f"[dim]参数: sigma={sigma}, k={k}, n_scales={n_scales}, method={partition_method}, plot={plot_distribution}[/dim]")
        
        # 步骤1：计算邻接矩阵
        print_step(1, 3, "计算邻接矩阵")
        with Timer("邻接矩阵"):
            adjacency = compute_adjacency_matrix(correlation_matrix, sigma, k)
        
        # 步骤2：计算全连接拉普拉斯矩阵
        print_step(2, 3, "计算全连接拉普拉斯矩阵")
        with Timer("拉普拉斯矩阵"):
            full_laplacian = compute_laplacian_matrix(adjacency)
        
        # 步骤3：计算多尺度拉普拉斯矩阵
        print_step(3, 3, "计算多尺度拉普拉斯矩阵")
        with Timer("多尺度拉普拉斯"):
            laplacian_family, alphas = compute_multiscale_laplacians(
                adjacency, n_scales, partition_method
            )
        
        # 绘制权重分布图
        if plot_distribution and self.output_dir is not None:
            # 提取非对角非零权重用于绘图
            A_no_diag = adjacency.copy()
            np.fill_diagonal(A_no_diag, 0)
            nonzero_weights = A_no_diag[A_no_diag > 0]
            
            if len(nonzero_weights) > 0:
                ensure_dir(self.output_dir)
                dist_path = self.output_dir / f"{prefix + '_' if prefix else ''}weight_distribution.png"
                plot_weight_distribution(
                    nonzero_weights, 
                    alphas, 
                    dist_path,
                    title=f"Edge Weight Distribution ({partition_method})"
                )
        
        # 保存结果
        if save_results and self.output_dir is not None:
            self._save_results(
                adjacency, full_laplacian, laplacian_family, alphas, prefix
            )
        
        # 打印统计
        stats_rows = [
            ["样本数", correlation_matrix.shape[0]],
            ["尺度数", n_scales],
            ["划分方法", partition_method],
            ["阈值范围", f"[{alphas[0]:.4f}, {alphas[-1]:.4f}]"],
        ]
        print_table("拉普拉斯特征统计", ["项目", "值"], stats_rows)
        
        print_success("拉普拉斯特征提取完成")
        
        return laplacian_family
    
    def _save_results(
        self,
        adjacency: np.ndarray,
        full_laplacian: np.ndarray,
        laplacian_family: np.ndarray,
        alphas: np.ndarray,
        prefix: str = "",
    ) -> None:
        """保存结果"""
        ensure_dir(self.output_dir)
        
        prefix_str = f"{prefix}_" if prefix else ""
        
        # 保存邻接矩阵
        save_numpy(
            adjacency,
            self.output_dir / f"{prefix_str}adjacency.npy"
        )
        
        # 保存全连接拉普拉斯矩阵
        save_numpy(
            full_laplacian,
            self.output_dir / f"{prefix_str}laplacian.npy"
        )
        
        # 保存拉普拉斯矩阵族
        save_numpy(
            laplacian_family,
            self.output_dir / f"{prefix_str}laplacian_family.npy"
        )
        
        # 保存每个尺度的拉普拉斯矩阵
        for k in range(laplacian_family.shape[0]):
            save_numpy(
                laplacian_family[k],
                self.output_dir / f"{prefix_str}laplacian_alpha_{k+1:02d}.npy"
            )

