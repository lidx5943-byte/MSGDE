"""
Trajectory Visualization Module
===============================

Provides trajectory and phase space visualization functions.

Usage
-----
>>> from src.visualization.trajectories import plot_phase_space, plot_butterfly_attractor
>>> 
>>> plot_phase_space(trajectories, save_path="phase_space.png")
>>> plot_butterfly_attractor(trajectories, figures_dir="figures")
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple

from ..utils.logger import get_logger, console, create_progress, print_success

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logger = get_logger(__name__)


def plot_phase_space(
    trajectories: np.ndarray,
    node_idx: int = 0,
    title: str = "Phase Space (X-Z Plane)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
) -> None:
    """
    Plot phase space diagram (X-Z plane)
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_nodes, n_times, 3)
    node_idx : int
        Node index
    title : str
        Plot title
    save_path : str, optional
        Save path
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        Resolution
    """
    x = trajectories[node_idx, :, 0]
    z = trajectories[node_idx, :, 2]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, z, 'b-', linewidth=0.5, alpha=0.8)
    ax.plot(x[0], z[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], z[-1], 'ro', markersize=8, label='End')
    
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Z', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add info text
    info_text = f'Node {node_idx}\nSteps: {trajectories.shape[1]}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print_success(f"Saved: {save_path}")
    
    plt.close()


def plot_trajectory_evolution(
    trajectories: np.ndarray,
    node_idx: int = 0,
    title: str = "Trajectory Evolution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300,
) -> None:
    """
    Plot trajectory evolution (three components over time)
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_nodes, n_times, 3)
    node_idx : int
        Node index
    title : str
        Plot title
    save_path : str, optional
        Save path
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        Resolution
    """
    node_traj = trajectories[node_idx]
    n_times = node_traj.shape[0]
    time_steps = np.arange(n_times)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    labels = ['X', 'Y', 'Z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time_steps, node_traj[:, i], color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend([label], loc='upper right')
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print_success(f"Saved: {save_path}")
    
    plt.close()


def plot_butterfly_attractor(
    trajectories: np.ndarray,
    figures_dir: Optional[str] = None,
    node_idx: int = 0,
    step_interval: int = 100,
    window_size: Optional[int] = None,
    interval_mode: str = "uniform",
    custom_intervals: Optional[List[int]] = None,
    dpi: int = 300,
) -> str:
    """
    Plot butterfly attractor sequence
    
    Generate a sequence of plots at specified step intervals.
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_nodes, n_times, 3)
    figures_dir : str, optional
        Figures save directory
    node_idx : int
        Node index
    step_interval : int
        Uniform step interval
    window_size : int, optional
        Window size (None for cumulative mode)
    interval_mode : str
        Interval mode: "uniform" or "custom"
    custom_intervals : List[int], optional
        Custom interval list
    dpi : int
        Resolution
        
    Returns
    -------
    str
        Save directory path
    """
    x = trajectories[node_idx, :, 0]
    z = trajectories[node_idx, :, 2]
    n_steps = trajectories.shape[1]
    
    # Create save directory
    if figures_dir:
        save_dir = Path(figures_dir) / 'butterfly_evolution'
    else:
        save_dir = Path('butterfly_evolution')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate plot step list
    if interval_mode == "custom" and custom_intervals is not None:
        step_list = []
        current_step = 0
        for interval in custom_intervals:
            current_step += interval
            if current_step <= n_steps:
                step_list.append(current_step)
        if not step_list or step_list[-1] < n_steps:
            step_list.append(n_steps)
    else:
        n_figures = (n_steps - 1) // step_interval + 1
        step_list = [min((i + 1) * step_interval, n_steps) for i in range(n_figures)]
    
    use_window = window_size is not None and window_size > 0
    
    console.print(f"[dim]Generating {len(step_list)} butterfly plots...[/dim]")
    
    with create_progress() as progress:
        task = progress.add_task("Plotting", total=len(step_list))
        
        for fig_idx, end_step in enumerate(step_list):
            # Determine data range
            if use_window:
                start_step = max(0, end_step - window_size)
                x_current = x[start_step:end_step]
                z_current = z[start_step:end_step]
                step_range = f"{start_step}-{end_step}"
            else:
                x_current = x[:end_step]
                z_current = z[:end_step]
                step_range = f"0-{end_step}"
            
            # Downsampling
            if use_window and len(x_current) <= 500:
                downsample = 1
            else:
                downsample = max(1, len(x_current) // 500)
            
            x_plot = x_current[::downsample]
            z_plot = z_current[::downsample]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.plot(x_plot, z_plot, 'b-', linewidth=0.5, alpha=0.8)
            
            if len(x_plot) > 0:
                ax.plot(x_plot[0], z_plot[0], 'go', markersize=8, label='Start', zorder=5)
                ax.plot(x_plot[-1], z_plot[-1], 'ro', markersize=8, label='End', zorder=5)
            
            ax.set_xlabel('X', fontsize=14, fontweight='bold')
            ax.set_ylabel('Z', fontsize=14, fontweight='bold')
            ax.set_title(f'Lorenz Butterfly Attractor (Steps {step_range})', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')
            
            info_text = f'Node {node_idx}\nSteps: {step_range}\nPoints: {len(x_plot)}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plt.tight_layout()
            
            # Save
            filename = save_dir / f'butterfly_step_{end_step:05d}.png'
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            progress.update(task, advance=1)
    
    print_success(f"Butterfly plots saved to: {save_dir}")
    
    return str(save_dir)


def plot_3d_trajectory(
    trajectories: np.ndarray,
    node_idx: int = 0,
    title: str = "3D Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300,
) -> None:
    """
    Plot 3D trajectory
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_nodes, n_times, 3)
    node_idx : int
        Node index
    title : str
        Plot title
    save_path : str, optional
        Save path
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        Resolution
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    node_traj = trajectories[node_idx]
    x, y, z = node_traj[:, 0], node_traj[:, 1], node_traj[:, 2]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Use color gradient for time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=0.5)
    
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='o', label='End')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print_success(f"Saved: {save_path}")
    
    plt.close()


def plot_multi_scale_comparison(
    trajectories: np.ndarray,
    node_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300,
) -> None:
    """
    Plot multi-scale trajectory comparison
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_scales, n_nodes, n_times, 3)
    node_idx : int
        Node index
    save_path : str, optional
        Save path
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        Resolution
    """
    n_scales = trajectories.shape[0]
    
    # Calculate subplot layout
    n_cols = min(3, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for scale_idx in range(n_scales):
        row = scale_idx // n_cols
        col = scale_idx % n_cols
        ax = axes[row, col]
        
        x = trajectories[scale_idx, node_idx, :, 0]
        z = trajectories[scale_idx, node_idx, :, 2]
        
        ax.plot(x, z, 'b-', linewidth=0.3, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Scale {scale_idx + 1}')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_scales, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'Multi-Scale Phase Space Comparison (Node {node_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print_success(f"Saved: {save_path}")
    
    plt.close()
