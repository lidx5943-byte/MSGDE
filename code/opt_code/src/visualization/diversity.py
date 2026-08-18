"""
Feature Diversity Visualization Module
======================================

Provides feature diversity analysis visualization functions.

Usage
-----
>>> from src.visualization.diversity import analyze_feature_diversity
>>> 
>>> analyze_feature_diversity(trajectories, figures_dir="figures")
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

from ..dynamics.features import compute_diversity_over_steps
from ..utils.logger import get_logger, console, print_success, print_header
from ..utils.io import save_numpy, ensure_dir

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logger = get_logger(__name__)


def plot_feature_diversity(
    steps: np.ndarray,
    diversity_scores: np.ndarray,
    method: str = "combined",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
) -> None:
    """
    Plot feature diversity over evolution steps
    
    Parameters
    ----------
    steps : np.ndarray
        Evolution step array
    diversity_scores : np.ndarray
        Diversity score array
    method : str
        Diversity calculation method
    title : str, optional
        Plot title
    save_path : str, optional
        Save path
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        Resolution
    """
    plt.figure(figsize=figsize)
    
    # Plot main curve
    plt.plot(steps, diversity_scores, 'b-', linewidth=2, alpha=0.8, label='Feature Diversity')
    
    # Smoothed curve
    if len(diversity_scores) > 10:
        window = min(10, len(diversity_scores) // 10)
        smoothed = np.convolve(diversity_scores, np.ones(window)/window, mode='same')
        plt.plot(steps, smoothed, 'r--', linewidth=1.5, alpha=0.6,
                label=f'Smoothed (window={window})')
    
    # Mark extreme points
    max_idx = np.argmax(diversity_scores)
    min_idx = np.argmin(diversity_scores)
    
    plt.plot(steps[max_idx], diversity_scores[max_idx],
            'ro', markersize=10, label=f'Max (step={steps[max_idx]})')
    plt.plot(steps[min_idx], diversity_scores[min_idx],
            'go', markersize=10, label=f'Min (step={steps[min_idx]})')
    
    # Labels and title
    plt.xlabel('Evolution Step', fontsize=14, fontweight='bold')
    plt.ylabel('Feature Diversity', fontsize=14, fontweight='bold')
    
    if title is None:
        title = f'Feature Diversity Evolution (Method: {method})'
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    # Statistics info
    mean_div = np.mean(diversity_scores)
    std_div = np.std(diversity_scores)
    info_text = f'Mean: {mean_div:.4f}\nStd: {std_div:.4f}'
    info_text += f'\nMax: {diversity_scores[max_idx]:.4f}'
    info_text += f'\nMin: {diversity_scores[min_idx]:.4f}'
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print_success(f"Saved: {save_path}")
    
    plt.close()


def analyze_feature_diversity(
    trajectories: np.ndarray,
    figures_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    step_interval: int = 100,
    window_size: Optional[int] = None,
    method: str = "combined",
    center_node_idx: int = 0,
    scale_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze feature diversity and save results
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_nodes, n_times, 3)
    figures_dir : str, optional
        Figures save directory
    data_dir : str, optional
        Data save directory
    step_interval : int
        Calculation interval
    window_size : int, optional
        Window size
    method : str
        Diversity calculation method
    center_node_idx : int
        Center node index
    scale_idx : int
        Scale index
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (steps array, diversity scores array)
    """
    console.print(f"[dim]Analyzing feature diversity (Scale {scale_idx + 1})...[/dim]")
    
    # Calculate diversity
    steps, diversity_scores = compute_diversity_over_steps(
        trajectories,
        step_interval=step_interval,
        window_size=window_size,
        method=method,
        center_node_idx=center_node_idx,
    )
    
    # Save figure
    if figures_dir:
        figures_dir = Path(figures_dir)
        ensure_dir(figures_dir)
        
        save_path = figures_dir / f'feature_diversity_scale{scale_idx+1}_{method}.png'
        plot_feature_diversity(
            steps, diversity_scores,
            method=method,
            title=f'Feature Diversity Evolution - Scale {scale_idx + 1}',
            save_path=str(save_path),
        )
    
    # Save data
    if data_dir:
        data_dir = Path(data_dir)
        ensure_dir(data_dir)
        
        data_path = data_dir / f'feature_diversity_scale{scale_idx+1}_{method}.npz'
        np.savez(
            data_path,
            steps=steps,
            diversity_scores=diversity_scores,
            method=method,
            window_size=window_size,
            scale_idx=scale_idx,
        )
        print_success(f"Data saved: {data_path}")
    
    return steps, diversity_scores


def analyze_all_scales(
    trajectories: np.ndarray,
    figures_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    n_scales: int = None,
    step_interval: int = 100,
    window_size: Optional[int] = None,
    method: str = "combined",
) -> dict:
    """
    Analyze feature diversity for all scales
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_scales, n_nodes, n_times, 3)
    figures_dir : str, optional
        Figures save directory
    data_dir : str, optional
        Data save directory
    n_scales : int, optional
        Number of scales to analyze
    step_interval : int
        Calculation interval
    window_size : int, optional
        Window size
    method : str
        Diversity calculation method
        
    Returns
    -------
    dict
        {scale_idx: (steps, diversity_scores)}
    """
    print_header("Multi-Scale Feature Diversity Analysis")
    
    if trajectories.ndim == 3:
        # Single scale
        trajectories = trajectories[np.newaxis, ...]
    
    available_scales = trajectories.shape[0]
    if n_scales is None:
        n_scales = available_scales
    else:
        n_scales = min(n_scales, available_scales)
    
    results = {}
    
    for scale_idx in range(n_scales):
        scale_traj = trajectories[scale_idx]
        steps, scores = analyze_feature_diversity(
            scale_traj,
            figures_dir=figures_dir,
            data_dir=data_dir,
            step_interval=step_interval,
            window_size=window_size,
            method=method,
            scale_idx=scale_idx,
        )
        results[scale_idx] = (steps, scores)
    
    # Plot comparison
    if figures_dir and len(results) > 1:
        plot_diversity_comparison(results, figures_dir, method)
    
    return results


def plot_diversity_comparison(
    results: dict,
    figures_dir: str,
    method: str = "combined",
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 300,
) -> None:
    """
    Plot multi-scale diversity comparison
    
    Parameters
    ----------
    results : dict
        {scale_idx: (steps, diversity_scores)}
    figures_dir : str
        Save directory
    method : str
        Method name
    figsize : Tuple[int, int]
        Figure size
    dpi : int
        Resolution
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (scale_idx, (steps, scores)), color in zip(results.items(), colors):
        plt.plot(steps, scores, linewidth=1.5, alpha=0.8,
                label=f'Scale {scale_idx + 1}', color=color)
    
    plt.xlabel('Evolution Step', fontsize=14, fontweight='bold')
    plt.ylabel('Feature Diversity', fontsize=14, fontweight='bold')
    plt.title(f'Multi-Scale Feature Diversity Comparison (Method: {method})', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    save_path = Path(figures_dir) / f'diversity_comparison_{method}.png'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print_success(f"Comparison plot saved: {save_path}")
    
    plt.close()
