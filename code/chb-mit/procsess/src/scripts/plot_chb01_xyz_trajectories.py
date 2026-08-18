# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB01患者X轨迹和X-Z平面可视化脚本

功能：
1. 加载chb01患者数据
2. 构建多尺度图并运行Lorenz动力学模拟
3. 提取X轨迹（尺度1标注为ε=0.416，全部10个尺度使用不同颜色图例）
4. 绘制X-Z平面轨迹图
5. 绘制第一个尺度的x, y, z三个方向轨迹（分三个文件，无标题）
6. 按SCI顶刊风格输出高清晰度图像
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.visualization.nature_style import set_nature_style
from src.matrix.similarity import compute_similarity_matrix_2d
from src.matrix.graph_builder import (
    filter_negative_correlations, apply_gaussian_kernel,
    partition_by_quantile, binarize_cutoff
)
from src.features.lorenz import (
    simulate_lorenz_sparse, LorenzConfig, LorenzOscillator
)


def load_chb01_data(data_dir: str = "/mnt/3M/chbmit-allchannels/patients_data") -> Tuple[np.ndarray, np.ndarray]:
    """加载chb01患者数据"""
    patient_dir = Path(data_dir) / "chb01"
    x_path = patient_dir / "x_data.npy"
    y_path = patient_dir / "y_labels.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"chb01数据文件不存在: {patient_dir}")

    x_data = np.load(x_path)
    y_labels = np.load(y_path)

    print(f"Loaded chb01 data: x_data shape={x_data.shape}, y_labels shape={y_labels.shape}")
    print(f"Class distribution: {np.bincount(y_labels)}")

    return x_data, y_labels


def build_graphs(x_data: np.ndarray, n_scales: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """构建多尺度图"""
    print("Building multi-scale graphs...")

    # 1. 计算相似度矩阵
    P = compute_similarity_matrix_2d(x_data, method="pearson", chunk_size=200, verbose=False)

    # 2. 过滤负相关
    A, _ = filter_negative_correlations(P)

    # 3. 应用高斯核
    A_prime = apply_gaussian_kernel(A, exponent=1)

    # 4. 按分位数分区
    Cutoff_weight, thresholds, _ = partition_by_quantile(A_prime, k=n_scales)

    # 5. 二值化
    Cutoff_0_1 = binarize_cutoff(Cutoff_weight, thresholds)

    print(f"Graphs built: Cutoff_weight shape={Cutoff_weight.shape}, Cutoff_0_1 shape={Cutoff_0_1.shape}")
    print(f"  -> Number of nodes per sample: {Cutoff_0_1.shape[2]}")

    return Cutoff_weight, Cutoff_0_1


def extract_trajectories(Cutoff_0_1: np.ndarray, seed: int = 42) -> Dict:
    """
    提取Lorenz动力学轨迹
    """
    num_scales, num_samples, n_nodes = Cutoff_0_1.shape
    print(f"Extracting trajectories for {num_scales} scales, {num_samples} samples, {n_nodes} nodes...")

    # Lorenz配置
    dyn_config = LorenzConfig(
        coupling_strength=0.42,
        coupling_mode="xyz_all",
        dt=0.01,
        total_steps=500,
        steady_steps=200,
    )

    # 输出演化参数信息
    print("=" * 60)
    print("Lorenz Simulation Parameters:")
    print(f"  Total evolution steps: {dyn_config.total_steps}")
    print(f"  Steady-state steps (saved): {dyn_config.steady_steps}")
    print(f"  Time step (dt): {dyn_config.dt}")
    print(f"  Number of nodes per sample: {n_nodes}")
    print(f"  Coupling strength: {dyn_config.coupling_strength}")
    print(f"  Coupling mode: {dyn_config.coupling_mode}")
    print("=" * 60)

    oscillator = LorenzOscillator(
        delta=10.0,
        gamma=60.0,
        beta=8.0/3.0,
        rk=7.0,
    )

    # 存储轨迹
    trajectories = {
        'scale_1': {'x': [], 'y': [], 'z': [], 'sample_info': []},
        'all_scales': {'x': [], 'z': [], 'scale_info': [], 'sample_info': []},
        'first_scale_xyz': {'x': None, 'y': None, 'z': None},  # 第一个尺度的xyz
        'time': None,
        'dt': dyn_config.dt,
        'steady_steps': dyn_config.steady_steps,
        'n_nodes': n_nodes,
    }

    # 选择代表性样本
    selected_samples = [0, num_samples // 4, num_samples // 2, 3 * num_samples // 4, num_samples - 1]

    # 1. 提取尺度1的轨迹
    print("Processing scale 1 (single scale)...")
    scale_idx = 0

    for sample_idx in selected_samples:
        if sample_idx >= num_samples:
            continue

        print(f"  Running simulation for sample {sample_idx}, scale {scale_idx}...")
        print(f"    -> Evolving {dyn_config.total_steps} steps, saving last {dyn_config.steady_steps} steps")
        print(f"    -> Using {n_nodes} nodes (all nodes averaged)")

        res = simulate_lorenz_sparse(
            Cutoff_0_1[scale_idx],
            dyn_config,
            oscillator,
            seed + scale_idx + sample_idx
        )

        steady_traj = res['steady_trajectory']
        print(f"    -> Steady trajectory shape: {steady_traj.shape} (nodes, steps, components)")

        x_traj = np.mean(steady_traj[:, :, 0], axis=0)
        y_traj = np.mean(steady_traj[:, :, 1], axis=0)
        z_traj = np.mean(steady_traj[:, :, 2], axis=0)

        trajectories['scale_1']['x'].append(x_traj)
        trajectories['scale_1']['y'].append(y_traj)
        trajectories['scale_1']['z'].append(z_traj)
        trajectories['scale_1']['sample_info'].append(sample_idx)

        if trajectories['time'] is None:
            trajectories['time'] = (np.arange(dyn_config.steady_steps) * dyn_config.dt)

    # 保存第一个样本的第一个尺度的完整xyz用于单独绘图
    trajectories['first_scale_xyz']['x'] = trajectories['scale_1']['x'][0]
    trajectories['first_scale_xyz']['y'] = trajectories['scale_1']['y'][0]
    trajectories['first_scale_xyz']['z'] = trajectories['scale_1']['z'][0]

    # 2. 提取全部10个尺度的轨迹（用于X轨迹图）
    print("Processing all 10 scales...")
    for scale_idx in range(num_scales):
        for sample_idx in selected_samples[:2]:
            if sample_idx >= num_samples:
                continue

            res = simulate_lorenz_sparse(
                Cutoff_0_1[scale_idx],
                dyn_config,
                oscillator,
                seed + scale_idx + sample_idx
            )

            steady_traj = res['steady_trajectory']
            x_traj = np.mean(steady_traj[:, :, 0], axis=0)
            z_traj = np.mean(steady_traj[:, :, 2], axis=0)

            trajectories['all_scales']['x'].append(x_traj)
            trajectories['all_scales']['z'].append(z_traj)
            trajectories['all_scales']['scale_info'].append(scale_idx)
            trajectories['all_scales']['sample_info'].append(sample_idx)

    return trajectories


def create_x_trajectory_plot(trajectories: Dict, output_dir: Path):
    """
    创建X轨迹可视化图
    左图：尺度1的三个方向x,y,z（不同颜色，图例标注）
    右图：全部10个尺度的X方向（只绘制x方向）
    """
    print("Creating trajectory comparison visualization...")

    set_nature_style()

    time = trajectories['time']

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # 颜色定义
    color_x = '#E67E22'  # 橙色 - X
    color_y = '#1B4F72'  # 深蓝 - Y
    color_z = '#145A32'  # 深绿 - Z
    colors_all = ['#1B4F72', '#2874A6', '#3498DB', '#5DADE2', '#85C1E9',
                  '#A9CCE3', '#D4E6F1', '#EBF5FB', '#FADBD8', '#E6B0AA']

    # 左图：尺度1的三个方向x, y, z（不同颜色，图例标注，归一化显示）
    # 绘制第一个样本的三个方向
    x_data = trajectories['scale_1']['x'][0]
    y_data = trajectories['scale_1']['y'][0]
    z_data = trajectories['scale_1']['z'][0]

    # 归一化到[0, 1]范围
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    x_norm = normalize(x_data)
    y_norm = normalize(y_data)
    z_norm = normalize(z_data)

    axes[0].plot(time, x_norm, color=color_x, alpha=0.8, linewidth=1.5, label='X')
    axes[0].plot(time, y_norm, color=color_y, alpha=0.8, linewidth=1.5, label='Y')
    axes[0].plot(time, z_norm, color=color_z, alpha=0.8, linewidth=1.5, label='Z')

    axes[0].set_xlabel('Time (s)', fontweight='bold', fontsize=9)
    axes[0].set_ylabel('Amplitude', fontweight='bold', fontsize=9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    # 图例标注 x, y, z
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    # 添加epsilon标注
    axes[0].text(0.02, 0.98, r'$\epsilon = 0.416$', transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', fontweight='bold')

    # 右图：全部10个尺度的X方向（只绘制x方向）
    for scale_idx in range(10):
        scale_x_data = [trajectories['all_scales']['x'][i]
                       for i in range(len(trajectories['all_scales']['x']))
                       if trajectories['all_scales']['scale_info'][i] == scale_idx][:1]
        if scale_x_data:
            axes[1].plot(time, scale_x_data[0], color=colors_all[scale_idx],
                        alpha=0.7, linewidth=1.0, label=f'Scale {scale_idx+1}')

    axes[1].set_xlabel('Time (s)', fontweight='bold', fontsize=9)
    axes[1].set_ylabel('X Amplitude', fontweight='bold', fontsize=9)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    axes[1].legend(loc='upper right', frameon=False, fontsize=7, ncol=2)

    plt.tight_layout()

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "chb01_x_trajectories_comparison"

    formats = ['png', 'pdf', 'svg', 'tif']
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'tif':
            fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")

    plt.close(fig)


def create_xz_plane_plot(trajectories: Dict, output_dir: Path):
    """
    创建X-Z平面轨迹图
    """
    print("Creating X-Z plane trajectory visualization...")

    set_nature_style()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    # 颜色
    color_scale_1 = '#E67E22'  # 橙色
    colors_all = ['#1B4F72', '#2874A6', '#3498DB', '#5DADE2', '#85C1E9',
                  '#A9CCE3', '#D4E6F1', '#EBF5FB', '#FADBD8', '#E6B0AA']

    # 左图：尺度1的X-Z平面轨迹
    for i in range(len(trajectories['scale_1']['x'])):
        x_traj = trajectories['scale_1']['x'][i]
        z_traj = trajectories['scale_1']['z'][i]

        axes[0].plot(x_traj, z_traj, color=color_scale_1, alpha=0.7, linewidth=1.2)
        # 标记起点和终点
        axes[0].scatter(x_traj[0], z_traj[0], color='green', s=30, zorder=5, marker='o')
        axes[0].scatter(x_traj[-1], z_traj[-1], color='red', s=30, zorder=5, marker='s')

    axes[0].set_xlabel('X', fontweight='bold', fontsize=9)
    axes[0].set_ylabel('Z', fontweight='bold', fontsize=9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    # 图例
    axes[0].plot([], [], color=color_scale_1, linewidth=2, label=r'$\epsilon = 0.416$')
    axes[0].scatter([], [], color='green', s=30, marker='o', label='Start')
    axes[0].scatter([], [], color='red', s=30, marker='s', label='End')
    axes[0].legend(loc='upper right', frameon=False, fontsize=8)

    # 右图：全部尺度的X-Z平面轨迹
    for scale_idx in range(10):
        scale_indices = [i for i in range(len(trajectories['all_scales']['x']))
                        if trajectories['all_scales']['scale_info'][i] == scale_idx][:1]
        for i in scale_indices:
            x_traj = trajectories['all_scales']['x'][i]
            z_traj = trajectories['all_scales']['z'][i]
            axes[1].plot(x_traj, z_traj, color=colors_all[scale_idx],
                        alpha=0.6, linewidth=0.8, label=f'Scale {scale_idx+1}')

    axes[1].set_xlabel('X', fontweight='bold', fontsize=9)
    axes[1].set_ylabel('Z', fontweight='bold', fontsize=9)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    axes[1].legend(loc='upper right', frameon=False, fontsize=7, ncol=2)

    plt.tight_layout()

    # 保存
    filename = "chb01_xz_plane_trajectories"

    formats = ['png', 'pdf', 'svg', 'tif']
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'tif':
            fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")

    plt.close(fig)


def create_scale1_xyz_plots(trajectories: Dict, output_dir: Path):
    """
    创建第一个尺度的x, y, z三个方向轨迹图（分三个文件，无标题）
    """
    print("Creating Scale 1 XYZ trajectory plots (no title)...")

    set_nature_style()

    time = trajectories['time']
    x_data = trajectories['first_scale_xyz']['x']
    y_data = trajectories['first_scale_xyz']['y']
    z_data = trajectories['first_scale_xyz']['z']

    components = [
        ('x', x_data, 'X Amplitude'),
        ('y', y_data, 'Y Amplitude'),
        ('z', z_data, 'Z Amplitude')
    ]

    color = '#1B4F72'  # 深蓝色

    for comp_name, comp_data, ylabel in components:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))

        ax.plot(time, comp_data, color=color, alpha=0.8, linewidth=1.2)

        ax.set_xlabel('Time (s)', fontweight='bold', fontsize=9)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        # 无标题

        plt.tight_layout()

        # 保存
        filename = f"chb01_scale1_{comp_name}_trajectory"

        formats = ['png', 'pdf', 'svg', 'tif']
        for fmt in formats:
            filepath = output_dir / f"{filename}.{fmt}"
            if fmt == 'tif':
                fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none', pil_kwargs={"compression": "tiff_lzw"})
            else:
                fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            print(f"  Saved: {filepath}")

        plt.close(fig)


def save_trajectory_data_json(trajectories: Dict, output_dir: Path):
    """保存轨迹数据到JSON文件"""
    print("Saving trajectory data to JSON...")

    json_data = {
        'patient_id': 'chb01',
        'dt': trajectories['dt'],
        'steady_steps': trajectories['steady_steps'],
        'n_nodes': trajectories['n_nodes'],
        'time': trajectories['time'].tolist() if isinstance(trajectories['time'], np.ndarray) else trajectories['time'],
        'scale_1': {
            'epsilon': 0.416,
            'samples': trajectories['scale_1']['sample_info'],
            'x_trajectories': [x.tolist() for x in trajectories['scale_1']['x']],
            'y_trajectories': [y.tolist() for y in trajectories['scale_1']['y']],
            'z_trajectories': [z.tolist() for z in trajectories['scale_1']['z']],
        },
        'all_scales': {
            'scales': trajectories['all_scales']['scale_info'],
            'samples': trajectories['all_scales']['sample_info'],
            'x_trajectories': [x.tolist() for x in trajectories['all_scales']['x']],
            'z_trajectories': [z.tolist() for z in trajectories['all_scales']['z']],
        },
        'first_scale_xyz': {
            'x': trajectories['first_scale_xyz']['x'].tolist(),
            'y': trajectories['first_scale_xyz']['y'].tolist(),
            'z': trajectories['first_scale_xyz']['z'].tolist(),
        }
    }

    output_path = output_dir / "chb01_x_trajectory_data.json"
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  Saved: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("CHB01 X Trajectory and X-Z Plane Visualization")
    print("=" * 60)

    # 设置输出目录
    output_dir = Path("/srv/New_eeg_code/3M/supplyment_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    x_data, y_labels = load_chb01_data()

    # 2. 构建图（只使用前100个样本以加快速度）
    sample_limit = 100
    x_subset = x_data[:sample_limit]

    Cutoff_weight, Cutoff_0_1 = build_graphs(x_subset, n_scales=10)

    # 3. 提取轨迹
    trajectories = extract_trajectories(Cutoff_0_1, seed=42)

    # 4. 创建可视化
    create_x_trajectory_plot(trajectories, output_dir)
    create_xz_plane_plot(trajectories, output_dir)
    create_scale1_xyz_plots(trajectories, output_dir)

    # 5. 保存JSON数据
    save_trajectory_data_json(trajectories, output_dir)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
