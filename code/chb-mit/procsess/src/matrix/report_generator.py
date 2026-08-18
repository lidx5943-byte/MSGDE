# Author: 王梓涵 <wangzh011031@163.com>
"""
报告生成模块
=============

生成矩阵生成过程的详细统计报告
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from rich.panel import Panel


def generate_matrix_report(
    output_dir: Path,
    config: Dict[str, Any],
    matrices_info: Dict[str, Any],
    filter_stats: Dict[str, Any],
    laplacian_stats: Dict[str, Any],
    subgraph_stats: Dict[str, Any],
    thresholds: np.ndarray,
    execution_time: float
) -> str:
    """
    生成详细的统计报告
    
    Args:
        output_dir: 输出目录
        config: 配置信息
        matrices_info: 矩阵形状信息
        filter_stats: 负相关过滤统计
        laplacian_stats: 拉普拉斯矩阵统计
        subgraph_stats: 子图统计
        execution_time: 执行时间 (秒)
    
    Returns:
        报告文本内容
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "=" * 60,
        "EEG 相似度矩阵生成报告",
        f"生成时间: {timestamp}",
        "=" * 60,
        "",
        "[基本配置]",
        f"实验名称: {config.get('experiment', {}).get('name', 'unknown')}",
        f"相似度方法: {config.get('similarity', {}).get('method', 'pearson')}",
        f"高斯核指数: {config.get('gaussian_kernel', {}).get('exponent', 1)}",
        f"子图划分数 K: {config.get('subgraph', {}).get('k', 10)}",
        f"划分方式: {config.get('subgraph', {}).get('partition_method', 'uniform')}",
        f"执行时间: {execution_time:.2f} 秒",
        "",
        "[输出矩阵]",
    ]
    
    # 矩阵信息
    for name, info in matrices_info.items():
        lines.append(f"{name}:")
        lines.append(f"  形状: {info['shape']}")
        lines.append(f"  数据类型: {info['dtype']}")
        lines.append(f"  文件大小: {info['size_mb']:.2f} MB")
        lines.append("")
    
    # 负相关过滤统计
    lines.extend([
        "[负相关过滤统计]",
        f"总配对数 (不含对角线): {filter_stats.get('n_total_pairs', 0):,}",
        f"过滤的负相关数: {filter_stats.get('n_negative_filtered', 0):,}",
        f"负相关占比: {filter_stats.get('negative_ratio', 0) * 100:.2f}%",
        f"过滤后边数: {filter_stats.get('n_edges_after', 0):,}",
        "",
    ])
    
    # 阈值信息
    lines.extend([
        "[阈值信息]",
        f"阈值数量: {len(thresholds)}",
        f"阈值范围: [{thresholds[0]:.6f}, {thresholds[-1]:.6f}]",
        "",
    ])
    
    # 拉普拉斯矩阵统计（多子图）
    if laplacian_stats.get('n_subgraphs'):
        lines.extend([
            "[拉普拉斯矩阵统计]",
            f"子图数量: {laplacian_stats.get('n_subgraphs', 0)}",
            f"节点数: {laplacian_stats.get('n_nodes', 0):,}",
            "",
        ])
        
        # 每个子图的统计
        subgraph_lap_stats = laplacian_stats.get('subgraphs', [])
        if subgraph_lap_stats:
            lines.extend([
                "[各子图拉普拉斯矩阵统计]",
                f"{'索引':<8}{'边数':<12}{'孤立节点':<12}{'平均度':<12}{'最大度':<12}",
                "-" * 60,
            ])
            for s in subgraph_lap_stats:
                lines.append(
                    f"{s['index']:<8}{s['n_edges']:<12}{s['n_isolated_nodes']:<12}{s['avg_degree']:<12.4f}{s['max_degree']:<12.4f}"
                )
            lines.append("")
    
    # 子图统计
    subgraphs = subgraph_stats.get('subgraphs', [])
    if subgraphs:
        # 检查是否有分位数信息（分位数划分）
        has_quantile = 'quantile_low' in subgraphs[0] if subgraphs else False
        
        if has_quantile:
            lines.extend([
                "[子图统计（分位数划分）]",
                f"{'索引':<8}{'分位数范围':<15}{'阈值':<15}{'边数':<12}{'孤立节点':<12}{'活跃节点':<12}",
                "-" * 80,
            ])
            
            for sg in subgraphs:
                threshold = sg.get('threshold', 0.0)
                quantile_low = sg.get('quantile_low', 0.0)
                quantile_high = sg.get('quantile_high', 0.0)
                quantile_str = f"[{quantile_low:.1f}%, {quantile_high:.1f}%)"
                lines.append(
                    f"{sg['index']:<8}{quantile_str:<15}{threshold:<15.6f}{sg['n_edges']:<12}{sg['n_isolated_nodes']:<12}{sg['n_active_nodes']:<12}"
                )
        else:
            lines.extend([
                "[子图统计（均匀阈值划分）]",
                f"{'索引':<8}{'阈值':<15}{'边数':<12}{'孤立节点':<12}{'活跃节点':<12}",
                "-" * 60,
            ])
            
            for sg in subgraphs:
                threshold = sg.get('threshold', 0.0)
                lines.append(
                    f"{sg['index']:<8}{threshold:<15.6f}{sg['n_edges']:<12}{sg['n_isolated_nodes']:<12}{sg['n_active_nodes']:<12}"
                )
        lines.append("")
    
    lines.append("=" * 60)
    
    report_text = "\n".join(lines)
    
    # 保存报告
    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    return report_text


def print_summary(report_text: str) -> None:
    """
    在控制台打印报告摘要
    
    Args:
        report_text: 报告文本
    """
    from ..utils.console import console
    console.print(Panel(report_text, title="[bold green]生成报告[/bold green]", expand=False))
