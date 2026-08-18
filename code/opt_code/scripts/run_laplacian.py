#!/usr/bin/env python3
"""
拉普拉斯特征提取脚本
===================

从相似度矩阵计算多尺度拉普拉斯特征。

使用方法
--------
python run_laplacian.py --config config/config.yaml --similarity data/similarity_pearson.npy

参数说明
--------
--config : 配置文件路径
--similarity : 相似度矩阵路径
--output : 输出目录（可选）
--n-scales : 拉普拉斯尺度数量（可选）
--sigma : 高斯核参数（可选）
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.laplacian.multiscale import LaplacianPipeline
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console, print_table
)
from src.utils.io import save_numpy, load_numpy, ensure_dir
from src.utils.timer import Timer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="拉普拉斯特征提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    
    parser.add_argument(
        "--similarity", "-s",
        type=str,
        required=True,
        help="相似度矩阵路径 (npy文件)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录",
    )
    
    parser.add_argument(
        "--n-scales",
        type=int,
        default=None,
        help="拉普拉斯尺度数量",
    )
    
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="高斯核参数sigma",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print_panel(
        "拉普拉斯特征提取\n\n"
        f"配置文件: {args.config}\n"
        f"相似度矩阵: {args.similarity}",
        title="拉普拉斯脚本",
        style="blue"
    )
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 覆盖配置参数
        if args.n_scales is not None:
            config.laplacian.n_scales = args.n_scales
        if args.sigma is not None:
            config.laplacian.sigma = args.sigma
        
        # 加载相似度矩阵
        console.print(f"\n[bold]加载相似度矩阵: {args.similarity}[/bold]")
        similarity_matrix = load_numpy(args.similarity)
        console.print(f"[dim]矩阵形状: {similarity_matrix.shape}[/dim]")
        
        # 验证矩阵
        if similarity_matrix.ndim != 2:
            raise ValueError(f"相似度矩阵应为2维，当前为 {similarity_matrix.ndim} 维")
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError(f"相似度矩阵应为方阵，当前形状: {similarity_matrix.shape}")
        
        # 确定输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(args.similarity).parent / "laplacian"
        ensure_dir(output_dir)
        
        # 计算拉普拉斯特征
        print_header("多尺度拉普拉斯特征提取")
        
        pipeline = LaplacianPipeline(config, output_dir=str(output_dir))
        laplacian_family = pipeline.run(similarity_matrix, save_results=True)
        
        # 打印额外统计
        console.print("\n[bold]拉普拉斯矩阵族统计:[/bold]")
        rows = []
        for k in range(laplacian_family.shape[0]):
            L = laplacian_family[k]
            # 计算边数（非零非对角元素）
            L_offdiag = L.copy()
            np.fill_diagonal(L_offdiag, 0)
            n_edges = np.sum(L_offdiag != 0) // 2
            
            rows.append([
                f"尺度 {k+1}",
                f"{np.mean(np.diag(L)):.4f}",
                f"{n_edges}",
                f"[{np.min(L):.4f}, {np.max(L):.4f}]"
            ])
        print_table("各尺度统计", ["尺度", "平均度", "边数", "值范围"], rows)
        
        print_success(f"拉普拉斯特征提取完成！结果已保存到: {output_dir}")
        
    except Exception as e:
        print_error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
