#!/usr/bin/env python3
"""
相似度矩阵计算脚本
=================

计算EEG数据的样本间相似度矩阵。

使用方法
--------
python run_similarity.py --config config/config.yaml --data data/preprocessed_data.npy

参数说明
--------
--config : 配置文件路径
--data : 预处理后的数据路径
--output : 输出目录（可选）
--method : 相似度计算方法（可选）
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.similarity.correlation import compute_similarity_matrix, compute_multiple_similarities
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console, print_table
)
from src.utils.io import save_numpy, load_numpy, ensure_dir
from src.utils.timer import Timer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="相似度矩阵计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="预处理后的数据路径 (npy文件)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录",
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        default=None,
        choices=["pearson", "spearman", "cosine", "rbf_kernel", "laplacian_kernel", "all"],
        help="相似度计算方法，'all'表示计算所有方法",
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="RBF/Laplacian核参数gamma",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print_panel(
        "相似度矩阵计算\n\n"
        f"配置文件: {args.config}\n"
        f"输入数据: {args.data}",
        title="相似度计算脚本",
        style="blue"
    )
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载数据
        console.print(f"\n[bold]加载数据: {args.data}[/bold]")
        data = load_numpy(args.data)
        console.print(f"[dim]数据形状: {data.shape}[/dim]")
        console.print(f"[dim]样本数: {data.shape[0]}[/dim]")
        
        # 确定输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(args.data).parent / "similarity"
        ensure_dir(output_dir)
        
        # 确定计算方法
        method = args.method or config.similarity.method
        
        # 准备额外参数
        kwargs = {}
        if args.gamma is not None:
            kwargs['gamma'] = args.gamma
        elif hasattr(config.similarity, 'rbf_gamma'):
            kwargs['gamma'] = config.similarity.rbf_gamma
        
        print_header("计算相似度矩阵")
        
        if method == "all":
            # 计算所有方法
            methods = ["pearson", "spearman", "cosine"]
            results = {}
            
            for m in methods:
                console.print(f"\n[bold cyan]计算 {m} 相似度...[/bold cyan]")
                with Timer(f"{m}相似度"):
                    similarity = compute_similarity_matrix(data, method=m, **kwargs)
                    results[m] = similarity
                
                # 保存
                save_path = output_dir / f"similarity_{m}.npy"
                save_numpy(similarity, save_path)
            
            # 打印统计
            rows = []
            for m, sim in results.items():
                rows.append([
                    m,
                    f"{sim.shape}",
                    f"{np.mean(sim):.4f}",
                    f"[{np.min(sim):.4f}, {np.max(sim):.4f}]"
                ])
            print_table("相似度矩阵统计", ["方法", "形状", "均值", "范围"], rows)
            
        else:
            # 计算单个方法
            console.print(f"\n[bold cyan]计算 {method} 相似度...[/bold cyan]")
            
            with Timer(f"{method}相似度"):
                similarity = compute_similarity_matrix(data, method=method, **kwargs)
            
            # 保存
            save_path = output_dir / f"similarity_{method}.npy"
            save_numpy(similarity, save_path)
            
            # 打印统计
            rows = [
                ["形状", str(similarity.shape)],
                ["均值", f"{np.mean(similarity):.6f}"],
                ["标准差", f"{np.std(similarity):.6f}"],
                ["最小值", f"{np.min(similarity):.6f}"],
                ["最大值", f"{np.max(similarity):.6f}"],
                ["对角线均值", f"{np.mean(np.diag(similarity)):.6f}"],
            ]
            print_table(f"{method}相似度统计", ["项目", "值"], rows)
        
        print_success(f"相似度矩阵计算完成！结果已保存到: {output_dir}")
        
    except Exception as e:
        print_error(f"计算失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

