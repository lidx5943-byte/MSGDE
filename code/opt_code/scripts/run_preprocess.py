#!/usr/bin/env python3
"""
EEG数据预处理脚本
=================

执行完整的EEG数据预处理流程。

使用方法
--------
python run_preprocess.py --config config/config.yaml --data data/eeg_data.npy

参数说明
--------
--config : 配置文件路径
--data : 输入数据路径
--labels : 标签数据路径（可选）
--output : 输出目录（可选）
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console
)
from src.utils.io import save_numpy, load_numpy, ensure_dir


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="EEG数据预处理",
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
        help="输入数据路径 (npy文件)",
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help="标签数据路径 (可选)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录",
    )
    
    parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="采样频率 (Hz)，覆盖配置文件",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print_panel(
        "EEG数据预处理\n\n"
        f"配置文件: {args.config}\n"
        f"输入数据: {args.data}",
        title="预处理脚本",
        style="blue"
    )
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载数据
        console.print(f"\n[bold]加载数据: {args.data}[/bold]")
        data = load_numpy(args.data)
        console.print(f"[dim]数据形状: {data.shape}[/dim]")
        
        # 加载标签（如果有）
        labels = None
        if args.labels:
            console.print(f"[bold]加载标签: {args.labels}[/bold]")
            labels = load_numpy(args.labels)
            console.print(f"[dim]标签形状: {labels.shape}[/dim]")
        
        # 创建流水线
        pipeline = PreprocessingPipeline(config)
        
        # 执行预处理
        sfreq = args.sfreq or config.preprocessing.sampling_rate
        cleaned_data, cleaned_labels, stats = pipeline.run(data, labels, sfreq)
        
        # 确定输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(args.data).parent / "preprocessed"
        ensure_dir(output_dir)
        
        # 保存结果
        output_data_path = output_dir / "preprocessed_data.npy"
        save_numpy(cleaned_data, output_data_path)
        
        if cleaned_labels is not None:
            output_labels_path = output_dir / "preprocessed_labels.npy"
            save_numpy(cleaned_labels, output_labels_path)
        
        # 保存统计信息
        import json
        stats_path = output_dir / "preprocess_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json_stats = {
                "original_shape": list(stats["original_shape"]),
                "final_shape": list(stats["final_shape"]),
                "n_removed_trials": stats["n_removed_trials"],
                "bad_trial_indices": stats["bad_trial_indices"],
            }
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print_success(f"预处理完成！结果已保存到: {output_dir}")
        
    except Exception as e:
        print_error(f"预处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

