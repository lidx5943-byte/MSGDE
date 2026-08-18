# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bonn 数据提取/转换脚本
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.bonn_loader import BonnLoader
from src.utils.console import console, print_error

def main():
    parser = argparse.ArgumentParser(description="Bonn 数据提取")
    parser.add_argument("--config", type=str, default="configs/data_extraction/bonn.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / args.config
        
    if not config_path.exists():
        print_error(f"配置文件未找到: {args.config}")
        return
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print_error(f"读取配置文件失败: {e}")
        return
        
    # BonnLoader 主要是加载器，如果需要将其转存为统一格式供 pipeline 使用
    # 我们可以在这里实例化 loader 并执行保存逻辑
    
    # 假设 BonnLoader 逻辑不同，仅供 pipeline 使用，这里暂时只是作为测试入口
    # 或者如果我们需要预处理 Bonn 数据，应该在 BonnLoader 中添加 process_and_save 方法
    # 目前 BonnLoader 只有 loads raw npy funcs.
    
    console.print("[yellow]BonnLoader 主要用于直接加载现有 .npy 数据集。 此脚本仅验证加载功能。[/yellow]")
    loader = BonnLoader(config)
    
    try:
        x_train, y_train, x_test, y_test = loader.load_data()
        console.print(f"[green]✓ 数据加载成功:[/green]")
        console.print(f"  Train: x={x_train.shape}, y={y_train.shape}")
        console.print(f"  Test:  x={x_test.shape}, y={y_test.shape}")
        
    except Exception as e:
        console.print(f"[red]数据加载失败: {e}[/red]")

if __name__ == "__main__":
    main()
