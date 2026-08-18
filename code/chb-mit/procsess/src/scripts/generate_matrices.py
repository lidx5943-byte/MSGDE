# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矩阵生成入口脚本
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.matrix_pipeline import run_matrix_generation_pipeline
from src.utils.console import console, print_error

def main():
    parser = argparse.ArgumentParser(description="Step 2: 矩阵生成")
    parser.add_argument("--config", type=str, default="configs/matrix_generation.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        # 尝试相对于项目根目录查找
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / args.config
        
    if not config_path.exists():
        print_error(f"配置文件未找到: {args.config}")
        return
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 还可以加载 base.yaml 进行合并 (如果需要)
        # 这里简化处理，假设 config 文件是完整的或者不需要 base
        # 实际上我们的设计是在 pipeline 内部可能会做合并，或者在这里做
        # 暂时只加载指定的 config
            
    except Exception as e:
        print_error(f"读取配置文件失败: {e}")
        return
        
    run_matrix_generation_pipeline(config)

if __name__ == "__main__":
    main()
