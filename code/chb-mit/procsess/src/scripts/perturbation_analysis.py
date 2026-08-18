# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微扰分析入口脚本 (Step 4)
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.perturbation_pipeline import run_perturbation_pipeline
from src.utils.console import console, print_error

def main():
    parser = argparse.ArgumentParser(description="Step 4: 微扰分析")
    parser.add_argument("--config", type=str, default="configs/perturbation.yaml", help="配置文件路径")
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
        
    run_perturbation_pipeline(config)

if __name__ == "__main__":
    main()
