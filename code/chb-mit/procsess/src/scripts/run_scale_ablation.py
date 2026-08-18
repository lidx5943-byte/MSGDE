# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尺度消融实验脚本入口

用法:
    python3 -m src.scripts.run_scale_ablation --config configs/scale_ablation.yaml
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.scale_ablation_pipeline import run_scale_ablation_pipeline
from src.utils.console import console, print_error

def main():
    parser = argparse.ArgumentParser(description="尺度消融实验: 评估每个独立尺度的性能")
    parser.add_argument("--config", type=str, default="configs/scale_ablation.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    # 尝试在相对路径查找
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
        
    # 运行流水线
    run_scale_ablation_pipeline(config)

if __name__ == "__main__":
    main()
