# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耦合强度×尺度 联合消融实验入口脚本

    python3 -m src.scripts.run_coupling_scale_ablation --config configs/coupling_scale_ablation.yaml
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.coupling_scale_pipeline import run_coupling_scale_ablation_pipeline
from src.utils.console import console, print_error

def main():
    parser = argparse.ArgumentParser(description="耦合强度×尺度 联合消融实验")
    parser.add_argument("--config", type=str, default="configs/coupling_scale_ablation.yaml", help="配置文件路径")
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
    run_coupling_scale_ablation_pipeline(config)

if __name__ == "__main__":
    main()
