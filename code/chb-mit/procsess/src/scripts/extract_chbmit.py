# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB-MIT 数据提取脚本 (Step 1/6)

功能：
从原始 EDF 文件或预处理文件提取 EEG 数据段，
并按患者保存为 x_data.npy 和 y_labels.npy。
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.chbmit_loader import CHBMITLoader
from src.utils.console import console, print_error, print_success

def main():
    parser = argparse.ArgumentParser(description="CHB-MIT 数据提取")
    parser.add_argument("--config", type=str, default="configs/data_extraction/chbmit.yaml", help="配置文件路径")
    parser.add_argument("--patients", nargs="+", default=None, help="指定要处理的患者ID (如 chb01 chb02)")
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
    
    # 如果指定了患者，覆盖配置
    if args.patients:
        if "data" not in config:
            config["data"] = {}
        config["data"]["patients"] = args.patients
        
    loader = CHBMITLoader(config)
    
    # 调用正确的方法：extract_all 而不是 process_all_patients
    try:
        x_path, y_path, pid_path, ch_path, n_samples = loader.extract_all(
            output_dir=None,  # 使用配置文件中的 output.base_dir
            batch_size=50,
            show_progress=True
        )
        
        if n_samples > 0:
            print_success(f"数据提取完成！")
            print_success(f"  数据文件: {x_path}")
            print_success(f"  标签文件: {y_path}")
            print_success(f"  患者ID文件: {pid_path}")
            print_success(f"  通道索引文件: {ch_path}")
            print_success(f"  总样本数: {n_samples}")
        else:
            print_error("数据提取失败：没有生成任何样本")
            
    except Exception as e:
        print_error(f"数据提取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()