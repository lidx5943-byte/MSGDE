# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共识性分析入口脚本

用法:
    python3 -m src.scripts.run_consensus_analysis --config configs/consensus_analysis.yaml
"""

import sys
import os
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到 pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.consensus_pipeline import run_consensus_analysis_pipeline
from src.utils.console import print_error, print_info

def get_best_model(summary_path: Path):
    """从 summary.csv 中找到 ALL 被选特征下准确率最高的模型"""
    if not summary_path.exists():
        print_error(f"Summary file not found: {summary_path}")
        return "RF"
        
    df = pd.read_csv(summary_path)
    avg_row = df[df['patient_id'] == 'AVERAGE']
    if avg_row.empty:
        numeric_df = df.select_dtypes(include=[np.number])
        avg_data = numeric_df.mean()
    else:
        avg_data = avg_row.iloc[0]
        
    models = ['SVM', 'RF', 'KNN', 'LR', 'GBDT']
    best_model = "RF"
    max_acc = -1
    
    for model in models:
        col = f"ALL_{model}_Accuracy"
        if col in avg_data:
            acc = avg_data[col]
            if acc > max_acc:
                max_acc = acc
                best_model = model
                
    return best_model

def main():
    parser = argparse.ArgumentParser(description="共识性分析 (Consensus Analysis)")
    parser.add_argument("--config", type=str, default="configs/consensus_analysis.yaml", help="配置文件路径")
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
    
    # 自动选择最佳模型
    summary_path = Path("/mnt/3M/chbmit-allchannels/per_patient_results/summary.csv")
    best_model = get_best_model(summary_path)
    print_info(f"自动识别的最佳模型: {best_model}")
    
    # 更新配置
    if "classification" not in config:
        config["classification"] = {}
    config["classification"]["model"] = best_model
    
    # 如果输出路径没设，设一个独特的
    if "output" not in config:
        config["output"] = {}
    config["output"]["save_dir"] = f"./consensus_analysis_{best_model.lower()}"
        
    run_consensus_analysis_pipeline(config)

if __name__ == "__main__":
    main()
