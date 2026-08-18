# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Summary CSV directly from existing patient results.
"""

import os
import json
import pandas as pd
from pathlib import Path
from src.utils.console import console, print_success, print_error

def generate_summary_only(output_dir_str: str):
    output_dir = Path(output_dir_str)
    if not output_dir.exists():
        print_error(f"Output directory not found: {output_dir}")
        return

    # Scan for patient result files
    patients = [p for p in output_dir.iterdir() if p.is_dir()]
    patients.sort()
    
    df_data = []
    
    print(f"Scanning {len(patients)} patient directories in {output_dir}...")
    
    for p_dir in patients:
        res_json = p_dir / "result_metrics.json"
        if not res_json.exists():
            continue
            
        try:
            with open(res_json, "r") as f:
                res = json.load(f)
                
            # Flatten dict
            flat_entry = {"patient_id": res.get("patient_id", p_dir.name)}
            
            # Basic stats
            flat_entry["n_samples"] = res.get("n_samples", 0)
            flat_entry["n_class0"] = res.get("n_class0", 0)
            flat_entry["n_class1"] = res.get("n_class1", 0)
            
            # Classifier Results
            clf_res = res.get("classifier_results", {})
            for exp_key, metrics in clf_res.items():
                for m_key, m_val in metrics.items():
                    if isinstance(m_val, (int, float)):
                        flat_entry[f"{exp_key}_{m_key}"] = m_val
                        
            df_data.append(flat_entry)
            
        except Exception as e:
            print(f"Error reading {res_json}: {e}")

    if not df_data:
        print_error("No valid data found.")
        return

    df = pd.DataFrame(df_data)
    
    # Calculate Mean Row (Append to bottom)
    # Filter numeric columns that are likely metrics (exclude n_samples etc for mean if desired, or keep all)
    # Usually we want mean of metrics.
    
    numeric_cols = [c for c in df.columns if c not in ["patient_id"]]
    mean_row = df[numeric_cols].mean()
    mean_row["patient_id"] = "AVERAGE"
    
    # Append mean row
    df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Move patient_id to front
    cols = ["patient_id"] + [c for c in df_final.columns if c != "patient_id"]
    df_final = df_final[cols]

    csv_path = output_dir / "summary.csv"
    df_final.to_csv(csv_path, index=False)
    print_success(f"Summary generated with averages: {csv_path}")
    
    # Optional: Transpose for easier reading of Model Averages
    # Create a separate summary of just averages per model
    avg_data = []
    for col in df.columns:
        if "_" in col and col not in ["patient_id", "n_samples", "n_class0", "n_class1"]:
             parts = col.split("_")
             # Try to parse Exp_Model_Metric
             # Example: ALL_RF_Accuracy
             # But Exp could be "S+T"...
             # Strategy: Last part is Metric. Rest is Model/Exp.
             metric = parts[-1]
             model_exp = "_".join(parts[:-1])
             avg_val = df[col].mean()
             avg_data.append({"Model_Exp": model_exp, "Metric": metric, "Value": avg_val})
             
    if avg_data:
        df_avg = pd.DataFrame(avg_data)
        # Pivot: Model_Exp as Index, Metric as Columns
        df_avg_pivot = df_avg.pivot(index="Model_Exp", columns="Metric", values="Value")
        avg_csv_path = output_dir / "summary_model_averages.csv"
        df_avg_pivot.to_csv(avg_csv_path)
        print_success(f"Model averages summary: {avg_csv_path}")

if __name__ == "__main__":
    generate_summary_only("/mnt/3M/chbmit-allchannels/per_patient_results")
