# Author: 王梓涵 <wangzh011031@163.com>
import pandas as pd
import yaml
from pathlib import Path
from src.visualization.coupling_scale_plotter import plot_coupling_scale_heatmap
from src.utils.console import console, print_header, print_success

def main():
    print_header("快速绘图: 耦合强度×尺度结果")
    
    # 路径配置
    config_path = "configs/coupling_scale_ablation.yaml"
    results_path = "/mnt/3M/chbmit-allchannels/coupling_scale_ablation/results_perturbation_12d.csv"
    output_dir = Path("/mnt/3M/chbmit-allchannels/coupling_scale_ablation")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    if not Path(results_path).exists():
        console.print(f"[red]错误: 结果文件不存在 {results_path}[/red]")
        return
    
    df = pd.read_csv(results_path)
    console.print(f"成功加载数据: {len(df)} 条记录")
    
    # 确保数据类型正确并排序
    df['Scale'] = df['Scale'].astype(int)
    df['CouplingStrength'] = df['CouplingStrength'].astype(float)
    df = df.sort_values(['Scale', 'CouplingStrength'])
    
    # 调用绘图逻辑
    console.print("正在生成可视化图表...")
    plot_coupling_scale_heatmap(df, output_dir, config)
    
    print_success(f"绘图完成！图片已保存至: {output_dir}")

if __name__ == "__main__":
    main()
