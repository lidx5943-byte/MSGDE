#!/usr/bin/env python3
"""
完整流水线脚本
=============

执行完整的EEG动力学分析流程：
1. 数据预处理
2. 相似度矩阵计算
3. 拉普拉斯特征提取
4. 动力学分析
5. 可视化（在动力学分析后立即执行，便于早期查看结果）
6. ML数据转换
7. ML分类评估

使用方法
--------
python run_full_pipeline.py --config config/config.yaml --data data/eeg_data.npy

参数说明
--------
--config : 配置文件路径
--data : 输入数据路径
--labels : 标签数据路径（可选）
--output : 输出目录（可选）
--name : 实验名称（可选）
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.similarity.correlation import compute_similarity_matrix
from src.laplacian.multiscale import LaplacianPipeline
from src.dynamics.pipeline import DynamicsAnalysisPipeline
from src.visualization.trajectories import plot_phase_space, plot_butterfly_attractor
from src.visualization.diversity import analyze_all_scales
from src.ml.data_converter import MLDataConverter
from src.ml.classification import run_ml_classification
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console, print_step
)
from src.utils.io import save_numpy, load_numpy, ensure_dir, Experiment
from src.utils.timer import Timer, TimerGroup


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="EEG动力学分析完整流水线",
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
        help="标签数据路径（可选）",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="输出基础目录",
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="eeg_dynamics",
        help="实验名称",
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="并行作业数",
    )
    
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过预处理步骤（假设输入已预处理）",
    )
    
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="跳过可视化步骤",
    )
    
    parser.add_argument(
        "--skip-diversity",
        action="store_true",
        help="跳过特征差异性分析（仅跳过特征差异性，保留其他可视化）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建实验
    exp = Experiment(
        name=args.name,
        base_dir=args.output,
        config_path=args.config,
        description="EEG动力学分析完整流水线"
    )
    
    # 记录运行命令
    exp.log_command(sys.argv)
    
    # 记录输入文件
    exp.log_input("data", args.data)
    if args.labels:
        exp.log_input("labels", args.labels)
    
    timers = TimerGroup("完整流水线")
    timers.start_total()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # =====================================
        # 步骤1：加载数据
        # =====================================
        print_step(1, 8, "加载数据")
        
        with timers.timer("数据加载"):
            data = load_numpy(args.data)
            console.print(f"[dim]数据形状: {data.shape}[/dim]")
            
            labels = None
            if args.labels:
                labels = load_numpy(args.labels)
                console.print(f"[dim]标签形状: {labels.shape}[/dim]")
        
        # =====================================
        # 步骤2：数据预处理
        # =====================================
        if not args.skip_preprocess:
            print_step(2, 8, "数据预处理")
            
            with timers.timer("预处理"):
                preproc_pipeline = PreprocessingPipeline(config)
                data, labels, preproc_stats = preproc_pipeline.run(
                    data, labels, config.preprocessing.sampling_rate
                )
            
            # 保存预处理结果
            save_numpy(data, exp.output_dir / "preprocessed_data.npy")
            exp.log_output("preprocessed_data", exp.output_dir / "preprocessed_data.npy")
            
            if labels is not None:
                save_numpy(labels, exp.output_dir / "preprocessed_labels.npy")
        else:
            console.print("[dim]跳过预处理步骤[/dim]")
        
        # =====================================
        # 步骤3：相似度矩阵计算
        # =====================================
        print_step(3, 8, "相似度矩阵计算")
        
        with timers.timer("相似度计算"):
            similarity_matrix = compute_similarity_matrix(
                data, method=config.similarity.method
            )
            save_numpy(
                similarity_matrix,
                exp.output_dir / f"similarity_{config.similarity.method}.npy"
            )
            exp.log_output("similarity_matrix", exp.output_dir / f"similarity_{config.similarity.method}.npy")
        
        # =====================================
        # 步骤4：拉普拉斯特征提取
        # =====================================
        print_step(4, 8, "拉普拉斯特征提取")
        
        with timers.timer("拉普拉斯特征"):
            laplacian_pipeline = LaplacianPipeline(
                config, output_dir=str(exp.output_dir)
            )
            laplacian_family = laplacian_pipeline.run(similarity_matrix)
            exp.log_output("laplacian_family", exp.output_dir / "laplacian_family.npy")
        
        # =====================================
        # 步骤5：动力学分析
        # =====================================
        print_step(5, 8, "动力学分析")
        
        with timers.timer("动力学分析"):
            # 动力学分析使用实验的输出目录
            dynamics_pipeline = DynamicsAnalysisPipeline(
                config, output_dir=str(exp.root_dir)
            )
            dynamics_data, trajectories = dynamics_pipeline.run(
                laplacian_family,
                n_scales=config.dynamics.analysis.n_scales,
                n_jobs=args.n_jobs,
            )
            exp.log_output("dynamics_data", exp.output_dir / "dynamics_data.npy")
            exp.log_output("trajectories", exp.output_dir / "trajectories.npy")
        
        # =====================================
        # 步骤6：可视化（在动力学分析后立即执行）
        # =====================================
        if not args.skip_visualize:
            print_step(6, 8, "可视化")
            
            with timers.timer("可视化"):
                # 相空间图
                if trajectories.ndim == 4:
                    traj_for_viz = trajectories[0]  # 第一个尺度
                else:
                    traj_for_viz = trajectories
                
                plot_phase_space(
                    traj_for_viz,
                    node_idx=0,
                    save_path=str(exp.figures_dir / "phase_space.png"),
                )
                
                # 蝴蝶图
                plot_butterfly_attractor(
                    traj_for_viz,
                    figures_dir=str(exp.figures_dir),
                    step_interval=config.visualization.trajectory.uniform_interval,
                )
                
                # 特征差异性
                if not args.skip_diversity and trajectories.ndim == 4:
                    analyze_all_scales(
                        trajectories,
                        figures_dir=str(exp.figures_dir),
                        data_dir=str(exp.output_dir),
                        step_interval=config.visualization.feature_diversity.step_interval,
                        method=config.visualization.feature_diversity.method,
                    )
                elif args.skip_diversity:
                    console.print("[dim]跳过特征差异性分析[/dim]")
        else:
            console.print("[dim]跳过可视化步骤[/dim]")
        
        # =====================================
        # 步骤7：ML数据转换
        # =====================================
        print_step(7, 8, "ML数据转换")
        
        with timers.timer("ML数据转换"):
            # 创建ML数据转换器
            ml_converter = MLDataConverter(exp.root_dir)
            
            if labels is not None:
                # 生成传统ML数据
                X_ml, y_ml = ml_converter.generate_ml_data(
                    dynamics_data, labels, shuffle=True, random_state=42
                )
                exp.log_output("ml_features", ml_converter.ml_dir / "X_features.npy")
                exp.log_output("ml_labels", ml_converter.ml_dir / "y_labels.npy")
                exp.log_output("ml_all_data", ml_converter.ml_dir / "all_data.npy")
                
                # 生成Transformer格式数据（不打乱顺序，保持与原始数据一致）
                X_tf, y_tf = ml_converter.convert_to_transformer_format(
                    trajectories, labels, shuffle=False, random_state=42
                )
                exp.log_output("transformer_data", ml_converter.transformer_dir / "X_transformer.npy")
                exp.log_output("transformer_labels", ml_converter.transformer_dir / "y_labels.npy")
                
                # 记录ML相关指标
                exp.log_metric("ml_feature_shape", list(X_ml.shape))
                exp.log_metric("transformer_shape", list(X_tf.shape))
                exp.log_metric("n_classes", len(np.unique(labels)))
            else:
                console.print("[yellow]未提供标签，跳过ML数据转换[/yellow]")
        
        # =====================================
        # 步骤8：ML分类评估
        # =====================================
        if labels is not None:
            print_step(8, 8, "ML分类评估")
            
            with timers.timer("ML分类"):
                # 获取类别名称配置（如果有）
                class_names = None
                if hasattr(config, 'ml') and hasattr(config.ml, 'classification'):
                    if hasattr(config.ml.classification, 'class_names'):
                        class_names = config.ml.classification.class_names
                
                # 运行分类评估
                classification_results = run_ml_classification(
                    X_ml, y_ml,
                    output_dir=exp.root_dir / "ml_classification",
                    cv_folds=10,
                    knn_k=5,
                    random_state=42,
                    class_names=class_names,
                    save_figures=True,
                )
                
                # 记录分类结果
                exp.log_output("confusion_matrix_knn", exp.root_dir / "ml_classification" / "confusion_matrix_knn.png")
                exp.log_output("confusion_matrix_rf", exp.root_dir / "ml_classification" / "confusion_matrix_rf.png")
                exp.log_output("classification_report", exp.root_dir / "ml_classification" / "classification_report.txt")
                
                # 记录分类指标
                exp.log_metric("knn_cv_mean", classification_results['knn']['metrics']['cv_mean'])
                exp.log_metric("knn_cv_std", classification_results['knn']['metrics']['cv_std'])
                exp.log_metric("rf_cv_mean", classification_results['random_forest']['metrics']['cv_mean'])
                exp.log_metric("rf_cv_std", classification_results['random_forest']['metrics']['cv_std'])
        else:
            console.print("[yellow]未提供标签，跳过ML分类评估[/yellow]")
        
        timers.stop_total()
        
        # 记录指标
        exp.log_metric("data_shape", list(data.shape))
        exp.log_metric("feature_shape", list(dynamics_data.shape))
        exp.log_metric("trajectory_shape", list(trajectories.shape))
        exp.log_metric("total_time_seconds", timers.total_elapsed)
        
        timers.report()
        
        # 完成实验
        exp.finish("completed")
        
        # 构建输出信息
        output_info = [
            f"实验目录: {exp.root_dir}",
            "",
            f"📊 特征矩阵: {dynamics_data.shape}",
            f"🌀 轨迹数据: {trajectories.shape}",
        ]
        
        if labels is not None:
            output_info.append(f"🤖 ML数据: {ml_converter.ml_dir}")
            output_info.append(f"🔮 Transformer数据: {ml_converter.transformer_dir}")
            if 'classification_results' in locals():
                output_info.append(f"📊 KNN准确率: {classification_results['knn']['metrics']['cv_mean']:.4f} ± {classification_results['knn']['metrics']['cv_std']:.4f}")
                output_info.append(f"🌲 RF准确率: {classification_results['random_forest']['metrics']['cv_mean']:.4f} ± {classification_results['random_forest']['metrics']['cv_std']:.4f}")
        
        output_info.append(f"⏱️ 总耗时: {timers.total_elapsed:.1f} 秒")
        
        print_panel(
            "\n".join(output_info),
            title="✅ 流水线执行成功",
            style="bold green"
        )
        
    except Exception as e:
        exp.finish("failed")
        print_error(f"流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
