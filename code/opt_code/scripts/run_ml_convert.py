#!/usr/bin/env python3
"""
ML Data Conversion Script
=========================

Convert dynamics analysis outputs to ML-ready formats:
- ML format: Features + Labels for traditional classifiers (SVM, RF, KNN, etc.)
- Transformer format: Sequence data for transformer models

Usage
-----
python run_ml_convert.py --features dynamics_data.npy --trajectories trajectories.npy --labels labels.npy

Arguments
---------
--features : Path to dynamics features (dynamics_data.npy)
--trajectories : Path to trajectory data (trajectories.npy)
--labels : Path to labels
--output : Output directory
--format : Output format (ml, transformer, all)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.ml.data_converter import MLDataConverter, generate_ml_data, convert_to_transformer_format
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console
)
from src.utils.io import load_numpy, ensure_dir, Experiment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert dynamics data to ML-ready formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Config file path",
    )
    
    parser.add_argument(
        "--features", "-f",
        type=str,
        required=True,
        help="Path to dynamics features (dynamics_data.npy)",
    )
    
    parser.add_argument(
        "--trajectories", "-t",
        type=str,
        default=None,
        help="Path to trajectory data (trajectories.npy), required for transformer format",
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help="Path to labels file (optional if defined in config)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output base directory",
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="ml_data",
        help="Experiment name",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["ml", "transformer", "all"],
        help="Output format: ml, transformer, or all",
    )
    
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle data",
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print_panel(
        f"ML Data Conversion\n\n"
        f"Features: {args.features}\n"
        f"Labels: {args.labels}\n"
        f"Format: {args.format}",
        title="ML Data Converter",
        style="blue"
    )
    
    try:
        # Create experiment
        exp = Experiment(
            name=args.name,
            base_dir=args.output,
            config_path=args.config,
            description="ML Data Conversion"
        )
        
        exp.log_command(sys.argv)
        exp.log_input("features", args.features)
        
        # Determine labels path
        labels_path = args.labels
        config = load_config(args.config)
        
        if labels_path is None:
            if hasattr(config, 'paths') and hasattr(config.paths, 'labels'):
                labels_path = config.paths.labels
                console.print(f"[dim]Using labels from config: {labels_path}[/dim]")
            else:
                print_error("Labels path not provided in args or config")
                sys.exit(1)
        
        exp.log_input("labels", labels_path)
        if args.trajectories:
            exp.log_input("trajectories", args.trajectories)
        
        # Load data
        console.print(f"\n[bold]Loading features: {args.features}[/bold]")
        features = load_numpy(args.features)
        console.print(f"[dim]Features shape: {features.shape}[/dim]")
        
        console.print(f"\n[bold]Loading labels: {labels_path}[/bold]")
        # Load existing labels directly instead of generating them
        labels = load_numpy(labels_path)
        if labels.ndim > 1:
            labels = labels.flatten()
        console.print(f"[dim]Labels shape: {labels.shape}[/dim]")
        
        # Create converter
        converter = MLDataConverter(exp.root_dir)
        
        shuffle = not args.no_shuffle
        
        # Generate ML data
        if args.format in ["ml", "all"]:
            X_ml, y_ml = converter.generate_ml_data(
                features, labels, shuffle, args.random_state
            )
            exp.log_output("ml_features", converter.ml_dir / "X_features.npy")
            exp.log_output("ml_labels", converter.ml_dir / "y_labels.npy")
            exp.log_output("ml_all_data", converter.ml_dir / "all_data.npy")
            exp.log_metric("ml_feature_shape", list(X_ml.shape))
        
        # Generate Transformer data
        if args.format in ["transformer", "all"]:
            if args.trajectories is None:
                print_error("Trajectory file required for transformer format. Use --trajectories")
                if args.format == "transformer":
                    sys.exit(1)
            else:
                console.print(f"\n[bold]Loading trajectories: {args.trajectories}[/bold]")
                trajectories = load_numpy(args.trajectories)
                console.print(f"[dim]Trajectories shape: {trajectories.shape}[/dim]")
                
                X_tf, y_tf = converter.convert_to_transformer_format(
                    trajectories, labels, shuffle, args.random_state
                )
                exp.log_output("transformer_data", converter.transformer_dir / "X_transformer.npy")
                exp.log_output("transformer_labels", converter.transformer_dir / "y_labels.npy")
                exp.log_metric("transformer_shape", list(X_tf.shape))
        
        # Finish experiment
        exp.finish("completed")
        
        print_panel(
            f"Conversion completed!\n\n"
            f"Output: {exp.root_dir}\n"
            f"ML data: {converter.ml_dir}\n"
            f"Transformer data: {converter.transformer_dir}",
            title="Success",
            style="bold green"
        )
        
    except Exception as e:
        print_error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

