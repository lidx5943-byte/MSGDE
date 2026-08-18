#!/usr/bin/env python3
"""
ML Classification Script
=========================

Run traditional ML classification (KNN and Random Forest) with cross-validation.

Usage
-----
python run_ml_classification.py --features X_features.npy --labels y_labels.npy

Arguments
---------
--features : Path to feature matrix (X_features.npy)
--labels : Path to labels (y_labels.npy)
--output : Output directory
--cv-folds : Number of CV folds (default: 10)
--knn-k : KNN k parameter (default: 5)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.ml.classification import run_ml_classification
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console
)
from src.utils.io import load_numpy, ensure_dir, Experiment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run traditional ML classification",
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
        help="Path to feature matrix (X_features.npy)",
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=str,
        required=True,
        help="Path to labels (y_labels.npy)",
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
        default="ml_classification",
        help="Experiment name",
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of cross-validation folds",
    )
    
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="KNN k parameter",
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print_panel(
        f"ML Classification\n\n"
        f"Features: {args.features}\n"
        f"Labels: {args.labels}\n"
        f"CV Folds: {args.cv_folds}\n"
        f"KNN k: {args.knn_k}",
        title="ML Classifier",
        style="blue"
    )
    
    try:
        # Create experiment (only create necessary directories)
        exp = Experiment(
            name=args.name,
            base_dir=args.output,
            config_path=args.config,
            description="Traditional ML Classification",
            create_all_dirs=False  # Only create necessary directories
        )
        
        exp.log_command(sys.argv)
        exp.log_input("features", args.features)
        exp.log_input("labels", args.labels)
        
        # Load data
        console.print(f"\n[bold]Loading features: {args.features}[/bold]")
        X = load_numpy(args.features)
        console.print(f"[dim]Features shape: {X.shape}[/dim]")
        
        console.print(f"\n[bold]Loading labels: {args.labels}[/bold]")
        y = load_numpy(args.labels)
        if y.ndim > 1:
            y = y.flatten()
        console.print(f"[dim]Labels shape: {y.shape}[/dim]")
        
        # Load config for class names if available
        config = load_config(args.config)
        class_names = None
        if hasattr(config, 'ml') and hasattr(config.ml, 'classification'):
            if hasattr(config.ml.classification, 'class_names'):
                class_names = config.ml.classification.class_names
        
        # Run classification
        results = run_ml_classification(
            X, y,
            output_dir=exp.root_dir / "ml_classification",
            cv_folds=args.cv_folds,
            knn_k=args.knn_k,
            random_state=args.random_state,
            class_names=class_names,
            save_figures=True,
        )
        
        # Log outputs
        exp.log_output("confusion_matrix_knn", exp.root_dir / "ml_classification" / "confusion_matrix_knn.png")
        exp.log_output("confusion_matrix_rf", exp.root_dir / "ml_classification" / "confusion_matrix_rf.png")
        exp.log_output("classification_report", exp.root_dir / "ml_classification" / "classification_report.txt")
        
        # Log metrics
        exp.log_metric("knn_cv_mean", results['knn']['metrics']['cv_mean'])
        exp.log_metric("knn_cv_std", results['knn']['metrics']['cv_std'])
        exp.log_metric("rf_cv_mean", results['random_forest']['metrics']['cv_mean'])
        exp.log_metric("rf_cv_std", results['random_forest']['metrics']['cv_std'])
        exp.log_metric("n_classes", results['n_classes'])
        
        # Finish experiment
        exp.finish("completed")
        
        print_panel(
            f"Classification completed!\n\n"
            f"Output: {exp.root_dir / 'ml_classification'}\n"
            f"KNN CV Accuracy: {results['knn']['metrics']['cv_mean']:.4f} ± {results['knn']['metrics']['cv_std']:.4f}\n"
            f"RF CV Accuracy: {results['random_forest']['metrics']['cv_mean']:.4f} ± {results['random_forest']['metrics']['cv_std']:.4f}",
            title="Success",
            style="bold green"
        )
        
    except Exception as e:
        print_error(f"Classification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

