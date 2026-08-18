"""
Traditional ML Classification Module
====================================

Provides traditional machine learning classification with cross-validation:
- KNN classifier
- Random Forest classifier
- Confusion matrix visualization
- Classification reports

Usage
-----
>>> from src.ml.classification import run_ml_classification
>>> 
>>> results = run_ml_classification(
>>>     X, y, 
>>>     output_dir="output/ml_classification",
>>>     cv_folds=10,
>>>     random_state=42
>>> )
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from datetime import datetime

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..utils.logger import (
    get_logger, console, print_header, print_success, 
    print_warning, print_table, print_panel
)
from ..utils.io import ensure_dir

logger = get_logger(__name__)

# Set matplotlib backend for non-interactive environments
plt.switch_backend('Agg')


def run_ml_classification(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Union[str, Path],
    cv_folds: int = 10,
    knn_k: int = 5,
    random_state: int = 42,
    class_names: Optional[Dict[int, str]] = None,
    save_figures: bool = True,
) -> Dict[str, Any]:
    """
    Run traditional ML classification with cross-validation
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Label array, shape (n_samples,)
    output_dir : str or Path
        Output directory for results
    cv_folds : int
        Number of cross-validation folds
    knn_k : int
        Number of neighbors for KNN
    random_state : int
        Random seed
    class_names : dict, optional
        Mapping from class index to class name
    save_figures : bool
        Whether to save confusion matrix figures
        
    Returns
    -------
    Dict[str, Any]
        Classification results including scores, predictions, and reports
    """
    print_header("Traditional ML Classification")
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    # Validate inputs
    if X.shape[0] != len(y):
        raise ValueError(f"Sample count mismatch: X={X.shape[0]}, y={len(y)}")
    
    if y.ndim > 1:
        y = y.flatten()
    
    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Setup class names
    if class_names is None:
        class_names = {int(cls): f"Class {cls}" for cls in unique_classes}
    else:
        # Convert keys to int
        class_names = {int(k): str(v) for k, v in class_names.items()}
        # Fill missing classes
        for cls in unique_classes:
            if int(cls) not in class_names:
                class_names[int(cls)] = f"Class {cls}"
    
    # Print data statistics
    class_distribution = {cls: np.sum(y == cls) for cls in unique_classes}
    
    rows = [
        ["Samples", X.shape[0]],
        ["Features", X.shape[1]],
        ["Classes", n_classes],
        ["CV Folds", cv_folds],
    ]
    for cls in sorted(class_distribution.keys()):
        count = class_distribution[cls]
        pct = 100 * count / len(y)
        class_name = class_names.get(int(cls), f"Class {cls}")
        rows.append([f"  {class_name}", f"{count} ({pct:.1f}%)"])
    print_table("Classification Data Statistics", ["Item", "Value"], rows)
    
    # Create stratified K-fold CV
    console.print(f"\n[dim]Using {cv_folds}-fold stratified cross-validation...[/dim]")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Get last fold for confusion matrix
    cv_for_matrix = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    train_idx, test_idx = list(cv_for_matrix.split(X, y))[-1]
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    results = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "cv_folds": cv_folds,
        "class_distribution": {int(k): int(v) for k, v in class_distribution.items()},
        "class_names": class_names,
    }
    
    # ========== KNN Classification ==========
    console.print("\n[bold cyan]KNN Classifier[/bold cyan]")
    console.print(f"[dim]k={knn_k}[/dim]")
    
    # Use pipeline with scaler for KNN to handle feature scaling issues
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=knn_k))
    # knn = KNeighborsClassifier(n_neighbors=knn_k)
    knn_scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Train and predict for confusion matrix
    knn.fit(X_train_fold, y_train_fold)
    y_pred_knn = knn.predict(X_test_fold)
    
    # Calculate metrics
    knn_metrics = {
        "cv_scores": knn_scores.tolist(),
        "cv_mean": float(knn_scores.mean()),
        "cv_std": float(knn_scores.std()),
        "cv_min": float(knn_scores.min()),
        "cv_max": float(knn_scores.max()),
        "test_accuracy": float(accuracy_score(y_test_fold, y_pred_knn)),
        "test_precision": float(precision_score(y_test_fold, y_pred_knn, average='weighted', zero_division=0)),
        "test_recall": float(recall_score(y_test_fold, y_pred_knn, average='weighted', zero_division=0)),
        "test_f1": float(f1_score(y_test_fold, y_pred_knn, average='weighted', zero_division=0)),
    }
    
    # Print results
    rows = [
        ["CV Mean Accuracy", f"{knn_metrics['cv_mean']:.4f} ± {knn_metrics['cv_std']:.4f}"],
        ["CV Min", f"{knn_metrics['cv_min']:.4f}"],
        ["CV Max", f"{knn_metrics['cv_max']:.4f}"],
        ["Test Accuracy", f"{knn_metrics['test_accuracy']:.4f}"],
        ["Test F1-Score", f"{knn_metrics['test_f1']:.4f}"],
    ]
    print_table("KNN Results", ["Metric", "Value"], rows)
    
    # Confusion matrix
    cm_knn = confusion_matrix(y_test_fold, y_pred_knn, labels=unique_classes)
    class_labels = [class_names.get(int(cls), f"Class {cls}") for cls in unique_classes]
    
    if save_figures:
        _plot_confusion_matrix(
            cm_knn, class_labels, 
            f"KNN Confusion Matrix (k={knn_k}, {cv_folds}-fold CV)",
            output_dir / "confusion_matrix_knn.png",
            cmap='Blues'
        )
    
    # Classification report
    report_knn = classification_report(y_test_fold, y_pred_knn, output_dict=True, zero_division=0)
    results["knn"] = {
        "metrics": knn_metrics,
        "confusion_matrix": cm_knn.tolist(),
        "classification_report": report_knn,
    }
    
    # ========== Random Forest Classification ==========
    console.print("\n[bold cyan]Random Forest Classifier[/bold cyan]")
    
    # Use n_jobs=1 to ensure reproducibility consistent with old version
    rf = RandomForestClassifier(random_state=random_state)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Train and predict for confusion matrix
    rf.fit(X_train_fold, y_train_fold)
    y_pred_rf = rf.predict(X_test_fold)
    
    # Calculate metrics
    rf_metrics = {
        "cv_scores": rf_scores.tolist(),
        "cv_mean": float(rf_scores.mean()),
        "cv_std": float(rf_scores.std()),
        "cv_min": float(rf_scores.min()),
        "cv_max": float(rf_scores.max()),
        "test_accuracy": float(accuracy_score(y_test_fold, y_pred_rf)),
        "test_precision": float(precision_score(y_test_fold, y_pred_rf, average='weighted', zero_division=0)),
        "test_recall": float(recall_score(y_test_fold, y_pred_rf, average='weighted', zero_division=0)),
        "test_f1": float(f1_score(y_test_fold, y_pred_rf, average='weighted', zero_division=0)),
    }
    
    # Print results
    rows = [
        ["CV Mean Accuracy", f"{rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']:.4f}"],
        ["CV Min", f"{rf_metrics['cv_min']:.4f}"],
        ["CV Max", f"{rf_metrics['cv_max']:.4f}"],
        ["Test Accuracy", f"{rf_metrics['test_accuracy']:.4f}"],
        ["Test F1-Score", f"{rf_metrics['test_f1']:.4f}"],
    ]
    print_table("Random Forest Results", ["Metric", "Value"], rows)
    
    # Confusion matrix
    cm_rf = confusion_matrix(y_test_fold, y_pred_rf, labels=unique_classes)
    
    if save_figures:
        _plot_confusion_matrix(
            cm_rf, class_labels,
            f"Random Forest Confusion Matrix ({cv_folds}-fold CV)",
            output_dir / "confusion_matrix_rf.png",
            cmap='Greens'
        )
    
    # Classification report
    report_rf = classification_report(y_test_fold, y_pred_rf, output_dict=True, zero_division=0)
    results["random_forest"] = {
        "metrics": rf_metrics,
        "confusion_matrix": cm_rf.tolist(),
        "classification_report": report_rf,
    }
    
    # Save results
    _save_classification_report(results, output_dir)
    
    print_success(f"Classification results saved to: {output_dir}")
    
    return results


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list,
    title: str,
    save_path: Path,
    cmap: str = 'Blues',
):
    """Plot and save confusion matrix"""
    n_classes = len(class_labels)
    fig_size = (max(6, n_classes * 0.8), max(5, n_classes * 0.8))
    
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved: {save_path}")


def _save_classification_report(results: Dict[str, Any], output_dir: Path):
    """Save detailed classification report"""
    report_path = output_dir / "classification_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRADITIONAL ML CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        # Data statistics
        f.write("-" * 80 + "\n")
        f.write("DATA STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Samples:     {results['n_samples']}\n")
        f.write(f"  Feature Dimension: {results['n_features']}\n")
        f.write(f"  Number of Classes: {results['n_classes']}\n")
        f.write(f"  CV Folds:          {results['cv_folds']}\n\n")
        
        f.write("Class Distribution:\n")
        for cls, count in sorted(results['class_distribution'].items()):
            pct = 100 * count / results['n_samples']
            class_name = results['class_names'].get(cls, f"Class {cls}")
            f.write(f"  {class_name}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        
        # KNN Results
        f.write("-" * 80 + "\n")
        f.write("KNN CLASSIFIER RESULTS\n")
        f.write("-" * 80 + "\n")
        knn_metrics = results['knn']['metrics']
        f.write(f"  CV Mean Accuracy: {knn_metrics['cv_mean']:.4f} ± {knn_metrics['cv_std']:.4f}\n")
        f.write(f"  CV Min Accuracy: {knn_metrics['cv_min']:.4f}\n")
        f.write(f"  CV Max Accuracy: {knn_metrics['cv_max']:.4f}\n")
        f.write(f"  Test Accuracy:   {knn_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision:  {knn_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall:     {knn_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1-Score:  {knn_metrics['test_f1']:.4f}\n\n")
        
        f.write("CV Scores per Fold:\n")
        for i, score in enumerate(knn_metrics['cv_scores'], 1):
            f.write(f"  Fold {i}: {score:.4f}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        report = results['knn']['classification_report']
        for key, value in report.items():
            if isinstance(value, dict):
                f.write(f"  {key}:\n")
                for k, v in value.items():
                    if isinstance(v, float):
                        f.write(f"    {k}: {v:.4f}\n")
                    else:
                        f.write(f"    {k}: {v}\n")
            else:
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Random Forest Results
        f.write("-" * 80 + "\n")
        f.write("RANDOM FOREST CLASSIFIER RESULTS\n")
        f.write("-" * 80 + "\n")
        rf_metrics = results['random_forest']['metrics']
        f.write(f"  CV Mean Accuracy: {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']:.4f}\n")
        f.write(f"  CV Min Accuracy: {rf_metrics['cv_min']:.4f}\n")
        f.write(f"  CV Max Accuracy: {rf_metrics['cv_max']:.4f}\n")
        f.write(f"  Test Accuracy:   {rf_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision:  {rf_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall:     {rf_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1-Score:   {rf_metrics['test_f1']:.4f}\n\n")
        
        f.write("CV Scores per Fold:\n")
        for i, score in enumerate(rf_metrics['cv_scores'], 1):
            f.write(f"  Fold {i}: {score:.4f}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        report = results['random_forest']['classification_report']
        for key, value in report.items():
            if isinstance(value, dict):
                f.write(f"  {key}:\n")
                for k, v in value.items():
                    if isinstance(v, float):
                        f.write(f"    {k}: {v:.4f}\n")
                    else:
                        f.write(f"    {k}: {v}\n")
            else:
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"Classification report saved: {report_path}")

