"""
Data Converter Module
=====================

Convert dynamics analysis results to ML-ready formats:
- ML format: (n_samples, n_features) for traditional classifiers
- Transformer format: (n_samples, seq_len, features) for transformer models

Usage
-----
>>> from src.ml.data_converter import MLDataConverter
>>> 
>>> converter = MLDataConverter(output_dir="output/ml")
>>> ml_data = converter.generate_ml_data(features, labels)
>>> tf_data = converter.convert_to_transformer_format(trajectories, labels)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from datetime import datetime

from ..utils.logger import (
    get_logger, console, print_header, print_success, 
    print_warning, print_table, print_panel, create_progress
)
from ..utils.io import save_numpy, ensure_dir

logger = get_logger(__name__)


def generate_ml_data(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Union[str, Path],
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ML-ready data (features + labels)
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    labels : np.ndarray
        Label array, shape (n_samples,)
    output_dir : str or Path
        Output directory for ML data
    shuffle : bool
        Whether to shuffle data
    random_state : int
        Random seed for shuffling
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X, y) - Features and labels
    """
    print_header("Generate ML Data")
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    # Validate shapes
    n_samples = features.shape[0]
    n_features = features.shape[1]
    
    if labels.ndim > 1:
        labels = labels.flatten()
    
    if n_samples != len(labels):
        raise ValueError(f"Sample count mismatch: features={n_samples}, labels={len(labels)}")
    
    console.print(f"[dim]Input: {n_samples} samples, {n_features} features[/dim]")
    
    # Shuffle if requested
    if shuffle:
        np.random.seed(random_state)
        shuffle_idx = np.random.permutation(n_samples)
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]
        console.print(f"[dim]Shuffled with random_state={random_state}[/dim]")
    
    # Create combined data (features + label column)
    all_data = np.hstack([features, labels.reshape(-1, 1)])
    
    # Analyze labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    
    # Print statistics
    rows = [
        ["Samples", n_samples],
        ["Features", n_features],
        ["Classes", n_classes],
    ]
    for label, count in zip(unique_labels, counts):
        rows.append([f"  Class {int(label)}", f"{count} ({100*count/n_samples:.1f}%)"])
    print_table("ML Data Statistics", ["Item", "Value"], rows)
    
    # Save data
    save_numpy(features, output_dir / "X_features.npy")
    save_numpy(labels, output_dir / "y_labels.npy")
    save_numpy(all_data, output_dir / "all_data.npy")
    
    # Generate report
    _generate_ml_report(output_dir, features, labels, unique_labels, counts)
    
    print_success(f"ML data saved to: {output_dir}")
    
    return features, labels


def convert_to_transformer_format(
    trajectories: np.ndarray,
    labels: np.ndarray,
    output_dir: Union[str, Path],
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert trajectory data to Transformer input format
    
    Converts trajectory data from dynamics analysis to transformer-ready sequences:
    - Input shape: (n_scales, n_samples, n_times, n_vars)
    - Output shape: (n_samples, seq_len, n_scales * n_vars)
    
    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory data, shape (n_scales, n_samples, n_times, n_vars)
    labels : np.ndarray
        Label array, shape (n_samples,)
    output_dir : str or Path
        Output directory for transformer data
    shuffle : bool
        Whether to shuffle data
    random_state : int
        Random seed for shuffling
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_transformer, y) - Transformer features and labels
    """
    print_header("Convert to Transformer Format")
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    console.print(f"[dim]Input trajectory shape: {trajectories.shape}[/dim]")
    
    # Get dimensions
    if trajectories.ndim == 4:
        n_scales, n_samples, n_times, n_vars = trajectories.shape
    else:
        raise ValueError(f"Expected 4D trajectory array, got shape {trajectories.shape}")
    
    # Validate labels
    if labels.ndim > 1:
        labels = labels.flatten()
    
    if n_samples != len(labels):
        raise ValueError(f"Sample count mismatch: trajectories={n_samples}, labels={len(labels)}")
    
    # Transpose: (n_scales, n_samples, n_times, n_vars) -> (n_samples, n_scales, n_times, n_vars)
    # WARNING: This matches the legacy behavior of data_preprocess.py.
    # The resulting reshape will mix scales and times dimensions in C-order.
    console.print("[dim]Transposing dimensions (Legacy Mode)...[/dim]")
    X = trajectories.transpose(1, 0, 2, 3)
    
    # Shuffle if requested (before reshaping, to match data_preprocess.py)
    if shuffle:
        np.random.seed(random_state)
        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        labels = labels[shuffle_idx]
        console.print(f"[dim]Shuffled with random_state={random_state}[/dim]")
    
    # Reshape to Transformer format: (n_samples, seq_len, feature_dim)
    # seq_len = n_times, feature_dim = n_scales * n_vars
    console.print("[dim]Reshaping to Transformer format...[/dim]")
    feature_dim = n_scales * n_vars
    X_transformer = X.reshape(n_samples, n_times, feature_dim)
    
    # Analyze labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    
    # Print statistics
    rows = [
        ["Samples (batch_size)", n_samples],
        ["Sequence Length (seq_len)", n_times],
        ["Feature Dimension (d_model input)", feature_dim],
        ["  - Scales", n_scales],
        ["  - Variables per scale (x,y,z)", n_vars],
        ["Classes", n_classes],
    ]
    for label, count in zip(unique_labels, counts):
        rows.append([f"  Class {int(label)}", f"{count} ({100*count/n_samples:.1f}%)"])
    print_table("Transformer Data Statistics", ["Item", "Value"], rows)
    
    # Save data
    save_numpy(X_transformer, output_dir / "X_transformer.npy")
    save_numpy(labels, output_dir / "y_labels.npy")
    
    # Generate report
    _generate_transformer_report(
        output_dir, X_transformer, labels, 
        n_scales, n_times, n_vars, unique_labels, counts
    )
    
    print_success(f"Transformer data saved to: {output_dir}")
    
    return X_transformer, labels


def _generate_ml_report(
    output_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Generate ML data report"""
    report_path = output_dir / "ml_data_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MACHINE LEARNING DATA REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("DATA DIMENSIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Samples:     {features.shape[0]}\n")
        f.write(f"  Feature Dimension: {features.shape[1]}\n")
        f.write(f"  Data Shape:        {features.shape}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Number of Classes: {len(unique_labels)}\n")
        for label, count in zip(unique_labels, counts):
            pct = 100 * count / len(labels)
            f.write(f"  Class {int(label)}: {count} samples ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("FEATURE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:   {np.mean(features):.6f}\n")
        f.write(f"  Std:    {np.std(features):.6f}\n")
        f.write(f"  Min:    {np.min(features):.6f}\n")
        f.write(f"  Max:    {np.max(features):.6f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  X_features.npy: Features only, shape {features.shape}\n")
        f.write(f"  y_labels.npy:   Labels only, shape {labels.shape}\n")
        f.write(f"  all_data.npy:   Combined [features, label], shape ({features.shape[0]}, {features.shape[1]+1})\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("USAGE EXAMPLE\n")
        f.write("-" * 70 + "\n")
        f.write("  import numpy as np\n")
        f.write("  from sklearn.model_selection import train_test_split\n")
        f.write("  from sklearn.svm import SVC\n\n")
        f.write("  X = np.load('X_features.npy')\n")
        f.write("  y = np.load('y_labels.npy')\n")
        f.write("  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n")
        f.write("  clf = SVC().fit(X_train, y_train)\n")
        f.write("  accuracy = clf.score(X_test, y_test)\n\n")
        
        f.write("=" * 70 + "\n")
    
    print_success(f"ML report saved: {report_path}")


def _generate_transformer_report(
    output_dir: Path,
    X: np.ndarray,
    labels: np.ndarray,
    n_scales: int,
    n_times: int,
    n_vars: int,
    unique_labels: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Generate Transformer data report"""
    report_path = output_dir / "transformer_data_report.txt"
    
    n_samples = X.shape[0]
    seq_len = X.shape[1]
    feature_dim = X.shape[2]
    n_classes = len(unique_labels)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TRANSFORMER DATA REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("DATA DIMENSIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Samples:      {n_samples}\n")
        f.write(f"  Sequence Length:    {seq_len}\n")
        f.write(f"  Feature Dimension:  {feature_dim}\n")
        f.write(f"  Data Shape:         {X.shape}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("SEQUENCE STRUCTURE\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Number of Scales:   {n_scales}\n")
        f.write(f"  Time Steps:         {n_times}\n")
        f.write(f"  Variables (x,y,z):  {n_vars}\n")
        f.write(f"  Feature = scales * vars = {n_scales} * {n_vars} = {feature_dim}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TRANSFORMER MODEL PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Input Dimension (d_model input): {feature_dim}\n")
        f.write(f"  Sequence Length (max_seq_len):   {seq_len}\n")
        f.write(f"  Number of Classes (num_classes): {n_classes}\n")
        f.write(f"  Batch Size (suggested):          32-128\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Number of Classes: {n_classes}\n")
        for label, count in zip(unique_labels, counts):
            pct = 100 * count / len(labels)
            f.write(f"  Class {int(label)}: {count} samples ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("DATA STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:   {np.mean(X):.6f}\n")
        f.write(f"  Std:    {np.std(X):.6f}\n")
        f.write(f"  Min:    {np.min(X):.6f}\n")
        f.write(f"  Max:    {np.max(X):.6f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  X_transformer.npy: Shape {X.shape}\n")
        f.write(f"                     (n_samples, seq_len, feature_dim)\n")
        f.write(f"  y_labels.npy:      Shape {labels.shape}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PYTORCH DATASET EXAMPLE\n")
        f.write("-" * 70 + "\n")
        f.write("  import numpy as np\n")
        f.write("  import torch\n")
        f.write("  from torch.utils.data import Dataset, DataLoader\n\n")
        f.write("  class EEGDataset(Dataset):\n")
        f.write("      def __init__(self, X_path, y_path):\n")
        f.write("          self.X = torch.FloatTensor(np.load(X_path))\n")
        f.write("          self.y = torch.LongTensor(np.load(y_path))\n")
        f.write("      def __len__(self):\n")
        f.write("          return len(self.y)\n")
        f.write("      def __getitem__(self, idx):\n")
        f.write("          return self.X[idx], self.y[idx]\n\n")
        f.write("  dataset = EEGDataset('X_transformer.npy', 'y_labels.npy')\n")
        f.write("  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TRANSFORMER MODEL ARCHITECTURE SUGGESTION\n")
        f.write("-" * 70 + "\n")
        f.write("  class TransformerClassifier(nn.Module):\n")
        f.write(f"      def __init__(self, d_input={feature_dim}, d_model=128, nhead=8,\n")
        f.write(f"                   num_layers=4, num_classes={n_classes}):\n")
        f.write("          super().__init__()\n")
        f.write("          self.input_proj = nn.Linear(d_input, d_model)\n")
        f.write(f"          self.pos_enc = PositionalEncoding(d_model, max_len={seq_len})\n")
        f.write("          encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)\n")
        f.write("          self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)\n")
        f.write("          self.classifier = nn.Linear(d_model, num_classes)\n\n")
        
        f.write("=" * 70 + "\n")
    
    print_success(f"Transformer report saved: {report_path}")


class MLDataConverter:
    """
    ML Data Converter Class
    
    Converts dynamics analysis outputs to various ML-ready formats.
    
    Attributes
    ----------
    output_dir : Path
        Base output directory
    ml_dir : Path
        ML data output directory
    transformer_dir : Path
        Transformer data output directory
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the converter
        
        Parameters
        ----------
        output_dir : str or Path
            Base output directory
        """
        self.output_dir = Path(output_dir)
        self.ml_dir = self.output_dir / "ml"
        self.transformer_dir = self.output_dir / "transformer"
        
        ensure_dir(self.ml_dir)
        ensure_dir(self.transformer_dir)
    
    def generate_ml_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ML-ready data"""
        return generate_ml_data(
            features, labels, self.ml_dir, shuffle, random_state
        )
    
    def convert_to_transformer_format(
        self,
        trajectories: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to Transformer format"""
        return convert_to_transformer_format(
            trajectories, labels, self.transformer_dir, shuffle, random_state
        )
    
    def process_all(
        self,
        features: np.ndarray,
        trajectories: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Process all conversions
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix from dynamics analysis
        trajectories : np.ndarray
            Trajectory data from dynamics analysis
        labels : np.ndarray
            Labels
        shuffle : bool
            Whether to shuffle data
        random_state : int
            Random seed
            
        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary with 'ml' and 'transformer' data
        """
        results = {}
        
        # Generate ML data
        results['ml'] = self.generate_ml_data(features, labels, shuffle, random_state)
        
        # Generate Transformer data
        results['transformer'] = self.convert_to_transformer_format(
            trajectories, labels, shuffle, random_state
        )
        
        return results

