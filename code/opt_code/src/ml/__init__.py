"""
Machine Learning Data Preparation Module
=========================================

Provides data format conversion and classification for machine learning:
- ML data: Features + Labels for traditional classifiers
- Transformer data: Sequence format for transformer models
- Traditional ML classification: KNN and Random Forest with cross-validation
"""

from .data_converter import (
    generate_ml_data,
    convert_to_transformer_format,
    MLDataConverter,
)
from .classification import (
    run_ml_classification,
)

__all__ = [
    "generate_ml_data",
    "convert_to_transformer_format",
    "MLDataConverter",
    "run_ml_classification",
]

