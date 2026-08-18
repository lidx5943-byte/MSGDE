"""
EEG动力学分析工具包
===================

本包提供完整的EEG信号动力学分析工具，包括：
- 数据预处理（滤波、基线校正、异常值处理）
- 相似度矩阵计算
- 多尺度拉普拉斯特征提取
- 混沌动力学分析（Lorenz振荡器网络）
- ML数据格式转换（传统ML + Transformer）
- 可视化工具

使用示例
--------
>>> from src.config import load_config
>>> from src.dynamics.pipeline import DynamicsAnalysisPipeline
>>> from src.ml.data_converter import MLDataConverter
>>> 
>>> config = load_config("config/config.yaml")
>>> pipeline = DynamicsAnalysisPipeline(config)
>>> features, trajectories = pipeline.run()
>>> 
>>> # ML数据转换
>>> converter = MLDataConverter(output_dir="output")
>>> X_ml, y_ml = converter.generate_ml_data(features, labels)
>>> X_tf, y_tf = converter.convert_to_transformer_format(trajectories, labels)
"""

__version__ = "1.0.0"
__author__ = "EEG Analysis Team"

from .config import load_config, Config

__all__ = [
    "load_config",
    "Config",
    "__version__",
]

