"""
配置加载模块
============

提供配置文件加载和验证功能：
- 从YAML文件加载配置
- 配置参数验证
- 默认值处理

使用示例
--------
>>> from src.config import load_config, Config
>>> 
>>> # 加载配置
>>> config = load_config("config/config.yaml")
>>> 
>>> # 访问配置
>>> print(config.dynamics.oscillator.alpha)
>>> print(config.preprocessing.filter.low_freq)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

import yaml

from .utils.logger import console, print_success, print_warning, print_error


# ==============================================================================
# 配置数据类定义
# ==============================================================================

@dataclass
class FilterConfig:
    """滤波配置"""
    low_freq: float = 8.0
    high_freq: float = 12.5
    order: int = 4


@dataclass
class NotchConfig:
    """陷波滤波配置"""
    freq: float = 50.0
    quality_factor: int = 30


@dataclass
class BaselineConfig:
    """基线校正配置"""
    duration: float = 0.2


@dataclass
class OutlierConfig:
    """异常值处理配置"""
    mad_threshold: float = 8.0


@dataclass
class BadTrialConfig:
    """坏trial检测配置"""
    min_variance: float = 1e-6
    min_std: float = 1e-6


@dataclass
class PreprocessingConfig:
    """预处理配置"""
    sampling_rate: int = 160
    filter: FilterConfig = field(default_factory=FilterConfig)
    notch: NotchConfig = field(default_factory=NotchConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    bad_trial: BadTrialConfig = field(default_factory=BadTrialConfig)


@dataclass
class SimilarityConfig:
    """相似度计算配置"""
    method: str = "pearson"
    rbf_gamma: Union[str, float] = "scale"
    laplacian_gamma: float = 1.0


@dataclass
class LaplacianConfig:
    """拉普拉斯矩阵配置"""
    n_scales: int = 5
    sigma: float = 3.0
    k: int = 1
    partition_method: str = "uniform"  # 分割方法：uniform (均匀划分阈值) 或 quantile (均匀划分边数)
    plot_distribution: bool = False    # 是否绘制权重分布图


@dataclass
class OscillatorConfig:
    """振荡器配置"""
    type: str = "Lorenz"
    alpha: float = 10.0
    beta: float = 2.666666
    gamma: float = 60.0
    rk: float = 7.0


@dataclass
class NumericalConfig:
    """数值积分配置"""
    method: str = "Euler"
    time_step: float = 1e-3
    coupling_strength: float = 0.42
    coupling_mode: str = "x_only"
    use_periodic_boundary: bool = True


@dataclass
class AnalysisConfig:
    """分析参数配置"""
    n_scales: int = 5
    time_steps: int = 110000
    transient_steps: int = 100000
    max_nodes: int = 1840


@dataclass
class SamplingConfig:
    """采样配置"""
    method: str = "hybrid"
    hybrid_ratio: float = 0.3


@dataclass
class InitialValuesConfig:
    """初始值范围配置"""
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 0.0
    z_max: float = 50.0


@dataclass
class StabilityConfig:
    """数值稳定性配置"""
    state_clip_min: float = -200.0
    state_clip_max: float = 200.0
    dynamics_clip_min: float = -5000.0
    dynamics_clip_max: float = 5000.0


@dataclass
class FeatureExtractionConfig:
    """特征提取配置"""
    enable_chaos: bool = True      # 是否提取混沌特征（Lyapunov指数）
    enable_sync: bool = True        # 是否提取同步特征（与中心节点的相关系数）


@dataclass
class DynamicsConfig:
    """动力学分析配置"""
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    initial_values: InitialValuesConfig = field(default_factory=InitialValuesConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)


@dataclass
class TrajectoryVisualizationConfig:
    """轨迹可视化配置"""
    interval_mode: str = "uniform"
    uniform_interval: int = 100
    custom_intervals: list = field(default_factory=lambda: [50, 50, 100, 100, 200])
    window_size: Optional[int] = 200


@dataclass
class FeatureDiversityConfig:
    """特征差异性配置"""
    enable: bool = True
    step_interval: int = 100
    method: str = "combined"
    n_scales: int = 1
    center_node_idx: int = 0


@dataclass
class FigureConfig:
    """图像配置"""
    dpi: int = 300
    format: str = "png"


@dataclass
class VisualizationConfig:
    """可视化配置"""
    trajectory: TrajectoryVisualizationConfig = field(default_factory=TrajectoryVisualizationConfig)
    feature_diversity: FeatureDiversityConfig = field(default_factory=FeatureDiversityConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)


@dataclass
class SubdirsConfig:
    """子目录配置"""
    logs: str = "logs"
    data: str = "data"
    figures: str = "figures"
    reports: str = "reports"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    save_to_file: bool = True


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str = "output"
    use_timestamp: bool = True
    subdirs: SubdirsConfig = field(default_factory=SubdirsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class ParallelConfig:
    """并行处理配置"""
    n_jobs: int = -1
    max_jobs: int = 8


@dataclass
class MLPathsConfig:
    """ML数据路径配置"""
    ml_data_dir: str = "ml"
    ml_features: str = "X_features.npy"
    ml_labels: str = "y_labels.npy"
    ml_all_data: str = "all_data.npy"
    transformer_data_dir: str = "transformer"
    transformer_features: str = "X_transformer.npy"
    transformer_labels: str = "y_labels.npy"
    classification_dir: str = "ml_classification"


@dataclass
class PathsConfig:
    """路径配置"""
    data_dir: str = "/srv/wzh/mm_eeg/data"
    motor_data: str = "/srv/wzh/mm_eeg/data/motor_data.npy"
    imagery_data: str = "/srv/wzh/mm_eeg/data/imagery_data.npy"
    metadata: str = "/srv/wzh/mm_eeg/data/metadata.npy"
    clean_data: str = "/srv/wzh/mm_eeg/data/chbmit_training_data.npy"
    labels: str = "/srv/wzh/mm_eeg/data/chbmit_training_data_labels.npy"
    correlation_dir: str = "corr"
    correlation_file: str = "pearson.npy"
    laplacian_dir: str = "pl_data"
    laplacian_family: str = "laplacian_family.npy"
    dynamics_dir: str = "dynamics"
    features_dir: str = "features"
    ml: MLPathsConfig = field(default_factory=MLPathsConfig)


@dataclass
class DataConversionConfig:
    """数据转换配置"""
    shuffle: bool = True
    random_state: int = 42


@dataclass
class KNNConfig:
    """KNN分类器配置"""
    k: int = 5
    weights: str = "uniform"
    algorithm: str = "auto"


@dataclass
class RandomForestConfig:
    """随机森林分类器配置"""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float] = "sqrt"
    bootstrap: bool = True


@dataclass
class ClassificationOutputConfig:
    """分类输出配置"""
    save_figures: bool = True
    save_report: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300


@dataclass
class MLClassificationConfig:
    """ML分类配置"""
    cv_folds: int = 10
    random_state: int = 42
    knn: KNNConfig = field(default_factory=KNNConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    class_names: Optional[Dict[int, str]] = None
    output: ClassificationOutputConfig = field(default_factory=ClassificationOutputConfig)


@dataclass
class MLConfig:
    """机器学习配置"""
    data_conversion: DataConversionConfig = field(default_factory=DataConversionConfig)
    classification: MLClassificationConfig = field(default_factory=MLClassificationConfig)


@dataclass
class ClassificationConfig:
    """分类配置（向后兼容）"""
    test_size: float = 0.3
    random_state: int = 42
    cv_folds: int = 10
    knn_k: int = 5


@dataclass
class Config:
    """主配置类"""
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    laplacian: LaplacianConfig = field(default_factory=LaplacianConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    
    # 原始配置字典（用于访问未定义的字段）
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


# ==============================================================================
# 配置加载函数
# ==============================================================================

def _dict_to_dataclass(data: Dict[str, Any], cls: type) -> Any:
    """
    将字典转换为dataclass实例
    
    参数
    ----
    data : Dict[str, Any]
        配置字典
    cls : type
        目标dataclass类型
        
    返回
    ----
    Any
        dataclass实例
    """
    if data is None:
        return cls()
    
    # 获取dataclass的字段信息
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    
    for key, value in data.items():
        if key.startswith('_'):
            continue
        if key not in field_types:
            continue
        
        field_type = field_types[key]
        
        # 处理嵌套的dataclass
        if dataclasses.is_dataclass(field_type):
            if isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: Union[str, Path] = None) -> Config:
    """
    从YAML文件加载配置
    
    参数
    ----
    config_path : str or Path, optional
        配置文件路径，默认为 config/config.yaml
        
    返回
    ----
    Config
        配置对象
        
    异常
    ----
    FileNotFoundError
        配置文件不存在
    yaml.YAMLError
        YAML解析错误
    """
    # 确定配置文件路径
    if config_path is None:
        # 尝试多个默认路径
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            print_warning("未找到配置文件，使用默认配置")
            return Config()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载YAML文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print_error(f"YAML解析错误: {e}")
        raise
    
    if raw_config is None:
        raw_config = {}
    
    # 转换为Config对象
    config = Config(
        paths=_dict_to_dataclass(raw_config.get('paths', {}), PathsConfig),
        preprocessing=_dict_to_dataclass(raw_config.get('preprocessing', {}), PreprocessingConfig),
        similarity=_dict_to_dataclass(raw_config.get('similarity', {}), SimilarityConfig),
        laplacian=_dict_to_dataclass(raw_config.get('laplacian', {}), LaplacianConfig),
        dynamics=_dict_to_dataclass(raw_config.get('dynamics', {}), DynamicsConfig),
        visualization=_dict_to_dataclass(raw_config.get('visualization', {}), VisualizationConfig),
        output=_dict_to_dataclass(raw_config.get('output', {}), OutputConfig),
        parallel=_dict_to_dataclass(raw_config.get('parallel', {}), ParallelConfig),
        classification=_dict_to_dataclass(raw_config.get('classification', {}), ClassificationConfig),
        ml=_dict_to_dataclass(raw_config.get('ml', {}), MLConfig),
        _raw=raw_config,
    )
    
    print_success(f"已加载配置: {config_path}")
    
    return config


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    保存配置到YAML文件
    
    参数
    ----
    config : Config
        配置对象
    config_path : str or Path
        保存路径
    """
    import dataclasses
    
    def dataclass_to_dict(obj: Any) -> Any:
        """递归将dataclass转换为字典"""
        if dataclasses.is_dataclass(obj):
            result = {}
            for f in dataclasses.fields(obj):
                if f.name.startswith('_'):
                    continue
                value = getattr(obj, f.name)
                result[f.name] = dataclass_to_dict(value)
            return result
        elif isinstance(obj, list):
            return [dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = dataclass_to_dict(config)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print_success(f"已保存配置: {config_path}")


def print_config(config: Config) -> None:
    """
    打印配置摘要
    
    参数
    ----
    config : Config
        配置对象
    """
    from .utils.logger import print_header, print_table
    
    print_header("配置摘要")
    
    # 动力学参数
    dynamics_rows = [
        ["振荡器类型", config.dynamics.oscillator.type],
        ["α (sigma)", config.dynamics.oscillator.alpha],
        ["β (beta)", config.dynamics.oscillator.beta],
        ["γ (rho)", config.dynamics.oscillator.gamma],
        ["耦合强度 ε", config.dynamics.numerical.coupling_strength],
        ["时间步长 h", config.dynamics.numerical.time_step],
        ["数值方法", config.dynamics.numerical.method],
        ["总步数", config.dynamics.analysis.time_steps],
        ["暂态步数", config.dynamics.analysis.transient_steps],
        ["尺度数量", config.dynamics.analysis.n_scales],
    ]
    print_table("动力学参数", ["参数", "值"], dynamics_rows)
    
    # 预处理参数
    preproc_rows = [
        ["采样率", f"{config.preprocessing.sampling_rate} Hz"],
        ["带通滤波", f"{config.preprocessing.filter.low_freq}-{config.preprocessing.filter.high_freq} Hz"],
        ["陷波频率", f"{config.preprocessing.notch.freq} Hz"],
        ["MAD阈值", config.preprocessing.outlier.mad_threshold],
    ]
    print_table("预处理参数", ["参数", "值"], preproc_rows)
