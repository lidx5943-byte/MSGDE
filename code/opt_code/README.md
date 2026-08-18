# EEG动力学分析工具包

基于混沌振荡器网络的EEG信号动力学分析工具包。

## 项目结构

```
opt_code/
├── config/
│   └── config.yaml              # 主配置文件
├── src/
│   ├── __init__.py
│   ├── config.py                # 配置加载模块
│   ├── preprocessing/           # 数据预处理
│   ├── similarity/              # 相似度计算
│   ├── laplacian/               # 拉普拉斯特征
│   ├── dynamics/                # 动力学分析
│   ├── visualization/           # 可视化
│   └── utils/                   # 工具模块
├── scripts/
│   ├── run_preprocess.py        # 预处理脚本
│   ├── run_similarity.py        # 相似度计算脚本
│   ├── run_laplacian.py         # 拉普拉斯特征脚本
│   ├── run_dynamics.py          # 动力学分析脚本
│   ├── run_ml_convert.py        # ML数据转换脚本
│   ├── run_ml_classification.py # ML分类评估脚本
│   ├── run_visualize.py         # 可视化脚本
│   └── run_full_pipeline.py     # 完整流水线
├── requirements.txt
└── README.md
```

## 安装

```bash
cd opt_code
pip install -r requirements.txt
```

## 快速开始

### 1. 完整流水线（推荐）

```bash
python scripts/run_full_pipeline.py \
    --config config/config.yaml \
    --data /path/to/eeg_data.npy \
    --labels /path/to/labels.npy \
    --name my_experiment \
    --output output
```

### 2. 分步执行

**步骤1：数据预处理**
```bash
python scripts/run_preprocess.py \
    --config config/config.yaml \
    --data /path/to/eeg_data.npy \
    --output output/preprocessed
```

**步骤2：相似度矩阵计算**
```bash
python scripts/run_similarity.py \
    --config config/config.yaml \
    --data output/preprocessed/preprocessed_data.npy \
    --output output/similarity \
    --method pearson
```

**步骤3：拉普拉斯特征提取**
```bash
python scripts/run_laplacian.py \
    --config config/config.yaml \
    --similarity output/similarity/similarity_pearson.npy \
    --output output/laplacian
```

**步骤4：动力学分析**
```bash
python scripts/run_dynamics.py \
    --config config/config.yaml \
    --laplacian output/laplacian/laplacian_family.npy \
    --name dynamics_exp \
    --output output
```

**步骤5：ML数据转换**
```bash
python scripts/run_ml_convert.py \
    --config config/config.yaml \
    --features output/dynamics_exp_YYYYMMDD_HHMMSS/data/output/dynamics_data.npy \
    --trajectories output/dynamics_exp_YYYYMMDD_HHMMSS/data/output/trajectories.npy \
    --labels /path/to/labels.npy \
    --output output/ml_data \
    --format all
```

**步骤6：ML分类评估**
```bash
python scripts/run_ml_classification.py \
    --config config/config.yaml \
    --features output/ml_data/ml/X_features.npy \
    --labels output/ml_data/ml/y_labels.npy \
    --output output/ml_classification \
    --cv-folds 10 \
    --knn-k 5
```

**步骤7：可视化**
```bash
python scripts/run_visualize.py \
    --config config/config.yaml \
    --trajectories output/dynamics_exp_YYYYMMDD_HHMMSS/data/output/trajectories.npy \
    --output output/figures \
    --all
```

不需要预处理的代码

```bash
cd /srv/wzh/mm_eeg/opt_code

python scripts/run_full_pipeline.py \
    -c config/config.yaml \
    -d /srv/wzh/mm_eeg/data/chbmit_training_data.npy \
    -l /srv/wzh/mm_eeg/data/chbmit_training_data_labels.npy \
    -o output/full_analysis \
    --skip-preprocess \
    --skip-diversity

python scripts/run_full_pipeline.py \
    -c config/config.yaml \
    -d /srv/wzh/mm_eeg/data/300eeg/300eeg_data.npy \
    -l /srv/wzh/mm_eeg/data/300eeg/300eeg_labels.npy \
    -o output/full_analysis \
    --skip-preprocess \
    --skip-diversity

```

## 实验目录结构

每次运行会创建一个带时间戳的实验目录，包含完整的实验记录：

```
output/
└── my_experiment_20241126_143052/
    ├── config/
    │   ├── config.yaml          # 配置文件副本
    │   ├── command.txt          # 运行命令记录
    │   └── run.sh               # 可重复执行的脚本
    ├── data/
    │   ├── input/               # 输入数据信息
    │   └── output/              # 输出数据
    │       ├── preprocessed_data.npy
    │       ├── similarity_pearson.npy
    │       ├── laplacian_family.npy
    │       ├── dynamics_data.npy
    │       └── trajectories.npy
    ├── ml/                      # 传统ML数据
    │   ├── X_features.npy       # 特征矩阵 (n_samples, n_features)
    │   ├── y_labels.npy         # 标签 (n_samples,)
    │   ├── all_data.npy         # 合并数据 [features, label]
    │   └── ml_data_report.txt   # ML数据报告
    ├── transformer/             # Transformer数据
    │   ├── X_transformer.npy    # 序列特征 (n_samples, seq_len, feature_dim)
    │   ├── y_labels.npy         # 标签
    │   └── transformer_data_report.txt  # Transformer数据报告
    ├── ml_classification/       # ML分类结果
    │   ├── confusion_matrix_knn.png  # KNN混淆矩阵
    │   ├── confusion_matrix_rf.png   # 随机森林混淆矩阵
    │   └── classification_report.txt  # 分类报告
    ├── figures/
    │   ├── phase_space.png
    │   ├── butterfly_evolution/
    │   └── feature_diversity_*.png
    ├── logs/
    │   └── experiment_summary.log  # 实验摘要日志
    └── reports/
        ├── experiment_info.json    # 实验信息（JSON格式）
        ├── full_config_report.txt  # 完整配置报告
        └── analysis_report.txt     # 分析报告（含全部参数）
```

### 分析报告 (analysis_report.txt)

包含完整的配置参数：
- Lorenz振荡器参数 (alpha, beta, gamma, rk)
- 数值积分参数 (method, time_step, coupling_strength, coupling_mode)
- 分析参数 (n_scales, time_steps, transient_steps, max_nodes)
- 初始值范围 (x, y, z)
- 数值稳定性控制 (state_clip, dynamics_clip)
- 各尺度演化结果
- 输出统计
- 计时信息

### 实验信息文件 (experiment_info.json)

```json
{
  "name": "my_experiment",
  "description": "EEG Dynamics Analysis",
  "timestamp": "20241126_143052",
  "start_time": "2024-11-26 14:30:52",
  "end_time": "2024-11-26 14:35:23",
  "status": "completed",
  "duration_seconds": 271.5,
  "command": "python scripts/run_full_pipeline.py --config config/config.yaml ...",
  "inputs": {
    "data": "/path/to/eeg_data.npy",
    "labels": "/path/to/labels.npy"
  },
  "outputs": {
    "dynamics_data": "/output/.../dynamics_data.npy",
    "trajectories": "/output/.../trajectories.npy"
  },
  "metrics": {
    "data_shape": [1000, 64, 160],
    "feature_shape": [1840, 70],
    "trajectory_shape": [5, 1840, 1000, 3]
  },
  "config": { ... }  // 完整配置内容
}
```

## Python API使用

```python
from src.config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.similarity.correlation import compute_similarity_matrix
from src.laplacian.multiscale import LaplacianPipeline
from src.dynamics.pipeline import DynamicsAnalysisPipeline
from src.utils.io import Experiment
import numpy as np

# 创建实验
exp = Experiment("my_analysis", config_path="config/config.yaml")

# 加载配置和数据
config = load_config("config/config.yaml")
data = np.load("data/eeg_data.npy")

# 1. 预处理
preproc = PreprocessingPipeline(config)
cleaned_data, labels, stats = preproc.run(data)

# 2. 相似度矩阵
similarity = compute_similarity_matrix(cleaned_data, method="pearson")

# 3. 拉普拉斯特征
laplacian_pipe = LaplacianPipeline(config, output_dir=str(exp.output_dir))
laplacian_family = laplacian_pipe.run(similarity)

# 4. 动力学分析
dynamics_pipe = DynamicsAnalysisPipeline(config, output_dir=str(exp.root_dir))
features, trajectories = dynamics_pipe.run(laplacian_family)

# 5. ML数据转换
from src.ml.data_converter import MLDataConverter
ml_converter = MLDataConverter(exp.root_dir)

# 生成传统ML数据
X_ml, y_ml = ml_converter.generate_ml_data(features, labels)

# 生成Transformer数据
X_tf, y_tf = ml_converter.convert_to_transformer_format(trajectories, labels)

# 完成实验
exp.finish("completed")
```

## 配置说明

配置文件 `config/config.yaml` 包含所有分析参数：

### 预处理参数
```yaml
preprocessing:
  sampling_rate: 160        # 采样率 (Hz)
  filter:
    low_freq: 8.0           # 带通滤波下限
    high_freq: 12.5         # 带通滤波上限
  notch:
    freq: 50.0              # 陷波频率
  outlier:
    mad_threshold: 8.0      # MAD异常值阈值
```

### 动力学参数
```yaml
dynamics:
  oscillator:
    type: "Lorenz"
    alpha: 10.0             # σ参数
    beta: 2.666666          # β参数 (8/3)
    gamma: 60.0             # ρ参数
  numerical:
    method: "Euler"         # 数值方法: Euler, RK4
    time_step: 0.001        # 时间步长
    coupling_strength: 0.42 # 耦合强度ε
  analysis:
    n_scales: 5             # 尺度数量
    time_steps: 11000       # 总步数
    transient_steps: 10000  # 暂态步数
    max_nodes: 1840         # 最大节点数
```

## ML数据格式

### 传统机器学习格式 (ml/)

用于SVM、随机森林、KNN等传统分类器：

```python
# 加载数据
X = np.load('ml/X_features.npy')  # (n_samples, n_features)
y = np.load('ml/y_labels.npy')    # (n_samples,)

# 使用示例
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = SVC().fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

### Transformer格式 (transformer/)

用于Transformer、LSTM等序列模型：

```python
# 加载数据
X = np.load('transformer/X_transformer.npy')  # (n_samples, seq_len, feature_dim)
y = np.load('transformer/y_labels.npy')        # (n_samples,)

# PyTorch Dataset示例
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = torch.FloatTensor(np.load(X_path))
        self.y = torch.LongTensor(np.load(y_path))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = EEGDataset('transformer/X_transformer.npy', 'transformer/y_labels.npy')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Transformer数据报告 (transformer_data_report.txt)

包含模型设计所需的关键参数：
- 样本数 (batch_size)
- 序列长度 (seq_len / max_seq_len)
- 特征维度 (d_model input)
- 类别数 (num_classes)
- 尺度信息 (n_scales, n_vars)

## 内存注意事项

动力学演化会消耗大量内存。当检测到内存需求过大时，程序会自动切换为顺序处理模式。

**内存估算公式：**
```
内存(GB) ≈ max_nodes × (time_steps - transient_steps) × 3 × 8 / (1024³)
```

**推荐配置：**
| 可用内存 | max_nodes | 有效步数 | 并行数 |
|----------|-----------|----------|--------|
| 8GB      | 500       | 1000     | 2      |
| 16GB     | 1000      | 1000     | 4      |
| 32GB     | 1840      | 1000     | 4      |

## 数学原理

### Lorenz系统

耦合Lorenz振荡器网络的动力学方程：

$$
\frac{dx_i}{dt} = \alpha(y_i - x_i)
$$

$$
\frac{dy_i}{dt} = x_i(\gamma - z_i) - y_i + \varepsilon \sum_j L_{ij} x_j
$$

$$
\frac{dz_i}{dt} = x_i y_i - \beta z_i
$$

其中：
- $\alpha, \beta, \gamma$ 是Lorenz系统参数
- $\varepsilon$ 是耦合强度
- $L$ 是拉普拉斯矩阵

### 多尺度拉普拉斯

1. 从相关矩阵构建邻接矩阵：$A_{ij} = \exp(-d_{ij}^2/\sigma^2)$
2. 计算拉普拉斯矩阵：$L = D - A$
3. 通过阈值生成多尺度子图

## 依赖

- numpy >= 1.21.0
- scipy >= 1.7.0
- pyyaml >= 6.0
- rich >= 12.0.0
- joblib >= 1.0.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- seaborn >= 0.11.0

## 许可证

MIT License
