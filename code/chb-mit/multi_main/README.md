# 多任务版本 - 动力学耦合强度消融实验

本目录包含将实验拆分为多个独立任务的版本，适合在SLURM集群上并行提交大量作业。

## 目录结构

```
multi_main/
├── config.yaml              # 配置文件（包含laplacian_output_path）
├── multiscale_filter.py     # 多尺度过滤脚本（生成拉普拉斯矩阵族）
├── dynamics_evolution.py     # 动力学演化脚本（单个耦合强度）
├── run_multiscale.slurm     # 多尺度过滤的SLURM脚本
├── run_dynamics.slurm       # 动力学演化的SLURM脚本
├── submit_all_jobs.sh       # 批量提交所有任务的脚本
├── stop_all_jobs.sh         # 取消任务的脚本
├── job_submissions.log      # 提交历史日志（自动生成）
├── job_ids.txt              # 最新一批Job ID列表（自动生成）
├── slurm_out/               # SLURM输出目录
└── README.md                # 本文件
```

## 工作流程

### 1. 第一步：生成拉普拉斯矩阵族（只需运行一次）

```bash
# 方式1：使用SLURM提交
sbatch --export=CONFIG_FILE=config.yaml run_multiscale.slurm

# 方式2：直接运行（本地测试）
python multiscale_filter.py --config config.yaml
```

这个脚本会：
- 加载相关系数矩阵
- 生成K=10个尺度的拉普拉斯矩阵族
- 保存到 `laplacian_output_path` 指定的目录

### 2. 第二步：提交所有动力学演化任务

```bash
# 批量提交所有耦合强度的任务
./submit_all_jobs.sh config.yaml
```

这个脚本会：
- 读取配置文件中的耦合强度范围（0.0-20.0，步长0.5）
- 为每个耦合强度值提交一个独立的SLURM作业
- 总共提交41个任务（如果范围是0.0-20.0，步长0.5）

### 3. 单独提交单个任务（可选）

```bash
# 提交单个耦合强度的任务
sbatch \
  --job-name="dyn_0.5" \
  --export=ALL,COUPLING_STRENGTH=0.5,CONFIG_FILE=config.yaml \
  run_dynamics.slurm
```

## 配置文件说明

配置文件 `config.yaml` 相比原版本，新增了以下配置项：

```yaml
paths:
  # 新增：拉普拉斯矩阵族保存路径
  laplacian_output_path: "/mnt/gs21/scratch/jiangj33/EEG_EXP/output-xyz/laplacian"
```

## 任务特点

### 多尺度过滤任务
- **运行时间**: 约10-30分钟（取决于数据规模）
- **资源需求**: 20 CPU, 64G 内存
- **输出**: `laplacian_family_K10_sigma3.0.npy`

### 动力学演化任务
- **运行时间**: 约1.6小时/任务（每个耦合强度）
- **资源需求**: 10 CPU, 32G 内存
- **并行度**: 每个任务内部10个尺度并行演化
- **输出**: `{dataset_name}-{coupling_strength}/data/trajectories.npy`

## 优势

1. **真正的并行**: 41个耦合强度任务可以同时运行，充分利用集群资源
2. **灵活调度**: 每个任务独立，可以单独重跑失败的任务
3. **资源优化**: 每个任务只需要10个CPU，可以同时运行更多任务
4. **断点续传**: 支持跳过已完成的实验（`skip_completed: true`）

## 监控和管理

### 查看所有任务状态
```bash
squeue -u $USER
```

### 查看任务日志
```bash
# 查看标准输出
tail -f slurm_out/slurm-<JOB_ID>.out

# 查看错误输出
tail -f slurm_out/slurm-<JOB_ID>.err
```

### 查看提交历史
```bash
# 查看所有提交记录
cat job_submissions.log

# 实时监控新提交
tail -f job_submissions.log

# 查看当前批次的Job ID列表
cat job_ids.txt
```

### 取消任务

#### 取消最新提交的一批任务（推荐）
```bash
# 需要确认
./stop_all_jobs.sh

# 跳过确认
./stop_all_jobs.sh --latest --confirm
```

#### 取消所有任务
```bash
# 取消所有任务（包括其他作业）
./stop_all_jobs.sh --all

# 跳过确认
./stop_all_jobs.sh --all --confirm
```

#### 从文件读取Job ID取消
```bash
# 从job_ids.txt读取
./stop_all_jobs.sh --ids job_ids.txt

# 从自定义文件读取
./stop_all_jobs.sh --ids my_job_ids.txt
```

#### 取消特定任务（手动）
```bash
scancel <JOB_ID>
```

### 日志文件说明

- `job_submissions.log` - 详细的提交历史记录，包含时间戳、耦合强度、Job ID等信息
- `job_ids.txt` - 最新一批提交的Job ID列表（每行一个），用于快速取消任务

## 时间估算

假设每个耦合强度任务运行1.6小时：

- **串行模式**（原版本）: 41 × 1.6 = 65.6 小时
- **并行模式**（本版本）: 如果集群有足够资源，可以同时运行所有41个任务，总时间约1.6小时

## 注意事项

1. **必须先运行多尺度过滤**: 在提交动力学演化任务之前，确保拉普拉斯矩阵族已经生成
2. **检查资源限制**: 确保集群有足够的资源同时运行多个任务
3. **存储空间**: 确保输出目录有足够的存储空间
4. **配置文件路径**: 确保所有脚本都能找到正确的配置文件

## 故障排除

### 拉普拉斯矩阵族不存在
```
错误: 拉普拉斯矩阵族不存在: ...
请先运行 multiscale_filter.py 生成拉普拉斯矩阵族
```
**解决方案**: 先运行 `run_multiscale.slurm` 生成拉普拉斯矩阵族

### 任务失败
检查 `slurm_out/slurm-<JOB_ID>.err` 文件查看错误信息，然后重新提交失败的任务。
