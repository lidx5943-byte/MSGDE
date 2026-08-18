#!/bin/bash --login
#SBATCH --job-name=multiscale_filter
#SBATCH --partition=general-long
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --chdir=/mnt/gs21/scratch/jiangj33/ldx/case17/multi_main
#SBATCH --output=slurm_out/slurm-%j.out
#SBATCH --error=slurm_out/slurm-%j.err

set -euo pipefail

# 创建输出目录（如果不存在）
mkdir -p slurm_out

# 打印环境信息
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "Config File: ${CONFIG_FILE:-config.yaml}"
echo "=========================================="
echo ""

# 设置Python环境
PYTHON_ENV="/mnt/ffs24/home/jiangj33/anaconda3/envs/eeg/bin/python"
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

# 检查Python环境是否存在
if [ ! -f "$PYTHON_ENV" ]; then
    echo "错误: Python环境不存在: $PYTHON_ENV"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 打印Python和配置信息
echo "Python路径: $PYTHON_ENV"
echo "配置文件: $CONFIG_FILE"
echo "Python版本: $($PYTHON_ENV --version)"
echo ""

# 运行多尺度过滤
echo "开始运行多尺度过滤..."
echo "=========================================="
echo ""

$PYTHON_ENV multiscale_filter.py --config "$CONFIG_FILE"

# 检查退出状态
EXIT_CODE=$?
echo ""
echo "=========================================="
echo "End Time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "多尺度过滤成功完成！"
else
    echo "多尺度过滤失败，退出代码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
