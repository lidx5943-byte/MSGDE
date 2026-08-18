#!/bin/bash --login
#SBATCH --job-name=dyn_evolution
#SBATCH --partition=general-long
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --chdir=/mnt/gs21/scratch/jiangj33/ldx/case1/multi_main
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
echo "Coupling Strength: ${COUPLING_STRENGTH:-未设置}"
echo "Config File: ${CONFIG_FILE:-config.yaml}"
echo "=========================================="
echo ""

# 检查必需的环境变量
if [ -z "${COUPLING_STRENGTH:-}" ]; then
    echo "错误: 环境变量 COUPLING_STRENGTH 未设置！"
    echo "请使用 --export 选项传递耦合强度值"
    exit 1
fi

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
echo "耦合强度: $COUPLING_STRENGTH"
echo "Python版本: $($PYTHON_ENV --version)"
echo ""

# 运行动力学演化
echo "开始运行动力学演化..."
echo "=========================================="
echo ""

$PYTHON_ENV dynamics_evolution.py \
    --config "$CONFIG_FILE" \
    --coupling-strength "$COUPLING_STRENGTH"

# 检查退出状态
EXIT_CODE=$?
echo ""
echo "=========================================="
echo "End Time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "动力学演化成功完成！"
else
    echo "动力学演化失败，退出代码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
