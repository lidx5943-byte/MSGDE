#!/bin/bash
# ==============================================================================
# 批量提交所有耦合强度的动力学演化任务
# ==============================================================================
# 
# 使用方法:
#   ./submit_all_jobs.sh [配置文件路径]
#
# 例如:
#   ./submit_all_jobs.sh config.yaml
#   ./submit_all_jobs.sh                    # 使用默认配置文件 config.yaml
# ==============================================================================

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置文件路径（默认为 config.yaml）
CONFIG_FILE="${1:-$SCRIPT_DIR/config.yaml}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 '$CONFIG_FILE' 不存在！"
    exit 1
fi

# 使用Python读取配置并生成耦合强度列表
echo "=========================================="
echo "读取配置文件: $CONFIG_FILE"
echo "=========================================="

# 生成耦合强度数组
COUPLING_STRENGTHS=$(python3 << EOF
import yaml
import numpy as np

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

coupling_config = config['experiment']['coupling_strength']
start = coupling_config['start']
end = coupling_config['end']
step = coupling_config['step']

# 生成耦合强度数组
num_steps = int((end - start) / step) + 1
strengths = np.linspace(start, end, num_steps)
strengths = np.round(strengths, 1)

# 输出为空格分隔的字符串
print(' '.join([f'{s:.1f}' for s in strengths]))
EOF
)

# 转换为数组
STRENGTHS_ARRAY=($COUPLING_STRENGTHS)
TOTAL_JOBS=${#STRENGTHS_ARRAY[@]}

echo "发现 $TOTAL_JOBS 个耦合强度值"
echo "耦合强度范围: ${STRENGTHS_ARRAY[0]} 到 ${STRENGTHS_ARRAY[$((TOTAL_JOBS-1))]}"
echo ""

# 确认提交
read -p "是否提交所有 $TOTAL_JOBS 个任务? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 检查SLURM脚本是否存在
SLURM_SCRIPT="run_dynamics.slurm"
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "错误: SLURM脚本 '$SLURM_SCRIPT' 不存在！"
    exit 1
fi

# 提交所有任务
echo ""
echo "=========================================="
echo "开始提交任务..."
echo "=========================================="

SUBMITTED_JOBS=()
FAILED_JOBS=()
JOB_IDS_FILE="$SCRIPT_DIR/job_ids.txt"

# 清空或创建job_ids.txt文件（新的一批任务）
> "$JOB_IDS_FILE"

for coupling_strength in "${STRENGTHS_ARRAY[@]}"; do
    echo -n "提交耦合强度 $coupling_strength ... "
    
    # 提交SLURM任务并提取Job ID
    # sbatch输出格式: "Submitted batch job 12345"
    SBATCH_OUTPUT=$(sbatch \
        --job-name="dyn_${coupling_strength}" \
        --export=ALL,COUPLING_STRENGTH="$coupling_strength",CONFIG_FILE="$CONFIG_FILE" \
        "$SLURM_SCRIPT" 2>&1)
    
    SBATCH_EXIT_CODE=$?
    
    # 提取Job ID（兼容多种grep版本）
    if [ $SBATCH_EXIT_CODE -eq 0 ]; then
        # 尝试使用grep -oP（如果支持）
        JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+' 2>/dev/null || \
                 echo "$SBATCH_OUTPUT" | grep -oE '[0-9]+' | head -1)
        
        if [ -n "$JOB_ID" ]; then
            echo "成功 (Job ID: $JOB_ID)"
            SUBMITTED_JOBS+=("$coupling_strength:$JOB_ID")
            # 立即写入job_ids.txt
            echo "$JOB_ID" >> "$JOB_IDS_FILE"
        else
            echo "失败 (无法提取Job ID)"
            echo "  sbatch输出: $SBATCH_OUTPUT"
            FAILED_JOBS+=("$coupling_strength")
        fi
    else
        echo "失败"
        echo "  sbatch输出: $SBATCH_OUTPUT"
        FAILED_JOBS+=("$coupling_strength")
    fi
    
    # 短暂延迟，避免提交过快
    sleep 0.1
done

echo ""
echo "=========================================="
echo "提交完成"
echo "=========================================="
echo "成功提交: ${#SUBMITTED_JOBS[@]} 个任务"
echo "失败: ${#FAILED_JOBS[@]} 个任务"
echo ""

# 保存任务信息到文件
JOB_LOG="$SCRIPT_DIR/job_submissions.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 创建或追加到日志文件
{
    echo "=========================================="
    echo "[$TIMESTAMP] 提交了 ${#SUBMITTED_JOBS[@]} 个任务"
    echo "配置文件: $CONFIG_FILE"
    echo "耦合强度范围: ${STRENGTHS_ARRAY[0]} 到 ${STRENGTHS_ARRAY[$((TOTAL_JOBS-1))]}"
    echo "----------------------------------------"
    echo "成功提交的任务:"
    for job_info in "${SUBMITTED_JOBS[@]}"; do
        IFS=':' read -r coupling job_id <<< "$job_info"
        echo "  耦合强度: $coupling  ->  Job ID: $job_id"
    done
    
    if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
        echo "----------------------------------------"
        echo "失败的任务:"
        for failed in "${FAILED_JOBS[@]}"; do
            echo "  耦合强度: $failed"
        done
    fi
    echo "----------------------------------------"
    echo "Job ID列表已保存到: $JOB_IDS_FILE"
    echo "=========================================="
    echo ""
} >> "$JOB_LOG"

# 显示摘要
echo ""
echo "任务信息已保存到: $JOB_LOG"
echo "Job ID列表已保存到: $JOB_IDS_FILE"
echo ""
echo "常用命令:"
echo "  查看所有任务:     squeue -u \$USER"
echo "  取消所有任务:     ./stop_all_jobs.sh"
echo "  查看任务日志:     tail -f slurm_out/slurm-<JOB_ID>.out"
echo "  查看提交历史:     tail -f $JOB_LOG"
echo ""
