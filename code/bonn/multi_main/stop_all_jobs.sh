#!/bin/bash
# ==============================================================================
# 取消所有已提交的动力学演化任务
# ==============================================================================
# 
# 使用方法:
#   ./stop_all_jobs.sh [选项]
#
# 选项:
#   --all         取消所有任务（包括其他作业）
#   --latest      只取消最新提交的一批任务（默认）
#   --ids FILE    从指定文件读取Job ID列表
#   --confirm     跳过确认提示
#
# 例如:
#   ./stop_all_jobs.sh                    # 取消最新提交的任务（需要确认）
#   ./stop_all_jobs.sh --latest --confirm # 取消最新提交的任务（不确认）
#   ./stop_all_jobs.sh --all              # 取消所有任务
#   ./stop_all_jobs.sh --ids job_ids.txt  # 从文件读取Job ID
# ==============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
MODE="latest"
SKIP_CONFIRM=false
JOB_IDS_FILE="$SCRIPT_DIR/job_ids.txt"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MODE="all"
            shift
            ;;
        --latest)
            MODE="latest"
            shift
            ;;
        --ids)
            MODE="file"
            JOB_IDS_FILE="$2"
            shift 2
            ;;
        --confirm)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --all         取消所有任务（包括其他作业）"
            echo "  --latest      只取消最新提交的一批任务（默认）"
            echo "  --ids FILE    从指定文件读取Job ID列表"
            echo "  --confirm     跳过确认提示"
            echo "  -h, --help    显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 根据模式获取Job ID列表
case $MODE in
    all)
        echo "=========================================="
        echo "取消所有任务"
        echo "=========================================="
        
        # 获取所有任务
        ALL_JOBS=$(squeue -u "$USER" -h -o "%i" 2>/dev/null || echo "")
        
        if [ -z "$ALL_JOBS" ]; then
            echo "没有找到运行中的任务"
            exit 0
        fi
        
        JOB_COUNT=$(echo "$ALL_JOBS" | wc -l)
        echo "找到 $JOB_COUNT 个运行中的任务"
        echo ""
        echo "任务列表:"
        squeue -u "$USER" -o "%.10i %.20j %.10T %.10M %.6D %R"
        echo ""
        
        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "确认取消所有 $JOB_COUNT 个任务? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "已取消操作"
                exit 0
            fi
        fi
        
        # 取消所有任务
        echo "正在取消所有任务..."
        scancel -u "$USER"
        
        if [ $? -eq 0 ]; then
            echo "✓ 已取消所有任务"
        else
            echo "✗ 取消任务时出错"
            exit 1
        fi
        ;;
        
    latest)
        echo "=========================================="
        echo "取消最新提交的任务"
        echo "=========================================="
        
        # 检查job_ids.txt文件是否存在
        if [ ! -f "$JOB_IDS_FILE" ]; then
            echo "错误: Job ID文件不存在: $JOB_IDS_FILE"
            echo "提示: 请先运行 submit_all_jobs.sh 提交任务"
            exit 1
        fi
        
        # 读取Job ID列表
        JOB_IDS=$(cat "$JOB_IDS_FILE" 2>/dev/null | grep -v '^$' | grep -v '^#' || echo "")
        
        if [ -z "$JOB_IDS" ]; then
            echo "Job ID文件为空，没有任务需要取消"
            exit 0
        fi
        
        # 过滤出仍然存在的任务
        ACTIVE_JOBS=()
        for job_id in $JOB_IDS; do
            if squeue -j "$job_id" -h &>/dev/null; then
                ACTIVE_JOBS+=("$job_id")
            fi
        done
        
        if [ ${#ACTIVE_JOBS[@]} -eq 0 ]; then
            echo "没有找到运行中的任务（可能已经完成或被取消）"
            exit 0
        fi
        
        JOB_COUNT=${#ACTIVE_JOBS[@]}
        echo "找到 $JOB_COUNT 个运行中的任务"
        echo ""
        echo "任务列表:"
        for job_id in "${ACTIVE_JOBS[@]}"; do
            squeue -j "$job_id" -o "%.10i %.20j %.10T %.10M %.6D %R" 2>/dev/null || true
        done
        echo ""
        
        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "确认取消这 $JOB_COUNT 个任务? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "已取消操作"
                exit 0
            fi
        fi
        
        # 取消任务
        echo "正在取消任务..."
        CANCELLED=0
        FAILED=0
        
        for job_id in "${ACTIVE_JOBS[@]}"; do
            if scancel "$job_id" 2>/dev/null; then
                echo "  ✓ 已取消 Job ID: $job_id"
                ((CANCELLED++))
            else
                echo "  ✗ 取消失败 Job ID: $job_id"
                ((FAILED++))
            fi
        done
        
        echo ""
        echo "=========================================="
        echo "取消完成"
        echo "=========================================="
        echo "成功取消: $CANCELLED 个任务"
        if [ $FAILED -gt 0 ]; then
            echo "失败: $FAILED 个任务"
        fi
        
        # 记录到日志
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        {
            echo "[$TIMESTAMP] 取消了 $CANCELLED 个任务"
            echo "  取消的Job ID: ${ACTIVE_JOBS[*]}"
            echo ""
        } >> "$SCRIPT_DIR/job_submissions.log"
        ;;
        
    file)
        echo "=========================================="
        echo "从文件读取Job ID: $JOB_IDS_FILE"
        echo "=========================================="
        
        if [ ! -f "$JOB_IDS_FILE" ]; then
            echo "错误: 文件不存在: $JOB_IDS_FILE"
            exit 1
        fi
        
        # 读取Job ID列表
        JOB_IDS=$(cat "$JOB_IDS_FILE" 2>/dev/null | grep -v '^$' | grep -v '^#' || echo "")
        
        if [ -z "$JOB_IDS" ]; then
            echo "文件为空，没有任务需要取消"
            exit 0
        fi
        
        # 过滤出仍然存在的任务
        ACTIVE_JOBS=()
        for job_id in $JOB_IDS; do
            if squeue -j "$job_id" -h &>/dev/null; then
                ACTIVE_JOBS+=("$job_id")
            fi
        done
        
        if [ ${#ACTIVE_JOBS[@]} -eq 0 ]; then
            echo "没有找到运行中的任务"
            exit 0
        fi
        
        JOB_COUNT=${#ACTIVE_JOBS[@]}
        echo "找到 $JOB_COUNT 个运行中的任务"
        echo ""
        
        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "确认取消这 $JOB_COUNT 个任务? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "已取消操作"
                exit 0
            fi
        fi
        
        # 取消任务
        echo "正在取消任务..."
        for job_id in "${ACTIVE_JOBS[@]}"; do
            scancel "$job_id" && echo "  ✓ 已取消 Job ID: $job_id" || echo "  ✗ 取消失败 Job ID: $job_id"
        done
        ;;
esac

echo ""
