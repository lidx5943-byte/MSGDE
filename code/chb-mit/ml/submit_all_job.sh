#!/usr/bin/env bash
# submit_all_job_4.sh 纯CPU版本，最大并发9任务（<10），无GPU、兼容旧SLURM
set -eu
shopt -s inherit_errexit 2>/dev/null || true

# ===================== 【可自行修改配置区】======================
BASE_DIR="/mnt/gs21/scratch/jiangj33/ldx/case21/output"
DATA_ROOT="${BASE_DIR}/output-xyz-multi-3k-10scalech01"
ORIG_CONFIG="${BASE_DIR}/config.yaml"
JOBS_DIR="${BASE_DIR}/result_MLch01"
EEG_PYTHON_PATH="/mnt/ffs24/home/jiangj33/anaconda3/envs/eeg/bin/python"
SCALES=(0 1 2 3 4 5 6 7 8 9)

# 集群账户分区
SLURM_ACCOUNT="guowei-search"
SLURM_PARTITION="guowei-search-gpu"
SLURM_QOS="normal"
MAX_CONCURRENT_GPU=9  # 单次队列上限9个，小于10，减少排队

# 资源配置（纯CPU，关闭GPU申请）
MEM_VAL="48G"
CPUS=12
WALLTIME="48:00:00"
REQUEST_GPU=0  # 0=不用GPU，纯CPU运行

SUBMIT_SLEEP=0.8
CHECK_INTERVAL=15
# ==============================================================

# 路径校验
check_path() {
    local path="$1"
    local desc="$2"
    if [ ! -e "${path}" ]; then
        echo -e "\033[31mERROR: ${desc} 不存在 -> ${path}\033[0m"
        exit 2
    fi
}
check_path "${DATA_ROOT}" "数据集根目录"
check_path "${BASE_DIR}/train.py" "训练脚本train.py"
check_path "${ORIG_CONFIG}" "原始yaml配置文件"
check_path "${EEG_PYTHON_PATH}" "Python解释器"

mkdir -p "${JOBS_DIR}"

# 统计任务：兼容旧slurm，无-P参数，屏蔽查询报错
get_running_task_num() {
    set +e
    local count=$(squeue -u "${USER}" 2>/dev/null | grep "${SLURM_PARTITION}" | grep -E '^[0-9]+' | wc -l)
    set -e
    echo "${count}"
}

# 并发等待逻辑
wait_for_concurrency() {
    while true; do
        local curr_num=$(get_running_task_num)
        if [ "${curr_num}" -lt "${MAX_CONCURRENT_GPU}" ]; then
            break
        fi
        echo -e "\033[33m⚠️ 当前并发任务${curr_num}，已达上限${MAX_CONCURRENT_GPU}，等待${CHECK_INTERVAL}s...\033[0m"
        sleep "${CHECK_INTERVAL}"
    done
}

# 遍历CHB-MIT文件夹
shopt -s nullglob
dirs=( "${DATA_ROOT}"/CHB-MIT-* )
if [ ${#dirs[@]} -eq 0 ]; then
    echo -e "\033[34m未匹配到 CHB-MIT-* 数据目录：${DATA_ROOT}\033[0m"
    exit 0
fi

echo -e "\033[32m===== 任务提交初始化 =====\033[0m"
echo "内存配置：${MEM_VAL} | CPU核数：${CPUS} | 最大运行时长：${WALLTIME}"
echo "限制最大并发任务：${MAX_CONCURRENT_GPU}（纯CPU，不申请GPU资源）"
echo "待处理目录总数：${#dirs[@]}，每个目录尺度数量：${#SCALES[@]}"
echo -e "\033[32m==========================\033[0m"

for fullpath in "${dirs[@]}"; do
    name="$(basename "${fullpath}")"
    echo -e "\n\033[34m>> 开始处理数据集：${name}\033[0m"
    datafile="${fullpath}/data/trajectories.npy"
    [ ! -f "${datafile}" ] && echo -e "\033[33m   ⚠️ 警告：缺失数据文件 ${datafile}\033[0m"

    for k in "${SCALES[@]}"; do
        jobname="${name}_scale${k}"
        jobdir="${JOBS_DIR}/${jobname}"
        mkdir -p "${jobdir}"

        # 跳过已完成/已提交任务
        if compgen -G "${jobdir}/slurm-*.out" > /dev/null; then
            echo "   ⏭️ 跳过 ${jobname}：目录存在slurm日志，任务已提交/完成"
            continue
        fi
        set +e
        exist_job=$(squeue -u "${USER}" -n "ml_${jobname}" 2>/dev/null | grep -E '^[0-9]+')
        set -e
        if [ -n "${exist_job}" ]; then
            echo "   ⏭️ 跳过 ${jobname}：队列中已有同名任务运行/排队"
            continue
        fi

        # 并发阻塞
        wait_for_concurrency

        cp "${BASE_DIR}/train.py" "${jobdir}/train.py"

        # 修改yaml输入路径
        esc_path=$(printf '%s' "${datafile}" | sed 's/[\/&]/\\&/g')
        sed -E 's#^[[:space:]]*input_data:[[:space:]]*.*#input_data: "'"${esc_path}"'"#g' "${ORIG_CONFIG}" > "${jobdir}/config.yaml.tmp"

        # 修改实验名
        orig_exp_line=$(grep -E '^[[:space:]]*experiment_name:' "${ORIG_CONFIG}" || true)
        if [ -n "${orig_exp_line}" ]; then
            exp_val=$(sed -E 's/^[[:space:]]*experiment_name:[[:space:]]*//' <<< "$orig_exp_line")
            exp_val_clean="${exp_val#\"}"; exp_val_clean="${exp_val_clean%\"}"
            exp_val_clean="${exp_val_clean#\'}"; exp_val_clean="${exp_val_clean%\'}"
            new_exp="${exp_val_clean}_${name}_scale${k}"
            new_exp_escaped=$(printf '%s' "${new_exp}" | sed 's/[\/&]/\\&/g')
            sed -E 's#^[[:space:]]*experiment_name:[[:space:]]*.*#experiment_name: "'"${new_exp_escaped}"'"#g' "${jobdir}/config.yaml.tmp" > "${jobdir}/config.yaml"
            rm -f "${jobdir}/config.yaml.tmp"
        else
            mv "${jobdir}/config.yaml.tmp" "${jobdir}/config.yaml"
            echo -e "\nexperiment_name: \"ml_${name}_scale${k}\"" >> "${jobdir}/config.yaml"
        fi

        # 设置scale_k
        if grep -qE '^[[:space:]]*scale_k:' "${jobdir}/config.yaml"; then
            sed -E 's#^[[:space:]]*scale_k:[[:space:]]*.*#scale_k: '"${k}"'#g' "${jobdir}/config.yaml" > "${jobdir}/config.yaml.tmp" && mv "${jobdir}/config.yaml.tmp" "${jobdir}/config.yaml"
        else
            echo -e "\nscale_k: ${k}" >> "${jobdir}/config.yaml"
        fi

        GPU_LINE=""
        # REQUEST_GPU=0 不添加gpu申请行

        # 生成任务脚本job.sh
        cat > "${jobdir}/job.sh" <<SBATCH_EOF
#!/bin/bash --login
#SBATCH --job-name=ml_${jobname}
#SBATCH --output=slurm-%j-${jobname}.out
#SBATCH --error=slurm-%j-${jobname}.err
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --qos=${SLURM_QOS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM_VAL}
#SBATCH --time=${WALLTIME}

set -eu
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}

echo "===== 作业环境信息 ====="
echo "启动时间: \$(date)"
echo "工作目录: \$(pwd)"
echo "申请内存: ${MEM_VAL} | CPU: ${CPUS} | 无GPU"
EEG_PYTHON="${EEG_PYTHON_PATH}"
echo "Python路径: \${EEG_PYTHON}"
\${EEG_PYTHON} --version
echo "========================"

\${EEG_PYTHON} train.py

echo "===== 作业结束 \$(date) ====="
SBATCH_EOF

        chmod +x "${jobdir}/job.sh"

        pushd "${jobdir}" > /dev/null
        set +e
        sbatch_full=$(sbatch job.sh 2>&1)
        sbatch_ret=$?
        set -e
        popd > /dev/null

        if [ ${sbatch_ret} -ne 0 ]; then
            echo -e "   \033[31m❌ ${jobname} 提交失败：${sbatch_full}\033[0m"
            sleep "${SUBMIT_SLEEP}"
            continue
        fi
        sbatch_out=$(echo "${sbatch_full}" | grep -E '^Submitted batch job')
        echo -e "   \033[32m✅ 已提交: ${sbatch_out}\033[0m"
        sleep "${SUBMIT_SLEEP}"
    done
done

echo -e "\n\033[32m==== 全部任务遍历完成，等待队列自动调度 ====\033[0m"
echo "查看队列：squeue -u ${USER} | grep ${SLURM_PARTITION}"
echo "清空所有排队PD任务：squeue -u ${USER} | grep PD | awk '{print \$1}' | xargs scancel"
