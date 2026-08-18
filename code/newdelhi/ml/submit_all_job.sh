#!/usr/bin/env bash
# submit_all_jobs_fixed.sh
# 功能：遍历所有耦合强度目录 + 10个尺度，批量提交ML训练任务
set -euo pipefail

# ====== 根据实际情况修改 ======
BASE_DIR="/mnt/gs21/scratch/jiangj33/ldx/case17"
DATA_ROOT="${BASE_DIR}/output-xyz-multi-3k-10scale"
ORIG_CONFIG="${BASE_DIR}/config.yaml"
JOBS_DIR="${BASE_DIR}/result_ML"
EEG_PYTHON_PATH="/mnt/ffs24/home/jiangj33/anaconda3/envs/eeg/bin/python"
# 若集群必须指定分区，请取消下一行注释并修改分区名
# SBATCH_PARTITION="standard"
# ==============================
# 尺度列表：0..9（共10个尺度）
SCALES=(0 1 2 3 4 5 6 7 8 9)
# ==============================

# 基础环境校验
if [ ! -d "${DATA_ROOT}" ]; then
  echo "ERROR: 数据根目录不存在: ${DATA_ROOT}"
  exit 2
fi
if [ ! -f "${BASE_DIR}/train.py" ]; then
  echo "ERROR: 训练脚本不存在: ${BASE_DIR}/train.py"
  exit 2
fi
if [ ! -f "${ORIG_CONFIG}" ]; then
  echo "ERROR: 模板配置文件不存在: ${ORIG_CONFIG}"
  exit 2
fi
if [ ! -x "${EEG_PYTHON_PATH}" ]; then
  echo "ERROR: Python环境路径无效或不可执行: ${EEG_PYTHON_PATH}"
  exit 2
fi

mkdir -p "${JOBS_DIR}"

# 遍历所有耦合强度目录
shopt -s nullglob
dirs=( "${DATA_ROOT}"/newdeli-* )
if [ ${#dirs[@]} -eq 0 ]; then
  echo "未在 ${DATA_ROOT} 下找到 newdeli-* 目录"
  exit 0
fi

# 计算总任务数（算术计算，更严谨）
total_jobs=$(( ${#dirs[@]} * ${#SCALES[@]} ))
echo "发现 ${#dirs[@]} 个耦合强度目录，每个目录对应 ${#SCALES[@]} 个尺度，总计 ${total_jobs} 个任务"
echo "=================================================="

# 外层循环：遍历所有耦合强度目录
for fullpath in "${dirs[@]}"; do
  name="$(basename -- "$fullpath")"   # 示例: newdeli-0.0
  echo ">>> 处理耦合强度: ${name}"

  datafile="${fullpath}/data/trajectories.npy"
  if [ ! -f "${datafile}" ]; then
    echo "    警告: 数据文件不存在 ${datafile}，跳过该目录所有尺度任务"
    continue
  fi

  # 内层循环：遍历10个尺度
  for scale in "${SCALES[@]}"; do
    echo "    -> 尺度 ${scale} 处理中..."
    
    # 每个任务独立目录，避免文件覆盖
    jobdir="${JOBS_DIR}/${name}_scale${scale}"
    mkdir -p "${jobdir}"

    # 复制训练脚本到作业目录
    cp "${BASE_DIR}/train.py" "${jobdir}/train.py"

    # 转义数据路径，适配sed替换
    esc_path=$(printf '%s' "${datafile}" | sed 's/[\/&]/\\&/g')

    # 第一步：替换input_data数据路径
    sed -E "s#^[[:space:]]*input_data:.*#input_data: \"${esc_path}\"#g" "${ORIG_CONFIG}" > "${jobdir}/config.yaml"

    # 第二步：替换experiment_name，确保全局唯一
    orig_exp_line=$(grep -E '^[[:space:]]*experiment_name:' "${ORIG_CONFIG}" || true)
    if [ -n "${orig_exp_line}" ]; then
      # 提取原始实验名，去除两侧引号
      exp_val=$(grep -E '^[[:space:]]*experiment_name:' "${ORIG_CONFIG}" | head -n1 | sed -E 's/^[[:space:]]*experiment_name:[[:space:]]*//')
      exp_val_clean="${exp_val}"
      exp_val_clean="${exp_val_clean#\"}"; exp_val_clean="${exp_val_clean%\"}"
      exp_val_clean="${exp_val_clean#\'}"; exp_val_clean="${exp_val_clean%\'}"
      # 拼接新实验名：原名_耦合强度_尺度
      new_exp="${exp_val_clean}_${name}_scale${scale}"
      new_exp_escaped=$(printf '%s' "${new_exp}" | sed 's/[\/&]/\\&/g')
      # 执行替换
      sed -i -E "s#^[[:space:]]*experiment_name:.*#experiment_name: \"${new_exp_escaped}\"#g" "${jobdir}/config.yaml"
    else
      # 配置中无该字段则追加
      echo "" >> "${jobdir}/config.yaml"
      echo "experiment_name: \"ml_${name}_scale${scale}\"" >> "${jobdir}/config.yaml"
    fi

    # 第三步：替换尺度参数 scale_k（核心修正：与配置文件键名完全一致）
    if grep -E '^[[:space:]]*scale_k:' "${jobdir}/config.yaml" > /dev/null; then
      # 已有scale_k字段，直接替换值
      sed -i -E "s#^[[:space:]]*scale_k:.*#scale_k: ${scale}#g" "${jobdir}/config.yaml"
    else
      # 无该字段则追加到文件末尾
      echo "" >> "${jobdir}/config.yaml"
      echo "scale_k: ${scale}" >> "${jobdir}/config.yaml"
    fi

    # 第四步：生成sbatch提交脚本
    cat > "${jobdir}/job.sh" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=ml_${name}_scale${scale}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
# 若需要指定分区，取消下一行注释
##SBATCH --partition=${SBATCH_PARTITION:-standard}

set -euo pipefail
echo "作业开始时间: \$(date)"
echo "工作目录: \$(pwd)"
echo "耦合强度: ${name}"
echo "尺度编号: ${scale}"

# 指定Python环境
EEG_PYTHON="${EEG_PYTHON_PATH}"
echo "使用Python: \${EEG_PYTHON}"
\${EEG_PYTHON} --version

# 运行训练脚本
\${EEG_PYTHON} train.py

echo "作业结束时间: \$(date)"
SBATCH_EOF

    chmod +x "${jobdir}/job.sh"

    # 提交任务（测试阶段可注释下面这段，只生成配置文件）
    pushd "${jobdir}" > /dev/null
    sbatch_out=$(sbatch job.sh 2>&1) || { 
      echo "    ❌ 尺度${scale}提交失败: ${sbatch_out}"
      popd > /dev/null
      continue
    }
    echo "    ✅ 尺度${scale}提交成功: ${sbatch_out}"
    popd > /dev/null

  done  # 内层尺度循环结束
  echo ""

done  # 外层目录循环结束

echo "=================================================="
echo "所有任务提交完成，总计 ${total_jobs} 个任务"