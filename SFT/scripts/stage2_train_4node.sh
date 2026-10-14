#!/bin/bash
# =============================================================================
# MiniMax-H3 SFT - Stage 2: 单节点 16 卡 P800 训练 (1 node × 16 XPU)
# （原 4 节点脚本改造为单节点：去掉 mpirun/ssh 跨机编排，直接本地 torchrun）
#
# 要点：
#   1. 单节点直接启动 torchrun，nproc_per_node 由 xpu-smi 自动检测（P800=16）
#   2. 平台注入 TF_CONFIG 时自动解析（单节点任务 worker 数为 1）
#   3. P800 BKCL RDMA 网卡自动配置
#   4. /proc/kunlun 超时设为 4800000ms（80 分钟）
#
# 用法：
#   bash scripts/stage2_train_4node.sh                              # 默认配置（官方数据集）
#   bash scripts/stage2_train_4node.sh configs/train/sft_train_1node.yaml
#   bash scripts/stage2_train_4node.sh --resume /path/to/ckpt       # 断点续训
#
# 前提：
#   - 单台 16 卡 P800 机器
#   - 平台自动注入 TF_CONFIG（可选，无则按单节点处理）
#   - conda 环境名：minimaxh3_xpt260821
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 默认用官方数据集配置（pet 数据集已弃用；如需 pet 配置请显式传 configs/train/sft_train_4node.yaml）
CONFIG="${SFT_DIR}/configs/train/sft_train_1node.yaml"
RESUME_CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME_CKPT="$2"; shift 2 ;;
        *) CONFIG="$1"; shift ;;
    esac
done

HOSTFILE_PATH="/job/hostfile"

# =============================================================================
# 广播 kill：Ctrl+C 时杀掉所有节点的 python 进程
# =============================================================================
broadcast_kill() {
    local hosts=""
    if [ -f "$HOSTFILE_PATH" ]; then
        hosts=$(awk '!/^#/ && NF {print $1}' "$HOSTFILE_PATH" | sort -u)
    fi

    if [ -n "$hosts" ]; then
        echo "Broadcasting: pkill -9 python to the following hosts:"
        for h in $hosts; do
            echo "  -> $h"
            nohup ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 "$h" "pkill -9 python || true" >/dev/null 2>&1 &
        done
        wait || true
    else
        echo "No hosts found in $HOSTFILE_PATH or file missing; skipping remote kill."
    fi

    echo "  -> localhost"
    pkill -9 python >/dev/null 2>&1 || true
}

cleanup() {
    echo ""
    echo "[Trap] Caught interrupt/termination signal. Stopping training on all machines..."
    broadcast_kill
    pkill -9 mpirun >/dev/null 2>&1 || true
    echo "[Trap] Cleanup done. Exiting."
    exit 130
}

trap cleanup INT TERM

# =============================================================================
# run_train：训练函数（单节点由主逻辑直接调用）
# 参数: $1 = CONFIG, $2 = RESUME_ARG, $3 = SCRIPT_DIR, $4 = SFT_DIR
# =============================================================================
run_train() {
    local CONFIG="$1"
    local RESUME_ARG="$2"
    local SCRIPT_DIR="$3"
    local SFT_DIR="$4"

    # ---- conda 环境 ----
    . /root/miniconda/etc/profile.d/conda.sh
    source /etc/profile

    # 修复 xpu-smi 加载坏库问题
    if [[ -d /usr/local/xre/so ]]; then
        export LD_LIBRARY_PATH="/usr/local/xre/so:${LD_LIBRARY_PATH:-}"
    fi

    conda activate minimaxh3_xpt260821

    # ---- 从 TF_CONFIG_B64 恢复 TF_CONFIG（SSH 继承时用 base64 编码）----
    if [[ -z "${TF_CONFIG:-}" && -n "${TF_CONFIG_B64:-}" ]]; then
        export TF_CONFIG=$(echo -n "$TF_CONFIG_B64" | base64 -d)
    fi

    # ---- 从 TF_CONFIG 解析节点信息（单节点任务可能无 TF_CONFIG，默认 1 节点）----
    if [[ -n "${TF_CONFIG:-}" ]]; then
        local NNODES=$(echo $TF_CONFIG | python -c 'import sys,json;print(len(json.load(sys.stdin)["cluster"]["worker"]))')
        local WORKER_MASTER=$(echo $TF_CONFIG | python -c 'import sys,json;print(json.load(sys.stdin)["cluster"]["worker"][0].split(":")[0])')
        local MACHINE_RANK=$(echo $TF_CONFIG | python -c 'import sys, json; y=json.load(sys.stdin);print(json.dumps(y["task"]["index"]))' | sed 's/"//g')
    else
        local NNODES=1
        local WORKER_MASTER="127.0.0.1"
        local MACHINE_RANK=0
    fi
    local NPROC_PER_NODE=$(xpu-smi -m | wc -l)
    if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -le 0 ]]; then
        echo "[ERROR] xpu-smi -m failed"
        exit 1
    fi
    local NUM_PROCESSES=$(($NNODES * $NPROC_PER_NODE))
    local MAIN_PROCESS_PORT=8008

    if [ -z "${OMP_NUM_THREADS:-}" ]; then
        export OMP_NUM_THREADS=8
    fi

    echo "====== 单节点训练环境 ======"
    echo "NNODES: ${NNODES}"
    echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
    echo "WORKER_MASTER: ${WORKER_MASTER}"
    echo "MACHINE_RANK: ${MACHINE_RANK}"
    echo "NUM_PROCESSES: ${NUM_PROCESSES}"
    echo "MAIN_PROCESS_PORT: ${MAIN_PROCESS_PORT}"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "========================"

    # ---- P800 XPU 加速 ----
    export CUDA_DISABLE_PRINTF=1
    # socket 网卡自动检测默认路由网卡（不能写死 eth0，不同机器网卡名不同）
    local default_if=$(ip route show default 2>/dev/null | awk '{print $5; exit}' || true)
    export BKCL_SOCKET_IFNAME="${BKCL_SOCKET_IFNAME:-${default_if:-eth0}}"

    # BKCL RDMA 网卡自动配置：取 ACTIVE 的 RDMA 网卡，排除 socket/bootstrap 管理网卡
    local rdma_nics=$(rdma link show 2>/dev/null | awk -v skip="${BKCL_SOCKET_IFNAME}" '
        /state ACTIVE/ {
            nic = ""
            for (i = 1; i < NF; i++) if ($i == "netdev") nic = $(i+1)
            if (nic != "" && nic != skip) print nic
        }' | paste -sd, -)
    if [ -n "$rdma_nics" ]; then
        export BKCL_RDMA_NICS="${BKCL_RDMA_NICS:-$rdma_nics}"
        echo "[BKCL] RDMA 网卡: ${BKCL_RDMA_NICS}（socket/bootstrap: ${BKCL_SOCKET_IFNAME}）"
    fi

    # BKCL 通用设置
    export BKCL_ENABLE_XDR=1
    export BKCL_USE_AR=1
    export XPU_FORCE_SHARED_DEVICE_CONTEXT=1

    # XPU 计算加速
    export XMLIR_CUDNN_ENABLED=1
    export XPUAPI_TF32_ROUND_MODE="rne"
    export USE_FAST_BF16_FC=1
    export XMLIR_MATMUL_FAST_MODE=1
    export XMLIR_FA_GEMM_TYPE="float16"
    export XDNN_USE_FAST_GELU=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # ---- /proc/kunlun 超时：4800000ms（80 分钟） ----
    for (( i=0; i<NPROC_PER_NODE; i++ )); do
        echo "4800000" > /proc/kunlun/dev$i/task_timeout_detect_threshold_in_ms 2>/dev/null || true
    done

    # ---- 从 yaml 读取参数 ----
    local TRAIN_ARGS=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --train-args --stage stage2)
    local MIXED_PRECISION=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get accelerate.mixed_precision)
    local SKIP_DOWNLOAD=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.skip_download)
    local MODEL_BASE_PATH=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.share_dir)
    local DIFFSYNTH_ROOT="${SFT_DIR}/../third_party/DiffSynth-Studio"
    local TRAIN_SCRIPT="${DIFFSYNTH_ROOT}/examples/minimax_h3/model_training/train.py"

    # ---- DiffSynth 环境变量 ----
    export TOKENIZERS_PARALLELISM=false
    export DIFFSYNTH_SKIP_DOWNLOAD="${SKIP_DOWNLOAD}"
    export WANDB_MODE=disabled

    # ModelScope 软链
    local MODELSCOPE_CACHE="${HOME}/.cache/modelscope/hub"
    if [[ ! -e "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3" ]]; then
        mkdir -p "${MODELSCOPE_CACHE}/MiniMax"
        ln -sf "${MODEL_BASE_PATH}" "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3"
    fi
    export DIFFSYNTH_MODEL_BASE_PATH="${MODELSCOPE_CACHE}"
    # 注入 XPU rank 级设备隔离补丁（见 scripts/xpu_rank_fix/sitecustomize.py）
    export PYTHONPATH="${SFT_DIR}/scripts/xpu_rank_fix:${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
    cd "${DIFFSYNTH_ROOT}"

    # ---- 将环境导出到 /root/.ssh/environment（多节点 SSH 继承用，单节点不需要，保留备用）----
    # TF_CONFIG 是 JSON 多行格式，SSH environment 不支持，需要排除
    if [[ "${NNODES}" -gt 1 ]]; then
        mkdir -p /root/.ssh
        env | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | grep -v '^TF_CONFIG=' > /root/.ssh/environment
        # TF_CONFIG 单独用 base64 编码成单行
        echo "TF_CONFIG_B64=$(echo -n "$TF_CONFIG" | base64 -w 0)" >> /root/.ssh/environment
    fi

    echo ">>>>> BKCL_RDMA_NICS: ${BKCL_RDMA_NICS}"

    # ---- 启动训练（用 torchrun + DeepSpeed ZeRO-3，单节点 16 卡）----
    # train.py 通过 USE_DEEPSPEED 环境变量感知是否启用 DeepSpeed
    export USE_DEEPSPEED=1
    export DEEPSPEED_ZERO_STAGE=3
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE-1)))}
    set -x
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --node_rank="${MACHINE_RANK}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${WORKER_MASTER}:${MAIN_PROCESS_PORT}" \
        --rdzv_id=5235 \
        --max_restarts=0 \
        "${TRAIN_SCRIPT}" \
        ${TRAIN_ARGS} \
        ${RESUME_ARG}
    set +x
}

# =============================================================================
# 主逻辑：单节点（rank 0）直接本地启动训练；若平台仍注入多节点 TF_CONFIG
# 且当前节点非 rank 0，则保留等待逻辑（兼容误配）
# =============================================================================

# 从 yaml 读取日志/输出目录
LOG_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.log_dir)
OUTPUT_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.output_dir)
DATA_CACHE_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get dataset.cache_dir)
CACHE_COUNT=$(find "${DATA_CACHE_DIR}" -name '*.pth' 2>/dev/null | wc -l)

# resume 参数
RESUME_ARG=""
if [[ -n "${RESUME_CKPT}" ]]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_CKPT}"
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 自动分割 train/val（若根目录有未分割的 .pth）
ROOT_COUNT=$(find "${DATA_CACHE_DIR}" -maxdepth 2 -name '*.pth' 2>/dev/null | grep -v '/train/' | grep -v '/val/' | wc -l || true)
if [[ "${ROOT_COUNT}" -gt 0 ]]; then
    echo "[SPLIT] 根目录有 ${ROOT_COUNT} 个未分割 .pth，自动分割 train/val..."
    python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --split-cache
fi

# 判断是否续训
IS_RESUME=false
if [[ -n "${RESUME_CKPT}" ]]; then IS_RESUME=true; fi
if [[ "${IS_RESUME}" == "false" ]]; then
    RESUME_FROM_YAML=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.resume_train 2>/dev/null || echo "false")
    if [[ "${RESUME_FROM_YAML}" == "True" ]]; then IS_RESUME=true; fi
fi

# resume_train=true 时自动找最新 checkpoint（排除 _ema）
if [[ "${IS_RESUME}" == "true" && -z "${RESUME_CKPT}" ]]; then
    LATEST_CKPT=$(ls -t "${OUTPUT_DIR}"/step-[0-9]*.safetensors 2>/dev/null | grep -v '_ema' | head -1)
    if [[ -n "${LATEST_CKPT}" ]]; then
        RESUME_CKPT="${LATEST_CKPT}"
        RESUME_ARG="--resume_from_checkpoint ${RESUME_CKPT}"
        echo "[RESUME] 自动找到最新 checkpoint: ${RESUME_CKPT}"
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage2_train_16card_${TIMESTAMP}.log"
TB_LOG_DIR="${OUTPUT_DIR}/tensorboard_log"

# 后台压测采样：写入 TensorBoard（与训练同一 event 目录）
python "${SCRIPT_DIR}/perf_monitor.py" "${TB_LOG_DIR}" 30 2>/dev/null &
PERF_PID=$!

# 从 TF_CONFIG 获取当前节点 rank（无 TF_CONFIG 时默认 0，即单节点）
MACHINE_RANK=$(echo "${TF_CONFIG:-}" | python -c 'import sys, json; y=json.load(sys.stdin);print(json.dumps(y["task"]["index"]))' 2>/dev/null | sed 's/"//g' || true)
[[ -z "${MACHINE_RANK}" ]] && MACHINE_RANK=0

echo "========================================"
echo " MiniMax-H3 SFT - Stage 2: 单节点 16 卡训练"
echo "========================================"
echo "配置文件:       ${CONFIG}"
echo "数据缓存:       ${DATA_CACHE_DIR} (${CACHE_COUNT} files)"
echo "输出目录:       ${OUTPUT_DIR}"
echo "日志:           ${LOG_FILE}"
echo "当前节点 rank:  ${MACHINE_RANK}"
echo "========================================"
echo ""

if [ "${MACHINE_RANK}" -eq 0 ]; then
    echo ">>> 单节点 (Rank 0): 直接启动 torchrun..."
    run_train "${CONFIG}" "${RESUME_ARG}" "${SCRIPT_DIR}" "${SFT_DIR}" \
        2>&1 | tee "${LOG_FILE}"
else
    echo ">>> Worker node (Rank ${MACHINE_RANK}): Waiting for commands from Master..."
    sleep infinity
fi

# 停止压测采样
kill ${PERF_PID} 2>/dev/null || true

echo ""
echo "=== Stage 2 训练完成 ==="
echo "Checkpoint 目录: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -name '*.safetensors' -o -name '*.pt' 2>/dev/null | head -10
echo "TensorBoard: ${TB_LOG_DIR}"
echo "日志: ${LOG_FILE}"
