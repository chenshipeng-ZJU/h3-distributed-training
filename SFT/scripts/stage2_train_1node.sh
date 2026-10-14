#!/bin/bash
# ============================================================================
# ★ 性能加速开关区 —— 唯一改这里。开开关=删对应行行首的 "#"
# 基线 58s/step（TB loss 链实测，fwd~18s/bwd~29s，linear 占 70%）。
# 开任一开关跑 10-20 步：tqdm s/it 或 TB loss 链间隔中位数对比基线，
# 且 loss/grad_norm 曲线必须与基线重合（数值守恒）。一次只开一个。
# 原理详解：论文资料/Agent提示词/P800-适配prompt/03-performance/
# ----------------------------------------------------------------------------
# 稳定性（跑通底线，勿动；实验 D 组时才把 XMLIR_USE_HYDRA_LINEAR 改 1）：
export CUDA_DISABLE_PRINTF=1                  # 关内核 printf
export TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC=12000 # XCCL watchdog 600s→200min，存 ckpt 的 full-gather 不被杀
export PYTORCH_ALLOC_CONF=max_split_size_mb:1024
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

#export XMLIR_CUDNN_ENABLED=1          # 使能 cuDNN（支持 Conv3d 等，建议打开）
export XMLIR_MATMUL_FAST_MODE=1       # MATMUL 计算加速
export XMLIR_FA_GEMM_TYPE="float16"   # Flash Attention 计算加速
export XDNN_USE_FAST_GELU=1           # GELU 计算加速（⚠️ 有精度损失，见下方取值说明）
export H3_SELECTIVE_AC_VERBOSE=0      # 正式 XBLAS A/B 不打印 selective AC 算子日志
export HYDRA_DISPATCH=1               # 仅启用算子 dispatch；XMLIR_USE_HYDRA_LINEAR 仍保持关闭
export DIT_FUSED_NORM=1               # P800 融合 RMSNorm，失败时由模型代码回退 eager
export DIT_FUSED_SWIGLU=1             # P800 融合 SwiGLU，失败时由模型代码回退 eager
export DIT_FUSED_ROPE=1               # P800 融合 RoPE，失败时由模型代码回退 eager
export SELECTIVE_AC=1                 # 与 YAML selective_fa 对齐
export SAC_SAVE_FA=1                  # 只保存 Attention 输出，不保存 FC 输出
# 可选：
export DISABLE_CAST_CACHE=1           # 禁用 cast cache（部分场景需要）
#export XMLIR_FC_TINTER_RES_TYPE="bfloat16"   # Kontext 4K 用
# 历史/精度相关（resnet 文档，可能已在新版修复）：
#export XMLIR_ENABLE_XBLAS_ADDMM=0     # 精度问题，optional
#export XMLIR_BMM_DISPATCH_VALUE=1     # 精度问题，optional
# ----------------------------------------------------------------------------
# A. index_select 修复（代码内置默认开；下面两行是"回退对照"开关）：
export H3_FAST_INDEX_SELECT=1   # 设 0 回退原生路径（one-hot matmul 修复的对照组）
export H3_EMBED_INDEX_ADD=1     # 设 1 回退 index_add_；P800 上 index_copy_ 慢 121x，A/B 变慢先开这行
# B. XBLAS autotune（离线调优好的 gemm.json；验证用 XBLAS_PRINT_CACHE_HITS=1 看 autotune 列命中）：
#export XBLAS_AUTOTUNE_FILE=/path/to/gemm.json  
# D. Fast-FC 前向融合（须同时 XMLIR_USE_HYDRA_LINEAR=1；盯 [cuda_memory] 行跑满 20 步）：
export XMLIR_USE_HYDRA_LINEAR=0               # 关 fc_fusion（backward 多存 input+weight，OOM 根因）
#export XMLIR_ENABLE_FAST_FC_FWD_OUT=1
# E. offload 税（在 yaml 改 use_gradient_checkpointing_offload: false；
#    省每步 124 趟 PCIe，先确认 [cuda_memory] 的 free 余量够）
# ============================================================================
# MiniMax-H3 SFT - Stage 2: 单节点 16 卡 P800（Kunlun XPU），FSDP，
# conda 环境 minimax_h3_csp（对齐同事 minimax_h3：accelerate 1.14.0 / torchcodec 0.9.1）
# 前提：先跑 stage1_data_process.sh 生成 latent 缓存（data/processed/ 下 .pth）
# 用法：
#   bash scripts/stage2_train_1node.sh                          # 默认配置
#   bash scripts/stage2_train_1node.sh --deepspeed-zero2        # ZeRO-2 对照实验
#   bash scripts/stage2_train_1node.sh --resume                 # 找最新 checkpoint 续训
#   bash scripts/stage2_train_1node.sh --xblas-tune             # 生成每个 rank 的 GEMM autotune cache
#   bash scripts/stage2_train_1node.sh --xblas-cache PATH      # 加载合并后的 GEMM cache 并打印命中
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 激活 conda 环境
. /root/miniconda/etc/profile.d/conda.sh
conda activate minimax_h3_csp

# ---- 解析参数 ----
CONFIG="${SFT_DIR}/configs/train/sft_train_1node.yaml"
RESUME_CKPT=""
EXTRA_RESUME_FLAG=""
TRAIN_BACKEND="fsdp2"
# 正式 XBLAS A/B 默认直接加载已生成的 cache；需要重新调优时再显式传 --xblas-tune。
XBLAS_MODE="load"
XBLAS_TUNE_DIR=""
XBLAS_CACHE_FILE="${SFT_DIR}/tune/xblas_h3_20261011_134032/gemm.json"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --deepspeed-zero2)
            TRAIN_BACKEND="deepspeed-zero2"
            shift ;;
        --resume)
            EXTRA_RESUME_FLAG="true"
            shift ;;
        --xblas-tune)
            XBLAS_MODE="generate"
            shift ;;
        --xblas-tune-dir)
            if [[ $# -lt 2 ]]; then
                echo "[XBLAS] --xblas-tune-dir 需要目录参数" >&2
                exit 2
            fi
            XBLAS_TUNE_DIR="$2"
            XBLAS_MODE="generate"
            shift 2 ;;
        --xblas-cache)
            if [[ $# -lt 2 ]]; then
                echo "[XBLAS] --xblas-cache 需要 JSON 文件参数" >&2
                exit 2
            fi
            XBLAS_CACHE_FILE="$2"
            XBLAS_MODE="load"
            shift 2 ;;
            *) CONFIG="$1"; shift ;;
esac
done

# ---- 从 yaml 读取参数 ----
NUM_GPUS=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get accelerate.num_processes)
LOG_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.log_dir)
OUTPUT_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.output_dir)
DATA_CACHE_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get dataset.cache_dir)
MODEL_BASE_PATH=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.share_dir)
SKIP_DOWNLOAD=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.skip_download)
NUM_EPOCHS=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.num_epochs)
TRAIN_SCRIPT="${SFT_DIR}/../third_party/DiffSynth-Studio/examples/minimax_h3/model_training/train.py"
DIFFSYNTH_ROOT="${SFT_DIR}/../third_party/DiffSynth-Studio"
DEEPSPEED_ACCEL_CONFIG="${SFT_DIR}/configs/train/.accelerate_deepspeed_zero2.yaml"
DEEPSPEED_CONFIG="${SFT_DIR}/configs/train/deepspeed_zero2.json"

if [[ "${TRAIN_BACKEND}" == "deepspeed-zero2" ]]; then
    if ! python -c "import deepspeed" >/dev/null 2>&1; then
        echo "[DeepSpeed] 当前 minimax_h3_csp 环境没有可用的 deepspeed，无法启动 ZeRO-2 实验。" >&2
        echo "[DeepSpeed] 当前工程默认保留 FSDP2；请在包含 XPU 兼容 deepspeed 的独立环境中重试。" >&2
        exit 2
    fi
    if [[ ! -f "${DEEPSPEED_CONFIG}" || ! -f "${DEEPSPEED_ACCEL_CONFIG}" ]]; then
        echo "[DeepSpeed] 缺少 ZeRO-2 配置文件:" >&2
        echo "  ${DEEPSPEED_CONFIG}" >&2
        echo "  ${DEEPSPEED_ACCEL_CONFIG}" >&2
        exit 2
    fi
fi

# ---- P800 XPU 环境配置（16 卡单节点）----
# 环境变量已全部集中到文件开头【★ 性能加速开关区】（稳定性 5 个 export +
# 加速开关 A-E 组），改开关去那里，这里只留结构性配置。
# 修复 xpu-smi 加载坏库问题
if [[ -d /usr/local/xre/so ]]; then
    export LD_LIBRARY_PATH="/usr/local/xre/so:${LD_LIBRARY_PATH:-}"
fi

# 卡数自动检测（P800 单节点 16 卡），检测失败回退 yaml 配置
NUM_XPUS=$(xpu-smi -m 2>/dev/null | wc -l | tr -d '[:space:]' || true)
if [[ -z "${NUM_XPUS}" || "${NUM_XPUS}" -le 0 ]]; then
    NUM_XPUS="${NUM_GPUS}"
fi
# 【不设 CUDA_VISIBLE_DEVICES】同事实测：显式设置会破坏 XCCL 拓扑检测
# （找不到 XPULINK ring 回退 PCIE，通信劣化）。16 进程自动映射 0-15 卡。

# BKCL 通信配置
# 本机（P800 PCIe 16 卡，无 XPULINK）历史问题：旧环境 16 卡 allreduce 死锁/拆分通信域。
# 现切换到 minimax_h3_csp（对齐同事环境）+ FSDP（reduce_scatter/all_gather 路径），
# XCCL 新版（9d385bf，build Aug 18）已带 16 卡 butterfly 规避逻辑。
# 训练前建议先验证: bash scripts/xccl_smoke_test.sh 16
# 默认不设任何 BKCL_* 变量；需要调整时在运行前 export 对应变量即可，例如:
#   BKCL_SOCKET_IFNAME=ens109np0 bash scripts/stage2_train_1node.sh

# 注意（来自同事脚本的重要教训，勿改）:
#   - 不要设置 CUDA_DEVICE_ORDER=OAM_ID（XCCL addr_window 问题）
#   - 不要显式设 CUDA_VISIBLE_DEVICES——会破坏 XCCL 拓扑检测（找不到 XPULINK ring
#     会回退 PCIE 劣化性能）！下方 CUDA_VISIBLE_DEVICES 逻辑已删除

# /proc/kunlun 超时：4800000ms（80 分钟）
for (( i=0; i<NUM_XPUS; i++ )); do
    TASK_TIMEOUT_FILE="/proc/kunlun/dev${i}/task_timeout_detect_threshold_in_ms"
    if [[ -w "${TASK_TIMEOUT_FILE}" ]]; then
        echo "4800000" > "${TASK_TIMEOUT_FILE}" || true
    fi
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ---- 自动分割 train/val（若 cache 根目录有未分割的 .pth）----
ROOT_COUNT=$(find "${DATA_CACHE_DIR}" -maxdepth 2 -name '*.pth' 2>/dev/null | grep -v '/train/' | grep -v '/val/' | wc -l || true)
if [[ "${ROOT_COUNT}" -gt 0 ]]; then
    echo "[SPLIT] cache 根目录有 ${ROOT_COUNT} 个未分割 .pth，自动分割 train/val..."
    python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --split-cache
fi

# ---- 断点续训：--resume 命令行 或 yaml resume_train=true，自动找最新 checkpoint ----
IS_RESUME="${EXTRA_RESUME_FLAG}"
if [[ "${IS_RESUME}" != "true" ]]; then
    YAML_RESUME=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.resume_train 2>/dev/null || echo "false")
    [[ "${YAML_RESUME}" == "True" || "${YAML_RESUME}" == "true" ]] && IS_RESUME="true"
fi
RESUME_ARG=""
if [[ "${IS_RESUME}" == "true" ]]; then
    LATEST_CKPT=$(ls -t "${OUTPUT_DIR}"/step-[0-9]*.safetensors 2>/dev/null | grep -v '_ema' | head -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
        RESUME_ARG="--resume_from_checkpoint ${LATEST_CKPT}"
        echo "[RESUME] 从最新 checkpoint 续训: ${LATEST_CKPT}"
    fi
fi

# ---- 环境变量 ----
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD="${SKIP_DOWNLOAD}"
# ModelScope 软链: ~/.cache/modelscope/hub/MiniMax/MiniMax-H3 -> share_dir
MODELSCOPE_CACHE="${HOME}/.cache/modelscope/hub"
if [[ ! -e "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3" ]]; then
    mkdir -p "${MODELSCOPE_CACHE}/MiniMax"
    ln -sf "${MODEL_BASE_PATH}" "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3"
fi
export DIFFSYNTH_MODEL_BASE_PATH="${MODELSCOPE_CACHE}"
# PYTHONPATH 注入 barrier 规避补丁（barrier_fix/sitecustomize.py）：
# XCCL 16 卡 bare barrier 死锁，checkpoint 保存的 wait_for_everyone 会触发；
# 补丁用 16 卡已验证可用的 all_gather 实现 barrier 等价同步。
# 另包含 DiffSynth 根目录（barrier_fix 之后的路径不影响本补丁）
export PYTHONPATH="${SFT_DIR}/scripts/barrier_fix:${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
cd "${DIFFSYNTH_ROOT}"

# ---- XBLAS GEMM autotune ----
# generate: 每个 rank 写独立的 gemm.rankN.json，训练结束后再合并；
# load: 正式训练只读取合并后的 cache，并打印 autotune 命中数。
case "${XBLAS_MODE}" in
    generate)
        if [[ -z "${XBLAS_TUNE_DIR}" ]]; then
            XBLAS_TUNE_DIR="${SFT_DIR}/tune/xblas_h3_$(date +%Y%m%d_%H%M%S)"
        fi
        mkdir -p "${XBLAS_TUNE_DIR}"
        export H3_XBLAS_AUTOTUNE_MODE="generate"
        export H3_XBLAS_TUNE_DIR="${XBLAS_TUNE_DIR}"
        echo "[XBLAS] 调优生成模式"
        echo "[XBLAS] 每个 rank 的 cache 目录: ${XBLAS_TUNE_DIR}"
        echo "[XBLAS] 训练结束后执行:"
        echo "         cat ${XBLAS_TUNE_DIR}/gemm.rank*.json > ${XBLAS_TUNE_DIR}/gemm.json"
        ;;
    load)
        if [[ ! -f "${XBLAS_CACHE_FILE}" ]]; then
            echo "[XBLAS] cache 文件不存在: ${XBLAS_CACHE_FILE}" >&2
            exit 2
        fi
        unset H3_XBLAS_AUTOTUNE_MODE H3_XBLAS_TUNE_DIR
        unset XBLAS_AUTOTUNE XBLAS_AUTOTUNE_WRITEBACK
        export XBLAS_AUTOTUNE_FILTER="gemm"
        export XBLAS_AUTOTUNE_FILE="${XBLAS_CACHE_FILE}"
        unset XBLAS_PRINT_CACHE_HITS
        echo "[XBLAS] cache 加载模式: ${XBLAS_CACHE_FILE}"
        echo "[XBLAS] 已关闭逐 GEMM 命中日志，训练输出保持简洁"
        ;;
    off)
        unset H3_XBLAS_AUTOTUNE_MODE H3_XBLAS_TUNE_DIR
        ;;
    *)
        echo "[XBLAS] 未知模式: ${XBLAS_MODE}" >&2
        exit 2
        ;;
esac

# ---- 生成分布式配置与训练参数 ----
if [[ "${TRAIN_BACKEND}" == "deepspeed-zero2" ]]; then
    ACCEL_CONFIG="${DEEPSPEED_ACCEL_CONFIG}"
else
    ACCEL_CONFIG=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --accelerate-yaml)
fi
TRAIN_ARGS=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --train-args --stage stage2)
# 过滤与 stage2 训练冲突/不支持的参数：
#   --enable_model_cpu_offload: stage1 数据处理专用，FSDP 下模型必须经 accelerator.prepare
#     包装，该参数会绕过 FSDP 包装导致训练失败
#   --val_cache_dir / --val_steps: 当前 train.py / parsers.py 中不存在这些参数，
#     传入会导致 argparse 直接报错退出
TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -e 's/--enable_model_cpu_offload //g' -e 's/--val_cache_dir [^ ]*//g' -e 's/--val_steps [^ ]*//g')

echo "========================================"
echo " MiniMax-H3 SFT - Stage 2: 单节点训练"
echo "========================================"
echo "配置文件:       ${CONFIG}"
echo "数据缓存:       ${DATA_CACHE_DIR}"
echo "输出目录:       ${OUTPUT_DIR}"
echo "GPU 数量:       ${NUM_GPUS}"
echo "Epochs:         ${NUM_EPOCHS}"
echo "训练后端:       ${TRAIN_BACKEND}"
echo "========================================"
echo ""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage2_train_1node_${TIMESTAMP}.log"
# TensorBoard run 命名：每次训练写一个带时间戳的子目录，run 名直观、便于多次训练对比
# （logger.py 读 H3_TB_RUN_NAME 决定子目录；perf_monitor 也写同一目录）
RUN_NAME="run_${TIMESTAMP}"
export H3_TB_RUN_NAME="${RUN_NAME}"
TB_LOG_DIR="${OUTPUT_DIR}/tensorboard_log/${RUN_NAME}"
mkdir -p "${TB_LOG_DIR}"

# ---- 后台压测采样：每 30s 把 XPU 显存/利用率/功耗写入 TensorBoard（与训练同一 event 目录）----
python "${SCRIPT_DIR}/perf_monitor.py" "${TB_LOG_DIR}" 30 2>/dev/null &
PERF_PID=$!
trap 'kill ${PERF_PID} 2>/dev/null || true' EXIT

# ---- 启动训练（默认 FSDP2；--deepspeed-zero2 时使用 ZeRO-2 对照配置）----
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    ${TRAIN_ARGS} \
    ${RESUME_ARG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Stage 2 训练完成 ==="
echo "Checkpoint 目录: ${OUTPUT_DIR}"
ls -t "${OUTPUT_DIR}"/*.safetensors 2>/dev/null | head -5 || true
echo "TensorBoard: ${TB_LOG_DIR}（loss + grad_norm + perf/*，tensorboard --logdir 即可查看）"
echo "日志:      ${LOG_FILE}"

# 清理临时文件
rm -rf "${SCRIPT_DIR}/__pycache__"
