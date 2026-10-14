#!/bin/bash
# =============================================================================
# MiniMax-H3 - Stage 3: SFT 模型推理测试
#
# 用法：
#   bash scripts/stage3_infer_sft.sh                                    # 自动找最新 checkpoint
#   bash scripts/stage3_infer_sft.sh --prompt "你的prompt"              # 覆盖 prompt
#   bash scripts/stage3_infer_sft.sh --ckpt /path/to/step-500.safetensors  # 指定 checkpoint
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 激活 conda 环境
. /root/miniconda/etc/profile.d/conda.sh
conda activate minimaxh3_xpt260821

CONFIG="${SFT_DIR}/configs/infer/infer_1node.yaml"
DIFFSYNTH_ROOT="${SFT_DIR}/../third_party/DiffSynth-Studio"

# 从 yaml 读取参数
SHARE_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.share_dir)
SKIP_DOWNLOAD=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.skip_download)
OUTPUT_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get inference.output_dir)
NUM_GPUS=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get num_processes)
LOG_DIR="/data/vjuicefs_ai_hz_model_wl/public_data/kunlun/csp/H3/SFT/logs"

# 环境变量
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD="${SKIP_DOWNLOAD}"
MODELSCOPE_CACHE="${HOME}/.cache/modelscope/hub"
if [[ ! -e "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3" ]]; then
    mkdir -p "${MODELSCOPE_CACHE}/MiniMax"
    ln -sf "${SHARE_DIR}" "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3"
fi
export DIFFSYNTH_MODEL_BASE_PATH="${MODELSCOPE_CACHE}"
export PYTHONPATH="${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
# P800 XPU 加速
export XMLIR_CUDNN_ENABLED=1
export XPUAPI_TF32_ROUND_MODE="rne"
export USE_FAST_BF16_FC=1
export XMLIR_MATMUL_FAST_MODE=1
export XMLIR_FA_GEMM_TYPE="float16"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${DIFFSYNTH_ROOT}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo " MiniMax-H3 - Stage 3: SFT 模型推理"
echo "========================================"
echo "配置文件:       ${CONFIG}"
echo "输出目录:       ${OUTPUT_DIR}"
echo "GPU 数量:       ${NUM_GPUS}"
echo "========================================"
echo ""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage3_infer_${TIMESTAMP}.log"

accelerate launch \
    --num_processes "${NUM_GPUS}" \
    "${SCRIPT_DIR}/stage3_infer.py" "${CONFIG}" "$@" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Stage 3 推理完成 ==="
echo "输出文件:"
find "${OUTPUT_DIR}" -name '*.mp4' 2>/dev/null | head -20
echo "日志: ${LOG_FILE}"

# 清理临时文件
rm -rf "${SCRIPT_DIR}/__pycache__"
