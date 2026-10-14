#!/bin/bash
# =============================================================================
# MiniMax-H3 SFT - Stage 1: 数据预处理 (sft:data_process)
#
# 用法：
#   bash scripts/stage1_data_process.sh                              # 默认配置
#   bash scripts/stage1_data_process.sh configs/train/sft_train_1node.yaml # 指定配置
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 激活 conda 环境
. /root/miniconda/etc/profile.d/conda.sh
conda activate minimax_h3_csp

CONFIG="${1:-${SFT_DIR}/configs/train/sft_train_1node.yaml}"

# 从 yaml 读取参数
TRAIN_ARGS=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --train-args --stage stage1)
LOG_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get trainer.log_dir)
# stage1 是数据预处理（无梯度/优化器，无 ZeRO 分片）：每个进程独立加载一份
# text_encoder+VAE（约 43GB/卡）。数据集只有 1 个视频时用 16 进程是浪费
# （15 个 rank 空载各占 43GB），默认单进程；大数据集需要并行时:
#   STAGE1_NUM_PROCESSES=16 bash stage1_data_process.sh
NUM_GPUS="${STAGE1_NUM_PROCESSES:-1}"
DATA_CACHE_DIR=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get dataset.cache_dir)
DATASET_METADATA=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get dataset.metadata_path)
MODEL_BASE_PATH=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.share_dir)
SKIP_DOWNLOAD=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get model.skip_download)
TRAIN_SCRIPT="${SFT_DIR}/../third_party/DiffSynth-Studio/examples/minimax_h3/model_training/train.py"
DIFFSYNTH_ROOT="${SFT_DIR}/../third_party/DiffSynth-Studio"

# 检查前置条件：官方数据集不存在则自动下载（ModelScope: DiffSynth-Studio/diffsynth_example_dataset）
if [[ ! -f "${DATASET_METADATA}" ]]; then
    # local_dir 必须是仓库根目录（.../diffsynth_example_dataset），modelscope 会保留
    # 仓库内相对路径（minimax_h3/MiniMax-H3-FL2VA/...），否则会产生嵌套目录
    DATASET_ROOT="$(dirname "$(dirname "$(dirname "${DATASET_METADATA}")")")"   # .../diffsynth_example_dataset
    echo "[DATA] metadata 不存在: ${DATASET_METADATA}"
    echo "[DATA] 自动下载官方数据集到 ${DATASET_ROOT} ..."
    modelscope download \
        --dataset DiffSynth-Studio/diffsynth_example_dataset \
        --include "minimax_h3/MiniMax-H3-FL2VA/*" \
        --local_dir "${DATASET_ROOT}"
    if [[ ! -f "${DATASET_METADATA}" ]]; then
        echo "错误: 数据集下载后仍未找到 metadata: ${DATASET_METADATA}"
        echo "实际下载到的文件位置如下（供排查目录结构）:"
        find "${DATASET_ROOT}" -type f 2>/dev/null | head -10
        exit 1
    fi
    echo "[DATA] 官方数据集下载完成"
fi

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}"

# 若 train/ val/ 存在（stage2 正在读取），为其中已处理的 .pth 建立软链接到 {rank}/ 目录，
# 使 stage1 断点续传检查（os.path.exists({rank}/{data_id}.pth)）生效并跳过，
# 但不移动/删除 train/ val/ 中的文件，避免影响正在运行的 stage2。
# 文件名格式: {rank}_{data_id}.pth  →  软链接: {rank}/{data_id}.pth -> ../train|val/{rank}_{data_id}.pth
SKIP_COUNT=0
for sub in train val; do
    sub_dir="${DATA_CACHE_DIR}/${sub}"
    [[ -d "${sub_dir}" ]] || continue
    echo "[SKIP-SETUP] 扫描 ${sub}/ 建立软链接（不影响 stage2）..."
    shopt -s nullglob
    for f in "${sub_dir}"/*.pth; do
        fname=$(basename "${f}")
        rank="${fname%%_*}"
        data_id="${fname#*_}"
        data_id="${data_id%.pth}"
        rank_dir="${DATA_CACHE_DIR}/${rank}"
        mkdir -p "${rank_dir}"
        link_path="${rank_dir}/${data_id}.pth"
        if [[ ! -e "${link_path}" ]]; then
            ln -sf "../${sub}/${fname}" "${link_path}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
        fi
    done
    shopt -u nullglob
done
echo "[SKIP-SETUP] 已为 ${SKIP_COUNT} 个已处理 .pth 建立软链接，stage1 将跳过这些样本"

# 环境变量
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD="${SKIP_DOWNLOAD}"
# DIFFSYNTH_MODEL_BASE_PATH 必须指向能 glob 到 model_id 的目录
# config.py 会查找: MODEL_BASE_PATH/MiniMax/MiniMax-H3/FL2VA/...
# 软链接: ~/.cache/modelscope/hub/MiniMax/MiniMax-H3 -> share_dir
MODELSCOPE_CACHE="${HOME}/.cache/modelscope/hub"
if [[ ! -e "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3" ]]; then
    mkdir -p "${MODELSCOPE_CACHE}/MiniMax"
    ln -sf "${MODEL_BASE_PATH}" "${MODELSCOPE_CACHE}/MiniMax/MiniMax-H3"
fi
export DIFFSYNTH_MODEL_BASE_PATH="${MODELSCOPE_CACHE}"
export PYTHONPATH="${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
cd "${DIFFSYNTH_ROOT}"

echo "========================================"
echo " MiniMax-H3 SFT - Stage 1: 数据预处理"
echo "========================================"
echo "配置文件:       ${CONFIG}"
echo "metadata:      ${DATASET_METADATA}"
echo "缓存输出:       ${DATA_CACHE_DIR}"
echo "模型路径:       ${MODEL_BASE_PATH}"
echo "GPU 数量:       ${NUM_GPUS}"
echo "========================================"
echo ""

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage1_data_process_${TIMESTAMP}.log"

accelerate launch \
    --num_processes "${NUM_GPUS}" \
    "${TRAIN_SCRIPT}" \
    ${TRAIN_ARGS} \
    2>&1 | tee "${LOG_FILE}"

# ---- 样本复制扩充：官方数据集仅 1 个视频，复制到 duplicate_to_total 份 ----
# （16 卡 × 100 样本/卡 = 1600；硬链接实现，不占额外磁盘，cp 兜底）
DUP_TOTAL=$(python "${SCRIPT_DIR}/parse_config.py" "${CONFIG}" --get dataset.duplicate_to_total 2>/dev/null || echo "")
[[ "${DUP_TOTAL}" == "None" ]] && DUP_TOTAL=""
if [[ -n "${DUP_TOTAL}" && "${DUP_TOTAL}" -gt 0 ]]; then
    # 现有真实样本数（软链接不算，val 不算；train/ 里的也算，保证重跑幂等）
    CURRENT=$(find "${DATA_CACHE_DIR}" -name '*.pth' ! -type l ! -path '*/val/*' 2>/dev/null | wc -l | tr -d '[:space:]' || true)
    if [[ "${CURRENT}" -lt "${DUP_TOTAL}" ]]; then
        SRC_PTH=$(find "${DATA_CACHE_DIR}" -name '*.pth' ! -type l ! -path '*/val/*' 2>/dev/null | head -1)
        # 防护：一个真实样本都没有 = stage1 编码失败（模型没跑或输出被清），
        # 此时"复制"无源可拷，硬退出报真因，别让 cp 'cannot stat ""' 掩盖问题
        if [[ -z "${SRC_PTH}" ]]; then
            echo "[DUP] 错误: 缓存目录 ${DATA_CACHE_DIR} 中没有真实 .pth 样本，无法复制扩充。"
            echo "[DUP] 请先检查上方 stage1 编码是否成功（accelerate 是否正常结束、{rank}/ 目录是否有输出）"
            exit 1
        fi
        SRC_DIR=$(dirname "${SRC_PTH}")
        echo "[DUP] 当前真实样本 ${CURRENT} < 目标 ${DUP_TOTAL}，开始复制扩充..."
        NEED=$((DUP_TOTAL - CURRENT))
        IDX=1
        CREATED=0
        while [[ ${CREATED} -lt ${NEED} ]]; do
            DST="${SRC_DIR}/dup_$(printf '%04d' "${IDX}").pth"
            IDX=$((IDX + 1))
            [[ -e "${DST}" ]] && continue
            ln "${SRC_PTH}" "${DST}" 2>/dev/null || cp "${SRC_PTH}" "${DST}"
            CREATED=$((CREATED + 1))
        done
        echo "[DUP] 新增 ${CREATED} 份副本（源: ${SRC_PTH}）"
    else
        echo "[DUP] 当前真实样本 ${CURRENT} >= 目标 ${DUP_TOTAL}，跳过复制"
    fi
fi

echo ""
echo "=== Stage 1 完成 ==="
echo "缓存文件数: $(find ${DATA_CACHE_DIR} -name '*.pth' 2>/dev/null | wc -l)"
echo "日志: ${LOG_FILE}"

# 清理临时文件
rm -rf "${SCRIPT_DIR}/__pycache__"
