#!/bin/bash
# =============================================================================
# H3 P800 环境对齐清单 + 一键建环境 + 自检（昆仑 XPU 版）
# 面向新机器/新用户从 0 复现：环境的 conda 名统一为 minimax_h3_csp（与训练脚本一致）
#
# 推荐镜像（含 xpytorch 全套依赖与 python310_torch29_cuda 母本环境）：
#   https://klx-sdk-release-public.su.bcebos.com/v1/XECV/vision-train-release/images/xpytorch_torch29py310_v0.42_0602_vision.tar
# 若要跑 DeepSpeed ZeRO：用标准 release 覆盖装 xpytorch 3.6.2.1（其 XCCL 已修复 16 卡 allreduce）：
#   https://klx-sdk-release-public.su.bcebos.com/xpytorch/release/3.6.2.1/xpytorch-cp310-torch290-ubuntu2004-x64.run
#
# torch 三件套说明：昆仑定制版（torch_xmlir 提供 CUDA 兼容层），镜像母本环境自带，
# 从 python310_torch29_cuda clone 即得，无需任何手动安装。
#
# 两种建环境方式（效果等价，任选其一）：
#   方式一（脚本自动建，推荐小白）：bash setup_env.sh setup minimax_h3_csp
#   方式二（手动逐条执行）：见文档第二节第 2 步
# 建完自检：bash setup_env.sh check minimax_h3_csp
# =============================================================================
set -euo pipefail

# ============================================================
# 第一部分：环境清单（对照表，来自已验证环境快照）
# ============================================================
# ┌──────────────────┬──────────────────────────────────────────────┐
# │ 组件              │ P800 已验证值（minimax_h3_csp）                │
# ├──────────────────┼──────────────────────────────────────────────┤
# │ Python           │ 3.10.15                                      │
# │ torch            │ 2.9.0+cu129（torch_xmlir 定制，内部 XPU 源）    │
# │ torchvision      │ 0.24.0+cu129                                 │
# │ torchaudio       │ 2.9.0+cu129                                  │
# │ torchcodec       │ 0.9.1（★0.16.0+cpu 在 XPU 上 undefined symbol）│
# │ torch_xray       │ 2.0.4（精度测试用，训练非必需）                  │
# │ accelerate       │ 1.14.0（★1.2.1 的 FSDP 行为有差异）             │
# │ transformers     │ 5.16.1                                       │
# │ peft             │ 0.12.0                                       │
# │ tokenizers       │ 0.23.2                                       │
# │ safetensors      │ 0.8.0                                        │
# │ numpy            │ 1.26.4                                       │
# │ einops           │ 0.8.2                                        │
# │ tensorboard      │ 2.20.0                                       │
# │ av (PyAV)        │ 16.0.1（音频兜底）                             │
# │ huggingface_hub  │ 1.30.0                                       │
# │ imageio           │ 2.37.4 + imageio-ffmpeg 0.6.0                  │
# │                   │ （★新版走 pyav 插件，get_meta_data 直接崩）      │
# │ flash_attn       │ XPU 私有实现（flash_attn wheel 来自内部源）      │
# │ hydrax/xmlir   │ 0.1.0 / 1.0.0.1（内部 wheel，随 XPU 源）       │
# │ XPU 栈基线     │ XPYTORCH p800_release_v3.6.2_torch29           │
# │                │ XCCL 3.1.7.0（_versions.txt 为准）             │
# │ deepspeed        │ 不装（import 副作用干扰 XPU 栈）                │
# │ diffusers        │ 不装                                          │
# └──────────────────┴──────────────────────────────────────────────┘
#
# 系统层（镜像自带，无需 pip）：
#   基础镜像  xmlir_ubuntu_2004_x86_64:v0.42_0602_vision 系
#   驱动      xpu-smi 显示 Driver 5.33.0.0
#   XCCL      _versions.txt 为准（旧机器实测 9d385bf build Aug 18 / vision_0821 包为 3.1.7.0）
#
# ✅ 训练节点 实测（2026-10-08）：本机镜像的 python310_torch29_cuda 母本自带正式版
#   xpytorch（xmlir wheel → torch_xmlir，p800_release_v3.6.2_torch29+2026062614，
#   XCCL 3.1.7.0），clone 即得，**无需再装任何 xpytorch run 包**。
# ⚠ 勿在训练环境/母本里跑 vision_0821 的 .run：它的 xmlir-1.0.0.1 wheel 与母本自带的
#   版本号相同（pip 看不出差别，只有 torch_xmlir/_versions.txt 能分辨），装上会把
#   0626 正式版覆盖成 -dev+2026082114 dev 构建（自检 WARN"dev 可能符号不配套"）。
#   h3a 母本已被覆盖过一次（2026-10-08）；需纯正式版母本请用 h3b 的。
#   真要跑 DeepSpeed ZeRO 16 卡 allreduce 修复时才用标准 release 3.6.2.1（上面第 9 行 URL）。
#
# 训练必须的环境变量（stage2_train_1node.sh 已内置，勿删）：
#   XMLIR_USE_HYDRA_LINEAR=0          # 关 fc_fusion 算子（backward 多存显存，必关）
#   TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC=12000  # XCCL watchdog 默认 600s 会杀慢 checkpoint
#   PYTORCH_ALLOC_CONF=max_split_size_mb:1024
#   ★不要设 CUDA_VISIBLE_DEVICES（破坏 XCCL 拓扑检测 → 回退 PCIE 通信劣化）
#   ★不要设 CUDA_DEVICE_ORDER=OAM_ID（addr_window 问题）
#
# 必须的代码补丁（third_party/DiffSynth-Studio，工程包已内置，勿删）：
#   train.py    stage2 checkpoint 开关以命令行为准（防 stage1 缓存锁死→激活爆显存）
#   runner.py   FSDP 不整体搬模型上卡（防 47GB 隐形驻留）；每步打印 [cuda_memory]
#   logger.py   TensorBoard / grad_norm / save_dtype
#   operators.py 音频 PyAV 兜底
#   SFT/scripts/barrier_fix/  XCCL 16 卡 barrier 死锁规避（PYTHONPATH 自动注入）
#
# 性能剖析（xprofiler 2.0.2.12，训练节点 2026-10-08 已实测两种用法）：
#   工具：third_party/tools/xprofiler-Linux_x86_64-2.0.2.12（新拷贝需 chmod +x bin/xprofiler）
#   输出目录：SFT/output/profiling/
#   A. 系统级采样（无需重启，对正在跑的训练直接采，10s 探针实测无扰动）：
#      xprofiler -t60 -x all --xputx --report --gzip --socket-path=/tmp/ -e out.json
#      产出 out.json（Chrome Tracing 格式：chrome://tracing 或 ui.perfetto.dev 打开）
#      ⚠ --gzip 落盘的 .json 实为 gzip 流，Chrome 直接加载会报
#        "Couldn't create an importer"——先 gzip -dc out.json > out_plain.json 再加载
#      ⚠ 体积上限：注入模式 8 卡全采很大（180s≈4GB 解压后），超过 Chrome/V8
#        单字符串 ~512MB 上限，加载即空白/SyntaxError——用 scripts/slice_trace.py
#        按 pid+时间窗切出 <500MB 子集再加载；或采集时 -x 0 只采单卡（FSDP 对称）
#      + report CSV（每卡 kernel 统计）。局限：无 API/NVTX 归因（目标进程未注入）。
#   B. 注入模式（重启训练，得到 CUDA API + XPUTX range + kernel 对齐的完整 timeline）：
#      （带 5 个注入 env 启动过的训练，collector 可中途挂接/换新而不必重启训练——
#        2026-10-08 180s 实测：运行中 rank 主动 connect 新 collector，match rate 100%）
#      训练启动前 export 5 个变量（XPUTX 客户端在进程启动时连接 collector）：
#        XPU_ENABLE_PROFILER_TRACING=1  XPU_TRACING_OUTPUT_NAME=/tmp/xprofiler.sock
#        NVTX_INJECTION64_PATH=<工具>/so/libxpuToolsExt.so
#        XPUTX_RUN_MODE=client  XPUTX_LISTEN_ADDR=unix:/tmp/xputx.sock
#      collector 先起后跑训练：xprofiler --report -r500 --xputx --cpu -t600 -x all
#        --gzip --socket-path=/tmp/ -e out.json（-t600 到点自停，训练可继续跑）
#      精准窗口：daemon 模式（-d --sync --trace-count=1）+ train.py 内 import
#      pybind_xprofiler 后 cuda_profiler_start()/stop()（wheel 在 <工具>/python/，
#      已装入训练环境；import 需 export LD_LIBRARY_PATH=<工具>/so:$LD_LIBRARY_PATH，
#      否则报 libxpurt.so.2 找不到）

# ============================================================
# 第二部分：从零搭建（路径 B，在 P800 容器内执行）
# ============================================================
if [[ "${1:-}" == "setup" ]]; then
    ENV_NAME="${2:?用法: setup_env.sh setup <环境名>}"
    source /root/miniconda/etc/profile.d/conda.sh
    # ★从镜像自带的母本环境获得 torch 三件套 + xpytorch 全套依赖
    #   （不要从零 create——会缺 xpytorch，极易装成不可用环境）
    # ⚠ 训练节点 实测（2026-10-08）：conda create --clone 要访问 conda.anaconda.org，
    #   本网络 TLS 拦截（自签证书）→ SSL_CERTIFICATE_VERIFY_FAILED，clone 半途失败
    #   会留下没有 bin/python 的空壳环境（activate 不报错，python/pip 静默落到
    #   base 的 py3.13，pip 会误装整套 CUDA torch）。因此先试 --offline 的 conda
    #   clone，不行就走离线文件克隆（与母本字节等效，已在本机验证）。
    SRC_ENV=/root/miniconda/envs/python310_torch29_cuda
    DST_ENV=/root/miniconda/envs/${ENV_NAME}
    if [[ ! -x "${DST_ENV}/bin/python" ]]; then
        conda create --offline -n "${ENV_NAME}" --clone python310_torch29_cuda || echo "[setup] conda clone 失败，改用离线文件克隆"
    fi
    if [[ ! -x "${DST_ENV}/bin/python" ]]; then
        echo "[setup] 离线文件克隆 ${SRC_ENV} -> ${DST_ENV} ..."
        rm -rf "${DST_ENV}"
        cp -a -n "${SRC_ENV}/." "${DST_ENV}/"
        # bin 整目录重建：二进制与母本字节一致，只对文本脚本重写路径。
        # grep 必须 -I 跳过二进制——conda 的 python3.10 二进制内嵌母本前缀字符串，
        # 误 sed 会导致字节长度错位 → python -V 段错误（训练节点 实测踩过）
        rm -rf "${DST_ENV}/bin"
        cp -a "${SRC_ENV}/bin" "${DST_ENV}/bin"
        sed_targets=$(grep -rlI "envs/python310_torch29_cuda" "${DST_ENV}/bin" || true)
        if [[ -n "${sed_targets}" ]]; then
            echo "${sed_targets}" | xargs -r sed -i "s|envs/python310_torch29_cuda|envs/${ENV_NAME}|g"
        fi
    fi
    conda activate "${ENV_NAME}"
    # 守卫：绝不带着残缺环境往下走（否则 pip 落到 base 误装 CUDA torch）
    if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "❌ ${ENV_NAME} 创建失败：${CONDA_PREFIX}/bin/python 不存在"
        exit 1
    fi

    # 训练框架层（版本固定，勿改；母本镜像自带的是 accelerate 1.2.1 / transformers
    # 4.49.0 等旧版，必须经下面这些 pip 升到已验证版本）
    # 国内网络慢时走清华 TUNA 镜像（bcebos 内部源不受影响，只影响 pypi 下载）
    PIP_MIRROR_ARGS="--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
    # 显式用环境内 pip（防 PATH 异常回落 base）
    PIP="${CONDA_PREFIX}/bin/pip"
    ${PIP} install ${PIP_MIRROR_ARGS} accelerate==1.14.0 transformers==5.16.1 peft==0.12.0
    ${PIP} install ${PIP_MIRROR_ARGS} tokenizers==0.23.2 safetensors==0.8.0 numpy==1.26.4 einops==0.8.2
    ${PIP} install ${PIP_MIRROR_ARGS} huggingface_hub==1.30.0 tensorboard==2.20.0
    ${PIP} install ${PIP_MIRROR_ARGS} "av==16.0.1" torchcodec==0.9.1 pandas modelscope ftfy pillow
    ${PIP} install ${PIP_MIRROR_ARGS} imageio==2.37.4 imageio-ffmpeg==0.6.0
    echo ""
    echo "✅ 环境创建完成。下一步自检：bash setup_env.sh check ${ENV_NAME}"
fi

# ============================================================
# 第三部分：自检（对任意已有环境执行：bash setup_env.sh 或 bash setup_env.sh check <环境名>）
# ============================================================
if [[ "${1:-}" == "check" ]]; then
    source /root/miniconda/etc/profile.d/conda.sh
    conda activate "${2:?用法: setup_env.sh check <环境名>}"
    # 守卫：python 必须解析到目标环境内。曾发生：环境残缺时 python/pip 落到
    # base（py3.13），13 项依赖全 OK 的假象（查的全是 base 的包，torch 是误装
    # 的 CUDA 2.14.1+cu130，torch_xmlir/device_count 才暴露真相）
    if [[ "$(command -v python)" != "${CONDA_PREFIX}/bin/python" ]]; then
        echo "❌ python 解析到 $(command -v python)（$(python -V 2>&1)），不在 ${CONDA_PREFIX} 内——"
        echo "   环境残缺或未激活，自检结果不代表训练环境，已中止。"
        exit 1
    fi
fi

# 无参数直接运行时同样防呆：禁止在 base / 未激活状态下自检（结果无意义）
if [[ "${1:-}" != "setup" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "base" || -z "${CONDA_DEFAULT_ENV:-}" || "$(command -v python)" != "${CONDA_PREFIX:-}/bin/python" ]]; then
        echo "❌ 自检必须在目标环境激活后执行（当前: ${CONDA_DEFAULT_ENV:-未激活}, python: $(command -v python)）"
        echo "   用法: bash setup_env.sh check <环境名>"
        exit 1
    fi
fi

echo ""
echo "================ P800 环境自检 ================"
check() {  # $1=模块名 $2=期望版本
    actual=$(python -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo "NOT_INSTALLED")
    if [[ "$actual" == "$2" ]]; then echo "  OK   $1==$actual"
    else echo "  FAIL $1: 期望 $2, 实际 $actual"; fi
}
check accelerate 1.14.0
check transformers 5.16.1
check peft 0.12.0
check tokenizers 0.23.2
check safetensors 0.8.0
check numpy 1.26.4
check einops 0.8.2
check huggingface_hub 1.30.0
check imageio 2.37.4
check imageio-ffmpeg 0.6.0
check tensorboard 2.20.0
check av 16.0.1
check torchcodec 0.9.1

# XPU 栈基线（来自已验证环境 _versions.txt；换基线改这两行，或临时覆盖：
#   EXPECTED_XCCL=3.1.7.0 bash setup_env.sh check <环境名>）
export EXPECTED_XPYTORCH="${EXPECTED_XPYTORCH:-p800_release_v3.6.2_torch29}"
export EXPECTED_XCCL="${EXPECTED_XCCL:-3.1.7.0}"

python - <<'PY'
import os
import torch
print("  torch", torch.__version__, "| cuda_available:", torch.cuda.is_available(), "| device_count:", torch.cuda.device_count())
try:
    import torch_xmlir
    print("  OK   torch_xmlir 可用（XPU 栈正常）")
    # _versions.txt 是版本权威来源，全量打印
    vf = os.path.join(os.path.dirname(torch_xmlir.__file__), "_versions.txt")
    if os.path.exists(vf):
        print("  ---- _versions.txt ----")
        vers = {}
        for line in open(vf):
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            vers[k.strip()] = v.strip()
            print("       " + line)
        # 与基线核对
        exp_xp = os.environ.get("EXPECTED_XPYTORCH", "")
        exp_xc = os.environ.get("EXPECTED_XCCL", "")
        xp, xc = vers.get("XPYTORCH", ""), vers.get("XCCL", "")
        if not xp:
            print("  WARN  _versions.txt 无 XPYTORCH 行")
        elif "-dev" in xp:
            print(f"  WARN  xpytorch 是 dev 构建: {xp}（正式版为 p800_release_*，dev 可能符号不配套）")
        elif exp_xp and exp_xp not in xp:
            print(f"  WARN  xpytorch 与基线不一致: 期望含 {exp_xp}, 实际 {xp}")
        else:
            print(f"  OK   xpytorch 版本核对通过: {xp}")
        if not xc:
            print("  WARN  _versions.txt 无 XCCL 行")
        elif exp_xc and xc != exp_xc:
            print(f"  WARN  XCCL 与基线不一致: 期望 {exp_xc}, 实际 {xc}（换栈须重跑 xccl_smoke_test.sh）")
        else:
            print(f"  OK   XCCL 版本核对通过: {xc}")
    else:
        print("  WARN  未找到 _versions.txt:", vf)
except Exception as e:
    print("  FAIL torch_xmlir 不可用:", e)
if torch.cuda.device_count() == 0:
    print("  FAIL device_count=0：XPU 设备不可见（未装 xpytorch，或容器未挂 /dev/xpu* 设备）")
import torch.distributed as d
print("  nccl/bkcl available:", d.is_nccl_available())
PY

# 训练关键环境变量检查（stage2 脚本会设，这里只提示）
echo ""
echo "  提示: 以下变量由 stage2_train_1node.sh 自动设置，手跑命令时需自行 export:"
echo "    XMLIR_USE_HYDRA_LINEAR=0 / TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC=12000"
echo "    且不要设 CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER=OAM_ID"
echo "================================================"
echo "自检通过后: bash SFT/scripts/stage1_data_process.sh → stage2_train_1node.sh"
