# H3 Distributed Training on Kunlun P800

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/xpytorch-2.9%20%2F%20XPU-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Platform](https://img.shields.io/badge/platform-Kunlun%20P800%20%C3%9716-2563eb)](#)
[![Open Source](https://img.shields.io/badge/open-source-111827)](https://github.com/chenshipeng-ZJU/h3-distributed-training)

面向联合音视频生成模型（MiniMax/MiniMax-H3）的分布式 LoRA SFT 训练与性能优化工程：在单节点 16 卡昆仑芯 P800 上，基于 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 完成 FSDP2 全分片训练适配、显存与算子问题定位、多卡通信稳定性修复，跑通 124 帧 480P LoRA SFT（约 20s/步）。

## 核心内容

- **训练流水线**：Stage 1 数据预处理（latent 缓存）→ Stage 2 FSDP2 训练（断点续训 + loss 曲线自动绘制）→ Stage 3 SFT 推理对比，各 stage 独立成脚本，可单独复跑。
- **分布式适配**：FSDP2 全分片 + BF16 混合精度 + 梯度检查点 + 两阶段数据缓存，单节点 16 卡（torchrun，nproc 由 `xpu-smi` 自动检测）。
- **通信稳定性**：`SFT/scripts/barrier_fix/sitecustomize.py` 修复 FSDP 在 XCCL 上的 barrier 死锁；`SFT/scripts/xccl_smoke_test.sh` 验证 all-gather / reduce-scatter 通信路径，规避不稳定的 all-reduce 同步。
- **显存问题定位**：定位并修复分片前完整模型驻留、融合线性算子额外保存中间值/权重、选择性 attention 梯度检查点配置错误导致的首步 OOM 与激活累积。
- **环境封装**：`SFT/scripts/setup_env.sh` 一键环境校验（XPU 驱动 / XCCL / xpytorch 版本对齐），支持 `check` 子命令做发布前自检。
- **性能剖析**：`SFT/scripts/perf_monitor.py` 与 `slice_trace.py` 支持训练期采集与算子级 trace 切片分析。

## 目录结构

```
.
├── SFT/
│   ├── configs/                # 训练 / 推理 yaml + accelerate / DeepSpeed 配置
│   ├── scripts/
│   │   ├── stage0-3_*.sh       # 基线推理 / 数据处理 / 训练 / SFT 推理流水线
│   │   ├── barrier_fix/        # FSDP barrier 死锁修复（sitecustomize.py）
│   │   ├── xccl_smoke_test.sh  # XCCL 通信冒烟测试
│   │   ├── setup_env.sh        # 环境封装与版本校验
│   │   └── perf_monitor.py     # 训练期性能采集
│   ├── data/                   # （不入库）训练数据与 latent 缓存
│   ├── output/                 # （不入库）checkpoint 与推理产物
│   ├── logs/                   # （不入库）日志与 loss 曲线
│   ├── tune/                   # （不入库）算子级性能调优 dump
│   └── README.md               # SFT 详细说明（目录、流程、用法）
├── third_party/                # （不入库）DiffSynth-Studio 与厂商工具
├── LICENSE
└── README.md
```

## 快速开始

```bash
git clone https://github.com/chenshipeng-ZJU/h3-distributed-training.git
cd h3-distributed-training

# 准备上游训练框架（公开仓库，不入本库）
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio

# 环境校验（XPU 驱动 / XCCL / xpytorch 版本对齐）
bash SFT/scripts/setup_env.sh check

# 三阶段流水线
bash SFT/scripts/stage1_data_process.sh     # 数据预处理
bash SFT/scripts/stage2_train_1node.sh      # FSDP2 训练（断点续训见 --resume）
bash SFT/scripts/stage3_infer_sft.sh        # SFT 推理对比
```

模型权重通过 ModelScope ID（`MiniMax/MiniMax-H3`）自动解析，训练/推理配置见 `SFT/configs/`。

> **注意**：配置文件中的路径（如 `/data/...`）为示例占位符，请按自己的训练环境修改 `project / share_dir / base_path` 等字段后再运行。

## 致谢

- 上游训练框架：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- 训练平台：昆仑芯 P800（XPU）+ xpytorch / XCCL

## License

[MIT](LICENSE)
