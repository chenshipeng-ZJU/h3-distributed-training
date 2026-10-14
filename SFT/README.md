# MiniMax-H3 SFT 训练 (8 GPU)

基于 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 对 MiniMax-H3 模型进行 8 卡 SFT (Supervised Fine-Tuning)。

## 目录结构

```
SFT/
├── configs/
│   ├── train/sft_train_1node.yaml          # 单节点训练配置（16卡 P800）
│   ├── train/sft_train_4node.yaml          # 单节点16卡训练配置（原4节点，现同为1node×16卡）
│   └── infer/infer_1node.yaml              # 推理配置（Stage 0 / Stage 3）
├── scripts/
│   ├── parse_config.py                    # yaml 解析工具（被各 sh 脚本调用）
│   ├── plot_loss.py                       # loss 曲线绘图（stage2 自动调用）
│   ├── convert_pet_dataset.py             # Pet 数据集 → H3 metadata 转换
│   ├── stage0_infer_original.sh           # Stage 0: 原始模型推理（8卡 DDP）
│   ├── stage0_infer.py                    # Stage 0 推理逻辑
│   ├── stage1_data_process.sh             # Stage 1: 数据预处理 (sft:data_process)
│   ├── stage2_train.sh                    # Stage 2: 训练 (sft:train)
│   ├── stage3_infer_sft.sh                # Stage 3: SFT 模型推理（8卡 DDP）
│   └── stage3_infer.py                    # Stage 3 推理逻辑
├── data/
│   ├── train/metadata.csv                 # 训练数据 metadata（1821 样本）
│   ├── infer/prompts.txt                  # 推理 prompt 列表
│   └── processed/                         # 预处理缓存（latent .pth 文件）
├── output/
│   └── pet/
│       ├── models/                        # 训练 checkpoint（stage2 输出）
│       ├── inference/                     # 推理视频（stage0/stage3 输出）
│       │   ├── raw_H3_infer_50step_720p_90frame/   # 原始模型
│       │   └── sft_H3_infer_50step_720p_90frame/   # SFT 模型
│       └── loss.csv                        # 训练 loss 记录
└── logs/                                  # 所有 stage 日志 + loss 曲线
```

## 流程

| Stage | 脚本 | 说明 |
|-------|------|------|
| 0 | `stage0_infer_original.sh` | 原始模型推理（基线对比） |
| 1 | `stage1_data_process.sh` | 数据预处理，生成 latent .pth 缓存 |
| 2 | `stage2_train.sh` | SFT 训练（自动生成 loss 曲线） |
| 3 | `stage3_infer_sft.sh` | SFT 模型推理 |

## 用法

```bash
# Stage 0: 原始模型推理
bash scripts/stage0_infer_original.sh

# Stage 1: 数据预处理
bash scripts/stage1_data_process.sh

# Stage 2: 训练（loss 曲线自动生成到 logs/loss_plot.png）
bash scripts/stage2_train.sh

# Stage 3: SFT 模型推理
bash scripts/stage3_infer_sft.sh
```

## 配置

- **训练（单节点 accelerate）**: `configs/train/sft_train_1node.yaml`
- **训练（单节点 torchrun，原4节点脚本）**: `configs/train/sft_train_4node.yaml`
- **推理**: `configs/infer/infer_1node.yaml`

两个配置文件独立，各管各的。
