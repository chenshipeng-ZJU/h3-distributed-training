#!/usr/bin/env python3
"""
MiniMax-H3 Stage 3: SFT 模型推理（8卡 DDP 并行）
每张卡加载完整模型，各自处理一部分数据。
"""

import torch, os, json, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_config import load_config
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from accelerate import Accelerator

# ---- 加载配置 ----
config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "configs", "infer", "infer_1node.yaml")
config = load_config(config_path)

m = config["model"]
inf = config["inference"]

MODEL_ID = m["model_id"]
VARIANT = m["train_variant"]
PROMPT = inf["prompt"]
PROMPT_FILE = inf.get("prompt_file")
HEIGHT = inf["height"]
WIDTH = inf["width"]
NUM_FRAMES = inf["num_frames"]
NUM_STEPS = inf["num_inference_steps"]
SEED = inf["seed"]
FPS = inf["fps"]
# 输出子目录命名：sft_H3_infer_{steps}step_{width}p_{frames}frame
SUBDIR = f"sft_H3_infer_{NUM_STEPS}step_{WIDTH}p_{NUM_FRAMES}frame"
OUTPUT_DIR = os.path.join(inf["output_dir"], SUBDIR)
SFT_CKPT = inf.get("sft_ckpt")

# ---- 初始化 accelerator ----
accelerator = Accelerator()
rank = accelerator.process_index
world_size = accelerator.num_processes
device = accelerator.device

# 命令行 --prompt 覆盖
if "--prompt" in sys.argv:
    idx = sys.argv.index("--prompt")
    PROMPT = sys.argv[idx + 1]
    MODE = "single"
elif PROMPT_FILE and os.path.exists(PROMPT_FILE):
    MODE = "batch"
else:
    MODE = "single"

# 确定 SFT checkpoint 路径
if "--ckpt" in sys.argv:
    idx = sys.argv.index("--ckpt")
    DIT_PATH = sys.argv[idx + 1]
elif SFT_CKPT and SFT_CKPT != "None":
    DIT_PATH = SFT_CKPT
else:
    # 自动找最新 checkpoint（output_dir 是 inference 子目录，checkpoint 在 models 子目录）
    train_output_dir = inf["output_dir"].replace("/inference", "/models")
    ckpts = sorted([f for f in os.listdir(train_output_dir) if f.endswith(".safetensors")]) if os.path.exists(train_output_dir) else []
    if ckpts:
        DIT_PATH = os.path.join(train_output_dir, ckpts[-1])
    else:
        if rank == 0:
            print("错误: 未找到 SFT checkpoint")
        sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- VRAM 配置 ----
# 每张卡用自己的 cuda:N，避免全堆到 GPU 0
dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
cuda_device = f"cuda:{rank}"
vram_config = {
    "offload_dtype": dtype_map[inf["offload_dtype"]],
    "offload_device": cuda_device if inf["offload_device"] == "cuda" else inf["offload_device"],
    "onload_dtype": dtype_map[inf["onload_dtype"]],
    "onload_device": cuda_device if inf["onload_device"] == "cuda" else inf["onload_device"],
    "preparing_dtype": dtype_map[inf["preparing_dtype"]],
    "preparing_device": cuda_device if inf["preparing_device"] == "cuda" else inf["preparing_device"],
    "computation_dtype": dtype_map[inf["computation_dtype"]],
    "computation_device": cuda_device if inf["computation_device"] == "cuda" else inf["computation_device"],
}

if rank == 0:
    print(f"=== 加载 SFT 模型 ({world_size} 卡并行) ===")
    print(f"DiT checkpoint: {DIT_PATH}")
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=str(device),
    model_configs=[
        ModelConfig(model_id=MODEL_ID, origin_file_pattern=f"{VARIANT}/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(path=DIT_PATH, **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern=f"{VARIANT}/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id=MODEL_ID, origin_file_pattern=f"{VARIANT}/audio_vae/model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info(str(device))[1] / (1024 ** 3) - 2,
)

if MODE == "batch":
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]
    # 分片：每张卡处理一部分
    my_items = items[rank::world_size]
    if rank == 0:
        print(f"\n=== 批量 I2VA 推理: {len(items)} 条，{world_size} 卡并行，每卡约 {len(my_items)} 条 ===")
    for i, item in enumerate(my_items):
        prompt = item["prompt"]
        input_image = item["input_image"]
        output_name = item.get("output_name", f"output_{i}.mp4")
        output_path = os.path.join(OUTPUT_DIR, output_name)
        # 跳过已生成的
        if os.path.exists(output_path):
            print(f"[Rank {rank}] [{i+1}/{len(my_items)}] 跳过已存在: {output_name}")
            continue
        print(f"[Rank {rank}] [{i+1}/{len(my_items)}] {output_name}")
        print(f"  input_image: {input_image[-40:]}")
        print(f"  prompt: {prompt[:60]}...")
        first_frame = Image.open(input_image)
        # 推理只用首帧（FL2VA, keyframe_indices=[0]）；SFT 训练用首帧+尾帧（[0, -1]）
        video, audio = pipe(
            prompt=prompt,
            height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
            num_inference_steps=NUM_STEPS, seed=SEED,
            keyframes=[first_frame], keyframe_indices=[0],
        )
        write_video_audio(
            video=video, audio=audio,
            output_path=output_path,
            fps=FPS, audio_sample_rate=pipe.audio_vae.sample_rate,
        )
        print(f"[Rank {rank}]   saved: {output_path}, frames: {len(video)}")
else:
    # 单条 prompt 只在 rank 0 执行
    if rank == 0:
        print(f"\n=== T2VA 推理 (SFT) ===")
        print(f"Prompt: {PROMPT}")
        video, audio = pipe(
            prompt=PROMPT,
            height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
            num_inference_steps=NUM_STEPS, seed=SEED,
        )
        output_path = os.path.join(OUTPUT_DIR, "sft_t2va.mp4")
        write_video_audio(
            video=video, audio=audio,
            output_path=output_path,
            fps=FPS, audio_sample_rate=pipe.audio_vae.sample_rate,
        )
        print(f"saved: {output_path}, frames: {len(video)}, audio: {tuple(audio.shape)}")

accelerator.wait_for_everyone()
if rank == 0:
    print("\n=== SFT 模型推理完成 ===")
