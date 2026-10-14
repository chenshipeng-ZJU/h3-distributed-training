#!/usr/bin/env python3
"""
解析统一配置 yaml，供 shell 脚本调用。

用法：
    python parse_config.py <config.yaml> --accelerate-yaml     # 输出 accelerate 临时 yaml 路径
    python parse_config.py <config.yaml> --train-args --stage stage1  # 输出 train.py 命令行参数
    python parse_config.py <config.yaml> --get <dotted.key>   # 输出任意字段值
    python parse_config.py <config.yaml> --split-cache        # 将 cache_dir 按 9:1 分到 train/val 子目录
    python parse_config.py <config.yaml> --split-metadata      # 将 metadata.csv 按 9:1 分到 metadata_train/val.csv
"""

import sys
import yaml
import json
import os
import re
import argparse
import shlex


def resolve_vars(obj, root_config):
    """递归解析 ${var} 变量引用"""
    if isinstance(obj, str):
        def replacer(m):
            key = m.group(1)
            val = get_nested(root_config, key)
            return str(val) if val != "" else m.group(0)
        return re.sub(r"\$\{([^}]+)\}", replacer, obj)
    elif isinstance(obj, dict):
        return {k: resolve_vars(v, root_config) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_vars(v, root_config) for v in obj]
    return obj


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # 解析 ${var} 引用
    config = resolve_vars(config, config)
    return config


def get_nested(config, dotted_key):
    keys = dotted_key.split(".")
    val = config
    for k in keys:
        if val is None or k not in val:
            return ""
        val = val[k]
    return val


def write_accelerate_yaml(config, config_path):
    accel_config = config["accelerate"]
    tmp_dir = os.path.dirname(os.path.abspath(config_path))
    tmp_path = os.path.join(tmp_dir, ".accelerate_tmp.yaml")
    with open(tmp_path, "w") as f:
        yaml.dump(accel_config, f, default_flow_style=False, sort_keys=False)
    return tmp_path


def _bool_flag(val):
    """返回 flag 参数：True 返回 '--xxx'，False 返回 ''"""
    return "--xxx" if val else ""


def generate_train_args(config, stage):
    """生成 train.py 命令行参数字符串"""
    m = config.get("model", {})
    d = config.get("dataset", {})
    o = config.get("optimizer", {})
    t = config.get("trainer", {})
    l = config.get("logger", {})
    p = config.get("processor", {})

    model_id = m.get("model_id", "MiniMax/MiniMax-H3")
    variant = m.get("train_variant", "FL2VA")

    # ---- Stage 差异 ----
    if stage == "stage1":
        model_paths = (
            f"{model_id}:{variant}/text_encoder/model*.safetensors,"
            f"{model_id}:{variant}/video_vae/source/model.safetensors,"
            f"{model_id}:{variant}/audio_vae/model.safetensors"
        )
        dataset_base = d.get("base_path", "/")
        dataset_metadata = d.get("metadata_path", "")
        output_path = d.get("cache_dir", "")
        dataset_repeat = d.get("repeat_stage1", 1)
        task = "sft:data_process"
        num_epochs = 1
        find_unused = False
        save_steps = None  # stage1 不存 checkpoint
        resume_ckpt = None  # stage1 不需要断点续训
    elif stage == "stage2":
        model_paths = f"{model_id}:{variant}/transformer/model*.safetensors"
        # stage2 从 cache_dir/train 读，val_cache_dir 自动设为 cache_dir/val
        cache_dir = d.get("cache_dir", "")
        dataset_base = os.path.join(cache_dir, "train") if cache_dir else ""
        val_cache_dir = os.path.join(cache_dir, "val") if cache_dir else ""
        d["val_cache_dir"] = val_cache_dir
        dataset_metadata = ""
        output_path = t.get("output_dir", "")
        dataset_repeat = d.get("repeat_stage2", 100)
        task = t.get("task", "sft:train")
        num_epochs = t.get("num_epochs", 2)
        find_unused = t.get("find_unused_parameters", False)
        save_steps = t.get("save_steps")
        # 断点续训：resume_train=true 时从 resume_ckpt_dir 恢复
        resume_ckpt = None
        if t.get("resume_train") and t.get("resume_ckpt_dir"):
            resume_ckpt = t["resume_ckpt_dir"]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    args = []

    # ---- Dataset Arguments ----
    args.append(f"--dataset_base_path {dataset_base}")
    if dataset_metadata:
        args.append(f"--dataset_metadata_path {dataset_metadata}")
    args.append(f"--data_file_keys {d.get('data_file_keys', 'video,input_audio')}")
    args.append(f"--extra_inputs {d.get('extra_inputs', 'input_audio,input_image')}")
    args.append(f"--height {d.get('height', 480)}")
    args.append(f"--width {d.get('width', 832)}")
    args.append(f"--num_frames {d.get('num_frames', 124)}")
    args.append(f"--max_pixels {d.get('max_pixels', 1048576)}")
    args.append(f"--dataset_repeat {dataset_repeat}")
    args.append(f"--dataset_num_workers {d.get('num_workers', 0)}")

    # ---- Model Arguments ----
    args.append(f"--model_id_with_origin_paths {model_paths}")
    # trainable_models 为空（LoRA 复现官方用法）时不传，让 freeze_except([]) 冻结全部参数、
    # 仅由 LoRA patch 激活可训练参数；否则优化器会拿到整个 base 模型权重
    trainable_models = m.get("trainable_models")
    if trainable_models:
        args.append(f"--trainable_models {trainable_models}")
    args.append(f"--remove_prefix_in_ckpt {t.get('remove_prefix_in_ckpt', 'pipe.dit.')}")

    # LoRA
    if m.get("lora_base_model"):
        args.append(f"--lora_base_model {m['lora_base_model']}")
        args.append(f"--lora_target_modules {m.get('lora_target_modules', 'q,k,v,o,ffn.0,ffn.2')}")
        args.append(f"--lora_rank {m.get('lora_rank', 32)}")
    if m.get("lora_checkpoint"):
        args.append(f"--lora_checkpoint {m['lora_checkpoint']}")
    if m.get("preset_lora_path"):
        args.append(f"--preset_lora_path {m['preset_lora_path']}")
    if m.get("preset_lora_model"):
        args.append(f"--preset_lora_model {m['preset_lora_model']}")

    # Template model
    if m.get("template_model_id_or_path"):
        args.append(f"--template_model_id_or_path {m['template_model_id_or_path']}")

    # 量化
    if m.get("fp8_models"):
        args.append(f"--fp8_models {m['fp8_models']}")
    if m.get("offload_models"):
        args.append(f"--offload_models {m['offload_models']}")
    if m.get("quant_options"):
        args.append(f"--quant_options {m['quant_options']}")

    # ---- Processor ----
    if p.get("path"):
        args.append(f"--processor_path {p['path']}")

    # ---- Optimizer Arguments ----
    args.append(f"--learning_rate {o.get('learning_rate', 1e-5)}")
    args.append(f"--weight_decay {o.get('weight_decay', 0.01)}")
    if o.get("customized_optimizer"):
        args.append(f"--customized_optimizer {o['customized_optimizer']}")
    args.append(f"--gradient_accumulation_steps {t.get('gradient_accumulation_steps', 1)}")
    if m.get("use_gradient_checkpointing") and stage != "stage1":
        args.append("--use_gradient_checkpointing")
    if m.get("use_gradient_checkpointing_offload") and stage != "stage1":
        args.append("--use_gradient_checkpointing_offload")
    if m.get("gradient_checkpointing_skip_blocks") and stage != "stage1":
        args.append(f"--gradient_checkpointing_skip_blocks {shlex.quote(str(m['gradient_checkpointing_skip_blocks']))}")
    if m.get("gradient_checkpointing_mode") and stage != "stage1":
        args.append(f"--gradient_checkpointing_mode {shlex.quote(str(m['gradient_checkpointing_mode']))}")
    if o.get("enable_model_cpu_offload") and stage != "stage1":
        args.append("--enable_model_cpu_offload")
    if o.get("enable_optimizer_cpu_offload"):
        args.append("--enable_optimizer_cpu_offload")
    if o.get("cpu_offload_split_threshold"):
        args.append(f"--cpu_offload_split_threshold {o['cpu_offload_split_threshold']}")

    # ---- Training Pipeline Arguments ----
    args.append(f"--num_epochs {num_epochs}")
    args.append(f"--task {task}")
    if find_unused:
        args.append("--find_unused_parameters")
    if save_steps:
        args.append(f"--save_steps {save_steps}")
    if resume_ckpt:
        args.append(f"--resume_from_checkpoint {resume_ckpt}")
    if t.get("training_cfg_scale", 1.0) != 1.0:
        args.append(f"--training_cfg_scale {t['training_cfg_scale']}")
    if t.get("audio_loss_weight", 1.0) != 1.0:
        args.append(f"--audio_loss_weight {t['audio_loss_weight']}")
    if t.get("initialize_model_on_cpu") and stage != "stage1":
        args.append("--initialize_model_on_cpu")
    # EMA
    if t.get("ema_decay", 0.0) > 0:
        args.append(f"--ema_decay {t['ema_decay']}")
        if t.get("ema_cpu_offload", False):
            args.append("--ema_cpu_offload")
    args.append(f"--output_path {output_path}")

    # ---- Dataset: audio ----
    if d.get("silent_on_missing_audio"):
        args.append("--silent_on_missing_audio")

    # ---- Val dataset (stage2 only) ----
    if stage == "stage2" and d.get("val_cache_dir") and d.get("val_steps"):
        args.append(f"--val_cache_dir {d['val_cache_dir']}")
        args.append(f"--val_steps {d['val_steps']}")

    # ---- Val 视频生成 (stage2 only) ----
    dv = config.get("dataset_val", {}) or {}
    if stage == "stage2" and dv.get("val_generate", {}).get("enabled", False):
        vg = dv["val_generate"]
        args.append("--val_generate_enabled")
        if vg.get("metadata"):
            args.append(f"--val_generate_metadata {vg['metadata']}")
        if vg.get("num"):
            args.append(f"--val_generate_num {vg['num']}")
        if vg.get("steps"):
            args.append(f"--val_generate_steps {vg['steps']}")
        if vg.get("height"):
            args.append(f"--val_generate_height {vg['height']}")
        if vg.get("width"):
            args.append(f"--val_generate_width {vg['width']}")
        if vg.get("num_frames"):
            args.append(f"--val_generate_num_frames {vg['num_frames']}")
        if vg.get("inference_steps"):
            args.append(f"--val_generate_inference_steps {vg['inference_steps']}")
        if vg.get("seed") is not None:
            args.append(f"--val_generate_seed {vg['seed']}")

    # ---- Logger Arguments ----
    if l.get("enable_tensorboard_log"):
        args.append("--enable_tensorboard_log")
    if l.get("enable_wandb_log"):
        args.append("--enable_wandb_log")
        args.append(f"--wandb_project {l.get('wandb_project', 'DiffSynth-Studio')}")
    if l.get("enable_swanlab_log"):
        args.append("--enable_swanlab_log")
        args.append(f"--swanlab_project {l.get('swanlab_project', 'DiffSynth-Studio')}")
    if l.get("enable_csv_log"):
        args.append("--enable_csv_log")

    return " ".join(args)


def split_cache(config, ratio=0.9, val_num=None, seed=42):
    """将 cache_dir 下的 .pth 按 ratio 或 val_num 移到 train/ val/ 子目录。
    若 train/ val/ 已存在旧数据，先合并回根目录再重新分割。"""
    import os, shutil, random
    cache_dir = get_nested(config, "dataset.cache_dir")
    if not cache_dir:
        print("Error: dataset.cache_dir not set"); return
    cache_dir = os.path.abspath(cache_dir)
    train_dir = os.path.join(cache_dir, "train")
    val_dir = os.path.join(cache_dir, "val")

    # 先把 train/ val/ 旧数据合并回根目录
    for sub in ("train", "val"):
        sub_dir = os.path.join(cache_dir, sub)
        if not os.path.isdir(sub_dir): continue
        for rank in os.listdir(sub_dir):
            rank_src = os.path.join(sub_dir, rank)
            if not os.path.isdir(rank_src): continue
            for f in os.listdir(rank_src):
                if f.endswith(".pth"):
                    src = os.path.join(rank_src, f)
                    if not os.path.exists(src): continue
                    os.makedirs(os.path.join(cache_dir, rank), exist_ok=True)
                    try:
                        shutil.move(src, os.path.join(cache_dir, rank, f))
                    except (OSError, shutil.Error):
                        pass
        # 删空目录
        for rank in os.listdir(sub_dir):
            rank_path = os.path.join(sub_dir, rank)
            if os.path.isdir(rank_path) and not os.listdir(rank_path):
                os.rmdir(rank_path)
        if os.path.isdir(sub_dir) and not os.listdir(sub_dir):
            os.rmdir(sub_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    rng = random.Random(seed)
    # 聚合所有 rank 的 .pth（跳过软链接，只收集真实文件），全局 shuffle 后按 val_num 总数切分
    all_pths = []
    for rank in sorted(os.listdir(cache_dir)):
        rp = os.path.join(cache_dir, rank)
        if not os.path.isdir(rp) or rank in ("train", "val"): continue
        for f in os.listdir(rp):
            if f.endswith(".pth"):
                fp = os.path.join(rp, f)
                if os.path.islink(fp):
                    continue
                all_pths.append((rank, f, fp))
    rng.shuffle(all_pths)
    total = len(all_pths)
    if val_num is not None:
        n_val = min(val_num, total)
    else:
        n_val = int(total * (1 - ratio))
    n_train = total - n_val
    # 直接放到 train/ val/ 根目录，文件名加 rank 前缀避免冲突
    for i, (rank, f, src) in enumerate(all_pths):
        dst_dir = train_dir if i < n_train else val_dir
        dst = os.path.join(dst_dir, f"{rank}_{f}")
        shutil.move(src, dst)
    print(f"Total: {total}\nTrain: {n_train} -> {train_dir}\nVal:   {n_val} -> {val_dir}")


def split_metadata(config, ratio=0.9, val_num=None, seed=42):
    """将 metadata.csv 按 ratio 或 val_num 写到 metadata_train.csv / metadata_val.csv。"""
    import os, csv, random
    meta_path = get_nested(config, "dataset.metadata_path")
    if not meta_path or not os.path.isfile(meta_path):
        print("Error: dataset.metadata_path not found"); return
    rng = random.Random(seed)
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    rng.shuffle(rows)
    if val_num is not None:
        nt = max(len(rows) - val_num, 0)
    else:
        nt = int(len(rows) * ratio)
    base = os.path.dirname(meta_path)
    train_path = os.path.join(base, "metadata_train.csv")
    val_path = os.path.join(base, "metadata_val.csv")
    for path, rows in [(train_path, rows[:nt]), (val_path, rows[nt:])]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"Total: {len(rows)}\nTrain: {nt} -> {train_path}\nVal:   {len(rows)-nt} -> {val_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--accelerate-yaml", action="store_true")
    parser.add_argument("--train-args", action="store_true")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--get", default=None)
    parser.add_argument("--split-cache", action="store_true", help="Split cache_dir .pth into train/ val/ (9:1)")
    parser.add_argument("--split-metadata", action="store_true", help="Split metadata.csv into train/val (9:1)")
    parser.add_argument("--ratio", type=float, default=None, help="Override split_ratio from yaml")
    parser.add_argument("--val-num", type=int, default=None, help="Override val_num from yaml; exact val count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.get:
        val = get_nested(config, args.get)
        print(val)
    elif args.accelerate_yaml:
        print(write_accelerate_yaml(config, args.config_path))
    elif args.train_args:
        print(generate_train_args(config, args.stage))
    elif args.split_cache:
        val_num = args.val_num if args.val_num is not None else get_nested(config, "dataset.val_num")
        if isinstance(val_num, str) and val_num.strip() == "":
            val_num = None
        elif val_num is not None:
            val_num = int(val_num)
        ratio = args.ratio if args.ratio is not None else float(get_nested(config, "dataset.split_ratio") or 0.9)
        split_cache(config, ratio, val_num, args.seed)
    elif args.split_metadata:
        val_num = args.val_num if args.val_num is not None else get_nested(config, "dataset.val_num")
        if isinstance(val_num, str) and val_num.strip() == "":
            val_num = None
        elif val_num is not None:
            val_num = int(val_num)
        ratio = args.ratio if args.ratio is not None else float(get_nested(config, "dataset.split_ratio") or 0.9)
        split_metadata(config, ratio, val_num, args.seed)
    else:
        print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
