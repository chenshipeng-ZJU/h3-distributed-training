import os


def _configure_h3_xblas_autotune():
    """Configure per-rank XBLAS tuning before importing torch."""
    if os.environ.get("H3_XBLAS_AUTOTUNE_MODE") != "generate":
        return

    tune_dir = os.environ.get("H3_XBLAS_TUNE_DIR")
    if not tune_dir:
        raise RuntimeError("H3_XBLAS_TUNE_DIR is required in generate mode")

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    os.makedirs(tune_dir, exist_ok=True)
    os.environ["XBLAS_AUTOTUNE_FILTER"] = "gemm"
    os.environ["XBLAS_AUTOTUNE"] = "1001"
    os.environ["XBLAS_AUTOTUNE_WRITEBACK"] = "1"
    os.environ["XBLAS_AUTOTUNE_FILE"] = os.path.join(
        tune_dir,
        f"gemm.rank{rank}.json",
    )


_configure_h3_xblas_autotune()

import torch, argparse, accelerate
from accelerate.state import DistributedType
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath
from diffsynth.core.vram.disk_map import DiskMap
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MINIMAX_H3_FRAME_RATE = 24
MINIMAX_H3_TIME_DIVISION_FACTOR = 17
MINIMAX_H3_TIME_DIVISION_REMAINDER = 5


def enable_shared_cpu_disk_map_loading():
    # P800 专项（移植自同事验证过的分叉）：DiskMap 读权重时保留 mmap 后端而不克隆进
    # 匿名内存，16 个 rank 通过 OS page cache 共享同一份权重页，避免 CPU 侧 16 倍复制
    if getattr(DiskMap.__getitem__, "_diffsynth_shared_cpu_disk_map_patch", False):
        return

    def shared_cpu_getitem(self, name):
        if self.rename_dict is not None:
            name = self.rename_dict[name]
        file_id = self.name_map[name]
        param = self.files[file_id].get_tensor(name)
        if self.torch_dtype is not None and isinstance(param, torch.Tensor):
            param = param.to(self.torch_dtype)
        if isinstance(param, torch.Tensor):
            self.num_params += param.numel()
        if self.num_params > self.buffer_size:
            self.flush_files()
        return param

    shared_cpu_getitem._diffsynth_shared_cpu_disk_map_patch = True
    DiskMap.__getitem__ = shared_cpu_getitem


class MiniMaxH3TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        gradient_checkpointing_skip_blocks="",
        gradient_checkpointing_mode="regular",
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        silent_on_missing_audio=False,
        training_cfg_scale=1.0,
        audio_loss_weight=1.0,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        if training_cfg_scale < 1.0:
            raise ValueError("training_cfg_scale must be at least 1.0")
        self.training_cfg_scale = training_cfg_scale
        self.audio_loss_weight = audio_loss_weight
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        pipe_kwargs = {}
        if processor_path is not None:
            pipe_kwargs["processor_config"] = self.parse_path_or_model_id(processor_path)
        self.pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, **pipe_kwargs)
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
            force_remove_params_shared=("video_latents", "audio_latents"),
            force_remove_params_nega=("prompt_embeds", "text_token_tags", "packed") if training_cfg_scale == 1.0 else (),
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self.pipe.scheduler_audio.set_timesteps(1000, training=True)

        # Store other configs
        self.silent_on_missing_audio = silent_on_missing_audio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Stage1 cache may contain a stale True.  CPU activation offload adds a
        # blocking D2H for every checkpoint and is deliberately disabled for
        # the P800 training path.
        self.use_gradient_checkpointing_offload = False
        self.gradient_checkpointing_skip_blocks = gradient_checkpointing_skip_blocks
        self.gradient_checkpointing_mode = gradient_checkpointing_mode
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=self.training_cfg_scale, audio_loss_weight=self.audio_loss_weight,
                inputs_nega=inputs_nega, **inputs_shared, **inputs_posi,
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=self.training_cfg_scale, audio_loss_weight=self.audio_loss_weight,
                inputs_nega=inputs_nega, **inputs_shared, **inputs_posi,
            ),
        }

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        # First/last-frame conditioning is derived from the training video itself, following the
        # `input_image` / `end_image` convention used by the wanvideo series. The H3 pipeline takes
        # them as a `keyframes` list plus `keyframe_indices` in {0, -1}.
        keyframes, keyframe_indices = [], []
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                keyframes.append(data["video"][0])
                keyframe_indices.append(0)
            elif extra_input == "end_image":
                keyframes.append(data["video"][-1])
                keyframe_indices.append(-1)
            else:
                inputs_shared[extra_input] = data[extra_input]
        if keyframes:
            inputs_shared["keyframes"] = keyframes
            inputs_shared["keyframe_indices"] = keyframe_indices
        # If no audio tracks in the video file, we use a silence tensor for training.
        if self.silent_on_missing_audio:
            if "input_audio" in extra_inputs and "input_audio" in inputs_shared and inputs_shared["input_audio"] is None:
                inputs_shared["input_audio"] = (torch.zeros((2, 800 * round(len(data["video"]) / 24 * 40))), 32000)
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": " "}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "keyframes": None,
            "keyframe_indices": None,
            "references": None,
            "ref_image_short_edge": 2048,
            "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
            "tiled": True,
            "tile_size": 256,
            "tile_overlap": 64,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            # Reuse the pipeline's CFG preprocessing path to build unconditional
            # embeddings when CFG-aware training is enabled.
            "cfg_scale": self.training_cfg_scale,
            "seed": 42,
            "rand_device": "cpu",
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        inputs_shared["gradient_checkpointing_skip_blocks"] = self.gradient_checkpointing_skip_blocks
        inputs_shared["gradient_checkpointing_mode"] = self.gradient_checkpointing_mode
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        else:
            # stage2 cached 路径: inputs 直接来自 stage1 保存的 .pth，其中的
            # use_gradient_checkpointing(_offload) 是 stage1 运行时的值（通常 False）。
            # 训练期开关必须以 stage2 命令行为准，否则激活不被 checkpoint，
            # 124 帧会在第一个 step 因激活累积（~72GB）而 OOM。
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs = list(inputs)
                inputs[0] = {
                    **inputs[0],
                    "use_gradient_checkpointing": self.use_gradient_checkpointing,
                    "use_gradient_checkpointing_offload": False,
                    "gradient_checkpointing_skip_blocks": self.gradient_checkpointing_skip_blocks,
                    "gradient_checkpointing_mode": self.gradient_checkpointing_mode,
                }
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def minimax_h3_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--processor_path", type=str, default=None, help="Path or `model_id:pattern` of the Qwen3-VL processor.")
    parser.add_argument("--initialize_model_on_cpu", default=True, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--silent_on_missing_audio", default=False, action="store_true", help="Whether to use silent audio as a fallback when no audio track is present in the video data.")
    parser.add_argument("--training_cfg_scale", type=float, default=1.0, help="Inverse-CFG scale for preserving MiniMax-H3 guidance distillation during fine-tuning. Values greater than 1 enable a no-grad unconditional branch; 1 keeps the standard flow-matching loss.")
    parser.add_argument("--audio_loss_weight", type=float, default=1.0, help="Weight of the audio term in the MiniMax-H3 loss. 1 keeps video and audio equally weighted; 0 trains on the video term only while the audio stream is still noised and forwarded.")
    parser.add_argument("--dist_timeout_minutes", type=int, default=200, help="Timeout (minutes) for the default process group. The XCCL/kccl plugin swallows InitProcessGroupKwargs(timeout=...), so the default PG timeout is force-overridden after init; otherwise a slow FSDP full-gather at checkpoint save (>10min) trips the strong XCCL watchdog and kills the run (P800-KB FAQ 'DistBackendError').")
    parser.add_argument("--save_dtype", type=str, default=None, help="Dtype for saving checkpoints: 'bf16'/'fp32'/None (None = keep as-is). FSDP upcasts params to fp32 and full-state-dict gathers become very large; saving bf16 halves write time and memory.")
    return parser


if __name__ == "__main__":
    parser = minimax_h3_parser()
    args = parser.parse_args()
    if args.num_frames % MINIMAX_H3_TIME_DIVISION_FACTOR != MINIMAX_H3_TIME_DIVISION_REMAINDER:
        raise ValueError(
            f"--num_frames must be {MINIMAX_H3_TIME_DIVISION_FACTOR}n+{MINIMAX_H3_TIME_DIVISION_REMAINDER} "
            f"(e.g. 39, 56, 124) so it lands on the video VAE's temporal grouping, got {args.num_frames}."
        )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters),
        ],
    )
    # XCCL/kccl 的 ProcessGroupXCCL 会吞掉 InitProcessGroupKwargs 的 timeout（保留隐藏的
    # 10 分钟默认值），且其 watchdog 是强杀——FSDP 存 checkpoint 时的慢 full-gather
    # 超过 10 分钟就会被 DistBackendError 杀掉。此处强制覆盖默认 PG 超时（P800-KB FAQ）。
    try:
        import datetime
        from torch.distributed.distributed_c10d import _get_default_group
        dist_timeout = datetime.timedelta(minutes=args.dist_timeout_minutes)
        pg = _get_default_group()
        if hasattr(pg, "_set_default_timeout"):
            pg._set_default_timeout(dist_timeout)
            print(f"[dist_timeout] default PG timeout overridden to {args.dist_timeout_minutes} min")
        else:
            print("[dist_timeout] _set_default_timeout not available on default PG; relying on InitProcessGroupKwargs")
    except Exception as e:
        print(f"[dist_timeout] latest attempt to override timeout skipped: {e}")
    init_device = "cpu" if getattr(accelerator, "distributed_type", None) == DistributedType.FSDP else accelerator.device
    model_init_device = "cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else init_device
    if model_init_device == "cpu":
        enable_shared_cpu_disk_map_loading()
        print("[init] CPU model initialization will keep DiskMap tensors mmap-backed to reduce per-rank CPU duplication.")
    video_processor = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
        time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
        frame_rate=MINIMAX_H3_FRAME_RATE,
        fix_frame_rate=True,
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(
                num_frames=args.num_frames,
                time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
                time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
                frame_rate=MINIMAX_H3_FRAME_RATE,
                fix_frame_rate=True,
            ),
            "references": MiniMaxH3ReferenceLoader(
                base_path=args.dataset_base_path,
                height=args.height,
                width=args.width,
                max_pixels=args.max_pixels,
                num_frames=args.num_frames,
                frame_rate=MINIMAX_H3_FRAME_RATE,
            ),
        }
    )
    model = MiniMaxH3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        gradient_checkpointing_skip_blocks=args.gradient_checkpointing_skip_blocks,
        gradient_checkpointing_mode=args.gradient_checkpointing_mode,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        silent_on_missing_audio=args.silent_on_missing_audio,
        training_cfg_scale=args.training_cfg_scale,
        audio_loss_weight=args.audio_loss_weight,
        task=args.task,
        device=model_init_device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
        save_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}.get(args.save_dtype, None),
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
