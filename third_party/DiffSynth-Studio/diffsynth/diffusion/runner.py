import os, json, torch, importlib
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.state import DistributedType
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def compute_grad_norm(model):
    """计算全局梯度范数（FSDP 兼容版）。
    FSDP(use_orig_params) 下 backward 结束后每 rank 只持有自己分片参数的梯度，
    本地平方和后用 all_gather 汇总成全局值——不能用 all_reduce（本机 16 卡
    all_reduce 有死锁 bug，见 xccl_smoke_test.sh）；也不能用 barrier（同样死锁）。
    必须在 optimizer.step()/zero_grad() 之前调用（zero_grad 会清掉梯度）。"""
    import math
    local_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_sq += float(p.grad.detach().float().pow(2).sum().item())
    if not dist_available_and_initialized():
        return math.sqrt(local_sq)
    t = torch.tensor([local_sq], dtype=torch.float64, device="cuda")
    world = torch.distributed.get_world_size()
    out = [torch.zeros_like(t) for _ in range(world)]
    torch.distributed.all_gather(out, t)
    return math.sqrt(sum(float(o.item()) for o in out))


def dist_available_and_initialized():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def log_cuda_memory_usage(accelerator: Accelerator, step: int):
    # 每步打印真实设备显存（mem_get_info），区分 torch 统计与设备真值（P800 排查用）
    if not accelerator.is_main_process or accelerator.device.type != "cuda":
        return
    try:
        device = accelerator.device
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        free, total = torch.cuda.mem_get_info(device)
        print(
            f"[cuda_memory] rank=0 step={step} "
            f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
            f"free={free / (1024 ** 3):.2f}GiB total={total / (1024 ** 3):.2f}GiB",
            flush=True,
        )
    except Exception as e:
        print(f"[cuda_memory] rank=0 step={step} unavailable: {e}", flush=True)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    if accelerator.is_main_process:
        save_training_args(args)

    is_fsdp = getattr(accelerator, "distributed_type", None) == DistributedType.FSDP
    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        if not is_fsdp:
            model.to(device=accelerator.device)
        else:
            # FSDP: 模型必须留在 CPU，由 FSDP 在 prepare 时直接分片。
            # 若先 model.to(cuda)，每卡会驻留完整 bf16 模型（~72GB），再叠加
            # FSDP upcast/分片的中间驻留会撑爆 96GB 卡（我们 42GB+47GB 隐形
            # 占用 OOM 的根因，同事已验证的正确姿势是只设 pipe.device）。
            model.pipe.device = accelerator.device
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                grad_norm = compute_grad_norm(model)  # 必须在 step/zero_grad 前（梯度还在）
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, grad_norm=grad_norm)
                log_cuda_memory_usage(accelerator, model_logger.num_steps)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
