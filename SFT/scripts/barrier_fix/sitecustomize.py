# =============================================================================
# 16 卡 P800 XCCL bug 规避补丁（barrier → all_gather）
#
# 背景（2026-09-18 冒烟测试实测）:
#   - all_gather / reduce_scatter: 16 卡全部正确（FSDP 训练的通信路径，安全）
#   - bare barrier / all_reduce:   16 卡死锁（XCCL 9d385bf 已知 bug，mock_barrier
#                                  内部走 allreduce 路径，带 device_ids 也无效）
#
# 影响: DiffSynth checkpoint 保存 logger.py save_model() -> accelerator.
# wait_for_everyone() -> torch.distributed.barrier()，每 save_steps 步触发一次，
# 不打补丁会在第一次存 checkpoint 时死锁。
#
# 原理: all_gather 本身就是集合同步——每个 rank 贡献一个 token，所有 rank 的
# token 都到齐才算完成，与 barrier 语义等价（且 16 卡已验证可用）。
#
# 生效方式: 本文件放在 PYTHONPATH 首位，Python 启动时自动加载（sitecustomize）。
# 仅在 LOCAL_RANK 存在（torchrun/accelerate 多进程 worker）时打补丁，
# 不影响单进程脚本和其他环境。
# =============================================================================
import os

_LOCAL_RANK = os.environ.get("LOCAL_RANK")


def _configure_h3_xblas_autotune():
    # XBLAS 必须在 worker 导入 torch 前拿到每个 rank 独立的 cache 路径。
    # stage2_train_1node.sh 的 --xblas-tune 模式通过环境变量启用这里。
    if os.environ.get("H3_XBLAS_AUTOTUNE_MODE") != "generate":
        return

    tune_dir = os.environ.get("H3_XBLAS_TUNE_DIR")
    if not tune_dir:
        return

    rank = os.environ.get("RANK", _LOCAL_RANK or "0")
    os.makedirs(tune_dir, exist_ok=True)
    os.environ["XBLAS_AUTOTUNE_FILTER"] = "gemm"
    os.environ["XBLAS_AUTOTUNE"] = "1001"
    os.environ["XBLAS_AUTOTUNE_WRITEBACK"] = "1"
    os.environ["XBLAS_AUTOTUNE_FILE"] = os.path.join(
        tune_dir,
        f"gemm.rank{rank}.json",
    )


_configure_h3_xblas_autotune()


def _apply_barrier_patch():
    try:
        import torch
        import torch.distributed as dist
    except Exception:
        return

    # 确保 worker 的设备上下文指向自己的卡（all_gather token 落在正确设备上）
    try:
        torch.cuda.set_device(int(_LOCAL_RANK))
    except Exception:
        pass

    def _safe_barrier(group=None, async_op=False, device_ids=None, **kwargs):
        if not dist.is_initialized():
            return None
        g = group if group is not None else dist.group.WORLD
        world_size = dist.get_world_size(g)
        token = torch.zeros(1, dtype=torch.uint8, device="cuda")
        out = [torch.zeros_like(token) for _ in range(world_size)]
        work = dist.all_gather(out, token, group=g, async_op=True)
        if async_op:
            return work
        work.wait()
        return None

    # 覆盖函数调用形式（accelerate wait_for_everyone 走这条路）
    dist.barrier = _safe_barrier
    # 覆盖 ProcessGroup 实例方法调用形式（pg.barrier()），pybind 类可能不可改，失败则忽略
    try:
        from torch.distributed import ProcessGroup
        ProcessGroup.barrier = lambda self, *a, **kw: _safe_barrier(*a, **kw)
    except Exception:
        pass

    # 标记补丁已生效（诊断用，写到 /tmp 不影响训练）
    try:
        with open(f"/tmp/barrier_fix_{_LOCAL_RANK}.ok", "w") as f:
            f.write("patched\n")
    except Exception:
        pass


if _LOCAL_RANK is not None:
    _apply_barrier_patch()
