import functools
import os

import torch


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def create_custom_forward_use_reentrant(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def _build_selective_attention_ops():
    aten = torch.ops.aten
    ops = set()
    op_names = (
        "scaled_dot_product_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "_flash_attention_forward",
        "_efficient_attention_forward",
    )
    for name in op_names:
        packet = getattr(aten, name, None)
        if packet is None:
            continue
        ops.add(packet)
        try:
            for overload_name in packet.overloads():
                overload = getattr(packet, overload_name, None)
                if overload is not None:
                    ops.add(overload)
        except (AttributeError, RuntimeError):
            default = getattr(packet, "default", None)
            if default is not None:
                ops.add(default)
    return frozenset(ops)


_SELECTIVE_ATTENTION_OPS = _build_selective_attention_ops()
_SELECTIVE_FA_LOGGED_OPS = set()
_SELECTIVE_AC_DEBUGGED_OPS = set()
_SELECTIVE_AC_DEBUG_LIMIT = 128
_SELECTIVE_AC_DEBUG_TOKENS = ("attention", "flash", "attn", "sdpa", "xformers")


def _selective_fa_policy(ctx, op, *args, **kwargs):
    op_name = str(op)
    op_name_lower = op_name.lower()
    if (
        os.environ.get("H3_SELECTIVE_AC_DEBUG_OPS", "0") == "1"
        and os.environ.get("RANK", "0") == "0"
        and op_name not in _SELECTIVE_AC_DEBUGGED_OPS
        and (
            len(_SELECTIVE_AC_DEBUGGED_OPS) < _SELECTIVE_AC_DEBUG_LIMIT
            or any(token in op_name_lower for token in _SELECTIVE_AC_DEBUG_TOKENS)
        )
    ):
        prefix = (
            "OP_CANDIDATE"
            if any(token in op_name_lower for token in _SELECTIVE_AC_DEBUG_TOKENS)
            else "OP"
        )
        print(f"[H3 selective_ac] {prefix} {op_name}", flush=True)
        _SELECTIVE_AC_DEBUGGED_OPS.add(op_name)
    if op in _SELECTIVE_ATTENTION_OPS:
        if os.environ.get("H3_SELECTIVE_AC_VERBOSE", "0") == "1":
            if op_name not in _SELECTIVE_FA_LOGGED_OPS:
                print(f"[H3 selective_ac] MUST_SAVE {op_name}", flush=True)
                _SELECTIVE_FA_LOGGED_OPS.add(op_name)
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE


def _selective_fa_context_fn():
    create_contexts = getattr(
        torch.utils.checkpoint,
        "create_selective_checkpoint_contexts",
        None,
    )
    if create_contexts is None:
        raise RuntimeError(
            "gradient_checkpointing_mode=selective_fa requires "
            "torch.utils.checkpoint.create_selective_checkpoint_contexts"
        )
    return functools.partial(create_contexts, _selective_fa_policy)


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    gradient_checkpointing_mode="regular",
    **kwargs,
):
    if gradient_checkpointing_mode not in {"regular", "selective_fa"}:
        raise ValueError(
            "gradient_checkpointing_mode must be 'regular' or 'selective_fa', "
            f"got {gradient_checkpointing_mode!r}"
        )
    if gradient_checkpointing_mode == "selective_fa" and use_gradient_checkpointing_offload:
        raise ValueError(
            "gradient_checkpointing_mode=selective_fa cannot be combined with "
            "use_gradient_checkpointing_offload=true"
        )

    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        if gradient_checkpointing_mode == "selective_fa":
            raise RuntimeError(
                "gradient_checkpointing_mode=selective_fa is not supported with "
                "configured DeepSpeed checkpointing"
            )
        all_args = args + tuple(kwargs.values())
        if not judge_args_requires_grad(*all_args):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward_use_reentrant(model),
                *all_args,
            )
        return model_output
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        checkpoint_kwargs = {
            "use_reentrant": False,
        }
        if gradient_checkpointing_mode == "selective_fa":
            checkpoint_kwargs["context_fn"] = _selective_fa_context_fn()
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            **checkpoint_kwargs,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
