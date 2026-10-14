from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward

MINIMAX_H3_ADALN_MODALITY_NUM = 3
_PATCH_T, _PATCH_H, _PATCH_W = 1, 2, 2
_DIT_FUSED_NORM = os.environ.get("DIT_FUSED_NORM", "0") == "1"
_DIT_FUSED_SWIGLU = os.environ.get("DIT_FUSED_SWIGLU", "0") == "1"
_DIT_FUSED_ROPE = os.environ.get("DIT_FUSED_ROPE", "0") != "0"
_FUSED_ROPE_WARNED = False


def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    # [1,24,T,H,W] -> [T*(H/2)*(W/2), 96]
    b, c, ft, fh, fw = (int(x) for x in latent.shape)
    t, h, w = ft // _PATCH_T, fh // _PATCH_H, fw // _PATCH_W
    packed = latent.reshape(b, c, t, _PATCH_T, h, _PATCH_H, w, _PATCH_W)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(b * t * h * w, c * _PATCH_T * _PATCH_H * _PATCH_W).contiguous()


def unpatchify_video(rows: torch.Tensor, ft: int, fh: int, fw: int, channel: int = 24) -> torch.Tensor:
    # [T*(H/2)*(W/2), 96] -> [1,24,T,H,W]  (inverse of patchify_video; ft/fh/fw match its input)
    t, h, w = ft // _PATCH_T, fh // _PATCH_H, fw // _PATCH_W
    packed = rows.reshape(-1, t, h, w, channel, _PATCH_T, _PATCH_H, _PATCH_W)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, ft, fh, fw).contiguous()


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    # [audio_channel, 32, T] -> [audio_channel*T, 32]  (channel-major)
    ac, ld, steps = (int(x) for x in latent.shape)
    return latent.permute(0, 2, 1).reshape(ac * steps, ld).contiguous()


def unpack_audio(rows: torch.Tensor, audio_channel: int, steps: int, latent_dim: int = 32) -> torch.Tensor:
    # [audio_channel*T, 32] -> [audio_channel, 32, T]
    return rows.reshape(audio_channel, steps, latent_dim).permute(0, 2, 1).contiguous()


def _norm(size: int, *, eps: float) -> nn.Module:
    if _DIT_FUSED_NORM:
        try:
            from torch_xmlir.nn.rms_norm import RMSNormLayer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "DIT_FUSED_NORM=1 but torch_xmlir.nn.rms_norm.RMSNormLayer "
                "is unavailable in the active environment"
            ) from exc
        return RMSNormLayer(size, eps=eps)
    return nn.RMSNorm(size, eps=eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, freqs) -> torch.Tensor:
    global _FUSED_ROPE_WARNED
    if isinstance(freqs, (tuple, list)):
        cos, sin = freqs
    else:
        cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
        sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    rot_dim = cos.shape[-1]
    if _DIT_FUSED_ROPE:
        try:
            fused_rope = torch.ops.custom_ops.fused_rope_forward
            out = torch.empty_like(x)
            result = fused_rope(
                x.unsqueeze(0),
                None,
                None,
                cos.unsqueeze(0),
                sin.unsqueeze(0),
                None,
                "BLHD",
                False,
                q_emb=out.unsqueeze(0),
                k_emb=torch.empty(0, dtype=x.dtype, device=x.device),
            )
            if isinstance(result, torch.Tensor):
                return result.squeeze(0) if result.dim() == x.dim() + 1 else result
            return out
        except (AttributeError, RuntimeError, TypeError) as exc:
            if not _FUSED_ROPE_WARNED:
                print(f"[H3] fused RoPE unavailable, using eager fallback: {exc}", flush=True)
                _FUSED_ROPE_WARNED = True
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


class _FastRowIndexSelect(torch.autograd.Function):
    """[perf] index_select(0, idx) 的快速反向。

    背景：8 卡 FSDP 下反向的 index_add_ 走 indexFuncLargeIndex 老路径，
    目标是 [unique_t, hidden] 小矩阵、索引高重复（每个 timestep 上千 token），
    实测 190ms/次 × ~300 次/步 ≈ 57s，是 bwd 81s 的主要病灶
    （kernel 有效带宽仅 ~1GB/s，且在 FSDP 每层 RS 栅栏前被串行放大）。

    修法：forward 不变（gather 快）；backward 把 index_add 换成 one-hot matmul
    （按索引分组求和 = onehot^T @ grad，GEMM 路径），实测 190ms → 2ms（~100×）。
    数学等价：fp32 下与 index_add 全等（bf16 累加误差同量级）。
    开关：H3_FAST_INDEX_SELECT=0 回退原生路径（对照实验用）。
    """

    @staticmethod
    def forward(ctx, self, index):
        ctx.save_for_backward(index)
        ctx.self_shape = self.shape
        return self.index_select(0, index)

    @staticmethod
    def backward(ctx, grad_out):
        # onehot^T @ grad_out == 按索引分组求和（index_add 的数学等价）
        (index,) = ctx.saved_tensors
        n, m = ctx.self_shape[0], grad_out.shape[0]
        onehot = grad_out.new_zeros(m, n)
        onehot.scatter_(1, index.unsqueeze(1), 1.0)
        grad_self = onehot.transpose(0, 1).matmul(grad_out.to(onehot.dtype))
        return grad_self.to(grad_out.dtype), None


def _fast_index_select(param, indices):
    if os.environ.get("H3_FAST_INDEX_SELECT", "1") == "1" and param.requires_grad:
        return _FastRowIndexSelect.apply(param, indices)
    return param.index_select(0, indices)


def _modulate_scale_shift(x, shift, scale, indices):
    # Cast back to x: index_select on the AdaLN params can promote the expression.
    # [perf] index_select 反向走 index_add_ 老路径（8 卡 FSDP bwd 57s 病灶）→ _fast_index_select
    return (x * (1.0 + _fast_index_select(scale, indices)) + _fast_index_select(shift, indices)).to(x.dtype)


def _modulate_gate(x, gate, other, indices):
    return (x + _fast_index_select(gate, indices) * other).to(x.dtype)


def _parse_gradient_checkpointing_skip_blocks(skip_blocks):
    if not skip_blocks:
        return frozenset()
    values = skip_blocks.split(",") if isinstance(skip_blocks, str) else skip_blocks
    try:
        return frozenset(int(str(value).strip()) for value in values if str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient_checkpointing_skip_blocks must be a comma-separated list of integers") from exc


def _sdpa_varlen_attention(q, k, v, cu_seqlens, softmax_scale, force_torch_sdpa=False):
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg_out = attention_forward(
            seg_q,
            seg_k,
            seg_v,
            scale=softmax_scale,
            compatibility_mode=force_torch_sdpa,
        )
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


class MiniMaxH3Rope(nn.Module):
    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.inv_freq_len = inv_freq_len
        self.inv_freq = nn.Parameter(self._build_inv_freq())

    def _build_inv_freq(self, device=None) -> torch.Tensor:
        steps = torch.arange(0, self.inv_freq_len, dtype=torch.float32, device=device)
        return 1.0 / (10000.0 ** (steps / self.inv_freq_len))

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(torch.float32)
        inv_freq = self._build_inv_freq(pos.device)
        per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, timestep_input_dim, time_embed_hidden_size, time_embed_dim):
        super().__init__()
        self.frequency_embedding_size = timestep_input_dim
        self.proj_in = nn.Linear(timestep_input_dim, time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(time_embed_hidden_size, time_embed_dim, bias=True)

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden = self.proj_in(t_freq.to(dtype))
        hidden = nn.functional.silu(hidden)
        return self.proj_out(hidden)


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(hidden_size, inner_dim * 3, bias=False)
        self.q_norm = _norm(attention_head_dim, eps=qk_norm_eps)
        self.k_norm = _norm(attention_head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(inner_dim, hidden_size, bias=False)

    def forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None, force_torch_sdpa=False):
        total = x.shape[0]
        qkv = self.qkv_proj(x)
        qkv = qkv.view(total, self.num_heads, 3, self.head_dim)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_freqs is not None:
            q = _apply_rope(q, rope_freqs)
            k = _apply_rope(k, rope_freqs)
        out = _sdpa_varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            softmax_scale=self.softmax_scale,
            force_torch_sdpa=force_torch_sdpa,
        )
        out = out.reshape(total, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MiniMaxH3MLP(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        hidden = self.fc1(x)
        if _DIT_FUSED_SWIGLU:
            try:
                from torch_xmlir.nn.swiglu import SwiGLUFunction
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "DIT_FUSED_SWIGLU=1 but torch_xmlir.nn.swiglu.SwiGLUFunction "
                    "is unavailable in the active environment"
                ) from exc
            hidden = SwiGLUFunction.apply(hidden, -1, True)
        else:
            gate, up = hidden.chunk(2, dim=-1)
            hidden = nn.functional.silu(gate) * up
        return self.fc2(hidden)


class MiniMaxH3AdalnProj(nn.Module):
    def __init__(self, hidden_size, time_embed_dim, out_features, *, expand_ratio, modality_num):
        super().__init__()
        if out_features != expand_ratio * hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, out_features, bias=True)

    def forward(self, t_emb):
        x = nn.functional.silu(t_emb)
        x = self.linear(x)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps):
        super().__init__()
        self.norm1 = _norm(hidden_size, eps=norm_eps)
        self.norm2 = _norm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps)
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_hidden_size)

    def forward(self, x, *, cu_seqlens, max_seqlen):
        x = x + self.attn(self.norm1(x), rope_freqs=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps, final_norm_eps):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps) for _ in range(num_layers)]
        )
        self.final_norm = _norm(hidden_size, eps=final_norm_eps)

    def forward(self, x, *, cu_seqlens, max_seqlen):
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps):
        super().__init__()
        self.norm1 = _norm(hidden_size, eps=norm_eps)
        self.norm2 = _norm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps)
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_hidden_size)
        self.adaln_proj = MiniMaxH3AdalnProj(hidden_size, time_embed_dim, adaln_out_features, expand_ratio=6, modality_num=MINIMAX_H3_ADALN_MODALITY_NUM)

    def forward(
        self,
        x,
        *,
        t_emb,
        combined_indices,
        rope_freqs,
        cu_seqlens,
        max_seqlen,
        force_torch_sdpa=False,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices)
        h = self.attn(
            h,
            rope_freqs=rope_freqs,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            force_torch_sdpa=force_torch_sdpa,
        )
        x = _modulate_gate(residual, gate_msa, h, combined_indices)
        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(h, shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _modulate_gate(residual, gate_mlp, h, combined_indices)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, hidden_size, time_embed_dim, final_adaln_out_features, latents_dim, audio_latents_dim, patch_size, final_norm_eps):
        super().__init__()
        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.norm = _norm(hidden_size, eps=final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(hidden_size, time_embed_dim, final_adaln_out_features, expand_ratio=2, modality_num=1)
        self.video_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(hidden_size, audio_latents_dim, bias=True)

    def forward(self, x, *, t_emb, inverse_indices):
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = _modulate_scale_shift(h, shift, scale, inverse_indices)
        video = self.video_out(h)
        audio = self.audio_out(h)
        return video, audio


class MiniMaxH3DiT(nn.Module):
    _repeated_blocks = ["MiniMaxH3DiTBlock"]

    def __init__(
        self,
        num_layers: int = 50,
        token_refiner_num_layers: int = 2,
        hidden_size: int = 5376,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        audio_latents_dim: int = 32,
        patch_size: tuple = (1, 2, 2),
        text_dim: int = 5120,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        adaln_out_features: int = 96768,
        final_adaln_out_features: int = 10752,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_channels_latents = latents_dim
        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_embedder = MiniMaxH3TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
        self.rope = MiniMaxH3Rope(rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            token_refiner_num_layers, hidden_size, num_attention_heads, attention_head_dim,
            ffn_hidden_size, norm_eps, qk_norm_eps, final_norm_eps,
        )
        self.blocks = nn.ModuleList(
            [MiniMaxH3DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps) for _ in range(num_layers)]
        )
        self.final_layer = MiniMaxH3FinalLayer(hidden_size, time_embed_dim, final_adaln_out_features, latents_dim, audio_latents_dim, patch_size, final_norm_eps)

    def _embed(self, *, x, audio_x, text_embeddings_selected, unique_timesteps, img_pos, audio_pos, text_pos, refiner_cu_seqlens, refiner_max_seqlen, seq_len, device):
        dtype = text_embeddings_selected.dtype
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(dtype)
        video_embed = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(dtype)
        audio_embed = self.audio_patch_proj(audio_rows)
        text_rows = text_embeddings_selected.to(device=device)
        text_embed = self.condition_proj(text_rows)
        text_embed = self.token_refiner(text_embed, cu_seqlens=refiner_cu_seqlens, max_seqlen=refiner_max_seqlen)

        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
        # [perf] index_add_ → index_copy_：text/img/audio 三段 pos 互斥且段内唯一
        # （fl2va 布局 [text|cond|audio|video|pad] 顺序切片，见 pipeline _build_packed_fl2va），
        # index_add_ 落在零矩阵上数值等价，但走 indexFuncLargeIndex 非合并散写路径
        # （8 卡 FSDP bwd 81s 中 57s 病灶）。index_copy_ 走向量化 copy kernel。
        # 开关：H3_EMBED_INDEX_ADD=1 回退旧行为（对照实验用）。
        if os.environ.get("H3_EMBED_INDEX_ADD", "0") == "1":
            embeddings.index_add_(0, text_pos, text_embed.to(dtype)[: text_pos.shape[0]])
            embeddings.index_add_(0, img_pos, video_embed.to(dtype)[: img_pos.shape[0]])
            embeddings.index_add_(0, audio_pos, audio_embed.to(dtype)[: audio_pos.shape[0]])
        else:
            embeddings.index_copy_(0, text_pos, text_embed.to(dtype)[: text_pos.shape[0]])
            embeddings.index_copy_(0, img_pos, video_embed.to(dtype)[: img_pos.shape[0]])
            embeddings.index_copy_(0, audio_pos, audio_embed.to(dtype)[: audio_pos.shape[0]])

        t_emb = self.time_embedder(unique_timesteps, dtype=dtype)
        return embeddings, t_emb

    def forward(
        self,
        x,
        audio_x,
        img_position_ids,
        unique_timesteps,
        inverse_indices,
        update_mask,
        token_tags,
        prompt_embeds,
        img_pos_info,
        audio_pos_info,
        text_pos_info,
        img_pos_for_infer_output_info,
        packed_seq_params,
        refiner_packed_seq_params,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        gradient_checkpointing_skip_blocks="",
        gradient_checkpointing_mode="regular",
        update_audio_mask=None,
        skip_mask_out_condition=False,
        control_hints=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_indices = inverse_indices.view(-1).to(torch.long)
        token_tags = token_tags.view(-1).to(torch.long)
        text_selected = prompt_embeds

        img_pos = img_pos_info["position_ids"].view(-1).to(torch.long)
        audio_pos = audio_pos_info["position_ids"].view(-1).to(torch.long)
        text_pos = text_pos_info["position_ids"].view(-1).to(torch.long)
        infer_out_pos = img_pos_for_infer_output_info["position_ids"].view(-1).to(torch.long)

        cu_seqlens = packed_seq_params["cu_seqlens_q"].to(torch.int32)
        max_seqlen = int(packed_seq_params["max_seqlen_q"])
        refiner_cu = refiner_packed_seq_params["cu_seqlens_q"].to(torch.int32)
        refiner_max = int(refiner_packed_seq_params["max_seqlen_q"])

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        device = x.device

        rope_raw = self.rope(img_position_ids).to(device)
        if _DIT_FUSED_ROPE:
            rope_freqs = (
                torch.cos(rope_raw).to(text_selected.dtype).unsqueeze(1),
                torch.sin(rope_raw).to(text_selected.dtype).unsqueeze(1),
            )
        else:
            rope_freqs = rope_raw

        decoder_input, t_emb = self._embed(
            x=x, audio_x=audio_x, text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device), audio_pos=audio_pos.to(device), text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu.to(device), refiner_max_seqlen=refiner_max,
            seq_len=seq_len, device=device,
        )

        combined_indices = (inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        gradient_checkpointing_skip_blocks = _parse_gradient_checkpointing_skip_blocks(
            gradient_checkpointing_skip_blocks
        )
        for block_id, block in enumerate(self.blocks):
            skip_checkpoint = block_id in gradient_checkpointing_skip_blocks
            hidden = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing and not skip_checkpoint,
                use_gradient_checkpointing_offload and not skip_checkpoint,
                hidden,
                gradient_checkpointing_mode=gradient_checkpointing_mode,
                force_torch_sdpa=(gradient_checkpointing_mode == "selective_fa"),
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if control_hints is not None and block_id in control_hints:
                hidden = hidden + control_hints[block_id].to(hidden.device, hidden.dtype)

        video_logits, audio_logits = self.final_layer(hidden, t_emb=t_emb, inverse_indices=inverse_indices)

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits
