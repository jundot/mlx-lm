# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    gather_last_axis,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, CacheList, KVCache
from .gated_delta import gated_delta_update, normalize_qk
from .pipeline import PipelineMixin
from .qwen3_next import Qwen3NextAttention as TextAttention
from .qwen3_next import Qwen3NextMLP as MLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


@dataclass
class VisionArgs(BaseModelArgs):
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    out_hidden_size: int
    num_position_embeddings: int
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3


def _rotate(x, angles):
    dtype = x.dtype
    half = angles.shape[-1]
    a, b = x[..., :half].astype(mx.float32), x[..., half : 2 * half].astype(mx.float32)
    c, s = mx.cos(angles), mx.sin(angles)
    return mx.concatenate(
        [
            (a * c - b * s).astype(dtype),
            (b * c + a * s).astype(dtype),
            x[..., 2 * half :],
        ],
        axis=-1,
    )


class VisionAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_heads
        self.qkv = nn.Linear(args.hidden_size, 3 * args.hidden_size)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, x, angles):
        n = x.shape[0]
        q, k, v = self.qkv(x).reshape(n, 3, self.num_heads, -1).transpose(1, 2, 0, 3)
        q, k = _rotate(q, angles), _rotate(k, angles)
        y = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=q.shape[-1] ** -0.5
        )
        return self.proj(y[0].transpose(1, 0, 2).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, dims, hidden_dims, out_dims, approximate=True):
        super().__init__()
        self.linear_fc1 = nn.Linear(dims, hidden_dims)
        self.linear_fc2 = nn.Linear(hidden_dims, out_dims)
        self.act = nn.GELU(approx="tanh" if approximate else "none")

    def __call__(self, x):
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.attn = VisionAttention(args)
        self.mlp = VisionMLP(args.hidden_size, args.intermediate_size, args.hidden_size)

    def __call__(self, x, angles):
        x = x + self.attn(self.norm1(x), angles)
        return x + self.mlp(self.norm2(x))


class VisionMerger(VisionMLP):
    def __init__(self, args):
        self.dims = args.hidden_size * args.spatial_merge_size**2
        super().__init__(self.dims, self.dims, args.out_hidden_size, approximate=False)
        self.norm = nn.LayerNorm(args.hidden_size, eps=1e-6)

    def __call__(self, x):
        return super().__call__(self.norm(x).reshape(-1, self.dims))


class VisionPatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(
            args.in_channels * args.temporal_patch_size * args.patch_size**2,
            args.hidden_size,
        )

    def __call__(self, x):
        return self.proj(x.astype(self.proj.weight.dtype))


class VisionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = VisionPatchEmbed(args)
        self.pos_embed = nn.Embedding(args.num_position_embeddings, args.hidden_size)
        self.blocks = [VisionBlock(args) for _ in range(args.depth)]
        self.merger = VisionMerger(args)

    def __call__(self, pixels):
        h, w, _ = pixels.shape
        m = self.args.spatial_merge_size
        rows = mx.broadcast_to(mx.arange(h)[:, None], (h, w))
        cols = mx.broadcast_to(mx.arange(w)[None, :], (h, w))

        def reorder(x):
            return x.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)

        rows, cols = reorder(rows), reorder(cols)
        side = int(self.args.num_position_embeddings**0.5)
        rh = rows.astype(mx.float32) * ((side - 1) / max(h - 1, 1))
        rw = cols.astype(mx.float32) * ((side - 1) / max(w - 1, 1))
        lo_h, lo_w = rh.astype(mx.int32), rw.astype(mx.int32)
        hi_h, hi_w = mx.minimum(lo_h + 1, side - 1), mx.minimum(lo_w + 1, side - 1)
        dh, dw = rh - lo_h, rw - lo_w
        x = self.patch_embed(pixels.reshape(h * w, -1))
        pos = 0
        for ih, iw, weight in [
            (lo_h, lo_w, (1 - dh) * (1 - dw)),
            (lo_h, hi_w, (1 - dh) * dw),
            (hi_h, lo_w, dh * (1 - dw)),
            (hi_h, hi_w, dh * dw),
        ]:
            pos = (
                pos
                + self.pos_embed(ih * side + iw).astype(mx.float32) * weight[:, None]
            )
        x = x + pos.astype(x.dtype)
        dim = self.args.hidden_size // self.args.num_heads // 2
        inv = 10000.0 ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        angles = mx.concatenate([rows[:, None] * inv, cols[:, None] * inv], axis=-1)
        for block in self.blocks:
            x = block(x, angles)
        return self.merger(x)


class Attention(TextAttention):
    def __init__(self, args):
        super().__init__(args)
        self.rotary_dims = int(args.head_dim * args.partial_rotary_factor)
        self.theta = args.rope_theta
        self.sections = (args.rope_parameters or {}).get("mrope_section", [11, 11, 10])

    def __call__(self, x, mask=None, cache=None, position_ids=None):
        state = None
        if isinstance(cache, CacheList):
            cache, state = cache[0], cache[1]
        if position_ids is None and (state is None or state[0] is None):
            return super().__call__(x, mask, cache)
        b, length, _ = x.shape
        offset = cache.offset if cache is not None else 0
        if position_ids is None:
            positions = mx.arange(length)[None, :] + offset + state[0]
            position_ids = mx.broadcast_to(positions[None], (3, b, length))
        if state is not None:
            state[0] = (
                mx.max(position_ids, axis=(0, 2))[:, None] + 1 - (offset + length)
            )
        q, gate = mx.split(
            self.q_proj(x).reshape(b, length, self.num_attention_heads, -1), 2, axis=-1
        )
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(x).reshape(b, length, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(x)
            .reshape(b, length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        inv = self.theta ** (
            -mx.arange(0, self.rotary_dims, 2, dtype=mx.float32) / self.rotary_dims
        )
        angles = position_ids[..., None].astype(mx.float32) * inv
        frequencies = angles[0]
        for axis in (1, 2):
            frequencies[..., axis : self.sections[axis] * 3 : 3] = angles[
                axis, ..., axis : self.sections[axis] * 3 : 3
            ]
        q, k = _rotate(q, frequencies[:, None]), _rotate(k, frequencies[:, None])
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out * mx.sigmoid(gate.reshape(b, length, -1)))


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids=None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        position_ids=None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_cache = cache[self.fa_idx]
            if isinstance(fa_cache, CacheList):
                fa_cache = fa_cache[0]
            fa_mask = create_attention_mask(hidden_states, fa_cache)
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=c, position_ids=position_ids
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                last_cache = cache[-1]
                if isinstance(last_cache, CacheList):
                    last_cache = last_cache[0]
                if hasattr(last_cache, "keys"):
                    last_cache.keys = mx.depends(last_cache.keys, hidden_states)
                else:
                    last_cache[0] = mx.depends(last_cache[0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return self.norm(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        position_ids=None,
    ) -> mx.array:
        out = self.model(
            inputs, cache, input_embeddings=input_embeddings, position_ids=position_ids
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
            if self.sharding_group is not None:
                out = gather_last_axis(out, self.sharding_group)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: Optional[dict] = None
    image_token_id: int = 248056

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))
        if args.vision_config:
            self.vision_tower = VisionModel(VisionArgs.from_dict(args.vision_config))

    def encode_image(self, pixel_values):
        return mx.concatenate([self.vision_tower(p) for p in pixel_values])

    def embed_inputs(self, inputs, *, pixel_values=None):
        embeddings = self.model.embed_tokens(inputs)
        if pixel_values is None:
            return embeddings
        if inputs.ndim != 1:
            raise ValueError("Prepare one multimodal prompt at a time.")
        features = self.encode_image(pixel_values)
        mask = inputs == self.args.image_token_id
        if (
            sum(t == self.args.image_token_id for t in inputs.tolist())
            != features.shape[0]
        ):
            raise ValueError("Image features and placeholder tokens do not match.")
        index = mx.clip(mx.cumsum(mask) - 1, 0, features.shape[0] - 1)
        return mx.where(
            mask[:, None], features[index].astype(embeddings.dtype), embeddings
        )

    def _multimodal_positions(self, inputs, grids):
        if inputs.ndim != 1:
            raise ValueError("Prepare one multimodal prompt at a time.")
        ids = inputs.tolist()
        positions, cursor, next_position = [], 0, 0
        for h, w in grids:
            start = ids.index(self.args.image_token_id, cursor)
            n = h * w
            if ids[start : start + n] != [self.args.image_token_id] * n:
                raise ValueError("Image features and placeholder tokens do not match.")
            text_positions = mx.arange(next_position, next_position + start - cursor)
            positions.append(mx.broadcast_to(text_positions, (3, start - cursor)))
            base = next_position + start - cursor
            grid = mx.stack(
                [
                    mx.zeros(n, mx.int32),
                    mx.repeat(mx.arange(h), w),
                    mx.tile(mx.arange(w), h),
                ]
            )
            positions.append(grid + base)
            next_position = base + max(h, w)
            cursor = start + n
        if self.args.image_token_id in ids[cursor:]:
            raise ValueError("More image placeholders than image features.")
        tail = mx.arange(next_position, next_position + len(ids) - cursor)
        positions.append(mx.broadcast_to(tail, (3, len(ids) - cursor)))
        position_ids = mx.concatenate(positions, axis=-1)[:, None, :]
        return position_ids

    def get_position_ids(self, inputs, *, pixel_values, cache=None):
        merge = self.vision_tower.args.spatial_merge_size
        grids = [(p.shape[0] // merge, p.shape[1] // merge) for p in pixel_values]
        position_ids = self._multimodal_positions(inputs, grids)
        if cache is not None:
            c = cache[self.model.fa_idx]
            delta = c[1][0] if isinstance(c, CacheList) else None
            kv = c[0] if isinstance(c, CacheList) else c
            position_ids = position_ids + kv.offset + (0 if delta is None else delta)
        return position_ids

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        position_ids=None,
        pixel_values=None,
    ):
        if pixel_values is not None:
            if input_embeddings is not None or inputs.shape[0] != 1:
                raise ValueError(
                    "Raw images require one prompt and no input embeddings."
                )
            input_embeddings = self.embed_inputs(inputs[0], pixel_values=pixel_values)[
                None
            ]
            position_ids = self.get_position_ids(
                inputs[0], pixel_values=pixel_values, cache=cache
            )
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            position_ids=position_ids,
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(("model.visual.", "vision_tower.", "visual.")):
                if not self.args.vision_config:
                    continue
                key = (
                    key.replace("model.visual.", "vision_tower.", 1)
                    if key.startswith("model.visual.")
                    else key.replace("visual.", "vision_tower.", 1)
                )
                if key == "vision_tower.patch_embed.proj.weight" and value.ndim == 5:
                    value = value.reshape(value.shape[0], -1)
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # Each rank computes a slice of the vocabulary
        lm = self.language_model
        if not lm.args.tie_word_embeddings:
            vocab = lm.args.vocab_size
            assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
            lm.lm_head = shard_linear(lm.lm_head, "all-to-sharded", group=group)
            lm.sharding_group = group

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return [
            (
                ArraysCache(size=2)
                if layer.is_linear
                else (
                    CacheList(KVCache(), ArraysCache(1))
                    if self.args.vision_config
                    else KVCache()
                )
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        predicate = self.language_model.quant_predicate
        return lambda path, module: (
            False
            if path.startswith("vision_tower.")
            else (predicate(path, module) if predicate else True)
        )

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
