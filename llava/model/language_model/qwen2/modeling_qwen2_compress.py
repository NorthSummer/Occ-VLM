# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Qwen2 model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from cuml.cluster import DBSCAN
from .dbscan import DBSCANVectorized
from .mrf import DifferentiableMRFSingleSparseEdge
import cudf

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"


def invert_grid_selected_token_order(grid_selected_token_order):

    reversed_dict = {}

    for grid, token_list in grid_selected_token_order.items():
        for tok in token_list:
            tok_i = int(tok)  
            reversed_dict[tok_i] = (int(grid[0]), int(grid[1]), int(grid[2]))

    return reversed_dict

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.rope_type = "default"
        self.attention_scaling = 1
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # if "dynamic" in self.rope_type:
        #     self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (4, ...)
        # position_ids: [bs, seq_len, 3]
        position_ids = position_ids.permute(2, 0, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Modified from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = [32, 16, 16]
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->Qwen2
class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


import torch
import torch.nn.functional as F
import math
from collections import defaultdict

class MarkovTokenSelector:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, grid_size=0.2, device='cpu', d_max=1.0):
        """
        alpha: 几何距离权重
        beta: 特征相似度权重
        gamma: attention权重
        grid_size: x-y平面网格大小
        device: 计算设备
        d_max: 距离归一化最大值
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.grid_size = grid_size
        self.device = device
        self.d_max = d_max

        self.base_probs = torch.tensor([
            [0.3, 0.5, 0.2],  # from drop
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ], device=self.device)

        self.eps = 1e-6

    def feature_similarity(self, f1, f2):
        f1_norm = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-8)
        f2_norm = f2 / (f2.norm(dim=-1, keepdim=True) + 1e-8)
        return (f1_norm * f2_norm).sum(dim=-1).clamp(0,1)  # 保证在[0,1]

    def geometric_distance(self, c1, c2):
        dist_raw = torch.norm(c1 - c2, p=2)
        dist_norm = torch.clamp(dist_raw / self.d_max, 0, 1)  # 归一化到[0,1]
        return dist_norm

    def viterbi(self, init_probs, trans_probs):
        seq_len = trans_probs.shape[0] + 1
        num_states = 3
        dp = torch.full((seq_len, num_states), -float('inf'), device=self.device)
        ptr = torch.zeros((seq_len, num_states), dtype=torch.long, device=self.device)

        dp[0] = torch.log(init_probs + 1e-10)

        for t in range(1, seq_len):
            for curr_state in range(num_states):
                prob_candidates = dp[t-1] + torch.log(trans_probs[t-1, :, curr_state] + 1e-10)
                dp[t, curr_state], ptr[t, curr_state] = torch.max(prob_candidates, dim=0)

        states = [0] * seq_len
        states[-1] = torch.argmax(dp[-1]).item()
        for t in reversed(range(1, seq_len)):
            states[t-1] = ptr[t, states[t]].item()

        return states

    def viterbi_optimized(self, init_probs, trans_probs):
        seq_len = trans_probs.shape[0] + 1
        num_states = 3
        
        # 使用更高效的动态规划
        log_init_probs = torch.log(init_probs + self.eps)
        log_trans_probs = torch.log(trans_probs + self.eps)
        
        # 初始化
        dp = torch.full((seq_len, num_states), -float('inf'), device=self.device)
        dp[0] = log_init_probs
        
        # 向量化动态规划更新
        for t in range(1, seq_len):
            # 使用广播避免显式循环
            dp[t] = torch.max(dp[t-1].unsqueeze(1) + log_trans_probs[t-1], dim=0)[0]
        
        # 回溯（这部分较难向量化，保持原样）
        ptr = torch.zeros((seq_len, num_states), dtype=torch.long, device=self.device)
        for t in range(1, seq_len):
            for curr_state in range(num_states):
                prob_candidates = dp[t-1] + log_trans_probs[t-1, :, curr_state]
                _, ptr[t, curr_state] = torch.max(prob_candidates, dim=0)
        
        # 状态回溯
        states = [0] * seq_len
        states[-1] = torch.argmax(dp[-1]).item()
        for t in reversed(range(1, seq_len)):
            states[t-1] = ptr[t, states[t]].item()
            
        return states
    
    def compute_init_probs(self, attention_scores):
        # attention_scores shape (seq_len,)
        attn = attention_scores.clamp(0,1)
        # 初始sim和dist取1，保证retain和merge概率合理分布
        sim = torch.ones_like(attn)
        dist = torch.ones_like(attn)

        z0 = 1 - attn[0]
        z2 = attn[0] * sim[0] * dist[0]
        z1 = attn[0] * (1 - sim[0] * dist[0])

        init_probs = torch.tensor([z0.item(), z1.item(), z2.item()], device=self.device)
        # 归一化（理论上和为1，但数值误差时归一化）
        init_probs /= init_probs.sum()
        return init_probs
    
    def compute_transition_probs_v1(self, coords, features, attention_scores):
        seq_len = coords.shape[0]
        num_states = 3
        trans_probs = torch.zeros(seq_len - 1, num_states, num_states, device=self.device)

        # base_probs = torch.tensor([
        #     [0.4, 0.3, 0.3],  # from drop
        #     [0.3, 0.4, 0.3],
        #     [0.3, 0.3, 0.4],
        # ], device=self.device)

        base_probs = torch.tensor([
            [0.3, 0.5, 0.2],  # from drop
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ], device=self.device)

        # 计算两两距离矩阵，归一化并转成相似度
        coords_exp1 = coords.unsqueeze(1)  # [seq_len,1,3]
        coords_exp2 = coords.unsqueeze(0)  # [1,seq_len,3]
        dist_matrix = torch.norm(coords_exp1 - coords_exp2, dim=-1)  # [seq_len, seq_len]
        dist_matrix = torch.clamp(dist_matrix / self.d_max, 0, 1)
        dist_sim_matrix = 1 - dist_matrix  # 距离越小相似度越大

        # 计算特征相似度矩阵（余弦相似度）
        f_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(f_norm, f_norm.transpose(0,1))  # [seq_len, seq_len]
        sim_matrix = sim_matrix.clamp(0,1)

        # 对相似度矩阵和距离相似度矩阵分别做softmax归一化（按行）
        sim_weights = torch.softmax(sim_matrix, dim=-1)  # [seq_len, seq_len]
        dist_weights = torch.softmax(dist_sim_matrix, dim=-1)  # [seq_len, seq_len]

        eps = 1e-6

        attn_min = attention_scores.min()
        attn_max = attention_scores.max()
        attention_scores = (attention_scores - attn_min) / (attn_max - attn_min + 1e-8)

        for t in range(seq_len - 1):
            attn2 = attention_scores[t+1].clamp(0,1)
            # 计算第t个token相对于其他token的加权相似度和距离相似度
            sim_feat = (sim_weights[t+1] * sim_matrix[t+1]).sum()
            d_geo_sim = (dist_weights[t+1] * dist_sim_matrix[t+1]).sum()

            # 加平滑防止为0
            sim_feat = sim_feat + eps
            d_geo_sim = d_geo_sim + eps

            # 计算三状态概率z0,z1,z2
            z0 = self.alpha * (1 - attn2)
            z2 = self.beta * (attn2 * sim_feat * d_geo_sim)
            z1 = self.gamma * (attn2 * (1 - sim_feat * d_geo_sim))

            z = torch.tensor([z0, z1, z2], device=self.device)
            z = torch.clamp(z, min=eps)
            z /= z.sum()

            for i_state in range(num_states):
                trans_probs[t, i_state, :] = base_probs[i_state, :] * z
                trans_probs[t, i_state, :] /= trans_probs[t, i_state, :].sum()

        return trans_probs

    def compute_transition_probs_optimized(self, coords, features, attention_scores):
        seq_len = coords.shape[0]
        num_states = 3
        
        # 向量化归一化
        attn_min, attn_max = torch.aminmax(attention_scores)
        attention_scores_norm = (attention_scores - attn_min) / (attn_max - attn_min + self.eps)
        
        # 预计算所有距离和相似度（向量化）
        # 使用 torch.cdist 计算所有点对距离
        dist_matrix = torch.cdist(coords.float(), coords.float(), p=2).to(torch.bfloat16)  # [seq_len, seq_len]
        dist_norm_matrix = torch.clamp(dist_matrix / self.d_max, 0, 1)
        dist_sim_matrix = 1 - dist_norm_matrix
        
        # 向量化特征相似度计算
        f_norm = features / (features.norm(dim=-1, keepdim=True) + self.eps)
        sim_matrix = torch.mm(f_norm, f_norm.transpose(0, 1))
        sim_matrix = sim_matrix.clamp(0, 1)
        
        # 向量化状态概率计算
        attn2 = attention_scores_norm[1:].clamp(0, 1)  # [seq_len-1]
        
        # 计算相邻点的相似度（避免全局计算）
        feat_sim = torch.einsum('nd,nd->n', f_norm[:-1], f_norm[1:])  # [seq_len-1]
        dist_sim = 1 - torch.clamp(
            torch.norm(coords[:-1] - coords[1:], p=2, dim=1) / self.d_max, 0, 1
        )  # [seq_len-1]
        
        # 向量化 z 计算
        z0 = self.alpha * (1 - attn2)  # [seq_len-1]
        z2 = self.beta * (attn2 * feat_sim * dist_sim)  # [seq_len-1]
        z1 = self.gamma * (attn2 * (1 - feat_sim * dist_sim))  # [seq_len-1]
        
        # 组合 z 向量
        z = torch.stack([z0, z1, z2], dim=1)  # [seq_len-1, 3]
        z = torch.clamp(z, min=self.eps)
        z = z / z.sum(dim=1, keepdim=True)  # [seq_len-1, 3]
        
        # 向量化转移概率计算
        base_probs_expanded = self.base_probs.unsqueeze(0).expand(seq_len-1, -1, -1)
        z_expanded = z.unsqueeze(1).expand(-1, num_states, -1)
        
        trans_probs = base_probs_expanded * z_expanded
        trans_probs = trans_probs / trans_probs.sum(dim=2, keepdim=True)
        
        return trans_probs
    
    def compute_transition_probs(self, coords, features, attention_scores):

        seq_len = coords.shape[0]
        num_states = 3
        trans_probs = torch.zeros(seq_len - 1, num_states, num_states, device=self.device)


        base_probs = torch.tensor([
            [0.3, 0.5, 0.2],  # from drop
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ], device=self.device)

        eps = 1e-6

        # 归一化attention scores到[0,1]
        attn_min = attention_scores.min()
        attn_max = attention_scores.max()
        attention_scores_norm = (attention_scores - attn_min) / (attn_max - attn_min + eps)

        for t in range(seq_len - 1):
            # 计算token t和t+1的归一化欧氏距离相似度
            dist_raw = torch.norm(coords[t] - coords[t+1], p=2)
            dist_norm = torch.clamp(dist_raw / self.d_max, 0, 1)
            dist_sim = 1 - dist_norm  # 距离越小相似度越大

            # 计算token t和t+1的特征余弦相似度
            f1_norm = features[t] / (features[t].norm() + eps)
            f2_norm = features[t+1] / (features[t+1].norm() + eps)
            feat_sim = (f1_norm * f2_norm).sum().clamp(0, 1)

            attn2 = attention_scores_norm[t+1].clamp(0, 1)

            # 计算三状态概率分量，权重参数alpha, beta, gamma来自初始化
            z0 = self.alpha * (1 - attn2)
            z2 = self.beta * (attn2 * feat_sim * dist_sim)
            z1 = self.gamma * (attn2 * (1 - feat_sim * dist_sim))

            z = torch.tensor([z0, z1, z2], device=self.device)
            z = torch.clamp(z, min=eps)
            z /= z.sum()

            # 将基础转移概率矩阵和状态概率分量相乘，得到最终转移概率
            for i_state in range(num_states):
                trans_probs[t, i_state, :] = base_probs[i_state, :] * z
                trans_probs[t, i_state, :] /= trans_probs[t, i_state, :].sum()

        return trans_probs


    def merge_tokens_by_states(self, states, features_grid, idx_tensor):
        """
        根据states合并token：
        - 状态0丢弃
        - 状态1单独保留
        - 连续两个及以上的状态2合并，合并后token索引为第一个2的索引
        - 单独的状态2单独保留
        """
        device = features_grid.device
        merged_features = []
        merged_indices = []

        i = 0
        while i < len(states):
            state = states[i]
            if state == 0:
                # 丢弃
                i += 1
            elif state == 1:
                # 单独保留
                merged_features.append(features_grid[i])
                merged_indices.append(idx_tensor[i].item())
                i += 1
            elif state == 2:
                # 检查是否有连续的2
                j = i + 1
                while j < len(states) and states[j] == 2:
                    j += 1
                length = j - i
                if length >= 2:
                    # 连续2，合并i到j-1的token
                    merge_range = list(range(i, j))
                    merged_feat = features_grid[merge_range].mean(dim=0)
                    merged_features.append(merged_feat)
                    merged_indices.append(idx_tensor[i].item())  # 以第一个2的索引为准
                    i = j
                else:
                    # 单独的2，直接保留
                    merged_features.append(features_grid[i])
                    merged_indices.append(idx_tensor[i].item())
                    i += 1
            else:
                # 其他状态，跳过
                i += 1

        if len(merged_features) == 0:
            # 防止空列表导致stack报错
            return torch.empty((0, features_grid.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)

        merged_features = torch.stack(merged_features)
        merged_indices = torch.tensor(merged_indices, dtype=torch.long, device=device)
        return merged_features, merged_indices
    

    def select_tokens(self, token_coords, features, attention):
        device = self.device
        N = features.shape[0]

        x_coords = token_coords[:, 0]
        y_coords = token_coords[:, 1]
        z_coords = token_coords[:, 2]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        num_grids_x = math.ceil((x_max - x_min).item() / self.grid_size)
        num_grids_y = math.ceil((y_max - y_min).item() / self.grid_size)

        if num_grids_x == 0:
            num_grids_x = 1
        if num_grids_y == 0:
            num_grids_y = 1

        x_indices = ((x_coords - x_min) / self.grid_size).floor().long()
        y_indices = ((y_coords - y_min) / self.grid_size).floor().long()

        x_indices = torch.clamp(x_indices, 0, num_grids_x - 1)
        y_indices = torch.clamp(y_indices, 0, num_grids_y - 1)

        grid_token_indices = defaultdict(list)
        for i in range(N):
            key = (x_indices[i].item(), y_indices[i].item())
            grid_token_indices[key].append(i)

        selected_features_list = []
        selected_indices_list = []

        for key, idx_list in grid_token_indices.items():
            if len(idx_list) == 1:
                # 只有一个token，直接保留，注意unsqueeze保持二维
                selected_features_list.append(features[idx_list[0]].unsqueeze(0))  # [1, D]
                selected_indices_list.append(torch.tensor([idx_list[0]], dtype=torch.long, device=device))  # [1]
                continue

            idx_tensor = torch.tensor(idx_list, device=device)
            coords_grid = token_coords[idx_tensor]
            features_grid = features[idx_tensor]
            attention_grid = attention[idx_tensor]

            # 按z排序
            z_sorted_indices = torch.argsort(coords_grid[:, 2])
            idx_tensor = idx_tensor[z_sorted_indices]
            coords_grid = coords_grid[z_sorted_indices]
            features_grid = features_grid[z_sorted_indices]
            attention_grid = attention_grid[z_sorted_indices]

            init_probs = self.compute_init_probs(attention_grid)
            trans_probs = self.compute_transition_probs_optimized(coords_grid, features_grid, attention_grid)

            states = self.viterbi_optimized(init_probs, trans_probs)

            # 保留状态1和2的token，状态0为drop
            merged_features, merged_indices = self.merge_tokens_by_states(states, features_grid, idx_tensor)

            # 确保 merged_features 是二维， merged_indices 是一维
            if merged_features.dim() == 1:
                merged_features = merged_features.unsqueeze(0)
            if merged_indices.dim() == 0:
                merged_indices = merged_indices.unsqueeze(0)

            selected_features_list.append(merged_features)
            selected_indices_list.append(merged_indices.to(torch.long))

        if len(selected_features_list) == 0:
            # 防止空返回
            return torch.empty((0, features.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)

        # 拼接所有格子选中的token
        recovered_features = torch.cat(selected_features_list, dim=0)
        recovered_token_indices = torch.cat(selected_indices_list, dim=0)

        # 按索引排序
        sorted_indices = torch.argsort(recovered_token_indices)
        recovered_token_indices = recovered_token_indices[sorted_indices]
        recovered_features = recovered_features[sorted_indices]

        return recovered_features, recovered_token_indices.to(torch.long)

class MRFTokenSelector(nn.Module):
    def __init__(self, grid_size=0.2, device='cpu', d_max=1.0, random_init=False):
        super().__init__()
        self.grid_size = grid_size
        self.device = device
        self.d_max = d_max
        self.eps = 1e-8
        
        # 可学习的权重参数
        if random_init:
            self._random_initialize_parameters()
        else:
            self._initialize_parameters()
    
    def _initialize_parameters(self):
        """默认参数初始化"""
        # 单节点势函数权重
        self.drop_weight = nn.Parameter(torch.tensor([1.0], device=self.device))
        self.keep_weight = nn.Parameter(torch.tensor([1.0], device=self.device))
        
        # 成对势函数权重
        self.pairwise_weight = nn.Parameter(torch.tensor([1.0], device=self.device))
        self.geo_weight = nn.Parameter(torch.tensor([0.5], device=self.device))
        self.feat_weight = nn.Parameter(torch.tensor([0.5], device=self.device))
        
        # 兼容性矩阵（状态间转移倾向）
        self.compatibility_matrix = nn.Parameter(torch.tensor([
            [0.1, 0.3, 0.2],  # drop -> [drop, keep, merge]
            [0.3, 0.1, 0.4],  # keep -> [drop, keep, merge]
            [0.2, 0.4, 0.8]   # merge -> [drop, keep, merge]
        ], device=self.device))
    
    def _random_initialize_parameters(self):
        """随机参数初始化"""
        # 单节点势函数权重
        self.drop_weight = nn.Parameter(torch.rand(1, device=self.device) * 2.0)
        self.keep_weight = nn.Parameter(torch.rand(1, device=self.device) * 2.0)
        
        # 成对势函数权重
        self.pairwise_weight = nn.Parameter(torch.rand(1, device=self.device) * 2.0)
        self.geo_weight = nn.Parameter(torch.rand(1, device=self.device))
        self.feat_weight = nn.Parameter(torch.rand(1, device=self.device))
        
        # 兼容性矩阵随机初始化
        random_matrix = torch.rand(3, 3, device=self.device) * 0.5 + 0.1
        self.compatibility_matrix = nn.Parameter(random_matrix)
    
    def compute_unary_potential(self, token_idx, state, attention):
        """单节点势函数：仅基于attention"""
        attn_score = attention[token_idx]
        
        if state == 0:  # drop
            # attention越高，越不应该drop
            potential = self.drop_weight * attn_score
        else:  # keep or merge (state == 1 or 2)
            # attention越低，越不应该keep/merge  
            potential = -self.keep_weight * (1 - attn_score)
            
        return potential
    
    def compute_pairwise_potential(self, i, j, state_i, state_j, features, coords):
        """成对势函数：基于几何距离和特征相似度"""
        # 几何距离相似度
        geo_distance = torch.norm(coords[i] - coords[j], p=2)
        geo_similarity = torch.exp(-self.pairwise_weight * geo_distance / self.d_max)
        
        # 特征相似度
        feat_similarity = F.cosine_similarity(
            features[i].unsqueeze(0), 
            features[j].unsqueeze(0), 
            dim=1
        ).squeeze()
        
        # 组合相似度
        combined_similarity = (
            self.geo_weight * geo_similarity + 
            self.feat_weight * feat_similarity
        )
        
        # 状态兼容性
        compatibility = self.compatibility_matrix[state_i, state_j]
        
        # 成对势函数：鼓励相似token有兼容状态
        pairwise_potential = -compatibility * combined_similarity
        
        return pairwise_potential
    
    def compute_total_energy(self, states, features, coords, attention):
        """计算给定状态配置的总能量"""
        if len(states) == 0:
            return torch.tensor(0.0, device=self.device)
        
        unary_energy = 0
        pairwise_energy = 0
        num_tokens = len(states)
        
        # 计算单节点能量
        for i in range(num_tokens):
            unary_energy += self.compute_unary_potential(i, states[i], attention)
        
        # 计算成对能量（相邻token）
        for i in range(num_tokens):
            neighbors = self._get_spatial_neighbors(i, coords)
            for j in neighbors:
                if j > i:  # 避免重复计算
                    pairwise_energy += self.compute_pairwise_potential(
                        i, j, states[i], states[j], features, coords
                    )
        
        return unary_energy + pairwise_energy
    
    def _get_spatial_neighbors(self, token_idx, coords, max_neighbors=8):
        """获取空间相邻的token（基于距离）"""
        neighbors = []
        if len(coords) <= 1:
            return neighbors
            
        coord_i = coords[token_idx]
        distances = []
        
        for j, coord_j in enumerate(coords):
            if j != token_idx:
                dist = torch.norm(coord_i - coord_j, p=2)
                distances.append((dist, j))
        
        # 取最近的邻居
        distances.sort()
        neighbors = [idx for _, idx in distances[:min(max_neighbors, len(distances))]]
        
        return neighbors
    
    def greedy_state_inference(self, features, coords, attention):
        """贪心状态推断（简化版MRF推断）"""
        num_tokens = features.shape[0]
        states = []
        
        for i in range(num_tokens):
            # 计算每种状态的能量
            energies = []
            for state in [0, 1, 2]:  # drop, keep, merge
                # 考虑局部环境的能量
                local_energy = self.compute_unary_potential(i, state, attention)
                
                # 简单考虑与最近邻居的成对能量
                neighbors = self._get_spatial_neighbors(i, coords, max_neighbors=2)
                for j in neighbors:
                    if j < len(states):  # 已经推断的状态
                        # 假设邻居状态为已推断状态
                        pairwise_energy = self.compute_pairwise_potential(
                            i, j, state, states[j], features, coords
                        )
                        local_energy += pairwise_energy * 0.1  # 权重较小
                
                energies.append(local_energy)
            
            # 选择能量最低的状态
            best_state = torch.argmin(torch.stack(energies)).item()
            states.append(best_state)
        
        return states
    
    def merge_tokens_by_states(self, states, features_grid, idx_tensor):
        """根据states合并token"""
        device = features_grid.device
        merged_features = []
        merged_indices = []

        i = 0
        while i < len(states):
            state = states[i]
            if state == 0:
                # 丢弃
                i += 1
            elif state == 1:
                # 单独保留
                merged_features.append(features_grid[i])
                merged_indices.append(idx_tensor[i].item())
                i += 1
            elif state == 2:
                # 检查是否有连续的2
                j = i + 1
                while j < len(states) and states[j] == 2:
                    j += 1
                length = j - i
                if length >= 2:
                    # 连续2，合并i到j-1的token
                    merge_range = list(range(i, j))
                    merged_feat = features_grid[merge_range].mean(dim=0)
                    merged_features.append(merged_feat)
                    merged_indices.append(idx_tensor[i].item())  # 以第一个2的索引为准
                    i = j
                else:
                    # 单独的2，直接保留
                    merged_features.append(features_grid[i])
                    merged_indices.append(idx_tensor[i].item())
                    i += 1
            else:
                # 其他状态，跳过
                i += 1

        if len(merged_features) == 0:
            # 防止空列表导致stack报错
            return torch.empty((0, features_grid.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)

        merged_features = torch.stack(merged_features)
        merged_indices = torch.tensor(merged_indices, dtype=torch.long, device=device)
        return merged_features, merged_indices
    
    def select_tokens(self, token_coords, features, attention):
        """主函数：选择和压缩token"""
        device = self.device
        N = features.shape[0]

        if N == 0:
            return torch.empty((0, features.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)

        x_coords = token_coords[:, 0]
        y_coords = token_coords[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # 处理边界情况
        if x_max - x_min < self.eps:
            num_grids_x = 1
        else:
            num_grids_x = max(1, math.ceil((x_max - x_min).item() / self.grid_size))
            
        if y_max - y_min < self.eps:
            num_grids_y = 1
        else:
            num_grids_y = max(1, math.ceil((y_max - y_min).item() / self.grid_size))

        x_indices = ((x_coords - x_min) / self.grid_size).floor().long()
        y_indices = ((y_coords - y_min) / self.grid_size).floor().long()

        x_indices = torch.clamp(x_indices, 0, num_grids_x - 1)
        y_indices = torch.clamp(y_indices, 0, num_grids_y - 1)

        grid_token_indices = defaultdict(list)
        for i in range(N):
            key = (x_indices[i].item(), y_indices[i].item())
            grid_token_indices[key].append(i)

        selected_features_list = []
        selected_indices_list = []

        for key, idx_list in grid_token_indices.items():
            if len(idx_list) == 1:
                # 只有一个token，直接保留
                selected_features_list.append(features[idx_list[0]].unsqueeze(0))
                selected_indices_list.append(torch.tensor([idx_list[0]], dtype=torch.long, device=device))
                continue

            idx_tensor = torch.tensor(idx_list, device=device)
            coords_grid = token_coords[idx_tensor]
            features_grid = features[idx_tensor]
            attention_grid = attention[idx_tensor]

            # 按z排序
            if coords_grid.shape[0] > 1:
                z_sorted_indices = torch.argsort(coords_grid[:, 2])
                idx_tensor = idx_tensor[z_sorted_indices]
                coords_grid = coords_grid[z_sorted_indices]
                features_grid = features_grid[z_sorted_indices]
                attention_grid = attention_grid[z_sorted_indices]

            # MRF状态推断
            states = self.greedy_state_inference(features_grid, coords_grid, attention_grid)

            # 合并token
            merged_features, merged_indices = self.merge_tokens_by_states(states, features_grid, idx_tensor)

            # 确保维度正确
            if merged_features.dim() == 1:
                merged_features = merged_features.unsqueeze(0)
            if merged_indices.dim() == 0:
                merged_indices = merged_indices.unsqueeze(0)

            selected_features_list.append(merged_features)
            selected_indices_list.append(merged_indices.to(torch.long))

        if len(selected_features_list) == 0:
            return torch.empty((0, features.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)

        # 拼接所有格子选中的token
        recovered_features = torch.cat(selected_features_list, dim=0)
        recovered_token_indices = torch.cat(selected_indices_list, dim=0)

        # 按索引排序
        if recovered_token_indices.shape[0] > 0:
            sorted_indices = torch.argsort(recovered_token_indices)
            recovered_token_indices = recovered_token_indices[sorted_indices]
            recovered_features = recovered_features[sorted_indices]

        return recovered_features, recovered_token_indices.to(torch.long)
    
    def get_parameters_info(self):
        """获取当前参数信息"""
        info = {
            'drop_weight': self.drop_weight.item(),
            'keep_weight': self.keep_weight.item(),
            'pairwise_weight': self.pairwise_weight.item(),
            'geo_weight': self.geo_weight.item(),
            'feat_weight': self.feat_weight.item(),
            'compatibility_matrix': self.compatibility_matrix.detach().cpu().numpy()
        }
        return info
    
import os
def cache_tokens(save_dir, coords, visual_tokens, attention):
    """
    把一段推理得到的图像 token 特征缓存到磁盘

    参数:
    - save_dir: 保存目录
    - sample_id: 当前样本的编号（可用 batch_id 或 step_id）
    - features: 全部 token features [seq_len, hidden_dim]
    - image_index: 图像 token 起始位置
    - num_image_tokens: 图像 token 数量
    - attention_scores: 对应的注意力分数 [num_image_tokens]
    - device: torch device
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取出本 sample 的图像 tokens


    # 构造存储 dict
    data = {
        "coords": coords.to(torch.float16),
        "features": visual_tokens.to(torch.float16),
        "attention": attention.to(torch.float16)
    }
    with open("./cache.txt", "r") as f:
        sample_id = f.readline().strip()  
    # 保存
    save_path = os.path.join(save_dir, f"sample_{sample_id}.pt")
    torch.save(data, save_path)
    print(f"[CACHE] Saved tokens to {save_path}")


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length, 1).repeat(1, 1, 3)
        elif position_ids.dim() == 2:
            position_ids = position_ids[..., None].repeat(1, 1, 3)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_coords: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = True
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if isinstance(past_key_values, (tuple, list)):
            seq_length_with_past = seq_length_with_past 
        

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        elif position_ids.dim() == 2:
            position_ids = position_ids[..., None].repeat(1, 1, 3)
        else:
            position_ids = position_ids.view(-1, seq_length).long()


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:

                USE_FAST_V = self.use_fast_v
                SYS_LENGTH= self.fast_v_sys_length
                IMAGE_TOKEN_LENGTH = self.fast_v_image_token_length
                ATTENTION_RANK = self.fast_v_attention_rank
                AGG_LAYER = self.fast_v_agg_layer
                FASTV_INPLACE = self.fast_v_inplace
                
                if AGG_LAYER:
                    assert AGG_LAYER > 0 , "K should be larger than 0"

                if inputs_embeds.shape[1] < 100:
                    USE_FAST_V = False
        
                # FastV Token Rerank, Token Drop Implementation, KV-Cache not supported
                if USE_FAST_V and FASTV_INPLACE:
                    #print("using inplace")

                    if idx<AGG_LAYER:
                        new_attention_mask = attention_mask

                    elif idx==AGG_LAYER:
                        # compute pruned tokens, generate fastv sign
                        last_layer_attention = layer_outputs[1]
                        # compute average attention over different head
                        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                        # generate new attention mask based on the average attention, sample the top ATTENTION_RANK tokens with highest attention
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        # get the attention in image token
                        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                        # get the indexs of the top ATTENTION_RANK tokens
                        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
                        # keep index
                        keep_indexs = torch.cat( (torch.arange(SYS_LENGTH,device=device), top_attention_rank_index, torch.arange(SYS_LENGTH+IMAGE_TOKEN_LENGTH,seq_length_with_past,device=device)))
                        # sort index
                        keep_indexs = keep_indexs.sort().values
                        # update seq length
                        new_seq_length = keep_indexs.shape[0]
                        # filter hidden states
                        hidden_states = hidden_states[:,keep_indexs,:]
                        # update position ids
                        position_ids = keep_indexs.unsqueeze(0)
                        if position_ids.dim() == 2:
                            position_ids = position_ids[..., None].repeat(1, 1, 3)
                        # update attention mask
                        new_attention_mask = _prepare_4d_causal_attention_mask(
                            None, (batch_size, new_seq_length), inputs_embeds, 0
                        )

                    else:
                        new_attention_mask = gen_attention_mask
                
                else: 
                    new_attention_mask=None

                if hidden_states.shape[1] == 1:
                    # new_attention_mask = new_attention_mask[..., 0].unsqueeze(0)
                    print(new_attention_mask)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_coords: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length, 1).repeat(1, 1, 3)
        elif position_ids.dim() == 2:
            position_ids = position_ids[..., None].repeat(1, 1, 3)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            # 0~27 qwen2
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            rank_layer = layer_idx+1
            
            if rank_layer in self.layer_list:
                if hidden_states.shape[1]!=1:  
                    stage = self.layer_list.index(rank_layer) # determine current stage
                    (
                        position_ids,
                        attention_mask,
                        hidden_states,
                        # recovered_token_indices
                    ) = self.pdrop_rank_drop(    
                        cur_num = stage,
                        rank_layer = rank_layer,
                        features = hidden_states,   
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        # token_coords=token_coords[0]
                    )
                    

                    recovered_token_indices = None
                    # process attention_mask again after undating 
                    if self._attn_implementation == "flash_attention_2":
                        # 2d mask is passed through the layers
                        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                    elif self._attn_implementation == "sdpa" and not output_attentions:
                        # output_attentions=True can not be supported when using SDPA, and we fall back on
                        # the manual implementation that requires a 4D causal mask in all cases.
                        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                            attention_mask,
                            (batch_size, seq_length),
                            inputs_embeds,
                            past_key_values_length,
                        )
                    else:
                        # 4d mask is passed through the layers
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                        )
                        
                else:
                    # update position_ids in decoding stage when inference
                    stage = self.layer_list.index(rank_layer) # determine current stage
                    cur_visual_length = [int(cur_image_token * self.image_token_ratio_list[stage]) for cur_image_token in self.image_tokens]
                    next_visual_length = [int(cur_image_token * self.image_token_ratio_list[stage + 1]) for cur_image_token in self.image_tokens]
                    # next_visual_length = [recovered_token_indices.shape[0].item()]
                    
                    new_position_ids = []
                    for idx, cur_position_ids in enumerate(position_ids):
                        cur_position_ids = cur_position_ids - (cur_visual_length[idx] - next_visual_length[idx])
                        new_position_ids.append(cur_position_ids)
                    position_ids = torch.stack(new_position_ids, dim=0)         
                    
        hidden_states = self.norm(hidden_states)



        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        
        if not return_dict: # ←
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, recovered_token_indices] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def pdrop_rank_drop(
        self, cur_num, rank_layer, features ,
        position_ids, attention_mask
    ):

        _position_ids = position_ids
        _attention_mask = attention_mask

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
        
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "right":
            
            batch_size = features.shape[0]
            image_tokens = [int(cur_image_token * self.image_token_ratio_list[cur_num]) for cur_image_token in self.image_tokens]
            keep_length = [int(cur_image_token * self.image_token_ratio_list[cur_num + 1]) for cur_image_token in self.image_tokens]

            features_list = []
            attention_mask_list = []

            if attention_mask is None:
                attention_mask = torch.ones((batch_size,features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()
           
            # obtain query_states and key_states to calculate attention map
            hidden_states=features.clone().detach()
            self_attn = self.layers[rank_layer].self_attn
            hidden_states = self.layers[rank_layer].input_layernorm(hidden_states)

            num_heads = self_attn.num_heads
            num_key_value_heads = self_attn.num_key_value_heads
            head_dim = self_attn.head_dim

            bsz, q_len, _ = hidden_states.size()

            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
            value_states = self_attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            # cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            cos, sin = self_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # attention_mask 
            # eager_attention_mask = _prepare_4d_causal_attention_mask(
            #     attention_mask, (batch_size, q_len), hidden_states, past_key_values_length=0
            # ).to(device=query_states.device)

            eager_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, q_len),
                hidden_states,
                past_key_values_length=0,
                sliding_window=self.config.sliding_window,
            )

            # take valid features
            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]
    
            attention_mask = [cur_attention_mask[cur_attention_mask] for cur_attention_mask, cur_attention_mask in zip(attention_mask, attention_mask)]

            # rank & drop
            for i in range(batch_size):
                image_index= self.image_token_posi[i]
                if image_index == -1:
                    cur_input_embeds = features[i]
                    features_list.append(cur_input_embeds)
                    attention_mask_list.append(attention_mask[i])
              
                    continue
                
                # obtain current states
                cur_key_states = key_states[i]
                cur_query_states = query_states[i] 
                cur_eager_attention_mask = eager_attention_mask[i] 
                
                # choose last instruction token as query
                if self.training:
                    index_before_answer=[]

                    if index_before_answer==[]:
                        print("========index_before_answer is []")
                        cur_input_embeds = features[i]
                        features_list.append(cur_input_embeds)
                        attention_mask_list.append(attention_mask[i])

                        continue

                    index_before_answer=torch.tensor(index_before_answer,device=attention_mask[0].device) 
                    text_query_states = cur_query_states[:,index_before_answer,:]  
                    text_eager_attention_mask = cur_eager_attention_mask[:,index_before_answer,:]

                else:
                    prompt_total_len = self.prompt_len[i] + image_tokens[i]
                    text_query_states = cur_query_states[:,prompt_total_len-1,:].unsqueeze(1)  
                    text_eager_attention_mask = cur_eager_attention_mask[:,prompt_total_len-1,:].unsqueeze(1)

                    num_q_heads = cur_query_states.size(0)   # 28
                    num_kv_heads = cur_key_states.size(0)    # 4
                    if num_q_heads != num_kv_heads:
                        # 推荐：要求可整除（通常语义是多个 query heads 共享同一个 kv head）
                        if num_q_heads % num_kv_heads == 0:
                            repeat = num_q_heads // num_kv_heads     # 7
                            cur_key_states = cur_key_states.repeat_interleave(repeat, dim=0)  # -> (28, seq_len, head_dim)
                        else:
                            # 兜底：重复并截断到 num_q_heads（不一定语义正确）
                            repeats = math.ceil(num_q_heads / num_kv_heads)
                            cur_key_states = cur_key_states.repeat_interleave(repeats, dim=0)[:num_q_heads]

                    # 现在做注意力点积：
                    head_dim = cur_query_states.size(-1)

                # calculate attention map
                attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(head_dim) #(num_head, text_token,seq_len)
                attn_weights = attn_weights + text_eager_attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) #(num_head, text_token,seq_len)
                
                attention_avg_head = torch.mean(attn_weights, dim=0) # ave across heads
                attention_avg_head = attention_avg_head[:,image_index:image_index+image_tokens[i]] # select image token as keys
                attention_avg_text = torch.mean(attention_avg_head, dim=0) # (576)

                # rank and drop by attention score
                top_rank_index = attention_avg_text.topk(keep_length[i]).indices
                top_rank_index = top_rank_index + image_index  
                top_rank_index= top_rank_index.sort().values  

                start_index = image_index + image_tokens[i]
                new_input_embeds = torch.cat([features[i][ :image_index, :] ,features[i][ top_rank_index, :], features[i][start_index:, :]], dim=0)
    
                new_attention_mask = torch.cat([attention_mask[i][:image_index], attention_mask[i][top_rank_index], attention_mask[i][start_index:]], dim=0)
                self.recovered_token_indices = top_rank_index

                features_list.append(new_input_embeds)
                attention_mask_list.append(new_attention_mask)
                   

            # Truncate sequences to max length as image embeddings can make the sequence longer
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 4096)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]
                new_attention_mask = [x[:tokenizer_model_max_length] for x in attention_mask_list]
    

            max_len = max(x.shape[0] for x in new_input_embeds)

            # padding the sequences to form batch
            embeds_padded=[]
    
            attention_mask_padded=[]
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
            for i, cur_new_embed in enumerate(new_input_embeds):
                cur_len_emb=cur_new_embed.shape[0]
                dif=max_len - cur_len_emb  # padding to longest seq
                
                cur_new_embed = torch.cat([cur_new_embed,torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)],dim=0)
            
                cur_attention_mask = new_attention_mask[i]
                cur_attention_mask = torch.cat([cur_attention_mask,torch.full((dif,),False, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)],dim=0)
                
                embeds_padded.append(cur_new_embed)
       
                attention_mask_padded.append(cur_attention_mask)

                cur_len = new_attention_mask[i].sum().item()
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            

            new_input_embeds = torch.stack(embeds_padded,dim=0)
            new_input_embeds = new_input_embeds.to(features[0].dtype)
            new_attention_mask = torch.stack(attention_mask_padded,dim=0)
           
            if _position_ids is None:
                position_ids = None
            if position_ids.dim() == 2:
                # (batch, seq_len) -> (batch, seq_len, 3)
                position_ids = position_ids.unsqueeze(-1).repeat(1, 1, 3)


            if _attention_mask is None:
                new_attention_mask = None
            else:
                new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)

            return position_ids, new_attention_mask, new_input_embeds
        
        else:
            raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")

    # implementation of pmerge
    def pdrop_rank_merge(
        self, cur_num, rank_layer, features ,
        position_ids, attention_mask, token_coords
    ):
        
        _position_ids = position_ids
        _attention_mask = attention_mask

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
        
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "right":
            
            batch_size = features.shape[0]
            image_tokens = [int(cur_image_token * self.image_token_ratio_list[cur_num]) for cur_image_token in self.image_tokens]
            keep_length = [int(cur_image_token * (self.image_token_ratio_list[cur_num + 1])) for cur_image_token in self.image_tokens]
            context_length = [int(cur_image_token * (self.image_token_ratio_list[cur_num + 1] - 0.3)) for cur_image_token in self.image_tokens]

            features_list = []
            attention_mask_list = []
        

            if attention_mask is None:
                attention_mask = torch.ones((batch_size,features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()
         

            # obtain query_states and key_states to calculate attention map
            hidden_states=features.clone().detach()
            self_attn = self.layers[rank_layer].self_attn
            hidden_states = self.layers[rank_layer].input_layernorm(hidden_states)

            num_heads = self_attn.num_heads
            num_key_value_heads = self_attn.num_key_value_heads
            head_dim = self_attn.head_dim

            bsz, q_len, _ = hidden_states.size()

            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
            value_states = self_attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            cos, sin = self_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # attention_mask 
            # eager_attention_mask = _prepare_4d_causal_attention_mask(
            #     attention_mask, (batch_size, q_len), hidden_states, past_key_values_length=0
            # ).to(device=query_states.device)

            eager_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, q_len),
                hidden_states,
                past_key_values_length=0,
                sliding_window=self.config.sliding_window,
            )

            # take valid features
            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]
            attention_mask = [cur_attention_mask[cur_attention_mask] for cur_attention_mask, cur_attention_mask in zip(attention_mask, attention_mask)]

            # rank & drop
            for i in range(batch_size):
                image_index= self.image_token_posi[i]
                if image_index == -1:
                    cur_input_embeds = features[i]
                    features_list.append(cur_input_embeds)
                    attention_mask_list.append(attention_mask[i])
                    continue
                
                # obtain current states
                cur_key_states = key_states[i]
                cur_query_states = query_states[i] 
                cur_eager_attention_mask = eager_attention_mask[i] 
                
                # choose last instruction token as query
                if self.training:
                    pass

                else:
                    prompt_total_len = self.prompt_len[i] + image_tokens[i]
                    text_query_states = cur_query_states[:,prompt_total_len-1,:].unsqueeze(1)  
                    text_eager_attention_mask = cur_eager_attention_mask[:,prompt_total_len-1,:].unsqueeze(1)
                
                    num_q_heads = cur_query_states.size(0)   # 28
                    num_kv_heads = cur_key_states.size(0)    # 4
                    if num_q_heads != num_kv_heads:
                        # 推荐：要求可整除（通常语义是多个 query heads 共享同一个 kv head）
                        if num_q_heads % num_kv_heads == 0:
                            repeat = num_q_heads // num_kv_heads     # 7
                            cur_key_states = cur_key_states.repeat_interleave(repeat, dim=0)  # -> (28, seq_len, head_dim)
                        else:
                            # 兜底：重复并截断到 num_q_heads（不一定语义正确）
                            repeats = math.ceil(num_q_heads / num_kv_heads)
                            cur_key_states = cur_key_states.repeat_interleave(repeats, dim=0)[:num_q_heads]

                    # 现在做注意力点积：
                    head_dim = cur_query_states.size(-1)

                # calculate attention map
                attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(head_dim) #(num_head, text_token,seq_len)
                attn_weights = attn_weights + text_eager_attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) #(num_head, text_token,seq_len)
                
                attention_avg_head = torch.mean(attn_weights, dim=0) # ave across heads
                attention_avg_head = attention_avg_head[:,image_index:image_index+image_tokens[i]] # select image token as keys
                attention_avg_text = torch.mean(attention_avg_head, dim=0) # (576)

                # rank and merge by attention score
                top_rank_index = attention_avg_text.topk(context_length[i]).indices
                rel_rank_index = top_rank_index
                top_rank_index = top_rank_index + image_index  
                top_rank_index = top_rank_index.sort().values  
               
                ### pmerge v2 start  ###
                # visual_tokens = features[i][image_index: image_index+image_tokens[i], :].clone() 
                visual_tokens = features[i][image_index: image_index+image_tokens[i], :].clone() 

                #selector = MarkovTokenSelector(alpha=0.12, beta=3.0, gamma=3.0, \
                #                                 grid_size=0.2, device=features[i].device)
                '''
                grid_tokens, grid_attention, recovered_features, recovered_token_indices = \
                    self.slice_and_grid_tokens(token_coords, visual_tokens, attention_avg_text, 
                    num_slices_z=8, grid_size=0.12, slice_select=True, max_tokens_per_grid=3, mode="max_attention")    


                import random
                with open("./cache.txt", "r") as f:
                    idx = f.readline().strip()
                save_dir= f"/data/ljn/code/Video-3D-LLM/results/scanqa/cached_tokens/{idx}.pt"
                if random.random() < 0.1: 
                    cache_tokens(save_dir, token_coords, visual_tokens, attention_avg_text)

                '''
                mrf = DifferentiableMRFSingleSparseEdge(num_classes=2,
                                        checkpoint_path="/data/ljn/code/Video-3D-LLM/ckpt/neural_mrf_token_selector.pth",
                                        device="cuda")
                                        
                recovered_features, recovered_token_indices, _, _ = mrf.inference(
                    token_coords = token_coords,
                    features = visual_tokens,
                    attention = attention_avg_text,   # 保证是 (N,1)
                    num_slices_z = 8,
                    grid_size = 0.12,
                    max_tokens_per_grid = 10,
                    mode = "max_attention",
                    fg_threshold = 0.5,
                    merge_method = "select_max_att"
                )
                
                ### pmerge v2 end  ### 
                recovered_token_indices = recovered_token_indices + image_index
                
                mask = torch.isin(top_rank_index, recovered_token_indices)  
                missing_from_recovered = top_rank_index[~mask] 

                all_indices = torch.cat([recovered_token_indices, missing_from_recovered])  # shape: [N + M]

                # 步骤2：获取排序顺序
                sort_order = torch.argsort(all_indices)  # 返回排序后的位置索引
                sorted_indices = all_indices[sort_order]  # 可选，用于调试或验证

                # 步骤3：确定哪些来自 recovered，哪些来自 missing
                n_recovered = len(recovered_token_indices)
                source_mask = sort_order < n_recovered  # BoolTensor, True 表示来自 recovered_features

                # 步骤4：初始化最终结果张量
                final_features = torch.empty(
                    (len(all_indices), features[i].size(1)),
                    dtype=features[i].dtype,
                    device=features[i].device
                )

                # 步骤5：填充 recovered 部分
                recovered_rows_to_take = sort_order[source_mask]  # 这些是 recovered_features 中的行号（0~N-1）
                final_features[source_mask] = recovered_features[recovered_rows_to_take]

                # 步骤6：填充 missing 部分
                missing_rows_to_take = sort_order[~source_mask] - n_recovered  # 映射到 missing_from_recovered 的行号（0~M-1）
                final_features[~source_mask] = features[i][missing_from_recovered][missing_rows_to_take]
                recovered_features = final_features
                
                recovered_token_indices = torch.unique(all_indices, sorted=True)   
                
                ratio = recovered_token_indices.shape[0] / visual_tokens.shape[0]
                self.image_token_ratio_list[cur_num + 1] = ratio
                with open('./results/multi3drefer/token_ratios_mrf_v3.txt', 'a') as f:  
                    f.write(f"{ratio}\n")  
                start_index = image_index + image_tokens[i]

                new_input_embeds = torch.cat([features[i][ :image_index, :] , recovered_features, features[i][start_index:, :]], dim=0)
                # new_input_embeds = torch.cat([features[i][ :image_index, :] ,features[i][recovered_token_indices, :], features[i][start_index:, :]], dim=0)
                new_attention_mask = torch.cat([attention_mask[i][:image_index], attention_mask[i][recovered_token_indices], attention_mask[i][start_index:]], dim=0)
                self.recovered_token_indices = recovered_token_indices
               
                features_list.append(new_input_embeds)
                attention_mask_list.append(new_attention_mask)
            
            # Truncate sequences to max length as image embeddings can make the sequence longer
            # print("Model max_position_embeddings:", self.config.max_position_embeddings)
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 4096)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]
                new_attention_mask = [x[:tokenizer_model_max_length] for x in attention_mask_list]
            
            max_len = max(x.shape[0] for x in new_input_embeds)
            # padding the sequences to form batch
            embeds_padded=[]
        
            attention_mask_padded=[]
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
            for i, cur_new_embed in enumerate(new_input_embeds):
                cur_len_emb=cur_new_embed.shape[0]
                dif=max_len - cur_len_emb  # padding to longest seq
                
                cur_new_embed = torch.cat([cur_new_embed,torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)],dim=0)
                
                cur_attention_mask = new_attention_mask[i]
                cur_attention_mask = torch.cat([cur_attention_mask,torch.full((dif,),False, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)],dim=0)
                
                embeds_padded.append(cur_new_embed)
                attention_mask_padded.append(cur_attention_mask)
                cur_len = new_attention_mask[i].sum().item()
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                
            new_input_embeds = torch.stack(embeds_padded,dim=0)
            new_input_embeds = new_input_embeds.to(features[0].dtype)
            new_attention_mask = torch.stack(attention_mask_padded,dim=0)
        
            
            if _position_ids is None:
                position_ids = None
            if position_ids.dim() == 2:
                # (batch, seq_len) -> (batch, seq_len, 3)
                position_ids = position_ids.unsqueeze(-1).repeat(1, 1, 3)

            if _attention_mask is None:
                new_attention_mask = None
            else:
                new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)

            return position_ids, new_attention_mask, new_input_embeds, recovered_token_indices
        
        else:
            raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")

    def slice_and_grid_tokens(self, token_coords, features, attention,
                            num_slices_z=4, grid_size=0.2, slice_select=False, max_tokens_per_grid=10,
                            mode='max_attention'):
        """
        mode: str, 'max_attention' 或 'average' 或 'top_k_attention'
            - 'max_attention': 选择网格内 attention 最大的 token 作为代表
            - 'average': 对网格内所有 token 特征求平均
            - 'top_k_attention': 根据attention，保留k=max_tokens_per_grid个tokens
        """
        device = features.device
        dim = features.shape[1]
        N = features.shape[0]

        x_coords = token_coords[:, 0]
        y_coords = token_coords[:, 1]
        z_coords = token_coords[:, 2]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min_coord, z_max_coord = z_coords.min(), z_coords.max()


        slice_bounds = torch.linspace(z_min_coord, z_max_coord, steps=num_slices_z+1, device=device)

        num_grids_x = math.ceil((x_max - x_min).item() / grid_size)
        num_grids_y = math.ceil((y_max - y_min).item() / grid_size)

        if num_grids_x == 0:
            num_grids_x = 1
        if num_grids_y == 0:
            num_grids_y = 1

        z_indices = torch.bucketize(z_coords, slice_bounds) - 1
        z_indices = torch.clamp(z_indices, 0, num_slices_z - 1)

        x_indices = ((x_coords - x_min) / grid_size).floor().long()
        y_indices = ((y_coords - y_min) / grid_size).floor().long()

        x_indices = torch.clamp(x_indices, 0, num_grids_x - 1)
        y_indices = torch.clamp(y_indices, 0, num_grids_y - 1)

        
        slice_counts = torch.bincount(z_indices, minlength=num_slices_z)
        slice_with_most_tokens = int(torch.argmax(slice_counts).item())

        grid_token_indices = defaultdict(list)
        token_indice_grids = {}
        for i in range(N):
            key = (z_indices[i].item(), x_indices[i].item(), y_indices[i].item())
            grid_token_indices[key].append(i)
            token_indice_grids[i] = [z_indices[i].item(), x_indices[i].item(), y_indices[i].item()]

        grid_features = torch.zeros((num_slices_z, num_grids_x, num_grids_y, dim), dtype=features.dtype, device=device)
        grid_attention = torch.zeros((num_slices_z, num_grids_x, num_grids_y), dtype=attention.dtype, device=device)

        grid_selected_token_order = defaultdict(list)

        for (z_idx, x_idx, y_idx), idx_list in grid_token_indices.items():
            idx_tensor = torch.tensor(idx_list, device=device)
            selected_idx = None

            if mode == 'max_attention':
                # 保留最多 max_tokens_per_grid 个注意力最大的token
                if len(idx_tensor) > max_tokens_per_grid:
                    selected_attention = attention[idx_tensor]
                    topk_indices = torch.topk(selected_attention, max_tokens_per_grid).indices
                    idx_tensor = idx_tensor[topk_indices]
                
                # 从这些token中选择注意力最大的
                if len(idx_tensor) == 1:
                    selected_idx = idx_tensor[0].unsqueeze(0)
                    avg_feat = features[selected_idx].squeeze(0)
                    avg_attention = attention[selected_idx].squeeze(0)
                else:
                    selected_attention = attention[idx_tensor]
                    max_idx = torch.argmax(selected_attention)
                    selected_idx = idx_tensor[max_idx].unsqueeze(0)  
                    avg_feat = features[selected_idx].squeeze(0)
                    avg_attention = selected_attention[max_idx]
                grid_selected_token_order[(z_idx, x_idx, y_idx)].append(selected_idx.item())

            elif mode == 'average':
                # 只对最多 max_tokens_per_grid 个token求平均
                if len(idx_tensor) > max_tokens_per_grid:
                    selected_attention = attention[idx_tensor]
                    topk_indices = torch.topk(selected_attention, max_tokens_per_grid).indices
                    idx_tensor = idx_tensor[topk_indices]
                
                avg_feat = features[idx_tensor].mean(dim=0)
                # attention仍然取最高的那个token的值
                selected_attention = attention[idx_tensor]
                max_idx = torch.argmax(selected_attention)
                selected_idx = idx_tensor[max_idx].unsqueeze(0)  
                avg_attention = selected_attention[max_idx]
                grid_selected_token_order[(z_idx, x_idx, y_idx)].append(selected_idx.item())

            elif mode == 'top_k_attention':
                # 根据attention，保留k=max_tokens_per_grid个tokens
                k = min(max_tokens_per_grid, len(idx_tensor))
                selected_attention = attention[idx_tensor]
                topk_indices = torch.topk(selected_attention, k).indices
                selected_idx_tensor = idx_tensor[topk_indices]  # 这里保留多个token的索引
                
                # 为每个选中的token都记录下来
                for i in range(len(selected_idx_tensor)):
                    token_idx = selected_idx_tensor[i].unsqueeze(0)
                    grid_selected_token_order[(z_idx, x_idx, y_idx)].append(token_idx.item())
                    
                    # 每个token都使用其对应的特征
                    feat = features[token_idx].squeeze(0)
                    att = attention[token_idx].squeeze(0)
                    
                    # 注意：这里需要特殊处理，因为一个网格要对应多个token
                    # 可以选择只保留第一个token的特征，或者有其他策略
                    if i == 0:  # 只保留第一个作为网格的代表特征
                        grid_features[z_idx, x_idx, y_idx] = feat
                        grid_attention[z_idx, x_idx, y_idx] = att

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # 对于非top_k_attention模式，正常设置网格特征
            if mode != 'top_k_attention':
                grid_features[z_idx, x_idx, y_idx] = avg_feat
                grid_attention[z_idx, x_idx, y_idx] = avg_attention

        recovered_tokens = []
        recovered_feats = []

        for key, token_idxs in grid_selected_token_order.items():
            z_idx, x_idx, y_idx = key
            for token_idx in token_idxs:
                # 对于top_k_attention模式，需要获取对应token的原始特征
                if mode == 'top_k_attention':
                    feat = features[token_idx]
                else:
                    feat = grid_features[z_idx, x_idx, y_idx]
                recovered_tokens.append(token_idx)
                recovered_feats.append(feat)

        recovered_tokens, recovered_feats = zip(*sorted(zip(recovered_tokens, recovered_feats), key=lambda x: x[0]))
        recovered_features = torch.stack(recovered_feats, dim=0)
        recovered_token_indices = torch.tensor(recovered_tokens, device=device)

        if slice_select:
            # slice_with_most_tokens 已提前计算
            
            z_sel = slice_with_most_tokens

            selected_grid_indices = []
            selected_token_indices = []
            selected_features = []

            for (z_idx, x_idx, y_idx), token_list in grid_selected_token_order.items():
                if z_idx != z_sel:
                    continue
                # selected_grid_indices.append((x_idx, y_idx))
                assert len(token_list) == 1
                selected_grid_indices.append([token_coords[int(token_list[0])][0].item(), token_coords[int(token_list[0])][1].item()]) 
                selected_token_indices.append(int(token_list[0]))
                selected_features.append(grid_features[z_sel, x_idx, y_idx])
            
            selected_grid_indices = np.array(selected_grid_indices) 
            dbscan = DBSCANVectorized(eps=0.15, minPts=1)

            selected_labels = dbscan.fit(selected_grid_indices)
            
            cluster_to_tokens = defaultdict(list)
            for token_idx, label in zip(selected_token_indices, selected_labels):
                cluster_to_tokens[int(label)].append(int(token_idx))
                
        return grid_features, grid_attention, recovered_features, recovered_token_indices


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_coords: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_coords=token_coords,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        recovered_token_indices = outputs[-1]
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None and not isinstance(past_key_values, tuple):
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if (attention_mask is not None and position_ids is None) or past_key_values:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The Qwen2 Model transformer with a sequence classification head on top (linear layer).

    [`Qwen2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    QWEN2_START_DOCSTRING,
)
class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
