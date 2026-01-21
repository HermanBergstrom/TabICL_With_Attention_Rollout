from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .rope import RotaryEmbedding
import math

# Efficient implementation equivalent to the following:
def custom_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

def sdpa_with_flattened_batch(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0
) -> tuple[Tensor, Tensor]:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's scaled_dot_product_attention and then reshaping the output
    back to the original shape. This flattening is necessary to properly trigger
    Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:
        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len)

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights

    Returns
    -------
    tuple[Tensor, Tensor]
        Attention output tensor of shape (..., nh, tgt_len, hs) and attention weights
        of shape (..., nh, tgt_len, src_len) preserving the original batch dimensions
    """

    q_shape = q.shape
    attn_weight_shape = (*q.shape[:-1], k.shape[-2])
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])
    
    #out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    out, attn_weight = custom_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    
    return out.view(q_shape), attn_weight.view(attn_weight_shape)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor | int] = None,
    rope: Optional[RotaryEmbedding] = None,
    need_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Multi-head attention with support for rotary position embeddings
    as well as specialized processing when attn_mask is an integer.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim)

    key : Tensor
        Key tensor of shape (..., src_len, embed_dim)

    value : Tensor
        Value tensor of shape (..., src_len, embed_dim)

    num_heads : int
        Number of attention heads

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections

    in_proj_bias : Tensor
        Combined bias vector for input projections

    dropout_p : float
        Dropout probability applied to attention weights

    out_proj_weight : Tensor
        Output projection weight matrix

    out_proj_bias : Tensor
        Output projection bias vector

    training : bool, default=True
        Whether the model is in training mode (affects dropout)

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:
            - For binary masks: True values indicate positions to ignore
            - For float masks: Values are directly added to attention scores

    attn_mask : Optional[Tensor | int], default=None
        Controls attention pattern in two possible ways:
        1. When provided as Tensor: Traditional mask preventing attention to certain positions
            - Shape: (tgt_len, src_len) or (..., num_heads, tgt_len, src_len)
        2. When provided as integer: Creates a split attention pattern where:
            - The first `attn_mask` tokens perform self-attention only (attend to themselves)
            - The remaining tokens attend only to the first `attn_mask` tokens

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding

    need_weights : bool, default=False
        Whether to return attention weights alongside the output

    Returns
    -------
    Tensor | tuple[Tensor, Tensor]
        If need_weights=False: Attention output tensor of shape (..., tgt_len, embed_dim)
        If need_weights=True: Tuple of (output tensor, attention weights tensor of shape (..., num_heads, tgt_len, src_len))
    """

    if isinstance(attn_mask, int):
        assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
        assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key.shape[-2]

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # Joint projection of query, key, value
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # Reshape for multi-head attention
    q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, tgt_len, hs)
    k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)
    v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)

    # Apply rotary position embeddings if provided
    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    if isinstance(attn_mask, int):
        cut_pos = attn_mask  # For better readability

        # Pre-allocate output tensor to avoid concatenation
        attn_output = torch.empty(*batch_shape, tgt_len, embed_dim, device=query.device, dtype=query.dtype)
        
        # Pre-allocate weights tensor if needed
        if need_weights:
            attn_weights = torch.zeros(*batch_shape, num_heads, tgt_len, src_len, device=query.device, dtype=query.dtype)
        else:
            attn_weights = None

        # Process left segment (self-attention within first cut_pos tokens)
        q_left = q[..., :cut_pos, :]  # (batch_shape, nh, cut_pos, hs)
        k_left = k[..., :cut_pos, :]
        v_left = v[..., :cut_pos, :]

        attn_left, weights_left = sdpa_with_flattened_batch(q_left, k_left, v_left, dropout_p=dropout_p)
        attn_left = attn_left.transpose(-3, -2).contiguous().view(*batch_shape, cut_pos, embed_dim)
        attn_output[..., :cut_pos, :] = F.linear(attn_left, out_proj_weight, out_proj_bias)

        if need_weights:
            attn_weights[..., :cut_pos, :cut_pos] = weights_left

        # Process right segment (tokens after cut_pos attending to tokens before cut_pos)
        if cut_pos < tgt_len:
            q_right = q[..., cut_pos:, :]  # (batch_shape, nh, tgt_len - cut_pos, hs)
            attn_right, weights_right = sdpa_with_flattened_batch(q_right, k_left, v_left, dropout_p=dropout_p)
            attn_right = attn_right.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len - cut_pos, embed_dim)
            attn_output[..., cut_pos:, :] = F.linear(attn_right, out_proj_weight, out_proj_bias)
            if need_weights:
                attn_weights[..., cut_pos:, :cut_pos] = weights_right
    else:
        # Process attention mask
        correct_2d_shape = (tgt_len, src_len)
        correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != correct_2d_shape:
                    raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
                attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
            elif attn_mask.dim() == len(correct_nd_shape):
                if attn_mask.shape != correct_nd_shape:
                    raise ValueError(
                        f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                        f"but got {attn_mask.shape}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

        # Process key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (*batch_shape, src_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
                *batch_shape, num_heads, tgt_len, src_len
            )

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        attn_output, attn_weights = sdpa_with_flattened_batch(q, k, v, attn_mask, dropout_p)  # (..., nh, tgt_len, hs)

        # Reshape and project output
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)
        
        # Only keep weights if requested
        if not need_weights:
            attn_weights = None    
    if need_weights:
        return attn_output, attn_weights
    else:
        return attn_output