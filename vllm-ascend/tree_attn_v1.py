#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
"""
NPU Tree Attention backend for EAGLE speculative decoding.

Supports the draft model attention phase where multiple tree-structured
candidate tokens per request are processed in parallel. The tree topology
is encoded in tree_attn_bias: a [tree_len, tree_len] additive bias where
-inf masks out ancestors that should not be attended to.

Key differences from GPU (Triton unified_attention with qq_bias):
  - KV is gathered from paged cache into dense tensors
  - PyTorch SDPA is used with a per-request custom mask
  - Mask = zeros for context tokens + tree_attn_bias for intra-tree tokens
"""

import ast
import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


# ---------------------------------------------------------------------------
# Tree bias helpers (ported from vllm/v1/attention/backends/tree_attn.py)
# ---------------------------------------------------------------------------

def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    """Count the number of nodes at each depth of the tree."""
    depth_counts: list[int] = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a [tree_len, tree_len] additive attention bias that encodes the
    tree topology.  Entry [i, j] is 0 if token i may attend to token j,
    -inf otherwise.  tree_len = len(sorted_tree_choices) + 1 (includes root).
    """
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), float("-inf"), dtype=dtype, device=device
    )
    # Each token attends to itself.
    for i in range(tree_len):
        tree_attn_mask[i, i] = 0.0
    # All tokens attend to the root (column 0).
    tree_attn_mask[:, 0] = 0.0
    # Each token attends to all its ancestors.
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur = sorted_tree_choices[start + j]
            if len(cur) == 1:
                continue
            ancestor_idx = [
                sorted_tree_choices.index(cur[: c + 1]) + 1
                for c in range(len(cur) - 1)
            ]
            tree_attn_mask[j + start + 1, ancestor_idx] = 0.0
        start += depth_counts[i]
    return tree_attn_mask


# ---------------------------------------------------------------------------
# AscendTreeAttentionMetadata
# ---------------------------------------------------------------------------

@dataclass
class AscendTreeAttentionMetadata(AscendMetadata):
    """
    Extends AscendMetadata with tree-attention bias for EAGLE draft model.

    tree_attn_bias: [tree_len, tree_len] additive bias (0 = allow, -inf = mask).
                    None for the root-level pass (draft_index == 0).
    tree_len:       Number of query tokens per request in the tree-decode phase.
    """
    tree_attn_bias: Optional[torch.Tensor] = None
    tree_len: int = 1


# ---------------------------------------------------------------------------
# Helper: gather KV from paged cache
# ---------------------------------------------------------------------------

def gather_kv_from_paged_cache(
    key_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,   # [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,  # [B, max_blocks_per_seq]  int32
    seq_lens: torch.Tensor,      # [B]  int64
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather KV tensors from the paged KV cache into contiguous
    [B, max_seq_len, num_kv_heads, head_size] dense tensors.

    Uses fully-vectorized indexing (no Python loops over tokens).
    """
    B = seq_lens.shape[0]
    max_seq_len = int(seq_lens.max().item())
    num_blocks, _, num_kv_heads, head_size = key_cache.shape

    # Position index for each slot: [max_seq_len]
    positions = torch.arange(
        max_seq_len, device=key_cache.device, dtype=torch.int64
    )

    # Which entry in block_tables to look up: [1, max_seq_len]
    block_col = (positions // block_size).unsqueeze(0)
    # Offset within a block: [1, max_seq_len]
    block_offset = (positions % block_size).unsqueeze(0)

    # Clamp to valid block range to avoid out-of-bounds index.
    max_cols = block_tables.shape[1]
    block_col_clamped = block_col.expand(B, -1).clamp(max=max_cols - 1)  # [B, max_seq_len]

    # Block IDs: [B, max_seq_len]
    block_ids = block_tables.long().gather(1, block_col_clamped)

    # Flat slot indices: [B, max_seq_len]
    slots = block_ids * block_size + block_offset  # [B, max_seq_len]
    slots_flat = slots.view(-1)  # [B * max_seq_len]

    # Flatten cache: [num_blocks * block_size, num_kv_heads, head_size]
    key_flat = key_cache.view(num_blocks * block_size, num_kv_heads, head_size)
    val_flat = value_cache.view(num_blocks * block_size, num_kv_heads, head_size)

    gathered_k = key_flat[slots_flat].view(B, max_seq_len, num_kv_heads, head_size)
    gathered_v = val_flat[slots_flat].view(B, max_seq_len, num_kv_heads, head_size)
    return gathered_k, gathered_v


# ---------------------------------------------------------------------------
# Tree-decode attention computation
# ---------------------------------------------------------------------------

def forward_tree_decode_attention(
    query: torch.Tensor,          # [B*T, num_heads, head_size]
    key_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,   # [B, max_blocks]  int32
    seq_lens: torch.Tensor,       # [B]  int32
    tree_attn_bias: torch.Tensor, # [T, T]  additive bias
    tree_len: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    output: torch.Tensor,         # [max_tokens, num_heads, head_size]  (pre-allocated)
) -> torch.Tensor:
    """
    Compute tree-attention for the draft model decode phase.

    For each request b:
      - query[b*T : (b+1)*T] attends to gathered_k[b, :seq_len[b]]
      - For context positions [0 : context_len]: no masking (0 bias)
      - For tree positions [context_len : seq_len]: tree_attn_bias applied

    Uses torch.nn.functional.scaled_dot_product_attention for correctness
    and NPU portability.
    """
    B = seq_lens.shape[0]
    block_size = key_cache.shape[1]
    head_size = key_cache.shape[3]

    # Gather full KV context from paged cache: [B, max_seq_len, nKH, H]
    gathered_k, gathered_v = gather_kv_from_paged_cache(
        key_cache, value_cache, block_tables[:B], seq_lens[:B].long(), block_size
    )

    for b in range(B):
        s = int(seq_lens[b].item())
        if s == 0:
            output[b * tree_len: (b + 1) * tree_len] = 0
            continue

        context_len = max(0, s - tree_len)  # tokens before the current tree nodes
        q_b = query[b * tree_len: (b + 1) * tree_len]  # [T, nH, H]
        k_b = gathered_k[b, :s]                         # [s, nKH, H]
        v_b = gathered_v[b, :s]                         # [s, nKH, H]

        # GQA: expand KV heads if needed
        if num_kv_heads < num_heads:
            repeats = num_heads // num_kv_heads
            k_b = k_b.repeat_interleave(repeats, dim=1)
            v_b = v_b.repeat_interleave(repeats, dim=1)

        # Build attention mask [T, s]:
        #   - Context part  [0 : context_len] : 0.0  (allow)
        #   - Tree part     [context_len : s] : tree_attn_bias[:, :tree_kv_len]
        mask_b = torch.zeros(tree_len, s, dtype=torch.float32, device=query.device)
        tree_kv_len = s - context_len
        if tree_kv_len > 0 and tree_attn_bias is not None:
            # tree_attn_bias is [T, T]; use only the first tree_kv_len columns
            mask_b[:, context_len:] = tree_attn_bias[:, :tree_kv_len].to(
                dtype=torch.float32
            )

        # SDPA expects [batch, heads, seq, dim]
        q_b4 = q_b.transpose(0, 1).unsqueeze(0).to(dtype=torch.float32)   # [1, nH, T, H]
        k_b4 = k_b.transpose(0, 1).unsqueeze(0).to(dtype=torch.float32)   # [1, nH, s, H]
        v_b4 = v_b.transpose(0, 1).unsqueeze(0).to(dtype=torch.float32)   # [1, nH, s, H]
        mask_b4 = mask_b.unsqueeze(0).unsqueeze(0)                         # [1, 1, T, s]

        attn_out = F.scaled_dot_product_attention(
            q_b4, k_b4, v_b4,
            attn_mask=mask_b4,
            scale=scale,
            dropout_p=0.0,
        )  # [1, nH, T, H]

        # [1, nH, T, H] -> [T, nH, H]
        attn_out = attn_out.squeeze(0).transpose(0, 1).to(dtype=query.dtype)
        output[b * tree_len: (b + 1) * tree_len] = attn_out

    return output


# ---------------------------------------------------------------------------
# AscendTreeAttentionMetadataBuilder
# ---------------------------------------------------------------------------

class AscendTreeAttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    """
    Extends AscendAttentionMetadataBuilder to support tree-attention metadata
    for EAGLE speculative decoding.

    Adds build_for_drafting() which slices tree_attn_bias appropriately for
    each depth level in the speculation tree.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        spec_config = vllm_config.speculative_config
        spec_token_tree: Optional[str] = None
        if spec_config is not None:
            spec_token_tree = spec_config.speculative_token_tree

        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree)
            if spec_token_tree is not None
            else [(0,)]
        )
        depth_counts = _get_depth_counts(tree_choices)
        self._full_tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices, depth_counts, dtype=torch.float32, device=device
        )
        # Active bias used during build(); swapped in build_for_drafting().
        self._active_tree_attn_bias: Optional[torch.Tensor] = self._full_tree_attn_bias

    # ------------------------------------------------------------------
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendTreeAttentionMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Promote AscendMetadata -> AscendTreeAttentionMetadata
        result = _promote_to_tree_metadata(
            base,
            tree_attn_bias=self._active_tree_attn_bias,
            tree_len=1,
        )
        return result

    # ------------------------------------------------------------------
    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
    ) -> AscendTreeAttentionMetadata:
        """
        Build attention metadata for EAGLE drafting at the given depth.

        draft_index == 0:  root level, standard prefill (no tree bias).
        draft_index >= 1:  tree-decode, slice the bias for current query_len.
        """
        orig_bias = self._active_tree_attn_bias

        if draft_index == 0:
            self._active_tree_attn_bias = None
            tree_len = 1
        else:
            start = 1
            end = 1 + common_attn_metadata.max_query_len
            self._active_tree_attn_bias = self._full_tree_attn_bias[
                start:end, start:end
            ].contiguous()
            tree_len = common_attn_metadata.max_query_len

        meta = self.build(0, common_attn_metadata, fast_build=True)

        # Restore original bias so subsequent calls are not affected.
        self._active_tree_attn_bias = orig_bias
        meta.tree_attn_bias = (
            None if draft_index == 0 else
            self._full_tree_attn_bias[1: 1 + tree_len, 1: 1 + tree_len].contiguous()
        )
        meta.tree_len = tree_len
        return meta

    # ------------------------------------------------------------------
    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ) -> AscendTreeAttentionMetadata:
        base = super().build_for_graph_capture(common_attn_metadata, attn_state)
        return _promote_to_tree_metadata(base, tree_attn_bias=None, tree_len=1)


# ---------------------------------------------------------------------------
# Utility: promote AscendMetadata -> AscendTreeAttentionMetadata
# ---------------------------------------------------------------------------

def _promote_to_tree_metadata(
    base: AscendMetadata,
    tree_attn_bias: Optional[torch.Tensor],
    tree_len: int,
) -> AscendTreeAttentionMetadata:
    """
    Shallow-copy base AscendMetadata and change its class to
    AscendTreeAttentionMetadata, adding the two extra fields.
    """
    result = copy.copy(base)
    result.__class__ = AscendTreeAttentionMetadata
    result.tree_attn_bias = tree_attn_bias
    result.tree_len = tree_len
    return result  # type: ignore[return-value]
