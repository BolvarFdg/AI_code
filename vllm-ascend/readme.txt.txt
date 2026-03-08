修改的文件
1. 新建文件
vllm_ascend/attention/tree_attn_v1.py

NPU Tree Attention 后端，包含：

_get_depth_counts() / _prepare_tree_attn_bias() — 从 speculative_token_tree 构建 [tree_len, tree_len] 的 additive mask
AscendTreeAttentionMetadata — 继承 AscendMetadata，新增 tree_attn_bias 和 tree_len 字段
gather_kv_from_paged_cache() — 向量化地从分页 KV 缓存（[num_blocks, block_size, nKH, H]）收集为稠密 [B, max_seq_len, nKH, H] 张量
forward_tree_decode_attention() — 每个请求独立调用 PyTorch SDPA，mask = 上下文部分全 0 + 树结构部分 tree_attn_bias
AscendTreeAttentionMetadataBuilder — 继承 AscendAttentionMetadataBuilder，新增 build_for_drafting(draft_index) 接口
_promote_to_tree_metadata() — 通过 copy.copy() + __class__ 重赋值将 AscendMetadata 升级为 AscendTreeAttentionMetadata

2. 修改文件
vllm_ascend/attention/attention_v1.py

在 forward_impl() 最前面增加 tree attention 分发：当 attn_metadata 是 AscendTreeAttentionMetadata 且 tree_attn_bias is not None 且 tree_len > 1 时，调用 forward_tree_decode_attention()，否则走原有路径不变。使用延迟导入避免循环引用。

vllm_ascend/spec_decode/eagle_proposer.py

主要改动：

import ast 头部导入
__init__() 新增树注意力字段：解析 speculative_token_tree，若为多分支树则设置 _use_tree_attention = True，计算 cu_drafts_per_level、child_drafts_per_level、tree_draft_pos_offsets
_get_attention_metadata_builder() — 覆盖父类方法，树模式下懒初始化并返回 AscendTreeAttentionMetadataBuilder
_propose() — 树模式下跳过线性 draft 步骤的循环，仅快照 _tree_common_attn_metadata
_run_merged_draft() — 深度 0 前向后，树模式调用 propose_tree() 并提前返回；线性路径保持不变
propose_tree() — 新方法，逐深度展开树：每层聚合所有候选节点、构建 AscendTreeAttentionMetadata、计算 slot mapping、调用草稿模型、采样下一层 token，最终返回 list[Tensor]