# SPDX-License-Identifier: Apache-2.0
"""Shared Copy-SFA metadata builders used by the fallback storage coverage.

This module provides the make_builder/common/populate helpers imported by
test_copy_sfa_fallback.py.  It was part of the dsa_offload_rebase_main_kernel_v2_1124
branch and was not carried into the v2_1125_fix branch; it is restored here so
the fallback storage tests can run.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadataBuilder

MODULE = "vllm_ascend.attention.sfa_kv_offload"
CACHE_BLOCK = 128


def make_builder(num_mtp_layers=1):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_fused_copy_sfa = True
    builder.store_on_host = True
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    num_nextn_predict_layers=num_mtp_layers,
                    index_share_for_mtp_iteration=True,
                )
            ),
        ),
        model_config=SimpleNamespace(
            max_model_len=16384,
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        ),
    )
    with patch(
        MODULE + ".get_ascend_config",
        return_value=SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=8192)
        ),
    ):
        builder._init_copy_sfa_metadata_buffers(config, torch.device("cpu"))
    return builder


def common(ends, lengths, pools=(1, 0), generations=(11, 12)):
    count = len(lengths)
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, *ends], dtype=torch.int32, device="cpu"),
        query_start_loc_cpu=torch.tensor([0, *ends], dtype=torch.int32),
        # Deliberately stale mirrors must never affect fused_copy_sfa geometry.
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cpu"),
        seq_lens_cpu=torch.full((count,), 777777, dtype=torch.int32),
        _seq_lens_cpu=torch.full((count,), 777777, dtype=torch.int32),
        copy_sfa_draft_index=None,
        # Most geometry tests exercise the optional restore descriptors too.
        copy_sfa_restore_tails=True,
        req_topk_buffer_slots=torch.tensor(pools, dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor(generations, dtype=torch.int64),
        req_topk_buffer_active=torch.ones(count, dtype=torch.bool),
        # Runner-owned block-aligned stable prefixes reach the builder via D2H.
        req_topk_buffer_stable_prefixes=torch.tensor(
            [(int(length) // CACHE_BLOCK) * CACHE_BLOCK for length in lengths], dtype=torch.int32
        ),
        block_table_tensor=torch.arange(count * 128, dtype=torch.int32, device="cpu").reshape(count, 128),
        slot_mapping=torch.arange(16, dtype=torch.int64, device="cpu") + 128,
        req_ids_tensor=None,
        token_to_req=None,
        offload_dummy=False,
        max_query_len=4,
        num_reqs=count,
        num_input_tokens=ends[-1],
    )


def populate(builder, cm, draft_index=None, *, reuse_topk=False):
    cm.copy_sfa_draft_index = draft_index
    builder.lim_reuse_topk = reuse_topk
    metadata = SimpleNamespace(slot_mapping=cm.slot_mapping[: cm.num_input_tokens])
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(cm.num_reqs, 0, cm.num_input_tokens, 0)):
        if draft_index in (None, 0):
            # The proposer uses ordinary build() for draft step 0 too.
            with patch(MODULE + ".AscendSFAMetadataBuilder.build", return_value=metadata):
                metadata = builder.build(0, cm)
        else:
            with patch(MODULE + ".AscendSFAMetadataBuilder.build_for_drafting", return_value=metadata):
                metadata = builder.build_for_drafting(cm, draft_index=draft_index)
    return metadata
