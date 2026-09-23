"""Prompt-last shadow features must use the same pruning as input logprobs."""

from types import SimpleNamespace

import torch

from test_c2kv_composition import extract_class


LogitsProcessor = extract_class(
    "layers/logits_processor.py", "LogitsProcessor",
    {"_get_pruned_states", "_get_hidden_states_to_store"},
    extra={"LogitsMetadata": SimpleNamespace},
)


def test_input_logprob_pruning_preserves_auxiliary_prompt_last_states():
    processor = LogitsProcessor()
    hidden = torch.arange(27, dtype=torch.float32).reshape(9, 3)
    aux = [hidden + 100, hidden + 200]
    metadata = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode_or_idle=lambda: False, is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False, is_extend=lambda: True),
        extend_return_logprob=True,
        extend_seq_lens_cpu=[4, 5],
        # The second request still samples its last token but has no input logprobs.
        extend_logprob_start_lens_cpu=[1, 5],
        capture_hidden_mode=SimpleNamespace(
            need_capture=lambda: True, is_full=lambda: False, is_last=lambda: True),
    )
    pruned, before_norm, aux_pruned, samples, _, _ = processor._get_pruned_states(
        hidden, None, aux, metadata)
    assert all(torch.equal(actual, source[[1, 2, 3, 8]])
               for actual, source in zip(aux_pruned, aux))
    captured = processor._get_hidden_states_to_store(
        hidden, None, aux, pruned, before_norm, aux_pruned, samples, metadata)
    assert torch.equal(captured, torch.cat([states[[3, 8]] for states in aux], dim=-1))
