"""CPU-only regressions for native C2KV mixed prefill/decode."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[3] / "python" / "sglang" / "srt" / "managers"


def _load_function(filename, name, *, class_name=None, namespace=None):
    tree = ast.parse((_ROOT / filename).read_text(encoding="utf-8"))
    owner = tree.body
    if class_name is not None:
        owner = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    node = copy.deepcopy(
        next(
            node
            for node in owner
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    scope = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(_ROOT / filename), "exec"), scope)
    return scope[name]


def _native_req(**overrides):
    values = dict(
        c2kv_outer_request_id="outer-1",
        c2kv_output_only_logprob=True,
        return_logprob=True,
        top_logprobs_num=0,
        token_ids_logprob=None,
        input_embeds=None,
        c2kv_rounds=[object()],
        c2kv_round_idx=1,
        is_prefill_only=False,
        return_hidden_states=False,
        c2kv_prompt_last_hidden_only=False,
        history_kv_eviction=None,
        history_kv_reference_config=None,
        history_kv_reference_state=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _can_mix(new_reqs, running_reqs, *, speculative=False):
    tree = ast.parse((_ROOT / "scheduler.py").read_text(encoding="utf-8"))
    gate = next(
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_is_native_output_only_mixed_chunk_req" in ast.unparse(node.test)
    )
    helper = _load_function("scheduler.py", "_is_native_output_only_mixed_chunk_req")
    new_batch = SimpleNamespace(
        reqs=new_reqs,
        return_logprob=any(req.return_logprob for req in new_reqs),
        input_embeds=None,
    )
    running_batch = SimpleNamespace(
        reqs=running_reqs,
        return_logprob=any(req.return_logprob for req in running_reqs),
        is_empty=lambda: not running_reqs,
    )
    scheduler = SimpleNamespace(
        is_mixed_chunk=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: not speculative),
        running_batch=running_batch,
    )
    scope = {
        "self": scheduler,
        "new_batch": new_batch,
        "_is_native_output_only_mixed_chunk_req": helper,
    }
    return eval(compile(ast.Expression(gate), "scheduler.py", "eval"), scope)


def test_native_output_only_contract_survives_logprob_start_normalization():
    tree = ast.parse((_ROOT / "scheduler.py").read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "c2kv_output_only_logprob"
            for target in node.targets
        )
    )
    scope = {
        "req": _native_req(c2kv_output_only_logprob=False),
        "recv_req": SimpleNamespace(
            c2kv_outer_request_id="outer-1",
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            input_embeds=None,
        ),
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "scheduler.py", "exec"), scope)
    scope["req"].logprob_start_len = 10  # Scheduler's normalized prompt length.
    gate = _load_function("scheduler.py", "_is_native_output_only_mixed_chunk_req")
    assert gate(scope["req"])
    assert _can_mix([scope["req"]], [_native_req()])

    for change in (
        {"logprob_start_len": 0},
        {"top_logprobs_num": 1},
        {"token_ids_logprob": [7]},
        {"c2kv_outer_request_id": None},
    ):
        for key, value in change.items():
            setattr(scope["recv_req"], key, value)
        exec(
            compile(ast.Module(body=[assignment], type_ignores=[]), "scheduler.py", "exec"),
            scope,
        )
        assert not gate(scope["req"])
        for key, value in (
            ("logprob_start_len", -1),
            ("top_logprobs_num", 0),
            ("token_ids_logprob", None),
            ("c2kv_outer_request_id", "outer-1"),
        ):
            setattr(scope["recv_req"], key, value)

    scope["req"].c2kv_output_only_logprob = True
    scope["req"].history_kv_reference_config = {"method": "h2o"}
    assert not gate(scope["req"])


def test_actual_scheduler_gate_excludes_unsupported_logprob_and_spec_modes():
    prefill = _native_req(c2kv_round_idx=0)
    decode = _native_req(c2kv_round_idx=1)
    assert _can_mix([prefill], [decode])
    assert not _can_mix([_native_req(c2kv_output_only_logprob=False)], [decode])
    assert not _can_mix([_native_req(history_kv_eviction={"method": "h2o"})], [decode])
    assert not _can_mix([prefill], [decode], speculative=True)
    assert not _can_mix([prefill], [_native_req(c2kv_round_idx=0)])
    legacy = _native_req(return_logprob=False, c2kv_output_only_logprob=False)
    assert _can_mix([legacy], [legacy], speculative=True)


@pytest.mark.parametrize("overlap", [False, True])
def test_mixed_decode_prefix_uses_prepared_physical_sequence(overlap):
    mix = _load_function(
        "schedule_batch.py",
        "mix_with_running",
        class_name="ScheduleBatch",
        namespace={"torch": torch, "ForwardMode": SimpleNamespace(MIXED="mixed")},
    )
    req = _native_req(
        c2kv_virtual_input_ids=[1, 2],
        origin_input_ids=list(range(100)),
        output_ids=[9, 10],
        kv_committed_len=13,
        c2kv_position_correction=5,
        set_extend_input_len=Mock(),
    )
    running = SimpleNamespace(
        reqs=[req],
        batch_size=lambda: 1,
        input_ids=torch.tensor([10]),
        out_cache_loc=torch.tensor([50]),
        seq_lens_cpu=torch.tensor([13]),  # Already advanced by prepare_for_decode.
    )
    prefill = SimpleNamespace(
        enable_overlap=overlap,
        reqs=[_native_req()],
        input_ids=torch.tensor([1, 2]),
        out_cache_loc=torch.tensor([40, 41]),
        prefix_lens=[4],
        extend_lens=[2],
        extend_num_tokens=2,
        extend_logprob_start_lens=[2],
        merge_batch=lambda other: prefill.reqs.extend(other.reqs),
    )
    mix(prefill, running)
    assert prefill.prefix_lens == [4, 12]
    assert prefill.extend_lens == [2, 1]
    assert prefill.extend_logprob_start_lens == [2, 1]
    assert prefill.prefix_lens[-1] + prefill.extend_lens[-1] == 13
    assert prefill.prefix_lens[-1] + req.c2kv_position_correction == 17
    assert prefill.input_ids.tolist() == [1, 2, 10]


def test_mixed_result_completes_decode_without_prefill_side_effects():
    telemetry = SimpleNamespace(set_phase=Mock(), mark_generation_start=Mock())
    release = Mock()
    namespace = {
        "paper_telemetry": telemetry,
        "release_kv_cache": release,
        "logger": SimpleNamespace(error=Mock()),
    }
    method_names = (
        "_process_mixed_decode_req",
        "_handle_finished_req",
        "_maybe_update_reasoning_tokens",
        "_mamba_prefix_cache_update",
        "process_batch_result_prefill",
    )
    methods = {
        name: _load_function(
            "scheduler_output_processor_mixin.py",
            name,
            class_name="SchedulerOutputProcessorMixin",
            namespace=namespace,
        )
        for name in method_names
    }
    scheduler = SimpleNamespace(
        is_generation=True,
        num_generated_tokens=0,
        model_config=SimpleNamespace(think_end_id=77),
        server_args=SimpleNamespace(disaggregation_decode_enable_offload_kvcache=False),
        enable_hisparse=False,
        tree_cache=object(),
        _release_c2kv_pins=Mock(),
        maybe_collect_routed_experts=Mock(),
        maybe_collect_customized_info=Mock(),
        stream_output=Mock(),
        report_prefill_stats=Mock(),
    )
    for name, method in methods.items():
        setattr(scheduler, name, MethodType(method, scheduler))

    prefill_time = SimpleNamespace(
        set_last_chunked_prefill_finish_time=Mock(),
        set_prefill_finished_time=Mock(),
    )
    prefill = _native_req(
        c2kv_round_idx=0,
        c2kv_rounds=[
            SimpleNamespace(post_history_kv_eviction=False, collect_history_kv_scores=False)
        ],
        is_chunked=1,
        is_retracted=False,
        time_stats=prefill_time,
        finished=lambda: False,
        output_ids=[],
    )
    decode_time = SimpleNamespace(
        set_last_decode_finish_time=Mock(),
        set_prefill_finished_time=Mock(),
        set_completion_time=Mock(),
    )
    grammar = SimpleNamespace(accept_token=Mock(), finished=False)
    held_checkpoint = object()
    decode = _native_req(
        rid="decode",
        c2kv_prompt_last_hidden_only=True,
        return_hidden_states=True,
        origin_input_ids=[1, 2],
        output_ids=[8],
        output_token_logprobs_val=[-0.5],
        output_token_logprobs_idx=[8],
        hidden_states=[],
        mamba_ping_pong_track_buffer=None,
        is_chunked=0,
        is_retracted=False,
        time_stats=decode_time,
        grammar=grammar,
        require_reasoning=True,
        update_reasoning_tokens=Mock(),
        multimodal_inputs=None,
        session=None,
        racer_held_generation=held_checkpoint,
        finished=lambda: len(decode.output_ids) >= 2,
        check_finished=Mock(),
    )
    batch = SimpleNamespace(
        reqs=[prefill, decode],
        decoding_reqs=[decode],
        return_logprob=True,
        prefill_stats=object(),
        dp_cooperation_info=None,
    )
    logits = SimpleNamespace(
        next_token_logprobs=torch.tensor([-0.1, -0.2]),
        input_token_logprobs=None,
        next_token_top_logprobs_val=None,
        next_token_token_ids_logprobs_val=None,
        hidden_states=torch.tensor([[1.0], [2.0]]),
    )
    result = SimpleNamespace(
        copy_done=None,
        logits_output=logits,
        next_token_ids=torch.tensor([31, 41]),
        extend_input_len_per_req=[2, 1],
        extend_logprob_start_len_per_req=[2, 1],
        can_run_cuda_graph=False,
    )

    scheduler.process_batch_result_prefill(batch, result)

    assert prefill.is_chunked == 0
    prefill_time.set_last_chunked_prefill_finish_time.assert_called_once()
    assert prefill.output_ids == []
    assert decode.output_ids == [8, 41]
    assert decode.output_token_logprobs_idx == [8, 41]
    assert decode.output_token_logprobs_val[-1] == pytest.approx(-0.2)
    assert decode.racer_held_generation is held_checkpoint
    assert decode.hidden_states == []
    decode_time.set_last_decode_finish_time.assert_called_once()
    decode_time.set_prefill_finished_time.assert_not_called()
    decode_time.set_completion_time.assert_called_once()
    grammar.accept_token.assert_called_once_with(41)
    assert grammar.finished
    decode.update_reasoning_tokens.assert_called_once_with(41, 77)
    release.assert_called_once_with(decode, scheduler.tree_cache, is_insert=False)
    telemetry.mark_generation_start.assert_not_called()
    assert scheduler.num_generated_tokens == 1
