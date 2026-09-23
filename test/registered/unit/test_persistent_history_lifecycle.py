"""CPU-only lifecycle regression tests. Never import a model or contact HTTP."""
import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import types
import math
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "python/sglang/srt/mem_cache"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ledger = load("persistent_ledger_test", CACHE / "history_kv_lifecycle.py")
eviction = load("physical_evictor_test", CACHE / "history_kv_eviction.py")
selection = load("history_kv_selection_test", CACHE / "history_kv_selection.py")


def method(path, cls, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == cls
        for n in c.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("method_name", ["streamingllm", "h2o", "snapkv_persistent", "pyramidkv"])
@pytest.mark.parametrize("layout", ["token", "ascend_page", "ascend_fia"])
def test_two_turns_reuse_only_resident_kv_and_new_tokens(method_name, layout):
    # Use nonzero physical allocator slots, independently of logical position.
    row = torch.zeros((1, 64), dtype=torch.int64)
    row[0, :8] = torch.arange(4, 12)
    keys = torch.zeros((64, 1, 1))
    values = torch.zeros_like(keys)
    keys[4:12, 0, 0] = torch.arange(8)
    values[4:12, 0, 0] = torch.arange(8) + 100
    shape = (16,4,1,1) if layout == "ascend_page" else (64,1,1,1)
    cache = SimpleNamespace(start_layer=0, layer_num=1,
        _get_key_buffer=lambda _: keys if layout == "token" else keys.view(shape),
        _get_value_buffer=lambda _: values if layout == "token" else values.view(shape))
    freed = []
    allocator = SimpleNamespace(page_size=4, get_kvcache=lambda: cache,
                                free=lambda x: freed.extend(x.tolist()), available_size=lambda: 64)
    req = SimpleNamespace(req_pool_idx=0, kv_committed_len=8, kv_allocated_len=8,
                          already_computed=8, c2kv_position_correction=0)
    evictor = eviction.PhysicalHistoryKVEvictor(SimpleNamespace(req_to_token=row), allocator)
    result = evictor.evict(req, method=method_name, history_start=2, history_end=8,
                          target_tokens=3, selected_history_indices=[0, 3, 5])
    assert result.success
    assert req.kv_committed_len == 5 and result.new_physical_kv_slots == 8
    assert result.next_rope_position_before == result.next_rope_position_after == 8
    retained = ledger.compact_positions(range(8), 2, 8, [0, 3, 5])
    assert retained == [0, 1, 2, 5, 7]
    assert keys[row[0, :5], 0, 0].tolist() == retained
    assert values[row[0, :5], 0, 0].tolist() == [p+100 for p in retained]
    # Turn 2 uses the *same physical prefix*. Only canonical new positions 8–11
    # are appended; logical old history is not materialized again.
    next_positions = ledger.append_resident_positions(retained, 8, 12)
    row[0, 5:9] = torch.arange(9, 13)
    keys[9:13, 0, 0] = torch.arange(8, 12)
    values[9:13, 0, 0] = torch.arange(8, 12) + 100
    req.kv_committed_len = req.kv_allocated_len = 9
    assert keys[row[0, :9], 0, 0].tolist() == next_positions
    hs, he = ledger.physical_history_range(next_positions, 2, 10)
    result = evictor.evict(req, method=method_name, history_start=hs, history_end=he,
                          target_tokens=3, selected_history_indices=[0, 3, 4])
    assert result.success and result.next_rope_position_after == 12
    final = ledger.compact_positions(next_positions, hs, he, [0, 3, 4])
    assert not {3, 4, 6}.intersection(final)
    assert keys[row[0, :req.kv_committed_len], 0, 0].tolist() == final
    assert values[row[0, :req.kv_committed_len], 0, 0].tolist() == [p+100 for p in final]


def test_within_turn_boundary_may_precede_cached_current_prefix():
    positions = ledger.append_resident_positions([0, 1, 4, 5, 6], 7, 10)
    assert ledger.physical_history_range(positions, 2, 5) == (2, 3)
    assert positions == [0, 1, 4, 5, 6, 7, 8, 9]


def test_attention_selection_window_uses_only_new_tail_queries():
    assert ledger.selection_query_window(
        "snapkv_persistent", 0, 6, 10, 3
    ) == (7, 10)
    assert ledger.selection_query_window("h2o", 5, 7, 9, 64) == (7, 9)
    assert ledger.selection_query_window("pyramidkv", 5, 7, 12, 3) == (9, 12)
    # With no current suffix, the last newly-prefilled history token still
    # completes the candidate span and is a valid query.
    assert ledger.selection_query_window("h2o", 0, 5, 5, 64) == (4, 5)
    assert ledger.selection_query_window("streamingllm", 5, 7, 9, 64) is None
    with pytest.raises(ValueError, match="REQUIRES_NEW_QUERY"):
        ledger.selection_query_window("h2o", 5, 5, 5, 64)


def test_long_history_score_logits_are_bounded_without_changing_scores(monkeypatch):
    collect = method(
        ROOT / "python/sglang/srt/models/qwen3.py",
        "Qwen3Attention",
        "_collect_history_kv_eviction_scores",
        {"torch": torch, "ForwardBatch": SimpleNamespace},
    )
    history_len = 100_000
    query_len = 20
    bmm_query_lens = []
    original_bmm = torch.bmm

    def checked(left, right):
        assert right.shape[0] == 1  # KV is not copied across four query heads.
        bmm_query_lens.append(left.shape[1] // 4)
        return original_bmm(left, right)

    monkeypatch.setattr(torch, "bmm", checked)
    config = {
        "method": "pyramidkv", "history_start": 0,
        "history_end": history_len, "history_kv_recent_window": 64,
    }
    key_buffer = torch.zeros(history_len + 1, 1, 1)
    fb = SimpleNamespace(
        c2kv_history_kv_eviction_configs=[config],
        forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda: True),
        extend_seq_lens_cpu=[query_len],
        extend_prefix_lens_cpu=[history_len],
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(1, history_len + 1).unsqueeze(0)
        ),
        token_to_kv_pool=SimpleNamespace(_get_key_buffer=lambda _: key_buffer),
    )
    attention = SimpleNamespace(
        num_heads=4, num_kv_heads=1, head_dim=1, scaling=1.,
        attn=SimpleNamespace(layer_id=0),
    )
    collect(
        attention,
        torch.zeros(query_len, 4),
        torch.zeros(query_len, 1),
        torch.arange(history_len, history_len + query_len),
        fb,
    )
    scores = fb.c2kv_history_kv_selection_scores[0]["layers"][0]
    expected = 4 * sum(1 / (history_len + i + 1) for i in range(query_len))
    assert bmm_query_lens == [10, 10]
    torch.testing.assert_close(scores[[0, -1]], torch.tensor([expected, expected]))


def test_grouped_score_respects_each_kv_heads_reference_positions():
    collect = method(
        ROOT / "python/sglang/srt/models/qwen3.py",
        "Qwen3Attention",
        "_collect_history_kv_eviction_scores",
        {"torch": torch, "ForwardBatch": SimpleNamespace},
    )
    layer = SimpleNamespace(
        key=torch.zeros(2, 2, 1),
        positions=torch.tensor([[0, 2], [1, 5]]),
        validate=lambda: None,
    )
    fb = SimpleNamespace(
        c2kv_history_kv_eviction_configs=[{
            "method": "pyramidkv", "history_start": 0, "history_end": 1,
        }],
        history_kv_reference_states=[SimpleNamespace(layer=lambda _: layer)],
        forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda: True),
        extend_seq_lens_cpu=[2], extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
    )
    attention = SimpleNamespace(
        num_heads=4, num_kv_heads=2, head_dim=1, scaling=1.,
        attn=SimpleNamespace(layer_id=0),
    )
    collect(
        attention, torch.zeros(2, 4), torch.zeros(2, 2),
        torch.tensor([3, 4]), fb,
    )
    scores = fb.c2kv_history_kv_selection_scores[0]
    torch.testing.assert_close(
        scores["headwise_layers"][0],
        torch.tensor([[2 / 3, 2 / 3, 2 / 3], [1., 0., 1.]]),
    )
    torch.testing.assert_close(scores["layers"][0], torch.tensor([5 / 3]))


def test_pyramid_scores_reference_only_history_after_source_replacement():
    collect = method(
        ROOT / "python/sglang/srt/models/qwen3.py",
        "Qwen3Attention",
        "_collect_history_kv_eviction_scores",
        {"torch": torch, "ForwardBatch": SimpleNamespace},
    )
    layer = SimpleNamespace(
        key=torch.zeros(1, 2, 1),
        positions=torch.tensor([[0, 1]]),
        validate=lambda: None,
    )
    key_buffer = torch.zeros(8, 1, 1)
    fb = SimpleNamespace(
        c2kv_history_kv_eviction_configs=[{
            "method": "pyramidkv",
            "history_start": 0,
            "history_end": 0,
            "selection_query_start": 12,
            "selection_query_end": 14,
        }],
        history_kv_reference_states=[SimpleNamespace(layer=lambda _: layer)],
        history_kv_resident_positions=[[10, 11, 12, 13]],
        forward_mode=SimpleNamespace(
            is_extend_or_draft_extend_or_mixed=lambda: True
        ),
        extend_seq_lens_cpu=[2],
        extend_prefix_lens_cpu=[2],
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[1, 2]])
        ),
        token_to_kv_pool=SimpleNamespace(
            _get_key_buffer=lambda _: key_buffer
        ),
    )
    attention = SimpleNamespace(
        num_heads=4,
        num_kv_heads=1,
        head_dim=1,
        scaling=1.0,
        attn=SimpleNamespace(layer_id=0),
    )

    collect(
        attention,
        torch.zeros(2, 4),
        torch.zeros(2, 1),
        torch.tensor([12, 13]),
        fb,
    )

    scores = fb.c2kv_history_kv_selection_scores[0]
    assert scores["query_tokens"] == 2
    assert scores["layers"][0].numel() == 0
    assert tuple(scores["headwise_layers"][0].shape) == (1, 2)
    assert bool((scores["headwise_layers"][0] > 0).all())


def test_overlap_processes_final_selection_round_before_decode_scheduling():
    path = ROOT / "python/sglang/srt/managers/scheduler.py"
    node = next(
        n for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_c2kv_pending_result_requires_early_process"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
         namespace)
    needs_early = namespace["_c2kv_pending_result_requires_early_process"]
    plain = SimpleNamespace(post_history_kv_eviction=False)
    selection_round = SimpleNamespace(post_history_kv_eviction=True)
    assert needs_early(SimpleNamespace(c2kv_rounds=[plain, selection_round],
                                       c2kv_round_idx=0))
    assert needs_early(SimpleNamespace(c2kv_rounds=[plain, selection_round],
                                       c2kv_round_idx=1))
    assert not needs_early(SimpleNamespace(c2kv_rounds=[plain],
                                           c2kv_round_idx=0))

    schedule_tree = ast.parse(
        (ROOT / "python/sglang/srt/managers/schedule_batch.py").read_text(
            encoding="utf-8"
        )
    )
    copy_node = next(
        n for c in schedule_tree.body
        if isinstance(c, ast.ClassDef) and c.name == "ScheduleBatch"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "copy"
    )
    return_call = next(
        n.value for n in ast.walk(copy_node)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
    )
    copy_fields = {kw.arg for kw in return_call.keywords}
    assert {"seq_lens", "seq_lens_cpu", "seq_lens_sum", "device"} <= copy_fields

    scheduler_source = path.read_text(encoding="utf-8")
    early_pop = scheduler_source.index(
        "pop_and_process(sync_c2kv_early_batch=True)"
    )
    next_batch = scheduler_source.index(
        "batch = self.get_next_batch_to_run()", early_pop
    )
    assert early_pop < next_batch
    assert "self.last_batch.seq_lens_sum = tmp_batch.seq_lens_sum" in (
        scheduler_source[early_pop - 2500:next_batch]
    )


def test_ledger_rejects_duplicate_and_resurrected_old_positions():
    for bad in ([0, 0], [0, 8], [2, 1]):
        with pytest.raises(ValueError):
            ledger.append_resident_positions(bad, 8, 12)
    with pytest.raises(ValueError):
        ledger.compact_positions([0, 1, 5], 1, 3, [0, 0])


def test_decode_cleanup_never_frees_prompt_pages_at_nonzero_allocator_offset():
    discard = method(CACHE / "session_aware_cache.py", "SessionAwareCache",
                     "_discard_persistent_decode_suffix", {"torch": torch, "Req": SimpleNamespace})
    row = torch.arange(40, 60).reshape(1, 20)
    freed = []
    self = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=row), page_size=4,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda x: freed.extend(x.tolist())))
    req = SimpleNamespace(origin_input_ids=list(range(5)), kv_committed_len=8, kv_allocated_len=10,
        req_pool_idx=0, persistent_decode_cache_locs=[torch.tensor(44), torch.tensor(49), torch.tensor(50)],
        kv_memory_report={})
    discard(self, req)
    assert freed == [48]  # page 12 only; prompt pages 10 and 11 must survive
    assert row[0, :5].tolist() == [40, 41, 42, 43, 44]
    assert row[0, 5:10].tolist() == [0]*5
    assert req.kv_committed_len == req.kv_allocated_len == 5


@pytest.mark.parametrize("reference_method", [None, "agentkv", "commitkv"])
def test_single_uncommitted_decode_token_is_reclaimed(reference_method):
    discard = method(
        CACHE / "session_aware_cache.py", "SessionAwareCache",
        "_discard_persistent_decode_suffix", {"torch": torch, "Req": SimpleNamespace},
    )
    row = torch.tensor([[40, 41, 42, 43, 44, 0, 0]])
    freed = []
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row), page_size=1,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda slots: freed.extend(slots.tolist())
        ),
    )
    req = SimpleNamespace(
        origin_input_ids=list(range(5)), req_pool_idx=0,
        kv_committed_len=5, kv_allocated_len=5,
        c2kv_position_correction=0,
        history_kv_resident_positions=list(range(5)),
        history_kv_reference_config=(
            {"method": reference_method} if reference_method else None
        ),
        reference_decode_logical_start=5,
        output_ids=[777],
        persistent_decode_cache_locs=[torch.tensor(45)],
        kv_memory_report={},
    )
    discard(owner, req)
    assert freed == [45]
    assert req.kv_committed_len == req.kv_allocated_len == 5
    assert req.persistent_decode_cache_locs == []


@pytest.mark.parametrize(
    ("receipt", "expected_active"),
    [
        ({"success": True, "kept_history_tokens": 3}, 3),
        ({"success": False, "kept_history_tokens": 3}, 0),
        (None, 0),
    ],
)
def test_persistent_decode_cleanup_preserves_measured_physical_history(
    receipt, expected_active
):
    discard = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "_discard_persistent_decode_suffix",
        {"torch": torch, "Req": SimpleNamespace},
    )
    row = torch.arange(40, 52).reshape(1, 12)
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        page_size=4,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda _: None),
    )
    report = {
        "active_history_kv_tokens": 0,
        "active_full_raw_tokens": 0,
        "active_history_kv_tokens_source": "scheduler_runtime",
    }
    if receipt is not None:
        report["history_kv_physical_eviction"] = receipt
    req = SimpleNamespace(
        origin_input_ids=list(range(5)),
        kv_committed_len=8,
        kv_allocated_len=10,
        req_pool_idx=0,
        persistent_decode_cache_locs=[],
        kv_memory_report=report,
    )

    discard(owner, req)

    assert report["active_history_kv_tokens"] == expected_active
    assert report["active_full_raw_tokens"] == expected_active
    assert report["active_history_kv_tokens_source"] == (
        "physical_eviction_measured" if receipt and receipt["success"]
        else "scheduler_runtime"
    )
    assert report["reference_history_token_slots"] == 0
    assert req.kv_committed_len == req.kv_allocated_len == 5


def test_legacy_streaming_slot_is_reconciled_before_persistent_continuation():
    adopt = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "_adopt_legacy_persistent_prefix",
        {"torch": torch, "SessionSlot": SimpleNamespace, "json": __import__("json"),
         "logging": __import__("logging")},
    )
    row = torch.arange(100, 112).reshape(1, 12)
    freed = []
    slot = SimpleNamespace(
        req_pool_idx=0,
        kv_committed_len=12,
        kv_allocated_len=12,
        cache_protected_len=0,
        history_kv_resident_positions=[],
        history_kv_score_state={"stale": {1: 2.0}},
    )
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        page_size=1,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.extend(indices.tolist())
        ),
    )
    adopt(owner, slot, 8)
    assert slot.kv_committed_len == slot.kv_allocated_len == 8
    assert slot.history_kv_resident_positions == list(range(8))
    assert slot.history_kv_score_state == {}
    assert row[0, 8:].tolist() == [0] * 4
    assert freed == [108, 109, 110, 111]


def test_persistent_history_unfinished_kv_stays_out_of_radix_tree():
    cache_unfinished = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "cache_unfinished_req",
        {"torch": torch, "Req": SimpleNamespace, "_is_streaming": lambda _: True},
    )
    row = torch.arange(400, 800).reshape(1, 400)
    inner_calls = []
    self = SimpleNamespace(
        _is_persistent_history_req=lambda _: True,
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        inner=SimpleNamespace(
            cache_unfinished_req=lambda *args, **kwargs: inner_calls.append(
                (args, kwargs)
            )
        ),
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        fill_ids=list(range(134)),
        prefix_indices=torch.empty(0, dtype=torch.int64),
        cache_protected_len=256,
    )

    cache_unfinished(self, req, chunked=False)

    assert inner_calls == []
    assert req.cache_protected_len == 0
    assert req.prefix_indices.tolist() == list(range(400, 534))


def test_chunked_persistent_history_stash_has_no_radix_protected_prefix():
    stash = method(
        ROOT / "python/sglang/srt/managers/scheduler.py",
        "Scheduler",
        "stash_chunked_request",
        {"torch": torch, "Req": SimpleNamespace},
    )
    self = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(800).reshape(1, 800)
        ),
        tree_cache=SimpleNamespace(
            cache_unfinished_req=lambda *args, **kwargs: pytest.fail(
                "multi-round stash must not enter tree cache"
            )
        ),
    )
    persistent = SimpleNamespace(
        c2kv_rounds=[object(), object()],
        c2kv_round_idx=0,
        req_pool_idx=0,
        kv_committed_len=256,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        already_computed=0,
        cache_protected_len=999,
        c2kv_kv_memory_hint={
            "persistent_history_session": {"enabled": True}
        },
    )
    stash(self, persistent)
    assert len(persistent.prefix_indices) == 256
    assert persistent.already_computed == 256
    assert persistent.cache_protected_len == 0

    ordinary = SimpleNamespace(**{
        **persistent.__dict__,
        "c2kv_kv_memory_hint": {},
        "cache_protected_len": 999,
    })
    stash(self, ordinary)
    assert ordinary.cache_protected_len == 256


def test_persistent_compacted_session_accounting_is_nonnegative_and_exact():
    session_held = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "session_held_tokens",
        {"Req": SimpleNamespace, "ceil_align": ledger.ceil_align if hasattr(ledger, "ceil_align") else lambda x, y: ((x + y - 1) // y) * y},
    )
    slot = SimpleNamespace(
        is_holding_kv=True,
        kv_allocated_len=134,
        cache_protected_len=0,
    )
    self = SimpleNamespace(slots={"history": slot}, page_size=1)
    assert session_held(self) == 134


@pytest.mark.parametrize("method_name", ["streamingllm", "h2o", "snapkv_persistent", "pyramidkv"])
@pytest.mark.parametrize("streaming", [False, True])
def test_multiround_finish_transfers_session_ownership_without_radix_insert(method_name, streaming):
    # Execute the real release function without importing SGLang/device code.
    path = CACHE / "common.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "release_kv_cache")
    class HybridPool:
        pass
    namespace = {"Req": SimpleNamespace, "BasePrefixCache": object,
                 "HybridReqToTokenPool": HybridPool}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    release = namespace["release_kv_cache"]
    owns = method(CACHE / "session_aware_cache.py", "SessionAwareCache",
                  "owns_finished_request", {"Req": SimpleNamespace,
                  "_is_streaming": lambda r: r.session is not None and r.session.streaming})
    # AST extraction leaves @staticmethod: resolve its function explicitly.
    owns = owns.__func__
    freed, pool_freed, saved = [], [], []
    pool = SimpleNamespace(req_to_token=torch.arange(40, 48).reshape(1, 8),
                           free=lambda r: pool_freed.append(r.req_pool_idx))
    def finish(req, is_insert):
        assert is_insert is False
        saved.append((req.req_pool_idx, req.kv_allocated_len))
        req.req_pool_idx = None  # SessionSlot.save_from_req ownership transfer
    cache = SimpleNamespace(req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda x: freed.extend(x.tolist())),
        owns_finished_request=owns, cache_finished_req=finish,
        dec_lock_ref=lambda _: None)
    req = SimpleNamespace(req_pool_idx=0, c2kv_rounds=[object()],
        c2kv_tree_cache_prefix_len=0, kv_allocated_len=8, last_node=None,
        session=SimpleNamespace(streaming=streaming), history_kv_eviction={"method": method_name})
    release(req, cache, is_insert=False)
    if streaming:
        assert saved == [(0, 8)] and req.req_pool_idx is None
        assert freed == [] and pool_freed == []
        # Session close, not normal request completion, releases these slots.
        cache.token_to_kv_pool_allocator.free(pool.req_to_token[0, :8])
        assert freed == list(range(40, 48))
    else:
        assert saved == [] and pool_freed == [0]
        assert freed == list(range(40, 48))
        assert req.kv_committed_freed and req.kv_overallocated_freed


def test_serving_delta_prefix_mismatch_fails_without_full_prefill_fallback():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    prepare = method(path, "OpenAIServingChat", "_prepare_persistent_history_delta",
                     {"ChatCompletionRequest": object, "List": list, "Optional": __import__('typing').Optional})
    self = SimpleNamespace(_is_persistent_history_request=lambda _: True,
                           _persistent_history_sessions={"s": [0, 1, 2, 3]},
                           _translate_tool_session_coordinates=lambda *_: None)
    req = SimpleNamespace(stream=False, session_params={"id": "s"},
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": True, "session_id": "s"},
                           "history_kv_eviction": {"history_start": 1, "history_end": 2}})
    delta, sid, canonical = prepare(self, req, [0, 1, 2, 3, 4, 5])
    assert delta == [4, 5] and sid == "s" and canonical == [0, 1, 2, 3, 4, 5]
    assert req.c2kv_kv_memory_hint['history_kv_eviction']['persistent_delta_history_tokens'] == 0
    with pytest.raises(ValueError, match="PREFIX_MISMATCH"):
        prepare(self, req, [9, 1, 2, 3, 4, 5])
    req.session_params['id'] = 'other'
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        prepare(self, req, [0, 1, 2, 3, 4, 5])


def test_recovery_append_replaces_only_server_verified_generation_prefix():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    prepare = method(
        path,
        "OpenAIServingChat",
        "_prepare_persistent_history_delta",
        {"ChatCompletionRequest": object, "List": list,
         "Optional": __import__('typing').Optional},
    )
    previous = [10, 11, 12, 90, 91]
    self = SimpleNamespace(
        _is_persistent_history_request=lambda _: True,
        _persistent_history_sessions={"s": previous},
        _persistent_history_generation_prefixes={"s": [90, 91]},
        _translate_tool_session_coordinates=lambda *_: None,
    )
    req = SimpleNamespace(
        stream=False,
        session_params={"id": "s"},
        c2kv_kv_memory_hint={
            "persistent_history_session": {
                "enabled": True,
                "session_id": "s",
                "recovery_append": {"enabled": True},
            },
            "history_kv_eviction": {
                "method": "pyramidkv", "history_start": 1,
                "history_end": 5,
            },
        },
    )
    full = [10, 11, 12, 70, 71, 92, 93]
    delta, sid, canonical = prepare(self, req, full)
    assert sid == "s" and canonical == full
    assert delta == [70, 71, 92, 93]
    hint = req.c2kv_kv_memory_hint
    assert hint["persistent_session_logical_prefix_tokens"] == 3
    assert hint["persistent_session_drop_generation_prefix_tokens"] == 2
    assert req.session_params["drop_previous_output"] is True

    # The exception is narrowly scoped: callers cannot nominate the removed
    # tokens or alter anything before the server-verified generation prefix.
    with pytest.raises(ValueError, match="RECOVERY_BODY_PREFIX_MISMATCH"):
        prepare(self, req, [10, 99, 12, 70, 71, 92, 93])
    self._persistent_history_generation_prefixes.clear()
    with pytest.raises(ValueError, match="GENERATION_PREFIX_UNAVAILABLE"):
        prepare(self, req, full)


def test_recovery_splice_physically_preserves_lossy_history_and_accounts_pages():
    trim = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "_trim_persistent_generation_prefix",
        {"torch": torch, "SessionSlot": SimpleNamespace, "Req": SimpleNamespace},
    )
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 0, 0]])
    freed = []
    # Canonical history positions 1 and 3 are already absent. Only the
    # contiguous generation scaffold [6, 7] may be removed.
    slot = SimpleNamespace(
        req_pool_idx=0, kv_committed_len=6, kv_allocated_len=6,
        history_kv_resident_positions=[0, 2, 4, 5, 6, 7],
        history_kv_score_state={0: {0: 1.0, 6: 2.0, 7: 3.0}},
    )
    req = SimpleNamespace(
        session=SimpleNamespace(session_id="s"), kv_memory_report={},
        c2kv_kv_memory_hint={
            "persistent_session_logical_prefix_tokens": 6,
            "persistent_session_drop_generation_prefix_tokens": 2,
        },
    )
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row), page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.extend(indices.tolist())
        ),
    )
    trim(owner, slot, req)
    assert slot.history_kv_resident_positions == [0, 2, 4, 5]
    assert slot.kv_committed_len == slot.kv_allocated_len == 4
    assert row[0].tolist() == [40, 41, 42, 43, 0, 0, 0, 0]
    assert freed == [44]
    assert slot.history_kv_score_state == {0: {0: 1.0}}
    receipt = req.kv_memory_report["persistent_session_generation_prefix_splice"]
    assert receipt["scope"] == "verified_generation_prefix_only"
    assert receipt["retained_body_physical_tokens"] == 4
    assert receipt["retained_body_resident_page_tokens"] == 4
    assert receipt["freed_physical_page_tokens"] == 2
    assert receipt["full_history_reprefill_performed"] is False


def test_failed_recovery_splice_closes_session_instead_of_reusing_trimmed_state():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"

    class Close:
        def __init__(self, session_id):
            self.session_id = session_id

    handle = method(
        path,
        "OpenAIServingChat",
        "_handle_non_streaming_request",
        {
            "GenerateReqInput": object, "ChatCompletionRequest": object,
            "Request": object, "Union": __import__('typing').Union,
            "ChatCompletionResponse": object, "ErrorResponse": object,
            "ORJSONResponse": object, "CloseSessionReqInput": Close,
            "logger": SimpleNamespace(exception=lambda *args, **kwargs: None),
        },
    )
    closed = []

    class Manager:
        async def close_session(self, obj, raw_request):
            closed.append(obj.session_id)

        async def generate_request(self, adapted, raw_request):
            if False:
                yield None
            raise ValueError("generation failed after recovery splice")

    released = []
    self = SimpleNamespace(
        tokenizer_manager=Manager(),
        create_error_response=lambda message: message,
        release_persistent_history_session=lambda sid: released.append(sid),
    )
    adapted = SimpleNamespace(
        c2kv_kv_memory_hint={
            "persistent_session_drop_generation_prefix_tokens": 2
        },
        _persistent_history_session_id="recovery",
    )
    result = asyncio.run(handle(self, adapted, SimpleNamespace(), object()))
    assert "generation failed" in result
    assert closed == ["recovery"] and released == ["recovery"]


@pytest.mark.parametrize(
    "method_name", ["h2o", "snapkv_persistent", "pyramidkv"]
)
def test_attention_persistent_empty_delta_is_request_level_error(
    method_name, monkeypatch
):
    monkeypatch.setitem(
        sys.modules, 'sglang.srt.mem_cache.history_kv_lifecycle', ledger
    )
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    prepare = method(
        path,
        "OpenAIServingChat",
        "_prepare_persistent_history_delta",
        {
            "ChatCompletionRequest": object,
            "List": list,
            "Optional": __import__('typing').Optional,
        },
    )
    previous = [0, 1, 2, 3]
    self = SimpleNamespace(
        _is_persistent_history_request=lambda _: True,
        _persistent_history_sessions={"s": previous},
    )
    req = SimpleNamespace(
        stream=False,
        session_params={"id": "s"},
        c2kv_kv_memory_hint={
            "persistent_history_session": {
                "enabled": True,
                "session_id": "s",
            },
            "history_kv_eviction": {
                "method": method_name,
                "history_start": 1,
                "history_end": 3,
            },
        },
    )
    with pytest.raises(
        ValueError, match="PERSISTENT_HISTORY_SELECTION_REQUIRES_NEW_QUERY"
    ):
        prepare(self, req, list(previous))
    assert "persistent_continuation" not in (
        req.c2kv_kv_memory_hint["history_kv_eviction"]
    )


def test_physical_ratio_budget_uses_exact_server_tokenized_history_span():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    resolve = method(
        path, "OpenAIServingChat", "_resolve_history_kv_eviction_range",
        {"ChatCompletionRequest": object, "List": list, "math": math},
    )
    self = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=None)),
        _chat_template_tools=lambda request: ["tool"],
        _c2kv_chat_template_input_ids=(
            lambda request, completed, tools: (
                [10] if len(completed) == 1 else [10, 20, 21])
        ),
        _find_token_subsequence=lambda haystack, needle: next(
            (i for i in range(len(haystack) - len(needle) + 1)
             if haystack[i:i + len(needle)] == needle), -1),
    )
    req = SimpleNamespace(
        messages=[object(), object(), object()],
        c2kv_kv_memory_hint={"history_kv_eviction": {
            "history_start_message_count": 1,
            "history_message_count": 2,
            "retention_ratio": 0.25,
        }},
    )
    resolve(self, req, [10, 20, 21, 30])
    config = req.c2kv_kv_memory_hint["history_kv_eviction"]
    assert config["history_start"] == 1 and config["history_end"] == 3
    assert config["target_tokens"] == 1
    assert config["target_tokens_source"] == "server_tokenized_retention_ratio"
    assert req.c2kv_kv_memory_hint["full_equivalent_history_tokens"] == 2


def test_paper_history_boundary_uses_server_tokens_and_accepts_first_turn():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    resolve = method(
        path,
        "OpenAIServingChat",
        "_resolve_paper_history_token_count",
        {"ChatCompletionRequest": object, "List": list, "Optional": Optional},
    )
    self = SimpleNamespace(
        _chat_template_tools=lambda request: ["tool"],
        _c2kv_chat_template_input_ids=lambda request, messages, tools: (
            [10] if len(messages) == 1 else [10, 20, 21]
        ),
        _find_token_subsequence=lambda haystack, needle: next(
            (
                i
                for i in range(len(haystack) - len(needle) + 1)
                if haystack[i : i + len(needle)] == needle
            ),
            -1,
        ),
        _c2kv_first_message_start_offset=lambda request, message, tools: 0,
    )
    config = {"history_start_message_count": 1, "history_message_count": 2}
    contextual = method(path, "OpenAIServingChat", "_c2kv_contextual_prefix_ids", {"List": list})
    self._c2kv_contextual_prefix_ids = lambda *args: contextual(self, *args)
    req = SimpleNamespace(
        messages=[object(), object(), object()],
        c2kv_kv_memory_hint={"paper_measurement": config},
    )
    assert resolve(self, req, [10, 20, 21, 30]) == 2
    assert config["history_start"] == 1
    assert config["history_end"] == 3
    assert config["server_tokenized"] is True

    first_turn_config = {
        "history_start_message_count": 0,
        "history_message_count": 0,
    }
    first_turn = SimpleNamespace(
        messages=[object()],
        c2kv_kv_memory_hint={"paper_measurement": first_turn_config},
    )
    assert resolve(self, first_turn, [10, 20]) == 0
    assert first_turn_config["history_full_kv_tokens"] == 0


def test_session_match_restores_prefix_and_builds_only_new_history_round(monkeypatch):
    # Inject only the two tiny modules imported by the method under test.
    # No SGLang/model imports, no endpoint, no allocator device initialization.
    rounds_module = types.ModuleType('sglang.srt.managers.schedule_batch')
    class Round:
        def __init__(self, tokens, segments, post_history_kv_eviction=False):
            self.tokens = tokens; self.post_history_kv_eviction = post_history_kv_eviction
    rounds_module.C2KVPrefillRound = Round
    monkeypatch.setitem(sys.modules, rounds_module.__name__, rounds_module)
    monkeypatch.setitem(sys.modules, 'sglang.srt.mem_cache.history_kv_lifecycle', ledger)
    match = method(CACHE/'session_aware_cache.py', 'SessionAwareCache', 'match_prefix', {
        'MatchPrefixParams': object, 'MatchResult': lambda **kw: SimpleNamespace(**kw),
        'torch': torch, '_is_streaming': lambda _: True})
    is_persistent = method(
        CACHE/'session_aware_cache.py', 'SessionAwareCache',
        '_is_persistent_history_req', {'Req': SimpleNamespace})
    row = torch.arange(4, 36).reshape(1, 32)
    prior = [0, 1, 2, 5, 7]
    req = SimpleNamespace(session=SimpleNamespace(session_id='s'), kv_memory_report={},
        history_kv_eviction={'persistent_continuation_pending': True,
                            'method': 'snapkv_persistent',
                            'persistent_protected_prefix_tokens': 2,
                            'persistent_canonical_history_end': 10,
                            'persistent_delta_history_tokens': 2},
        c2kv_kv_memory_hint={'persistent_session_logical_prefix_tokens': 8,
                            'persistent_session_canonical_prompt_tokens': 12},
        origin_input_ids=prior+[8,9,10,11], c2kv_tool_source_spans=[])
    def restore(req):
        req.req_pool_idx=0; req.kv_committed_len=5; req.c2kv_position_correction=3
    slot=SimpleNamespace(req_pool_idx=0, kv_committed_len=5,
                         c2kv_position_correction=3,
                         history_kv_resident_positions=prior,
                         restore_to_req=restore, cache_protected_len=0, virtual_node=object())
    self=SimpleNamespace(slots={'s':slot}, req_to_token_pool=SimpleNamespace(req_to_token=row),
                         _refresh_persistent_tool_prefix=lambda slot, req: None,
                         _is_persistent_history_req=is_persistent)
    result=match(self,SimpleNamespace(req=req,key=SimpleNamespace(token_ids=req.origin_input_ids)))
    assert result.device_indices.tolist()==[4,5,6,7,8]
    assert req.history_kv_resident_positions==prior+[8,9,10,11]
    assert req.history_kv_eviction['history_start']==2
    assert req.history_kv_eviction['history_end']==7
    assert req.c2kv_rounds[0].tokens==prior+[8,9]
    assert req.c2kv_rounds[1].tokens==[10,11]
    assert not req.c2kv_rounds[0].post_history_kv_eviction
    assert req.c2kv_rounds[1].post_history_kv_eviction
    assert req.history_kv_eviction['selection_query_start']==7
    assert req.history_kv_eviction['selection_query_end']==9
    assert req.history_kv_eviction['selection_query_tokens']==2
    assert req.kv_memory_report['selection_query_tokens']==2
    assert not {3,4,6}.intersection(req.history_kv_resident_positions)


def test_first_request_selection_round_ends_with_tail_queries(monkeypatch):
    rounds_module = types.ModuleType('sglang.srt.managers.schedule_batch')
    class Round:
        def __init__(self, tokens, segments, post_history_kv_eviction=False):
            self.tokens = tokens
            self.post_history_kv_eviction = post_history_kv_eviction
    rounds_module.C2KVPrefillRound = Round
    monkeypatch.setitem(sys.modules, rounds_module.__name__, rounds_module)
    monkeypatch.setitem(
        sys.modules, 'sglang.srt.mem_cache.history_kv_lifecycle', ledger
    )
    build = method(
        ROOT/'python/sglang/srt/managers/scheduler.py',
        'Scheduler', '_build_history_kv_eviction_rounds',
        {
            'Optional': Optional,
            '_persistent_history_session_error': lambda *_: None,
        },
    )
    owner = SimpleNamespace(_log_c2kv_token_usage=lambda *args, **kwargs: None)
    config = {
        'method': 'h2o', 'history_start': 1, 'history_end': 6,
        'history_kv_recent_window': 3, 'target_tokens': 2,
    }
    req = SimpleNamespace(
        history_kv_eviction=config, origin_input_ids=list(range(10)),
        prefix_indices=[], session=None, kv_memory_report={},
    )
    assert build(owner, req) is None
    assert [r.tokens for r in req.c2kv_rounds] == [
        list(range(7)), list(range(7, 10))
    ]
    assert [r.post_history_kv_eviction for r in req.c2kv_rounds] == [
        False, True
    ]
    assert req.kv_memory_report['selection_query_tokens'] == 3


def test_persistent_history_session_binding_validation():
    scheduler = ROOT / "python/sglang/srt/managers/scheduler.py"
    validate = function(scheduler, "_persistent_history_session_error", {
        "Optional": Optional,
    })
    marker = {
        "persistent_history_session": {
            "enabled": True,
            "session_id": "expected",
        }
    }
    persistent = {"persistent_session": True}

    assert validate(SimpleNamespace(session=None), {}) is None
    assert validate(SimpleNamespace(
        session=None, c2kv_kv_memory_hint=marker,
    ), persistent) == "PERSISTENT_HISTORY_SESSION_UNAVAILABLE"
    assert validate(SimpleNamespace(
        session=SimpleNamespace(streaming=False, session_id="expected"),
        c2kv_kv_memory_hint=marker,
    ), persistent) == "PERSISTENT_HISTORY_SESSION_REQUIRES_STREAMING_SESSION"
    assert validate(SimpleNamespace(
        session=SimpleNamespace(streaming=True, session_id="different"),
        c2kv_kv_memory_hint=marker,
    ), persistent) == "PERSISTENT_HISTORY_SESSION_ID_MISMATCH"
    assert validate(SimpleNamespace(
        session=SimpleNamespace(streaming=True, session_id="expected"),
        c2kv_kv_memory_hint=marker,
    ), persistent) is None


def test_persistent_first_turn_without_session_fails_before_round_construction():
    scheduler = ROOT / "python/sglang/srt/managers/scheduler.py"
    validate = function(scheduler, "_persistent_history_session_error", {
        "Optional": Optional,
    })
    build = method(
        scheduler,
        "Scheduler",
        "_build_history_kv_eviction_rounds",
        {
            "Optional": Optional,
            "_persistent_history_session_error": validate,
        },
    )
    req = SimpleNamespace(
        session=None,
        prefix_indices=[],
        origin_input_ids=list(range(6)),
        history_kv_eviction={
            "method": "agentkv",
            "persistent_session": True,
            "history_start": 0,
            "history_end": 0,
        },
        c2kv_kv_memory_hint={
            "persistent_history_session": {"enabled": True, "session_id": "s"}
        },
    )
    assert build(SimpleNamespace(), req) == "PERSISTENT_HISTORY_SESSION_UNAVAILABLE"
    assert not hasattr(req, "c2kv_rounds")


def test_closed_persistent_session_aborts_before_physical_eviction(monkeypatch):
    scheduler = ROOT / "python/sglang/srt/managers/scheduler.py"
    validate = function(scheduler, "_persistent_history_session_error", {
        "Optional": Optional,
    })

    class Abort:
        def __init__(self, message):
            self.message = message

    schedule_batch = types.ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.FINISH_ABORT = Abort
    monkeypatch.setitem(sys.modules, schedule_batch.__name__, schedule_batch)
    events = []
    telemetry = SimpleNamespace(
        set_phase=lambda phase: events.append(("phase", phase)),
        sample=lambda event: events.append(("sample", event)),
    )
    apply_eviction = method(
        scheduler,
        "Scheduler",
        "_apply_history_kv_eviction",
        {
            "_persistent_history_session_error": validate,
            "paper_telemetry": telemetry,
        },
    )
    checked = []
    req = SimpleNamespace(
        session=None,
        history_kv_eviction={"method": "agentkv", "persistent_session": True},
        c2kv_kv_memory_hint={
            "persistent_history_session": {"enabled": True, "session_id": "closed"}
        },
        kv_memory_report={},
        check_finished=lambda: checked.append(True),
    )
    assert not apply_eviction(SimpleNamespace(), req)
    assert isinstance(req.to_finish, Abort)
    assert req.to_finish.message == "PERSISTENT_HISTORY_SESSION_UNAVAILABLE"
    assert req.persistent_history_eviction_failed
    assert checked == [True]
    assert req.kv_memory_report["history_kv_physical_eviction"] == {
        "success": False,
        "error": "PERSISTENT_HISTORY_SESSION_UNAVAILABLE",
    }
    assert req.kv_memory_report["history_kv_runtime_status"] == (
        "persistent_session_unavailable"
    )
    assert req.kv_memory_report["persistent_history_session_error"] == (
        "PERSISTENT_HISTORY_SESSION_UNAVAILABLE"
    )
    assert events == [
        ("phase", "selection"),
        ("sample", "history_kv_eviction_failed"),
        ("phase", "prefill"),
    ]


@pytest.mark.parametrize("method_name", ["commitkv", "pyramidkv"])
def test_reference_recovery_with_no_selectable_history_keeps_empty_state(
    monkeypatch, method_name
):
    """Source replacement may leave only native recovery/current tokens."""

    class Abort:
        def __init__(self, message):
            self.message = message

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.schedule_batch",
        SimpleNamespace(FINISH_ABORT=Abort),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.history_kv_eviction",
        eviction,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.history_kv_lifecycle",
        ledger,
    )
    telemetry = SimpleNamespace(
        set_phase=lambda *_: None,
        sample=lambda *_: None,
    )
    apply_eviction = method(
        ROOT / "python/sglang/srt/managers/scheduler.py",
        "Scheduler",
        "_apply_history_kv_eviction",
        {
            "_persistent_history_session_error": lambda *_: None,
            "json": __import__("json"),
            "logger": SimpleNamespace(
                error=lambda *args, **kwargs: None,
                info=lambda *args, **kwargs: None,
            ),
            "math": math,
            "paper_telemetry": telemetry,
        },
    )
    empty_layer = SimpleNamespace(
        key=torch.empty(1, 0, 1),
        value=torch.empty(1, 0, 1),
        positions=torch.empty(1, 0, dtype=torch.long),
    )
    empty_state = SimpleNamespace(
        layers={0: empty_layer},
        resident_bytes=0,
        selection_metadata={"method": method_name},
    )
    allocator = SimpleNamespace(
        page_size=1,
        get_kvcache=lambda: SimpleNamespace(),
        available_size=lambda: 100,
    )
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=allocator,
        _bytes_per_kv_token=lambda: 8,
        _build_pyramidkv_reference_state=lambda *_: pytest.fail(
            "empty history must not rebuild PyramidKV state"
        ),
        _build_agentkv_reference_state=lambda *_: pytest.fail(
            "empty history must not rebuild AgentKV state"
        ),
        _build_commitkv_reference_state=lambda *_: pytest.fail(
            "empty history must not rebuild CommitKV state"
        ),
        _log_c2kv_token_usage=lambda *args, **kwargs: None,
    )
    req = SimpleNamespace(
        rid="empty-reference-recovery",
        history_kv_eviction={
            "method": method_name,
            "history_start": 0,
            "history_end": 0,
            "target_tokens": 8,
            "selection_query_tokens": 16,
            "persistent_session": True,
            "persistent_canonical_history_end": 27,
        },
        history_kv_selection_scores=None,
        history_kv_reference_state=empty_state,
        history_kv_resident_positions=[27, 28],
        kv_memory_report={},
        kv_committed_len=2,
        c2kv_position_correction=27,
        c2kv_virtual_input_ids=[100, 101],
        c2kv_kv_memory_hint={"persistent_session_delta_tokens": 2},
        session=SimpleNamespace(session_id="recovery"),
    )

    assert apply_eviction(owner, req)
    assert req.history_kv_reference_state is empty_state
    assert req.history_kv_resident_positions == [27, 28]
    assert req.c2kv_persistent_active_input_ids == [100, 101]
    assert req.kv_memory_report["selection_query_tokens_observed"] == 0
    assert req.kv_memory_report["active_history_kv_tokens"] == 0
    assert req.kv_memory_report["history_kv_runtime_status"] == (
        "reference_attention_ok"
    )
    assert req.kv_memory_report["history_kv_lifecycle"][
        "history_kv_backend"
    ] == "reference_attention"


def test_closed_persistent_request_uses_ordinary_cache_cleanup():
    finished, unfinished = [], []
    inner = SimpleNamespace(
        cache_finished_req=lambda req, **kwargs: finished.append((req, kwargs)),
        cache_unfinished_req=lambda req, **kwargs: unfinished.append((req, kwargs)),
    )
    cache_finished = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "cache_finished_req",
        {"Req": SimpleNamespace, "_is_streaming": lambda req: False},
    )
    cache_unfinished = method(
        CACHE / "session_aware_cache.py",
        "SessionAwareCache",
        "cache_unfinished_req",
        {"Req": SimpleNamespace, "_is_streaming": lambda req: False},
    )
    owner = SimpleNamespace(inner=inner)
    req = SimpleNamespace(
        session=None,
        persistent_history_eviction_failed=True,
        c2kv_kv_memory_hint={
            "persistent_history_session": {"enabled": True, "session_id": "closed"}
        },
    )
    cache_unfinished(owner, req, chunked=True)
    cache_finished(owner, req, is_insert=False, reason="abort")
    assert unfinished == [(req, {"chunked": True})]
    assert finished == [(req, {"is_insert": False, "reason": "abort"})]


def test_chunked_selection_scores_accumulate_before_single_eviction():
    merge = method(
        ROOT/'python/sglang/srt/managers/scheduler_output_processor_mixin.py',
        'SchedulerOutputProcessorMixin',
        '_accumulate_history_kv_selection_scores',
        {'Req': SimpleNamespace},
    )
    req = SimpleNamespace(req_pool_idx=3, rid='r',
                          history_kv_selection_scores=None)
    first = SimpleNamespace(history_kv_selection_scores={3: {
        'method': 'snapkv_persistent', 'history_start': 1,
        'history_end': 3, 'query_tokens': 2,
        'layers': [torch.tensor([1., 2.]), torch.tensor([3., 4.])],
    }})
    second = SimpleNamespace(history_kv_selection_scores={3: {
        'method': 'snapkv_persistent', 'history_start': 1,
        'history_end': 3, 'query_tokens': 1,
        'layers': [torch.tensor([10., 20.]), torch.tensor([30., 40.])],
    }})
    merge(None, req, first)
    merge(None, req, second)
    assert req.history_kv_selection_scores['query_tokens'] == 3
    torch.testing.assert_close(
        req.history_kv_selection_scores['layers'][0], torch.tensor([11., 22.])
    )
    torch.testing.assert_close(
        req.history_kv_selection_scores['layers'][1], torch.tensor([33., 44.])
    )


@pytest.mark.parametrize("method_name", ["h2o", "snapkv_persistent", "pyramidkv"])
@pytest.mark.parametrize("layout", ["token", "ascend_page", "ascend_fia"])
def test_attention_eviction_scores_cached_resident_keys_not_full_history(method_name, layout):
    collect=method(ROOT/'python/sglang/srt/models/qwen3.py', 'Qwen3Attention',
        '_collect_history_kv_eviction_scores', {'torch':torch, 'ForwardBatch':SimpleNamespace})
    keys=torch.full((32,1,1),1000.)  # nonresident pages must not enter scoring
    keys[[4,8,9,13],0,0]=torch.tensor([0.,1.,4.,7.])
    if layout == "ascend_page":
        keys = keys.reshape(8,4,1,1)
    elif layout == "ascend_fia":
        keys = keys.reshape(32,1,1,1)
    config={'method':method_name,'history_start':1,'history_end':4,
            'resident_logical_positions':[0,1,4,7,8]}
    fb=SimpleNamespace(c2kv_history_kv_eviction_configs=[config],
        forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda:True),
        extend_seq_lens_cpu=[3],extend_prefix_lens_cpu=[4],req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[4,8,9,13,14,15,16]])),
        token_to_kv_pool=SimpleNamespace(_get_key_buffer=lambda _:keys))
    self=SimpleNamespace(num_heads=1,num_kv_heads=1,head_dim=1,scaling=1.,attn=SimpleNamespace(layer_id=0))
    collect(self,torch.ones(3,1),torch.tensor([[8.],[9.],[10.]]),torch.tensor([8,9,10]),fb)
    scores=fb.c2kv_history_kv_selection_scores[0]['layers'][0]
    assert scores.numel()==3 and scores.argmax().item()==2
    # All current-tail queries contribute. Each query is normalized over every
    # causally visible key, including its own current key, before the history
    # candidate span is sliced out.
    all_keys = torch.tensor([0., 1., 4., 7., 8., 9., 10.])
    expected = torch.stack([
        torch.softmax(all_keys[:5], 0)[1:4],
        torch.softmax(all_keys[:6], 0)[1:4],
        torch.softmax(all_keys[:7], 0)[1:4],
    ]).sum(0)
    torch.testing.assert_close(scores, expected)


def test_missing_persistent_slot_never_falls_back_to_reprefill():
    match = method(CACHE/'session_aware_cache.py', 'SessionAwareCache', 'match_prefix', {
        'MatchPrefixParams': object, 'MatchResult': object, '_is_streaming': lambda _: True})
    req = SimpleNamespace(session=SimpleNamespace(session_id='lost'),
                          history_kv_eviction={'persistent_continuation':True},c2kv_kv_memory_hint={})
    self = SimpleNamespace(slots={})
    with pytest.raises(RuntimeError,match='RESIDENT_CACHE_MISSING'):
        match(self,SimpleNamespace(req=req))


def test_h2o_accumulates_scores_only_for_resident_positions_across_requests():
    select=method(ROOT/'python/sglang/srt/managers/scheduler.py', 'Scheduler',
                  '_select_history_kv_eviction_indices', {'torch':torch,'Optional':Optional})
    config={'method':'h2o','persistent_session':True,'history_start':1,'history_end':4,'target_tokens':1}
    req=SimpleNamespace(history_kv_eviction=config,history_kv_resident_positions=[0,1,4,7,8],
        history_kv_selection_scores={'layers':[torch.tensor([.1,.2,.7])]},
        history_kv_score_state={0:{3:1000.,4:10.}})
    assert select(None,req,config)==[1]  # canonical position 4 retains old score
    assert 3 not in req.history_kv_score_state[0]  # evicted position is not a candidate
    req.history_kv_resident_positions=[0,4,8,9]
    config['history_end']=3
    req.history_kv_selection_scores={'layers':[torch.tensor([.2,.9])]}
    assert select(None,req,config)==[0]  # old resident 4, not resurrected old 7
    assert set(req.history_kv_score_state[0])=={4,8}


def test_snapkv_pooling_uses_canonical_positions_across_persistent_gaps():
    select = method(
        ROOT / "python/sglang/srt/managers/scheduler.py",
        "Scheduler",
        "_select_history_kv_eviction_indices",
        {
            "torch": torch,
            "Optional": Optional,
            "pool_snapkv_scores_by_position": selection.pool_snapkv_scores_by_position,
        },
    )
    config = {
        "method": "snapkv_persistent",
        "persistent_session": True,
        "history_start": 0,
        "history_end": 7,
        "target_tokens": 4,
        "history_kv_recent_window": 2,
        "history_kv_kernel_size": 3,
        "history_kv_pooling": "avgpool",
    }
    req = SimpleNamespace(
        history_kv_eviction=config,
        history_kv_resident_positions=[0, 1, 100, 101, 102, 200, 201],
        history_kv_selection_scores={
            "layers": [torch.tensor([8.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])]
        },
    )

    assert select(None, req, config) == [2, 3, 5, 6]


@pytest.mark.parametrize("method_name", ["h2o", "snapkv_persistent"])
@pytest.mark.parametrize("layout", ["token", "ascend_page", "ascend_fia"])
@pytest.mark.parametrize("query_groups", [1, 2])
@pytest.mark.parametrize("cached_current", [False, True])
def test_history_scores_include_visible_current_keys_before_selection(
    method_name, layout, query_groups, cached_current
):
    collect = method(
        ROOT / "python/sglang/srt/models/qwen3.py", "Qwen3Attention",
        "_collect_history_kv_eviction_scores",
        {"torch": torch, "ForwardBatch": SimpleNamespace},
    )
    select = method(
        ROOT / "python/sglang/srt/managers/scheduler.py", "Scheduler",
        "_select_history_kv_eviction_indices",
        {"torch": torch, "Optional": Optional},
    )
    # Head 0 prefers old token 0, but attends mostly to the current content.
    # Head 1 prefers old token 1 and still attends to history. Renormalizing
    # over history alone incorrectly gives head 0 enough weight to win.
    history_keys = torch.tensor([[2., 0.], [0., 1.], [-10., -10.]])
    current_key = torch.tensor([[10., -10.]])
    prefix_keys = torch.cat([history_keys, current_key]) if cached_current else history_keys
    prefix_len = len(prefix_keys)
    # One new query isolates denominator semantics from the persistent-tail
    # query-window policy covered above.
    new_keys = current_key
    positions = torch.tensor([0, 4, 7, 8, 9])[:prefix_len + 1]
    slots = torch.tensor([4, 8, 9, 13])[:prefix_len]
    keys = torch.full((32, 2, 1), 1000.)  # Nonresident KV must remain invisible.
    keys[slots] = prefix_keys.unsqueeze(-1)
    if layout == "ascend_page":
        keys = keys.reshape(8, 4, 2, 1)
    elif layout == "ascend_fia":
        keys = keys.reshape(32, 1, 2, 1)
    config = {
        "method": method_name, "history_start": 0, "history_end": 3,
        "history_kv_recent_window": 1, "history_kv_kernel_size": 1,
        "history_kv_h2o_recent_fraction": 0.5, "target_tokens": 2,
        "persistent_session": True, "resident_logical_positions": positions.tolist(),
    }
    fb = SimpleNamespace(
        c2kv_history_kv_eviction_configs=[config],
        forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda: True),
        extend_seq_lens_cpu=[1], extend_prefix_lens_cpu=[prefix_len],
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=slots.unsqueeze(0)),
        token_to_kv_pool=SimpleNamespace(_get_key_buffer=lambda _: keys),
    )
    attention = SimpleNamespace(
        num_heads=2 * query_groups, num_kv_heads=2, head_dim=1,
        scaling=1., attn=SimpleNamespace(layer_id=0),
    )
    query = torch.ones(1, attention.num_heads)
    collect(attention, query, new_keys, positions[prefix_len:], fb)
    score_info = fb.c2kv_history_kv_selection_scores[0]
    scores = score_info["layers"][0]

    # Independent dense causal attention reference: normalize over every
    # visible key, then slice the history candidates. Keep the query policy.
    all_keys = torch.cat([prefix_keys, new_keys]).T.repeat_interleave(query_groups, dim=0)
    logits = query[0, :, None] * all_keys
    logits[:, positions > positions[prefix_len]] = -torch.inf
    expected = torch.softmax(logits, dim=-1)[:, :3].sum(dim=0)
    req = SimpleNamespace(
        history_kv_eviction=config, history_kv_resident_positions=positions.tolist(),
        history_kv_selection_scores=score_info, history_kv_score_state={},
    )
    # Current keys participate in normalization, never in history selection.
    assert select(None, req, config) == [1, 2]
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("prefix_len", [0, 2])
def test_history_scores_preserve_prefill_window_and_causality(prefix_len):
    collect = method(
        ROOT / "python/sglang/srt/models/qwen3.py", "Qwen3Attention",
        "_collect_history_kv_eviction_scores",
        {"torch": torch, "ForwardBatch": SimpleNamespace},
    )
    keys = torch.tensor([0., 1., 2., 3., 1000., 1000.]).view(6, 1)
    cache = torch.full((16, 1, 1), 2000.)
    slots = torch.tensor([4, 8])[:prefix_len]
    cache[slots] = keys[:prefix_len].unsqueeze(-1)
    fb = SimpleNamespace(
        c2kv_history_kv_eviction_configs=[{
            "method": "snapkv_persistent", "history_start": 1,
            "history_end": 4, "history_kv_recent_window": 2,
        }],
        forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda: True),
        extend_seq_lens_cpu=[6 - prefix_len], extend_prefix_lens_cpu=[prefix_len],
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=slots.unsqueeze(0)),
        token_to_kv_pool=SimpleNamespace(_get_key_buffer=lambda _: cache),
    )
    attention = SimpleNamespace(
        num_heads=1, num_kv_heads=1, head_dim=1, scaling=1.,
        attn=SimpleNamespace(layer_id=0),
    )
    collect(attention, torch.ones(6 - prefix_len, 1), keys[prefix_len:],
            torch.arange(prefix_len, 6), fb)
    # The history observation window remains queries 2 and 3; the current
    # suffix and each query's future history keys must stay causally masked.
    logits = keys.T.expand(2, -1).clone()
    logits[torch.arange(6)[None, :] > torch.tensor([2, 3])[:, None]] = -torch.inf
    expected = torch.softmax(logits, dim=-1)[:, 1:4].sum(dim=0)
    torch.testing.assert_close(fb.c2kv_history_kv_selection_scores[0]["layers"][0], expected)


def test_tool_h2o_scores_every_prefill_query_across_chunks():
    collect = method(
        ROOT / "python/sglang/srt/models/qwen3.py", "Qwen3Attention",
        "_collect_history_kv_eviction_scores",
        {"torch": torch, "ForwardBatch": SimpleNamespace},
    )
    merge = method(
        ROOT / "python/sglang/srt/managers/scheduler_output_processor_mixin.py",
        "SchedulerOutputProcessorMixin", "_accumulate_history_kv_selection_scores",
        {"Req": SimpleNamespace},
    )
    config = {
        "method": "h2o", "tool_kv_eviction": True,
        "history_start": 0, "history_end": 5,
        "selection_query_start": 0, "selection_query_end": 5,
    }
    cache = torch.zeros(16, 1, 1)
    cache[[4, 8, 9], 0, 0] = torch.tensor([0., 1., 2.])
    attention = SimpleNamespace(
        num_heads=1, num_kv_heads=1, head_dim=1, scaling=1.,
        attn=SimpleNamespace(layer_id=0),
    )
    req = SimpleNamespace(req_pool_idx=0, rid="tool", history_kv_selection_scores=None)
    for prefix_len, new_keys in ((0, [0., 1., 2.]), (3, [3., 4.])):
        fb = SimpleNamespace(
            c2kv_history_kv_eviction_configs=[config],
            forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda: True),
            extend_seq_lens_cpu=[len(new_keys)], extend_prefix_lens_cpu=[prefix_len],
            req_pool_indices=torch.tensor([0]),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[4, 8, 9]])),
            token_to_kv_pool=SimpleNamespace(_get_key_buffer=lambda _: cache),
        )
        collect(attention, torch.ones(len(new_keys), 1),
                torch.tensor(new_keys).view(-1, 1),
                torch.arange(prefix_len, prefix_len + len(new_keys)), fb)
        merge(None, req, SimpleNamespace(history_kv_selection_scores=fb.c2kv_history_kv_selection_scores))
    expected = torch.zeros(5)
    for query_position in range(5):
        expected[:query_position + 1] += torch.softmax(
            torch.arange(query_position + 1, dtype=torch.float32), dim=0
        )
    torch.testing.assert_close(req.history_kv_selection_scores["headwise_layers"][0][0], expected)
    assert req.history_kv_selection_scores["query_tokens"] == 5
