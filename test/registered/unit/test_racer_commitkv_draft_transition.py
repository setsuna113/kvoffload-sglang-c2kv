"""CommitKV lifecycle state that a RACER draft actually protects.

SZ BFCL Long b256 (racer_v2 CommitKV c1_v2_verified): every draft that
brought a new tool result protected 27-32 pending tokens, while admission
assumed the 0 left open by the previous request.  Box4 ToolSandbox b256: an
initial S0 source was the just-committed generation, whose decoded prefix a
decode checkpoint had already moved into the reference state.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python" / "sglang" / "srt"


def _load(relative):
    """Import one engine module; without the serving dependencies, load its file.

    The fallback registers the file under its package name, so the lazy
    in-function imports of production code resolve to the same module.
    """
    name = "sglang.srt." + relative[:-3].replace("/", ".")
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    importlib.import_module("sglang.srt.observability.paper_telemetry")
except ImportError:
    sys.modules.setdefault("sglang.srt.observability", SimpleNamespace(
        paper_telemetry=SimpleNamespace(sample=lambda *args, **kwargs: None)))
commitkv = _load("mem_cache/commitkv.py")
reference = _load("mem_cache/history_kv_reference.py")
racer_transaction = _load("mem_cache/racer_transaction.py")
_load("mem_cache/history_kv_eviction.py")
_load("mem_cache/c2kv_composition.py")
CommitKVConfig, CommitKVRuntimeState = commitkv.CommitKVConfig, commitkv.CommitKVRuntimeState
partition_event_span = commitkv.partition_event_span
CommitKVServingState = reference.CommitKVServingState
ReferenceHistoryKVState, ReferenceLayerKV = reference.ReferenceHistoryKVState, reference.ReferenceLayerKV
commitkv_event_transition = reference.commitkv_event_transition

SCHEDULER = SRT / "managers" / "scheduler.py"


def _method(name, **names):
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    node = next(
        item
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler"
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {
        "List": List,
        "math": math,
        "torch": torch,
        "Req": object,
        "paper_telemetry": SimpleNamespace(sample=lambda *args, **kwargs: None),
        **names,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCHEDULER), "exec"), namespace)
    return namespace[name]


class _Window:
    """A full W-query pre window whose page effects are fixed per page start."""

    def __init__(self, key_positions, effects, queries=8):
        self.key_positions = torch.tensor(key_positions, dtype=torch.long)
        self.query_positions = torch.arange(queries, dtype=torch.long)
        self._effects = effects
        self._local = {int(p): i for i, p in enumerate(key_positions)}

    def effect(self, indices):
        start = int(self.key_positions[min(indices)].item())
        return torch.tensor(self._effects.get(start, 0.0))


def _span(index, role, phase, start, end):
    return {"message_index": index, "role": role, "phase": phase, "start": start, "end": end}


HISTORY = [
    _span(0, "user", "others", 0, 10),
    _span(1, "assistant", "act", 10, 30),
    _span(2, "tool", "tool", 30, 60),
]


def _state_after_decode(*, queries=8, target_tokens=256):
    """A committed decision: events configured, then a decode left its pre window."""
    config = CommitKVConfig(measurement_layer_id=0, page_size=16)
    state = CommitKVServingState(policy=CommitKVRuntimeState(config), target_tokens=target_tokens)
    state.configure_events(HISTORY)
    assert state.policy.pending is None  # no pre window yet: nothing to open
    pages = (*partition_event_span(1, 10, 30, page_size=16),
             *partition_event_span(2, 30, 60, page_size=16))
    state.pre_pages = pages
    # Page effects rank (2,46) > (1,10) > (2,30) > (1,26); the B/8 = 32-token
    # pending budget takes the first two whole pages (14 + 16 tokens).
    state.pre_window = _Window(list(range(0, 90)), {46: 0.9, 10: 0.8, 30: 0.7, 26: 0.1},
                               queries=queries)
    return state


def test_projection_is_what_the_next_tool_event_protects_and_changes_nothing():
    state = _state_after_decode()
    receipts = list(state.receipts)
    transition = racer_transaction.commitkv_next_transition(state)
    assert transition["event_message_indices"] == [0, 1, 2]
    assert transition["tool_event"] == {
        "tokens": 30,
        "positions": list(range(10, 26)) + list(range(46, 60)),
        "source_message_indices": [1, 2],
    }
    assert state.policy.pending is None and state.receipts == receipts

    draft = HISTORY + [_span(3, "assistant", "act", 60, 70), _span(4, "tool", "tool", 70, 90)]
    assert commitkv_event_transition(transition["event_message_indices"], draft) == "tool_event"
    predicted = racer_transaction.draft_pending_positions(transition, draft, carried_positions=[])
    state.configure_events(draft)
    assert racer_transaction.protected_pending_positions(state) == predicted
    assert len(predicted) == 30


def test_new_events_without_a_tool_close_the_window_and_unchanged_events_keep_it():
    state = _state_after_decode()
    open_window = HISTORY + [_span(3, "assistant", "act", 60, 70), _span(4, "tool", "tool", 70, 90)]
    state.configure_events(open_window)
    carried = racer_transaction.protected_pending_positions(state)
    assert carried
    transition = racer_transaction.commitkv_next_transition(state)

    assert racer_transaction.draft_pending_positions(transition, open_window, carried) == carried
    turn_end = open_window + [_span(5, "assistant", "others", 90, 95), _span(6, "user", "others", 95, 99)]
    assert commitkv_event_transition(transition["event_message_indices"], turn_end) == "new_events"
    assert racer_transaction.draft_pending_positions(transition, turn_end, carried) == []
    state.configure_events(turn_end)
    assert racer_transaction.protected_pending_positions(state) == []


def test_short_pre_window_opens_nothing_and_other_methods_keep_the_old_value():
    state = _state_after_decode(queries=5)
    transition = racer_transaction.commitkv_next_transition(state)
    assert transition["tool_event"] == {"tokens": 0, "positions": [], "source_message_indices": []}
    draft = HISTORY + [_span(3, "assistant", "act", 60, 70), _span(4, "tool", "tool", 70, 90)]
    assert racer_transaction.draft_pending_positions(transition, draft, [7]) == []
    state.configure_events(draft)
    assert state.policy.pending is None

    assert racer_transaction.commitkv_next_transition(None) is None
    assert racer_transaction.commitkv_next_transition(SimpleNamespace(rows=[])) is None
    assert racer_transaction.draft_pending_positions(None, draft, [7, 8]) == [7, 8]


def test_admission_rejects_the_draft_pending_window_before_selection():
    """SZ s0 task 2 turn-3/step-3: 231 evidence tokens leave 25 of B=256."""
    transition = racer_transaction.commitkv_next_transition(_state_after_decode())
    draft = HISTORY + [_span(3, "assistant", "act", 60, 70), _span(4, "tool", "tool", 70, 90)]
    pending = racer_transaction.draft_pending_positions(transition, draft, carried_positions=[])
    hint = {
        "persistent_history_session": {
            "enabled": True, "history_budget_tokens": 256,
            "transaction": {"decision_id": "turn-3/step-3", "phase": "draft", "resolution": "commit"},
        },
        "history_kv_eviction": {"method": "commitkv", "history_start": 0, "history_end": 90,
                                "target_tokens": 256},
        "history_kv_reference_config": {"method": "commitkv", "target_tokens": 256},
        "racer_active_ephemeral_source_spans": [[100, 331]],
    }
    with pytest.raises(racer_transaction.RacerCapacityInfeasible) as failure:
        racer_transaction.enforce_request_budget(hint, pending_tokens=len(pending))
    assert failure.value.as_error()["capacity"]["stage"] == "draft"
    assert failure.value.as_error()["capacity"]["required_tokens"] == 231 + 30


def _commitkv_builder(normal_positions, *, existing_positions=(), target=8, capacity=None):
    count = len(normal_positions)
    keys = torch.arange(200, dtype=torch.float32).view(200, 1, 1)
    pool = SimpleNamespace(start_layer=0, layer_num=1,
                           get_kv_buffer=lambda layer: (keys, keys + 1000),
                           _get_key_buffer=lambda layer: keys,
                           _get_value_buffer=lambda layer: keys + 1000)
    row = torch.zeros((1, 200), dtype=torch.long)
    row[0, :count] = torch.arange(100, 100 + count)
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: pool, page_size=1, available_size=lambda: 1000,
            free=lambda slots: None),
        _bytes_per_kv_token=lambda: 16,
    )
    scheduler._build_commitkv_reference_state = MethodType(
        _method("_build_commitkv_reference_state"), scheduler)
    existing = None
    if existing_positions:
        positions = torch.tensor([list(existing_positions)], dtype=torch.long)
        existing = ReferenceHistoryKVState(method="commitkv", layers={0: ReferenceLayerKV(
            key=positions.to(torch.float32).unsqueeze(-1),
            value=positions.to(torch.float32).unsqueeze(-1) + 1000,
            positions=positions)}, expected_layer_ids=(0,))
    runtime = CommitKVServingState(
        policy=CommitKVRuntimeState(CommitKVConfig(measurement_layer_id=0, checkpoint_interval=4)),
        target_tokens=target)
    req = SimpleNamespace(
        req_pool_idx=0, history_kv_runtime_state=runtime,
        history_kv_reference_state=existing,
        history_kv_reference_config={"method": "commitkv", "target_tokens": target,
                                     "racer_effective_target_tokens": capacity or target},
        history_kv_resident_positions=list(normal_positions), c2kv_kv_memory_hint={},
        kv_memory_report={})
    return scheduler, req


def _positions(state):
    return state.layers[0].positions[0].tolist()


def test_reference_resident_prefix_of_a_committed_source_is_not_a_candidate():
    # A committed act spans [10, 16): 10-13 were moved to the reference by a
    # decode checkpoint, 14-15 are its normal-row tail (excluded as before).
    scheduler, req = _commitkv_builder([14, 15, 16, 17, 18, 19], existing_positions=(2, 3, 10, 11, 12, 13),
                                       capacity=6)
    req.c2kv_kv_memory_hint = {"racer_initial_replaced_source_spans": [[10, 16]]}
    config = {"history_start": 0, "history_end": 6, "racer_excluded_history_indices": [0, 1]}
    state = scheduler._build_commitkv_reference_state(req, config)
    assert _positions(state) == [2, 3, 16, 17, 18, 19]

    req.c2kv_kv_memory_hint["racer_protected_pending_source_positions"] = [11]
    state = scheduler._build_commitkv_reference_state(req, config)
    assert _positions(state) == [3, 11, 16, 17, 18, 19]


def test_selection_that_never_reached_the_source_prefix_is_unchanged():
    scheduler, req = _commitkv_builder([14, 15, 16, 17, 18, 19], existing_positions=(2, 3, 10, 11, 12, 13),
                                       capacity=3)
    config = {"history_start": 0, "history_end": 6, "racer_excluded_history_indices": [0, 1]}
    before = _positions(scheduler._build_commitkv_reference_state(req, dict(config)))
    req.c2kv_kv_memory_hint = {"racer_initial_replaced_source_spans": [[10, 16]]}
    after = _positions(scheduler._build_commitkv_reference_state(req, dict(config)))
    assert before == after == [17, 18, 19]


def test_decode_checkpoint_keeps_the_decoded_tail_despite_prefill_exclusions():
    """ToolSandbox step 2 kept decoded tokens 0-53 and dropped 54-127 of the tail."""
    scheduler, req = _commitkv_builder([0, 1], capacity=3)
    checkpoint = MethodType(_method("_apply_reference_decode_checkpoint"), scheduler)
    # The prefill excluded history indices 1 and 2 of its own range.
    req.history_kv_eviction = {"racer_excluded_history_indices": [1, 2]}
    req.history_kv_resident_positions = [0, 1]
    req.c2kv_kv_memory_hint = {"racer_initial_replaced_source_spans": [[0, 1]]}
    for name, value in dict(persistent_decode_cache_locs=[], decode_batch_idx=4,
                            reference_decode_protected_len=2, reference_decode_logical_start=2,
                            kv_committed_len=6, kv_allocated_len=6, already_computed=6,
                            c2kv_position_correction=0).items():
        setattr(req, name, value)
    assert checkpoint(req) == -4
    assert _positions(req.history_kv_reference_state) == [3, 4, 5]


def _draft_hint(spans, *, resolution="commit", evidence=()):
    persistent = {"enabled": True, "transaction": {
        "decision_id": "turn-1/step-2", "phase": "draft", "resolution": resolution}}
    if evidence:
        persistent["initial_s0_append"] = {"enabled": True, "evidence_message_indices": list(evidence),
                                           "source_message_indices": [1]}
    return {"persistent_history_session": persistent, "history_kv_event_token_spans": spans}


def test_resumed_draft_reads_the_receipt_of_the_state_it_resumes():
    transition = racer_transaction.commitkv_next_transition(_state_after_decode())
    closed = {"event_message_indices": [0, 1, 2],
              "tool_event": {"tokens": 0, "positions": [], "source_message_indices": []}}
    held = {"current_protected_pending_positions": [], "protected_pending_positions": [55],
            "current_commitkv_next_transition": transition, "commitkv_next_transition": closed}
    draft = HISTORY + [_span(3, "assistant", "act", 60, 70), _span(4, "tool", "tool", 70, 90)]
    resolve = racer_transaction.resumed_draft_pending_positions
    assert resolve(_draft_hint(draft), held) == transition["tool_event"]["positions"]
    assert resolve(_draft_hint(draft, resolution="discard"), held) == []
    # Only S0 evidence rows are new: CommitKV never sees them, the window stays.
    evidence_only = HISTORY + [_span(3, "assistant", "others", 60, 61), _span(4, "user", "others", 61, 90)]
    assert resolve(_draft_hint(evidence_only, resolution="discard", evidence=(3, 4)), held) == [55]
    assert resolve(_draft_hint(evidence_only, resolution="discard"), held) == []

    regeneration = _draft_hint(draft)
    regeneration["persistent_history_session"]["transaction"]["phase"] = "regenerate"
    assert resolve(regeneration, held) is None
    assert resolve(_draft_hint(draft), {}) is None
    # Other methods and older receipts keep the value admission used before.
    legacy = {"current_protected_pending_positions": [5, 6]}
    assert resolve(_draft_hint(draft), legacy) == [5, 6]


def test_commit_records_both_next_transition_receipts():
    serving = ROOT / "python" / "sglang" / "srt" / "entrypoints" / "openai" / "serving_chat.py"
    tree = ast.parse(serving.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingChat")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_commit_persistent_history_session"]
    namespace = {"GenerateReqInput": object, "List": list, "Dict": dict, "Any": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(serving), "exec"), namespace)
    chat = namespace["OpenAIServingChat"]()
    for name in ("_persistent_history_tool_segments", "_persistent_history_generation_bases",
                 "_persistent_history_sessions", "_persistent_history_computed_prefixes",
                 "_persistent_history_exact_output", "_persistent_history_generation_prefixes"):
        setattr(chat, name, {})
    held_transition = {"event_message_indices": [0], "tool_event": {"tokens": 1, "positions": [3],
                                                                   "source_message_indices": [0]}}
    current_transition = {"event_message_indices": [0, 1], "tool_event": {"tokens": 2, "positions": [4, 5],
                                                                          "source_message_indices": [1]}}
    request = SimpleNamespace(
        _persistent_history_session_id="session",
        _persistent_history_canonical_prompt_ids=list(range(12)),
        _persistent_history_generation_prefix_ids=None,
        c2kv_kv_memory_hint={"history_kv_reference_config": {"method": "commitkv"},
                             "persistent_session_canonical_prompt_tokens": 12,
                             **_draft_hint([])})
    report = {"persistent_session_computed_logical_horizon": 13,
              "racer_transaction": {"commitkv_next_transition": held_transition},
              "racer_current_commitkv_next_transition": current_transition}
    chat._persistent_history_requests = {("session", id(request)): request}
    chat._commit_persistent_history_session(
        request, [{"output_ids": [50, 51], "text": "ok", "meta_info": {"kv_memory_report": report}}])
    held = chat._persistent_history_transactions["session"]
    assert held["commitkv_next_transition"] == held_transition
    assert held["current_commitkv_next_transition"] == current_transition
