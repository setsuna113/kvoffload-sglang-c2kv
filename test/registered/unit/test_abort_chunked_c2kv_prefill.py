"""CPU-only tests: an abort reaches a chunked multi-round C2KV prefill.

Between chunks the request lives in ``Scheduler.chunked_req``; before this
fix an abort waited for the whole prefill, so an acknowledged timeout cleanup
returned 504 and the engine stayed busy. The request now finishes at the next
chunk boundary through the mid-prefill C2KV failure path.
"""

import ast
import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"
CACHE = ROOT / "python/sglang/srt/mem_cache"


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
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FINISH_ABORT:
    def __init__(self, message=None):
        self.message = message


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def persistent(enabled=True):
    return {"persistent_history_session": {"enabled": enabled}} if enabled else {}


class Persistent:
    """SessionAwareCache._is_persistent_history_req without importing SGLang."""

    @staticmethod
    def _is_persistent_history_req(req):
        hint = getattr(req, "c2kv_kv_memory_hint", None)
        return bool(isinstance(hint, dict)
                    and isinstance(hint.get("persistent_history_session"), dict)
                    and hint["persistent_history_session"].get("enabled"))


def chunked_req(**overrides):
    values = dict(
        rid="rid-1", c2kv_rounds=[object(), object()], to_finish=None,
        finished_reason=None, kv_committed_len=24, return_logprob=False,
        c2kv_kv_memory_hint=persistent(),
    )
    values.update(overrides)
    req = SimpleNamespace(**values)
    req.finished = lambda: req.finished_reason is not None

    def check_finished():
        if req.finished_reason is None and req.to_finish:
            req.finished_reason, req.to_finish = req.to_finish, None

    req.check_finished = check_finished
    return req


def abort_request_fn():
    return method(SCHEDULER, "Scheduler", "abort_request", {
        "FINISH_ABORT": FINISH_ABORT, "DisaggregationMode": DisaggregationMode,
        "AbortReq": SimpleNamespace, "logger": logging.getLogger(__name__),
        "release_kv_cache": lambda *a, **k: pytest.fail("no queued release"),
        "release_req_to_metadata_buffer": lambda *a, **k: None,
    })


def scheduler_for_abort(chunked):
    running = SimpleNamespace(reqs=[])
    return SimpleNamespace(
        waiting_queue=[], grammar_manager=SimpleNamespace(abort_requests=lambda r: None),
        disaggregation_mode=DisaggregationMode.NULL, running_batch=running,
        cur_batch=None, chunked_req=chunked, enable_hicache_storage=False,
    )


@pytest.mark.parametrize("request_rid,abort_all,marked", [
    ("rid-1", False, True), ("other", False, False), ("", True, True),
])
def test_abort_marks_only_the_matching_chunked_c2kv_request(request_rid, abort_all, marked):
    req = chunked_req()
    abort_request_fn()(scheduler_for_abort(req), SimpleNamespace(rid=request_rid, abort_all=abort_all))
    assert isinstance(req.to_finish, FINISH_ABORT) is marked


def test_abort_leaves_ordinary_chunked_requests_to_upstream_behavior():
    req = chunked_req(c2kv_rounds=None)
    abort_request_fn()(scheduler_for_abort(req), SimpleNamespace(rid="rid-1", abort_all=False))
    assert req.to_finish is None


def abort_chunked_fn(events):
    return method(SCHEDULER, "Scheduler", "_abort_chunked_c2kv_request", {
        "FINISH_ABORT": FINISH_ABORT, "SessionAwareCache": Persistent,
        "logger": logging.getLogger(__name__),
        "release_kv_cache": lambda req, cache, is_insert=True: events.append(
            ("release", req.persistent_history_eviction_failed, is_insert)),
    })


def scheduler_for_finish(events, overlap=False):
    return SimpleNamespace(
        enable_overlap=overlap, tree_cache=object(),
        _release_c2kv_pins=lambda req: events.append(("pins",)),
        stream_output=lambda reqs, logprob: events.append(
            ("stream", [type(r.finished_reason).__name__ for r in reqs])),
    )


def test_aborted_persistent_chunk_rolls_back_releases_and_reports_abort():
    events = []
    req = chunked_req(to_finish=FINISH_ABORT(), persistent_history_eviction_failed=False)
    assert abort_chunked_fn(events)(scheduler_for_finish(events), req) is True
    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert events == [("pins",), ("release", True, False), ("stream", ["FINISH_ABORT"])]


@pytest.mark.parametrize("change", ["not_aborted", "overlap", "not_c2kv"])
def test_chunk_boundary_abort_is_limited_to_aborted_c2kv_without_overlap(change):
    events = []
    req = chunked_req(
        to_finish=None if change == "not_aborted" else FINISH_ABORT(),
        c2kv_rounds=None if change == "not_c2kv" else [object()])
    finished = abort_chunked_fn(events)(scheduler_for_finish(events, overlap=change == "overlap"), req)
    assert finished is False and events == [] and req.finished_reason is None


def test_scheduler_finishes_the_chunked_request_before_it_is_re_added():
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    get_next = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Scheduler"
                    for n in c.body if isinstance(n, ast.FunctionDef)
                    and n.name == "get_next_batch_to_run")
    block = next(n for n in ast.walk(get_next) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.chunked_req is not None")
    lines = [ast.unparse(statement) for statement in block.body]
    assert lines[:2] == ["chunked_req_to_exclude.add(self.chunked_req)",
                         "self.stash_chunked_request(self.chunked_req)"]
    assert lines[2] == ("if self._abort_chunked_c2kv_request(self.chunked_req):\n"
                        "    self.chunked_req = None")


def test_waiting_c2kv_abort_rolls_back_only_persistent_turns():
    events = []
    cleanup = method(SCHEDULER, "Scheduler", "_cleanup_aborted_c2kv_waiting_req", {
        "SessionAwareCache": Persistent,
        "release_kv_cache": lambda req, cache, is_insert=True: events.append(
            (req.rid, getattr(req, "persistent_history_eviction_failed", False), is_insert)),
    })
    self = SimpleNamespace(tree_cache=object(), _release_c2kv_pins=lambda req: None)
    for rid, hint in (("persistent", persistent()), ("ordinary", persistent(False))):
        cleanup(self, SimpleNamespace(rid=rid, c2kv_rounds=[object()], req_pool_idx=0,
                                      kv_committed_freed=False, c2kv_kv_memory_hint=hint))
    assert events == [("persistent", True, False), ("ordinary", False, False)]


def test_aborted_turn_frees_every_page_once_through_rollback_and_close():
    """Real release/rollback/close code: prior turn kept, new chunk freed, no double free."""
    class HybridPool:
        pass

    release = function(CACHE / "common.py", "release_kv_cache", {
        "Req": SimpleNamespace, "BasePrefixCache": object, "HybridReqToTokenPool": HybridPool})
    cache_ns = {"Req": SimpleNamespace, "_is_streaming": lambda r: r.session is not None,
                "torch": torch, "DecLockRefParams": SimpleNamespace,
                "json": __import__("json"), "logging": logging, "Optional": Optional}
    finished = method(CACHE / "session_aware_cache.py", "SessionAwareCache", "cache_finished_req", cache_ns)
    rollback = method(CACHE / "session_aware_cache.py", "SessionAwareCache",
                      "_rollback_failed_persistent_request", cache_ns)
    owns = method(CACHE / "session_aware_cache.py", "SessionAwareCache",
                  "owns_finished_request", cache_ns).__func__
    close = method(CACHE / "session_aware_cache.py", "SessionAwareCache", "release_session", cache_ns)

    freed, free_slots = [], []
    row = torch.zeros(1, 16, dtype=torch.int64)
    row[0, :10] = torch.arange(100, 110)  # 4 resident session tokens + 6 prefilled chunk tokens
    pool = SimpleNamespace(req_to_token=row, free_slots=free_slots)
    slot = SimpleNamespace(req_pool_idx=0, kv_committed_len=4, kv_allocated_len=4,
                           cache_protected_len=0, last_node=None, swa_uuid_for_lock=None,
                           is_holding_kv=True)
    cache = SimpleNamespace(
        slots={"session": slot}, page_size=1, req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda x: freed.extend(x.tolist())),
        inner=SimpleNamespace(cache_finished_req=lambda *a, **k: pytest.fail("rolled back")),
        _is_persistent_history_req=Persistent._is_persistent_history_req, owns_finished_request=owns)
    cache._rollback_failed_persistent_request = lambda req: rollback(cache, req)
    cache.cache_finished_req = lambda req, is_insert=True, **kw: finished(cache, req, is_insert, **kw)

    req = SimpleNamespace(req_pool_idx=0, c2kv_rounds=[object()], kv_allocated_len=10,
                          session=SimpleNamespace(session_id="session"), last_node=None,
                          c2kv_kv_memory_hint=persistent(), persistent_history_eviction_failed=True,
                          kv_memory_report=None)
    release(req, cache, is_insert=False)
    assert sorted(freed) == list(range(104, 110)) and req.req_pool_idx is None
    close(cache, "session", active_req=req)
    assert sorted(freed) == list(range(100, 110)) and free_slots == [0]
