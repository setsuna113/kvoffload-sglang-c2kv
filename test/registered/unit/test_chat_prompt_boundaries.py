"""CPU regressions for structured assistant tails and grouped tool results."""
import ast
import copy
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (Path(__file__).resolve().parents[3]
          / "python/sglang/srt/entrypoints/openai/serving_chat.py")


def serving_method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                and cls.name == "OpenAIServingChat" for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
        names=[ast.alias(name="annotations")]), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("content", ["", "Calling the tool."])
def test_completed_assistant_tool_call_remains_history(content):
    messages = [{"role": "user", "content": "Find it"},
                {"role": "assistant", "content": content, "tool_calls": [
                    {"id": "call1", "function": {"name": "lookup", "arguments": {"key": "x"}}}]}]
    original = copy.deepcopy(messages)
    changed, prefix = serving_method("_handle_last_assistant_message")(
        None, messages, SimpleNamespace(continue_final_message=False))
    assert changed == original
    assert prefix is None


def test_cancelled_completion_cannot_recreate_closed_session_tracker():
    released = []
    owner = SimpleNamespace(release_persistent_history_session=released.append)
    request = SimpleNamespace(_persistent_history_session_id="s1",
                              _lifecycle_cancel_requested=True,
                              _persistent_history_canonical_prompt_ids=[1, 2])
    serving_method("_commit_persistent_history_session")(
        owner, request, [{"output_ids": [3]}])
    assert released == ["s1"]


def session_owner():
    owner = SimpleNamespace(_persistent_history_requests=weakref.WeakValueDictionary())
    for name in ("sessions", "generation_prefixes", "generation_bases", "computed_prefixes",
                 "exact_output", "tool_segments", "tool_source_digests"):
        setattr(owner, "_persistent_history_" + name, {})
    return owner


class Request:
    def __init__(self, session="s1"):
        self._persistent_history_session_id = session
        self._persistent_history_canonical_prompt_ids = [1, 2]


def test_completed_response_after_close_cannot_recreate_session():
    owner = session_owner()
    request = Request()
    serving_method("_track_persistent_history_request")(owner, request)
    # The terminal request was already removed from the manager's rid map.
    # Closing the session still invalidates its live response consumer.
    serving_method("release_persistent_history_session")(owner, "s1")
    serving_method("_commit_persistent_history_session")(
        owner, request, [{"output_ids": [3]}])
    assert owner._persistent_history_sessions == {}
    assert not owner._persistent_history_requests


def test_live_request_ownership_is_distinct_and_abandoned_state_is_weak():
    owner = session_owner()
    first, overlap = Request(), Request()
    track = serving_method("_track_persistent_history_request")
    track(owner, first)
    track(owner, overlap)
    assert len(owner._persistent_history_requests) == 2
    del overlap
    gc.collect()
    assert list(owner._persistent_history_requests.values()) == [first]
    serving_method("_commit_persistent_history_session")(
        owner, first, [{"output_ids": [3]}])
    assert owner._persistent_history_sessions["s1"] == [1, 2]
    assert not owner._persistent_history_requests


def test_explicit_continuation_and_legacy_text_tail_stay_unchanged():
    handle = serving_method("_handle_last_assistant_message")
    message = {"role": "assistant", "content": "Partial text"}
    assert handle(None, [message], SimpleNamespace(continue_final_message=True)) == ([], "Partial text")
    assert handle(None, [message], SimpleNamespace(continue_final_message=False)) == (
        [{"role": "user", "content": "Partial text"}], None)


def context(prefix, roles):
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids: {10: "\n", 11: " ", 12: "wrong"}.get(ids[0], "x"))
    obj = SimpleNamespace(tokenizer_manager=SimpleNamespace(tokenizer=tokenizer),
                         _c2kv_chat_template_input_ids=lambda *a: prefix)
    request = SimpleNamespace(messages=[SimpleNamespace(role=r) for r in roles])
    return obj, request


def test_history_boundary_inside_tool_group_excludes_synthetic_eos():
    obj, request = context([1, 2, 3, 99, 10], ["user", "tool", "tool"])
    resolve = serving_method("_c2kv_contextual_prefix_ids")
    assert resolve(obj, request, 2, None, [1, 2, 3, 4, 5, 99, 10]) == [1, 2, 3]


@pytest.mark.parametrize("prefix,roles", [
    ([1, 2, 12, 99, 10], ["user", "tool", "tool"]),
    ([1, 2, 3, 12], ["user", "tool", "tool"]),
    ([1, 2, 3, 99, 10], ["user", "assistant", "tool"]),
    ([1, 2, 3, 99, 10], ["user", "tool", "user"]),
])
def test_contextual_boundary_does_not_accept_changed_history(prefix, roles):
    obj, request = context(prefix, roles)
    assert serving_method("_c2kv_contextual_prefix_ids")(
        obj, request, 2, None, [1, 2, 3, 4, 99, 10]) == prefix
