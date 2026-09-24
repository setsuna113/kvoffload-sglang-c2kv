"""CPU-only exact text/coordinate binding for original and recovered instances."""
import importlib.util
import sys
from pathlib import Path

import pytest

SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, SRT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    composition = load("v2_composition", "mem_cache/c2kv_composition.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    resolver = load("v2_spans", "mem_cache/native_protection_spans.py")
    return composition, resolver


class Characters:
    def decode(self, ids, **kwargs):
        return "".join(chr(index) for index in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def plan(text):
    return {"history_kv_event_token_spans": [{"message_index": 2, "start": 3, "end": 3 + len(text)}],
            "persistent_history_session": {"extra_protection": {
                "schema": "racer-native-protection-v2", "enabled": True,
                "units": [{"unit_id": "u", "event_id": "e", "complete_event": False,
                           "instances": [{"kind": "original", "source_message_indices": [1], "fragments": []},
                                         {"kind": "recovery", "source_message_indices": [2],
                                          "fragments": [{"message_index": 2, "text": "Alpha"}]}]}],
                "events": [{"event_id": "e", "instances": [{"kind": "recovery", "complete_event": True,
                            "source_message_indices": [2], "fragments": [{"message_index": 2, "text": text}]}]}]}}}


def test_missing_original_does_not_erase_recovered_alias_and_expands_tool_coordinates(modules):
    _, resolver = modules
    text = 'Evidence: {"name":"Alpha","value":7}'
    hint = plan(text)
    hint["tool_memory_segments"] = [{"token_start": 0, "token_end": 1, "source_tokens": 10}]
    resolver.resolve_native_protection(hint, Characters(), list(map(ord, "###" + text)))
    instance, = hint["racer_native_protection_units"][0]["instances"]
    start = 3 + 9 + text.index("Alpha")
    assert instance["spans"] == [[start, start + 5]]
    assert instance["kind"] == "recovery"
    assert hint["racer_native_protection_events"][0]["instances"][0]["complete_event"] is True


def test_fragment_must_be_unique_within_exact_event_context(modules):
    _, resolver = modules
    text = 'Alpha elsewhere {"name":"Alpha","value":7}'
    hint = plan(text)
    ids = list(map(ord, "###" + text))
    resolver.resolve_native_protection(hint, Characters(), ids)
    assert hint["racer_native_protection_units"][0]["status"] == "source_span_unavailable"
    fragment = hint["persistent_history_session"]["extra_protection"]["units"][0]["instances"][1]["fragments"][0]
    fragment["context"] = '{"name":"Alpha","value":7}'
    resolver.resolve_native_protection(hint, Characters(), ids)
    start = 3 + text.rindex("Alpha")
    assert hint["racer_native_protection_units"][0]["instances"][0]["spans"] == [[start, start + len("Alpha")]]


def test_byte_fallback_keeps_complete_unicode_token_group(modules):
    _, resolver = modules

    class Bytes:
        def decode(self, ids, **kwargs):
            return bytes(ids).decode("utf8", errors="replace")

    text, offsets = resolver._decoded_offsets(Bytes(), list("A中B".encode()))
    assert text == "A中B"
    assert offsets == [(0, 1), (1, 2), (1, 2), (1, 2), (2, 3)]


def test_carrier_remap_covers_nested_original_and_evidence_fragments(modules):
    composition, _ = modules
    hint = plan("Alpha")
    composition.remap_message_metadata(hint, [0], 3)
    unit = hint["persistent_history_session"]["extra_protection"]["units"][0]
    assert unit["instances"][0]["source_message_indices"] == [0]
    assert unit["instances"][1]["source_message_indices"] == [1]
    assert unit["instances"][1]["fragments"][0]["message_index"] == 1


def test_off_removes_caller_supplied_resolved_spans(modules):
    _, resolver = modules
    hint = {"racer_native_protection_units": ["untrusted"]}
    resolver.resolve_native_protection(hint, Characters(), [])
    assert "racer_native_protection_units" not in hint


def test_serving_resolves_after_generated_bpe_and_before_internal_filter(modules, monkeypatch):
    from test_racer_transaction import _drifted_recovery_owner

    _, resolver = modules
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.native_protection_spans", resolver)
    owner, request = _drifted_recovery_owner()
    owner._persistent_history_transactions["s"]["internal_source_spans"] = [[1, 4]]
    hint = request.c2kv_kv_memory_hint
    hint["persistent_history_session"]["extra_protection"] = {
        "schema": "racer-native-protection-v2", "enabled": True,
        "units": [{"unit_id": "u", "event_id": "e", "complete_event": True,
                   "instances": [{"kind": "recovery", "source_message_indices": [1], "fragments": []}]}],
        "events": []}
    owner._prepare_persistent_history_delta(request, [1, 200, 2, 3, 5, 6, 1])
    assert hint["racer_native_protection_units"][0]["instances"][0]["spans"] == [[1, 4]]
    assert all(event["message_index"] != 1 for event in hint["history_kv_event_token_spans"])
