"""Resolve native protection against the tokenized canonical chat, without prefill."""
from __future__ import annotations

from sglang.srt.mem_cache.c2kv_composition import source_boundary


def _decoded_offsets(tokenizer, ids):
    """Use fast offsets only when they describe the exact existing token IDs."""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == ids:
            return text, list(encoded["offset_mapping"])
    except (TypeError, ValueError, NotImplementedError, KeyError):
        pass
    # Some templates have boundary tokens that cannot be re-encoded in isolation.
    # Only stable decoded prefixes establish boundaries (byte fallback may emit
    # incomplete Unicode). Unstable prefixes share the surrounding full span.
    stable = [(0, 0)]
    for count in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:count], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        if text.startswith(prefix):
            stable.append((count, len(prefix)))
    offsets = [(0, len(text)) for _ in ids]
    for (left, start), (right, end) in zip(stable, stable[1:]):
        for index in range(left, right):
            offsets[index] = (start, end)
    return text, offsets


def resolve_native_protection(hint, tokenizer, prompt_ids):
    """Run before internal recovery spans are filtered from native event metadata.

    Wire instances use message indices after carrier removal. Output spans use
    expanded canonical source coordinates, as do resident KV position ledgers.
    An ambiguous/unmappable instance is omitted independently of other aliases.
    """
    for key in ("racer_native_protection_units", "racer_native_protection_events"):
        hint.pop(key, None)
    plan = (hint.get("persistent_history_session") or {}).get("extra_protection") or {}
    if not isinstance(plan, dict) or plan.get("schema") != "racer-native-protection-v2" or plan.get("enabled") is not True:
        return
    spans = {item["message_index"]: (int(item["start"]), int(item["end"]))
             for item in hint.get("history_kv_event_token_spans") or []}
    segments = hint.get("tool_memory_segments") or []
    decoded = {}

    def fragment_span(fragment):
        index = fragment["message_index"]
        start, end = spans[index]
        if not 0 <= start < end <= len(prompt_ids):
            raise ValueError("Invalid message span")
        if index not in decoded:
            decoded[index] = _decoded_offsets(tokenizer, prompt_ids[start:end])
        text, offsets = decoded[index]
        needle = fragment["text"]
        if not isinstance(needle, str) or not needle:
            raise ValueError("Empty source fragment")
        context = fragment.get("context", text)
        if not isinstance(context, str) or not context or text.count(context) != 1:
            raise ValueError("Ambiguous source context")
        if context.count(needle) != 1:
            raise ValueError("Ambiguous source fragment")
        left = text.index(context) + context.index(needle)
        right = left + len(needle)
        indices = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
        if not indices:
            raise ValueError("Unmapped source fragment")
        lo, hi = min(indices), max(indices) + 1
        if offsets[lo][0] > left or offsets[hi - 1][1] < right:
            raise ValueError("Incomplete source fragment")
        return start + lo, start + hi

    def instance_spans(instance):
        fragments = instance.get("fragments") or []
        rendered = ([fragment_span(fragment) for fragment in fragments] if fragments else
                    [spans[index] for index in instance["source_message_indices"]])
        if not rendered or any(not 0 <= a < b <= len(prompt_ids) for a, b in rendered):
            raise ValueError("Empty source instance")
        return [[source_boundary(a, segments), source_boundary(b, segments)] for a, b in rendered]

    for key, output in (("units", "racer_native_protection_units"),
                        ("events", "racer_native_protection_events")):
        resolved = []
        for item in plan.get(key) or []:
            instances = []
            for instance in item.get("instances") or []:
                try:
                    ranges = instance_spans(instance)
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                instances.append({"kind": instance["kind"], "spans": ranges,
                                  "complete_event": instance.get("complete_event", False)})
            resolved.append({**{field: item[field] for field in ("unit_id", "event_id", "complete_event")
                                if field in item}, "instances": instances,
                             "status": "resolved" if instances else "source_span_unavailable"})
        hint[output] = resolved
