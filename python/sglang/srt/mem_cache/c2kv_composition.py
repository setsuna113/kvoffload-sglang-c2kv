"""CPU-only coordinate and round contracts for tool/history composition."""

from __future__ import annotations


def remap_message_metadata(hint, removed, message_count):
    """Remove carrier rows from boundary counts and event metadata together."""
    removed = set(removed)
    recovery = (hint.get("persistent_history_session") or {}).get("recovery_append")
    if isinstance(recovery, dict) and "source_message_indices" in recovery:
        indices = list(recovery["source_message_indices"])
        if any(type(index) is not int or not 0 <= index < message_count or index in removed for index in indices):
            raise ValueError("RACER_SOURCE_REPLACEMENT_MESSAGE_INVALID")
        recovery["source_message_indices"] = [index - sum(previous < index for previous in removed) for index in indices]
    for name in ("history_kv_eviction", "history_kv_reference_config"):
        config = hint.get(name)
        if not isinstance(config, dict):
            continue
        for key in ("history_start_message_count", "history_message_count"):
            if key in config:
                count = int(config[key])
                if not 0 <= count <= message_count:
                    raise ValueError("C2KV_COMPOSITION_MESSAGE_BOUNDARY_INVALID")
                config[key] = count - sum(index < count for index in removed)
    events = hint.get("history_kv_event_messages")
    if events is not None:
        if len(events) != message_count:
            raise ValueError("C2KV_COMPOSITION_EVENT_ALIGNMENT_INVALID")
        retained = [
            event for index, event in enumerate(events) if index not in removed
        ]
        hint["history_kv_event_messages"] = [
            {**event, "message_index": index} if isinstance(event, dict) else event
            for index, event in enumerate(retained)
        ]


def source_boundary(position, segments):
    """Map an input boundary to source space, including zero-width carriers."""
    position = int(position)
    result = position
    for segment in segments:
        start, end = int(segment["token_start"]), int(segment["token_end"])
        if start < position < end:
            raise ValueError("C2KV_COMPOSITION_BOUNDARY_INSIDE_SEGMENT")
        if end <= position:
            result += int(segment["source_tokens"]) - (end - start)
    return result


def trailing_source_horizon_to_input(source_horizon, input_prompt_len, segments):
    """Map a source-space horizon after the prompt back to its rendered input."""
    input_prompt_len = int(input_prompt_len)
    source_prompt_len = source_boundary(input_prompt_len, segments)
    source_horizon = int(source_horizon)
    if source_horizon < source_prompt_len:
        raise ValueError("C2KV_COMPOSITION_HORIZON_BEFORE_SOURCE_PROMPT")
    return input_prompt_len + source_horizon - source_prompt_len


def physical_boundary(position, segments):
    position = int(position)
    result = position
    for segment in segments:
        start, end = int(segment["token_start"]), int(segment["token_end"])
        if start < position < end:
            raise ValueError("C2KV_COMPOSITION_BOUNDARY_INSIDE_SEGMENT")
        if end <= position:
            result += len(segment["positions"]) - (end - start)
    return result


def resident_positions(input_len, segments, prefix_positions=(), canonical_prefix=0):
    """Build physical-order canonical positions without inventing gist tokens."""
    positions = list(prefix_positions)
    cursor = len(positions)
    source_cursor = int(canonical_prefix)
    for segment in segments:
        start, end = int(segment["token_start"]), int(segment["token_end"])
        if not cursor <= start <= end <= input_len:
            raise ValueError("C2KV_COMPOSITION_SEGMENT_ORDER_INVALID")
        source_cursor += start - cursor
        positions.extend(range(source_cursor - (start - cursor), source_cursor))
        local = list(segment["positions"])
        if local != sorted(set(local)) or any(
            position < 0 or position >= int(segment["source_tokens"])
            for position in local
        ):
            raise ValueError("C2KV_COMPOSITION_SEGMENT_POSITIONS_INVALID")
        positions.extend(source_cursor + position for position in local)
        source_cursor += int(segment["source_tokens"])
        cursor = end
    positions.extend(range(source_cursor, source_cursor + input_len - cursor))
    return positions


def protected_history_indices(positions, history_start, history_end, tool_spans):
    return [index - history_start for index in range(history_start, history_end)
            if any(start <= positions[index] < end for start, end in tool_spans)]


def raw_query_window(rounds, segments, query_window):
    """Injected KV supplies keys, never freshly executed selection queries."""
    cursor = 0
    raw_tail_start = 0
    for item in rounds:
        cursor += len(item.tokens)
        if item.post_inject_seg_indices:
            cursor += sum(len(segments[index]["positions"]) for index in item.post_inject_seg_indices)
            raw_tail_start = cursor
    start, end = query_window
    start = max(start, raw_tail_start)
    if start >= end:
        raise ValueError("C2KV_COMPOSITION_REQUIRES_RAW_QUERY_AFTER_INJECTION")
    return start, end


def split_rounds_at_query(rounds, segments, query_start, query_end, round_type):
    """Split raw-token rounds while preserving all existing injection events."""
    output = []
    physical_cursor = 0
    observed = 0
    for original in rounds:
        tokens = list(original.tokens)
        round_end = physical_cursor + len(tokens)
        cuts = sorted({physical_cursor, round_end} | {
            boundary for boundary in (query_start, query_end)
            if physical_cursor < boundary < round_end
        })
        for left, right in zip(cuts, cuts[1:]):
            selected = query_start <= left and right <= query_end
            if selected:
                observed += right - left
            output.append(round_type(
                tokens[left - physical_cursor:right - physical_cursor],
                list(original.post_inject_seg_indices) if right == round_end else [],
                collect_history_kv_scores=selected,
            ))
        physical_cursor = round_end + sum(
            len(segments[index]["positions"])
            for index in original.post_inject_seg_indices
        )
    if observed != query_end - query_start or not output:
        raise ValueError("C2KV_COMPOSITION_QUERY_OVERLAPS_INJECTION")
    # Keep NPU partial-page bridge rounds intact. Accumulate observations
    # across them and mutate the KV allocation only after the final query.
    selected = [index for index, item in enumerate(output) if item.collect_history_kv_scores]
    if not selected:
        raise ValueError("C2KV_COMPOSITION_QUERY_EMPTY")
    if any(output[index].post_inject_seg_indices for index in range(selected[0], selected[-1] + 1)):
        raise ValueError("C2KV_COMPOSITION_QUERY_OVERLAPS_INJECTION")
    output[selected[-1]].post_history_kv_eviction = True
    return output
