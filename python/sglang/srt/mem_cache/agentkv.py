"""Faithful AgentKV Stage-Q selector and semantic query rings.

This is a correctness-first port of the ``agentkv`` plan in
``LiuTaowen-Tony/kv-management-minisgl`` at commit
``254c57bc84e4a7895159ba1062bd46c83626f511``.  The upstream factory maps
AgentKV to StageQ-SnapKV's "q32 local strong" recipe: four independent query
rings with eight rows each, a 32-query anchor budget, SnapKV attention
scoring without score pooling, 16 sink tokens, eight recent tokens, and mean
pooling across GQA query groups.

The selector is independent of SGLang's paged request table.  It returns
per-layer, per-KV-head indices over the caller-provided history candidate
axis.  The common reference runtime gathers those indices into
``ReferenceHistoryKVState`` and uses Torch SDPA for attention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from sglang.srt.mem_cache.history_kv_events import (
    resolve_history_kv_event_token_spans,
)

AGENTKV_SOURCE_REPOSITORY = "https://github.com/LiuTaowen-Tony/kv-management-minisgl"
AGENTKV_SOURCE_COMMIT = "254c57bc84e4a7895159ba1062bd46c83626f511"
AGENTKV_ALGORITHM_VERSION = "agentkv_stageq_snapkv_q32_local_strong_254c57b"

AGENTKV_ANCHOR_BUDGET = 32
AGENTKV_NUM_STAGES = 4
AGENTKV_QUERY_RING_CAPACITY = AGENTKV_ANCHOR_BUDGET // AGENTKV_NUM_STAGES
AGENTKV_OBSERVATION_WINDOW = 8
AGENTKV_POOLING_KERNEL_SIZE = 1
AGENTKV_SINK_TOKENS = 16
AGENTKV_QUERY_GROUP_POOLING = "mean"

AGENTKV_STAGE_OTHERS = 0
AGENTKV_STAGE_THINK = 1
AGENTKV_STAGE_ACT = 2
AGENTKV_STAGE_TOOL = 3


def agentkv_stage_for_event(role: str, phase: str) -> int:
    """Map proxy event semantics onto AgentKV's canonical four stages.

    ``act`` and ``tool`` are semantic labels retained before the proxy
    normalizes native tool messages into OpenAI-compatible user text.  Plain
    assistant content is the reasoning/text stage; system and ordinary user
    messages use the catch-all stage.
    """

    role = str(role).strip().lower()
    phase = str(phase).strip().lower()
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported AgentKV event role: {role!r}")
    if phase not in {"others", "act", "tool"}:
        raise ValueError(f"unsupported AgentKV event phase: {phase!r}")
    if phase == "tool" or role == "tool":
        return AGENTKV_STAGE_TOOL
    if phase == "act":
        return AGENTKV_STAGE_ACT
    if role == "assistant":
        return AGENTKV_STAGE_THINK
    return AGENTKV_STAGE_OTHERS


def resolve_agentkv_message_stages(
    *,
    total_tokens: int,
    message_prefix_token_counts: Sequence[int],
    event_messages: Sequence[Mapping[str, object]],
    device: torch.device | str | None = None,
    trailing_stage: int = AGENTKV_STAGE_THINK,
) -> torch.Tensor:
    """Resolve message-level event hints to exact token stages.

    ``message_prefix_token_counts`` contains the token length before the first
    message followed by the length after every assembled message.  Each event
    must identify exactly one assembled ``message_index``.  Tokens after the
    final message boundary are the assistant generation scaffold and default
    to the reasoning/text stage.
    """

    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    boundaries = [int(item) for item in message_prefix_token_counts]
    event_token_count = boundaries[-1] if boundaries else -1
    if event_token_count > total_tokens:
        raise ValueError("message prefix counts are outside the token sequence")
    spans = resolve_history_kv_event_token_spans(
        total_tokens=event_token_count,
        message_prefix_token_counts=boundaries,
        event_messages=event_messages,
    )
    if trailing_stage not in range(AGENTKV_NUM_STAGES):
        raise ValueError("trailing_stage must be an AgentKV stage")

    stages = torch.full(
        (total_tokens,),
        int(trailing_stage),
        dtype=torch.int32,
        device=device,
    )
    if spans[0]["start"]:
        stages[: int(spans[0]["start"])] = AGENTKV_STAGE_OTHERS
    for span in spans:
        stage = agentkv_stage_for_event(str(span["role"]), str(span["phase"]))
        stages[int(span["start"]) : int(span["end"])] = stage
    return stages


def refine_agentkv_stages_with_markers(
    token_ids: Sequence[int] | torch.Tensor,
    base_stage_ids: torch.Tensor,
    marker_stage_sequences: Mapping[Sequence[int], int],
    *,
    reset_offsets: Sequence[int] = (),
) -> torch.Tensor:
    """Refine assistant spans using tokenizer-resolved native markers.

    The caller tokenizes markers such as ``<think>``, ``</think>``, and
    ``<tool_call>`` with the same tokenizer used for the prompt.  A marker
    changes the stage for itself and following tokens until another marker or
    a message-level stage boundary.  ``reset_offsets`` should contain exact
    message-prefix boundaries because adjacent messages can share the same
    base stage.  Longest markers win when one token sequence prefixes another.
    """

    tokens = [
        int(item)
        for item in (
            token_ids.detach().cpu().tolist()
            if torch.is_tensor(token_ids)
            else token_ids
        )
    ]
    if base_stage_ids.ndim != 1 or base_stage_ids.numel() != len(tokens):
        raise ValueError("base_stage_ids must match the one-dimensional token sequence")
    markers: list[tuple[tuple[int, ...], int]] = []
    for sequence, stage in marker_stage_sequences.items():
        normalized = tuple(int(item) for item in sequence)
        if not normalized:
            raise ValueError("AgentKV marker token sequences must be non-empty")
        if int(stage) not in range(AGENTKV_NUM_STAGES):
            raise ValueError(f"invalid AgentKV marker stage: {stage}")
        markers.append((normalized, int(stage)))
    markers.sort(key=lambda item: len(item[0]), reverse=True)
    resets = {int(item) for item in reset_offsets}
    if any(item < 0 or item > len(tokens) for item in resets):
        raise ValueError("AgentKV marker reset offset is outside the token sequence")

    base = base_stage_ids.detach().cpu().to(torch.int64).tolist()
    output = base_stage_ids.clone()
    current_stage = base[0] if base else AGENTKV_STAGE_OTHERS
    previous_base = current_stage
    offset = 0
    while offset < len(tokens):
        if offset in resets or base[offset] != previous_base:
            current_stage = base[offset]
            previous_base = base[offset]
        matched: tuple[tuple[int, ...], int] | None = None
        for sequence, stage in markers:
            end = offset + len(sequence)
            crosses_boundary = any(
                item in resets or base[item] != base[offset]
                for item in range(offset + 1, min(end, len(tokens)))
            )
            if not crosses_boundary and tuple(tokens[offset:end]) == sequence:
                matched = (sequence, stage)
                break
        if matched is None:
            output[offset] = current_stage
            offset += 1
            continue
        sequence, current_stage = matched
        output[offset : offset + len(sequence)] = current_stage
        offset += len(sequence)
    return output


@dataclass
class AgentKVQueryRing:
    """Per-request, per-layer Stage-Q query rings.

    The upstream GPU pool uses physical circular buffers.  This reference
    representation stores the same last-eight FIFO contents per stage.  Query
    row order does not affect AgentKV's mean reduction across observations.
    """

    num_stages: int = AGENTKV_NUM_STAGES
    capacity: int = AGENTKV_QUERY_RING_CAPACITY
    _queries: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)
    _positions: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_stages <= 0 or self.capacity <= 0:
            raise ValueError("AgentKV query ring stages and capacity must be positive")

    def clear(self) -> None:
        self._queries.clear()
        self._positions.clear()

    def write_layer(
        self,
        *,
        layer_id: int,
        query: torch.Tensor,
        positions: torch.Tensor,
        stage_ids: torch.Tensor,
    ) -> None:
        """Append ``query[T,Hq,D]`` rows, keeping the newest rows per stage."""

        if query.ndim != 3:
            raise ValueError("AgentKV query must have shape [tokens,Hq,D]")
        tokens = query.shape[0]
        if positions.shape != (tokens,) or stage_ids.shape != (tokens,):
            raise ValueError("AgentKV positions and stage_ids must match query tokens")
        if query.device != positions.device or query.device != stage_ids.device:
            raise ValueError("AgentKV query, positions, and stages must share a device")
        if positions.dtype != torch.long:
            raise ValueError("AgentKV query positions must use torch.long")

        layer = int(layer_id)
        queries = self._queries.setdefault(layer, {})
        saved_positions = self._positions.setdefault(layer, {})
        if tokens == 1:
            # Decode is one token per layer. A single stage scalar avoids four
            # shape-dependent GPU nonzero synchronizations for every token.
            stage = int(stage_ids[0].item())
            if stage < -1 or stage >= self.num_stages:
                raise ValueError(f"invalid AgentKV query stages: [{stage}]")
            stage_batches = [(stage, query, positions)] if stage >= 0 else []
        else:
            normalized_stages = stage_ids.to(torch.long)
            invalid = normalized_stages[
                (normalized_stages >= self.num_stages) | (normalized_stages < -1)
            ]
            if invalid.numel():
                raise ValueError(f"invalid AgentKV query stages: {invalid.tolist()}")
            stage_batches = []
            for stage in range(self.num_stages):
                selected = torch.nonzero(
                    normalized_stages == stage, as_tuple=False
                ).flatten()
                if selected.numel():
                    stage_batches.append(
                        (
                            stage,
                            query.index_select(0, selected),
                            positions.index_select(0, selected),
                        )
                    )
        for stage, new_query, new_positions in stage_batches:
            if stage in queries:
                if queries[stage].shape[1:] != new_query.shape[1:]:
                    raise ValueError("AgentKV query head shape changed within one ring")
                if (
                    queries[stage].device != new_query.device
                    or queries[stage].dtype != new_query.dtype
                ):
                    raise ValueError(
                        "AgentKV query device or dtype changed within one ring"
                    )
                new_query = torch.cat([queries[stage], new_query], dim=0)
                new_positions = torch.cat(
                    [saved_positions[stage], new_positions], dim=0
                )
            queries[stage] = new_query[-self.capacity :].contiguous().clone()
            saved_positions[stage] = (
                new_positions[-self.capacity :].contiguous().clone()
            )

    def read_layer(
        self,
        layer_id: int,
        *,
        stages: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stage-concatenated query rows and their canonical positions."""

        layer = int(layer_id)
        queries = self._queries.get(layer, {})
        positions = self._positions.get(layer, {})
        read_stages = tuple(range(self.num_stages)) if stages is None else tuple(stages)
        if any(stage < 0 or stage >= self.num_stages for stage in read_stages):
            raise ValueError("AgentKV read stage is outside the configured rings")
        parts = [queries[stage] for stage in read_stages if stage in queries]
        position_parts = [
            positions[stage] for stage in read_stages if stage in positions
        ]
        if parts:
            return torch.cat(parts, dim=0), torch.cat(position_parts, dim=0)
        # A never-written layer has no shape/device from which to make an empty
        # query tensor.  Runtime callers should treat this as unavailable.
        return torch.empty((0, 0, 0)), torch.empty((0,), dtype=torch.long)

    def rows_by_stage(self, layer_id: int) -> list[int]:
        queries = self._queries.get(int(layer_id), {})
        return [
            int(queries[stage].shape[0]) if stage in queries else 0
            for stage in range(self.num_stages)
        ]

    def reassign_positions(
        self, *, layer_id: int, positions: Sequence[int], stage: int
    ) -> None:
        """Move already-buffered marker rows to their resolved decode stage."""

        if stage < 0 or stage >= self.num_stages:
            raise ValueError("AgentKV reassignment stage is invalid")
        wanted = {int(item) for item in positions}
        if not wanted:
            return
        layer = int(layer_id)
        queries = self._queries.setdefault(layer, {})
        saved_positions = self._positions.setdefault(layer, {})
        moved_q = []
        moved_p = []
        for source_stage in range(self.num_stages):
            if source_stage not in saved_positions:
                continue
            pos = saved_positions[source_stage]
            mask = torch.tensor(
                [int(item) in wanted for item in pos.tolist()],
                dtype=torch.bool,
                device=pos.device,
            )
            if bool(mask.any()):
                moved_q.append(queries[source_stage][mask])
                moved_p.append(pos[mask])
                queries[source_stage] = queries[source_stage][~mask]
                saved_positions[source_stage] = pos[~mask]
        if moved_q:
            query = torch.cat(moved_q, dim=0)
            pos = torch.cat(moved_p, dim=0)
            order = torch.argsort(pos)
            query, pos = query[order], pos[order]
            if stage in queries:
                query = torch.cat([queries[stage], query], dim=0)
                pos = torch.cat([saved_positions[stage], pos], dim=0)
            queries[stage] = query[-self.capacity :].contiguous().clone()
            saved_positions[stage] = pos[-self.capacity :].contiguous().clone()


def _identity_indices(tokens: int, heads: int, device: torch.device) -> torch.Tensor:
    return torch.arange(tokens, dtype=torch.long, device=device).expand(heads, -1)


def select_agentkv_layer_indices(
    key: torch.Tensor,
    query_observation: torch.Tensor,
    *,
    target_tokens: int,
    sink_tokens: int = AGENTKV_SINK_TOKENS,
    recent_tokens: int = AGENTKV_OBSERVATION_WINDOW,
    optional_rank_output: list[list[int]] | None = None,
) -> torch.Tensor:
    """Run the upstream StageQ-SnapKV selection for one layer.

    ``key`` is ``[history,Hkv,D]`` and ``query_observation`` is
    ``[observations,Hq,D]``.  Returned indices have shape ``[Hkv,K]``.
    """

    if key.ndim != 3 or min(key.shape) <= 0:
        raise ValueError("AgentKV key must be non-empty [history,Hkv,D]")
    history_tokens, kv_heads, head_dim = key.shape
    if target_tokens <= 0:
        raise ValueError("AgentKV target_tokens must be positive")
    if sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("AgentKV sink and recent token counts must be non-negative")
    identity = _identity_indices(history_tokens, kv_heads, key.device)
    if history_tokens <= target_tokens:
        if optional_rank_output is not None:
            optional_rank_output.extend([] for _ in range(kv_heads))
        return identity
    if query_observation.ndim != 3:
        raise ValueError(
            "AgentKV query observation must have shape [observations,Hq,D]"
        )
    if query_observation.numel() == 0:
        # Faithful StageQ-SnapKV behavior: no valid query observation means no
        # eviction, rather than substituting another selector.
        if optional_rank_output is not None:
            optional_rank_output.extend([] for _ in range(kv_heads))
        return identity
    if query_observation.device != key.device:
        raise ValueError("AgentKV key and query observations must share a device")
    if query_observation.shape[-1] != head_dim:
        raise ValueError("AgentKV query and key head dimensions must match")
    query_heads = query_observation.shape[1]
    if query_heads % kv_heads:
        raise ValueError("AgentKV query heads must be divisible by KV heads")

    sink_len = min(sink_tokens, history_tokens)
    recent_len = min(recent_tokens, max(0, history_tokens - sink_len))
    mandatory = torch.unique(
        torch.cat(
            [
                torch.arange(sink_len, dtype=torch.long, device=key.device),
                torch.arange(
                    history_tokens - recent_len,
                    history_tokens,
                    dtype=torch.long,
                    device=key.device,
                ),
            ]
        ),
        sorted=True,
    )
    if mandatory.numel() >= target_tokens:
        if optional_rank_output is not None:
            optional_rank_output.extend([] for _ in range(kv_heads))
        return mandatory[:target_tokens].expand(kv_heads, -1).contiguous()

    candidate_start = sink_len
    candidate_end = max(candidate_start, history_tokens - recent_len)
    candidates = torch.arange(
        candidate_start, candidate_end, dtype=torch.long, device=key.device
    )
    if candidates.numel() == 0:
        if optional_rank_output is not None:
            optional_rank_output.extend([] for _ in range(kv_heads))
        return mandatory.expand(kv_heads, -1).contiguous()

    query_observation = query_observation[:AGENTKV_ANCHOR_BUDGET]
    group_size = query_heads // kv_heads
    grouped_query = query_observation.reshape(
        query_observation.shape[0], kv_heads, group_size, head_dim
    )
    candidate_key = key.index_select(0, candidates).permute(1, 0, 2)
    logits = torch.einsum("ohgd,hcd->ohgc", grouped_query, candidate_key)
    logits = logits / math.sqrt(head_dim)
    scores = torch.softmax(logits, dim=-1).mean(dim=(0, 2))
    topk_count = min(target_tokens - mandatory.numel(), candidates.numel())
    selected = candidates[
        torch.topk(scores, k=topk_count, dim=-1, sorted=False).indices
    ]
    if optional_rank_output is not None:
        optional_rank_output.extend(
            sorted(row.tolist(), key=lambda index: float(score_row[index - candidate_start]))
            for row, score_row in zip(selected, scores)
        )
    mandatory_by_head = mandatory.expand(kv_heads, -1)
    return torch.sort(torch.cat([mandatory_by_head, selected], dim=1), dim=1).values


def select_agentkv_headwise(
    keys_by_layer: Sequence[torch.Tensor],
    query_ring: AgentKVQueryRing,
    *,
    target_tokens: int,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """Select AgentKV indices for all layers for the reference runtime."""

    if not keys_by_layer:
        raise ValueError("AgentKV requires at least one layer of keys")
    selected: list[torch.Tensor] = []
    query_rows: list[int] = []
    rows_by_stage: list[list[int]] = []
    for layer_id, key in enumerate(keys_by_layer):
        query, _ = query_ring.read_layer(layer_id)
        if query.numel() == 0:
            query = key.new_empty((0, key.shape[1], key.shape[2]))
        indices = select_agentkv_layer_indices(key, query, target_tokens=target_tokens)
        selected.append(indices)
        query_rows.append(int(query.shape[0]))
        rows_by_stage.append(query_ring.rows_by_stage(layer_id))

    metadata: dict[str, object] = {
        "method": "agentkv",
        "algorithm_version": AGENTKV_ALGORITHM_VERSION,
        "source_repository": AGENTKV_SOURCE_REPOSITORY,
        "source_commit": AGENTKV_SOURCE_COMMIT,
        "reference_attention_backend": "torch_sdpa",
        "per_head_selection": True,
        "anchor_budget": AGENTKV_ANCHOR_BUDGET,
        "num_stages": AGENTKV_NUM_STAGES,
        "query_ring_capacity_per_stage": AGENTKV_QUERY_RING_CAPACITY,
        "observation_window": AGENTKV_OBSERVATION_WINDOW,
        "pooling_kernel_size": AGENTKV_POOLING_KERNEL_SIZE,
        "sink_tokens": AGENTKV_SINK_TOKENS,
        "query_group_pooling": AGENTKV_QUERY_GROUP_POOLING,
        "per_layer_budget_tokens": [int(item.shape[1]) for item in selected],
        "per_layer_query_rows": query_rows,
        "per_layer_query_rows_by_stage": rows_by_stage,
    }
    return selected, metadata


__all__ = [
    "AGENTKV_ALGORITHM_VERSION",
    "AGENTKV_ANCHOR_BUDGET",
    "AGENTKV_NUM_STAGES",
    "AGENTKV_OBSERVATION_WINDOW",
    "AGENTKV_QUERY_RING_CAPACITY",
    "AGENTKV_SINK_TOKENS",
    "AGENTKV_STAGE_ACT",
    "AGENTKV_STAGE_OTHERS",
    "AGENTKV_STAGE_THINK",
    "AGENTKV_STAGE_TOOL",
    "AgentKVQueryRing",
    "agentkv_stage_for_event",
    "refine_agentkv_stages_with_markers",
    "resolve_agentkv_message_stages",
    "select_agentkv_headwise",
    "select_agentkv_layer_indices",
]
