"""Correctness-first persistent history KV tensors and attention.

The ordinary SGLang request table has one token index and one sequence length
shared by every layer and KV head. Methods such as PyramidKV cannot be encoded
there: each layer and head retains a different set of canonical positions.
This module keeps that compressed history in method-owned tensors and combines
it with the ordinary paged prefix at attention time. It is a reference route;
it deliberately bypasses fused serving attention when active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F


# A headwise causal mask needs Hq * Q * K bytes. Keep its peak bounded across
# long extend requests without changing which keys any query can attend to.
REFERENCE_SDPA_MAX_MASK_BYTES = 16 * 1024 * 1024


@dataclass
class ReferenceLayerKV:
    """One layer's headwise history, with already-rotated keys."""

    key: torch.Tensor  # [Hkv, K, D]
    value: torch.Tensor  # [Hkv, K, D]
    positions: torch.Tensor  # [Hkv, K], canonical RoPE positions

    def validate(self) -> None:
        if self.key.ndim != 3 or self.value.shape != self.key.shape:
            raise ValueError("reference history K/V must share [Hkv, K, D]")
        if self.positions.shape != self.key.shape[:2]:
            raise ValueError("reference history positions must have shape [Hkv, K]")
        if self.positions.dtype != torch.long:
            raise ValueError("reference history positions must use torch.long")
        if self.key.device != self.value.device or self.key.device != self.positions.device:
            raise ValueError("reference history K/V/positions must share one device")
        if self.positions.numel() and not torch.all(
            self.positions[:, 1:] > self.positions[:, :-1]
        ):
            raise ValueError("reference history positions must increase per KV head")

    @property
    def resident_bytes(self) -> int:
        return (
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
            + self.positions.numel() * self.positions.element_size()
        )


@dataclass
class ReferenceHistoryKVState:
    method: str
    layers: Dict[int, ReferenceLayerKV] = field(default_factory=dict)
    selection_metadata: Dict[str, object] = field(default_factory=dict)
    expected_layer_ids: Optional[tuple[int, ...]] = None

    def validate(self) -> None:
        if not self.method:
            raise ValueError("reference history method is required")
        if not self.layers:
            raise ValueError("reference history needs at least one layer")
        if self.expected_layer_ids is not None and set(self.layers) != set(
            self.expected_layer_ids
        ):
            raise ValueError(
                "reference history layers do not match the expected local model layers"
            )
        for layer in self.layers.values():
            layer.validate()

    @property
    def resident_bytes(self) -> int:
        return sum(layer.resident_bytes for layer in self.layers.values())

    def layer(self, layer_id: int) -> Optional[ReferenceLayerKV]:
        return self.layers.get(int(layer_id))


def normalize_event_spans(spans: Sequence[dict]) -> tuple:
    """CommitKV's event signature: (message, role, phase, start, end) rows."""
    return tuple(
        (
            int(item.get("message_index", -1)),
            str(item.get("role") or ""),
            str(item.get("phase") or "others"),
            int(item.get("start", -1)),
            int(item.get("end", -1)),
        )
        for item in spans
    )


def new_event_rows(previous_signature: Sequence[tuple], normalized: Sequence[tuple]):
    """Rows whose message is new since ``previous_signature``, and their tool rows."""
    previous_indices = {item[0] for item in previous_signature}
    new_events = [item for item in normalized if item[0] not in previous_indices]
    new_tools = [
        item
        for item in new_events
        if item[1].lower() == "tool" or item[2].lower() == "tool"
    ]
    return new_events, new_tools


def commitkv_event_transition(
    previous_message_indices: Sequence[int], spans: Sequence[dict]
) -> str:
    """Classify what ``configure_events`` does with these spans.

    ``"tool_event"`` opens a new window (``tool_transition_protection``),
    ``"new_events"`` only closes the open one, and ``"none"`` keeps it.
    """
    previous = tuple((int(index), "", "others", -1, -1) for index in previous_message_indices)
    new_events, new_tools = new_event_rows(previous, normalize_event_spans(spans))
    if new_tools:
        return "tool_event"
    return "new_events" if new_events else "none"


@dataclass
class CommitKVServingState:
    """Serving capture around CommitKV's exact tensor-policy core."""

    policy: Any
    target_tokens: int
    event_signature: tuple = ()
    event_pages: tuple = ()
    pre_window: Any = None
    pre_pages: tuple = ()
    pending_commit_id: Any = None
    post_queries: list[torch.Tensor] = field(default_factory=list)
    post_positions: list[torch.Tensor] = field(default_factory=list)
    pre_queries: list[torch.Tensor] = field(default_factory=list)
    pre_positions: list[torch.Tensor] = field(default_factory=list)
    pre_scan_metadata: dict = field(default_factory=dict)
    receipts: list[dict] = field(default_factory=list)

    def event_message_indices(self) -> list[int]:
        """Message indices of the last configured event spans."""
        return sorted({item[0] for item in self.event_signature})

    def tool_transition_protection(self) -> dict:
        """Protection that a new tool event would open from this state.

        ``configure_events`` closes the open window on any new event and, on a
        new tool event with a full pre window, runs ``record_pre`` over the
        saved scan.  This is the same computation without mutation, so it can
        be reported before the next request is planned and admitted.
        """
        observed = (
            int(self.pre_window.query_positions.numel())
            if self.pre_window is not None
            else 0
        )
        if observed != self.policy.config.window_size:
            return {"tokens": 0, "positions": [], "source_message_indices": []}
        pages, _, _, protected_ids, _ = self.policy.preview_pre(
            self.pre_pages,
            self.pre_window,
            self.pre_window.key_positions,
            total_budget=self.target_tokens,
        )
        by_id = {page.page_id: page for page in pages}
        protected = [by_id[page_id] for page_id in protected_ids]
        positions = sorted(
            {int(index) for page in protected for index in page.token_indices}
        )
        return {
            "tokens": len(positions),
            "positions": positions,
            "source_message_indices": sorted({page.event_id for page in protected}),
        }

    def configure_events(self, spans: Sequence[dict]) -> None:
        from sglang.srt.mem_cache.commitkv import partition_event_span

        normalized = normalize_event_spans(spans)
        if normalized == self.event_signature:
            return
        new_events, new_tools = new_event_rows(self.event_signature, normalized)
        if new_events:
            # The next request starts a new agent turn. A short previous turn
            # cannot finish its post window using queries after this input.
            if self.pending_commit_id is not None or self.policy.pending is not None:
                self.receipts.append(
                    self.policy.record_incomplete_post(
                        self.pending_commit_id,
                        observed_query_count=sum(
                            item.shape[0] for item in self.post_queries
                        ),
                    )
                )
                self.pending_commit_id = None
                self.post_queries.clear()
                self.post_positions.clear()
        if new_tools:
            commit_id = new_tools[-1][0]
            observed = (
                int(self.pre_window.query_positions.numel())
                if self.pre_window is not None
                else 0
            )
            if observed == self.policy.config.window_size:
                receipt = self.policy.record_pre(
                    commit_id,
                    self.pre_pages,
                    self.pre_window,
                    self.pre_window.key_positions,
                    total_budget=self.target_tokens,
                )
                receipt.update(self.pre_scan_metadata)
                self.pending_commit_id = commit_id
            else:
                # Missing/short pre evidence must not reuse the prior action's
                # queries or become a partial-window lifecycle decision.
                receipt = {
                    "commit_id": commit_id,
                    "measurement_phase": "pre_commit_unavailable",
                    "measurement_layer_id": self.policy.config.measurement_layer_id,
                    "reason": "turn_ended_before_window",
                    "observed_query_count": observed,
                    "required_query_count": self.policy.config.window_size,
                    "incomplete_transition_policy": (
                        "full_window_or_unclassified_project_convention"
                    ),
                    "accepted_page_ids": [],
                }
            self.receipts.append(receipt)
        if new_events:
            # Query windows must stay within one generated segment. In
            # particular a zero-decode response must not retain a stale pre.
            self.pre_window = None
            self.pre_pages = ()
            self.pre_queries.clear()
            self.pre_positions.clear()
            self.pre_scan_metadata.clear()

        pages = []
        for index, role, phase, start, end in normalized:
            if phase.lower() not in {"act", "tool"} and role.lower() != "tool":
                continue
            pages.extend(
                partition_event_span(
                    index, start, end, page_size=self.policy.config.page_size
                )
            )
        # The paper supplies a scan cap but no truncation priority.  Canonical
        # message/page order is explicit and over-cap inputs fail in record_pre.
        self.event_signature = normalized
        self.event_pages = tuple(pages)

    def record_decode_window(
        self,
        query: torch.Tensor,
        query_positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_positions: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        from sglang.srt.mem_cache.commitkv import build_deletion_effect_window

        window_size = int(self.policy.config.window_size)
        if self.pending_commit_id is not None:
            if sum(item.shape[0] for item in self.post_queries) < window_size:
                self.post_queries.append(query.detach().clone())
                self.post_positions.append(query_positions.detach().clone())
            post_query = torch.cat(self.post_queries, dim=0)[:window_size]
            post_positions = torch.cat(self.post_positions, dim=0)[:window_size]
            if post_query.shape[0] == window_size:
                window = build_deletion_effect_window(
                    post_query.transpose(0, 1),
                    key,
                    value,
                    query_positions=post_positions,
                    key_positions=key_positions,
                    scale=scale,
                )
                self.receipts.append(
                    self.policy.record_post(
                        self.pending_commit_id, window, key_positions
                    )
                )
                self.pending_commit_id = None
                self.post_queries.clear()
                self.post_positions.clear()

        self.pre_queries.append(query.detach().clone())
        self.pre_positions.append(query_positions.detach().clone())
        while sum(item.shape[0] for item in self.pre_queries) > window_size:
            extra = sum(item.shape[0] for item in self.pre_queries) - window_size
            if self.pre_queries[0].shape[0] <= extra:
                self.pre_queries.pop(0)
                self.pre_positions.pop(0)
            else:
                self.pre_queries[0] = self.pre_queries[0][extra:]
                self.pre_positions[0] = self.pre_positions[0][extra:]
        pre_query = torch.cat(self.pre_queries, dim=0)
        pre_positions = torch.cat(self.pre_positions, dim=0)
        self.pre_window = build_deletion_effect_window(
            pre_query.transpose(0, 1),
            key,
            value,
            query_positions=pre_positions,
            key_positions=key_positions,
            scale=scale,
        )
        resident = {int(position) for position in key_positions.tolist()}
        fully_resident = [
            page
            for page in self.event_pages
            if all(position in resident for position in page.token_indices)
        ]
        max_pages = int(self.policy.config.max_scanned_pages)
        # CommitKV specifies a scan cap but not an over-cap priority. The
        # serving convention scans the latest fully resident pages in
        # canonical message/page order, avoiding both invalid partial effects
        # and an unusable fail-closed path for ordinary long histories.
        self.pre_pages = tuple(fully_resident[-max_pages:])
        self.pre_scan_metadata = {
            "scan_policy": "latest_fully_resident_pages_project_convention",
            "candidate_event_pages": len(self.event_pages),
            "fully_resident_event_pages": len(fully_resident),
            "scan_cap_pages": max_pages,
            "scan_truncated_pages": max(0, len(fully_resident) - max_pages),
        }


def pyramidkv_layer_budgets(
    history_tokens: int,
    target_tokens: int,
    num_layers: int,
    *,
    recent_window: int = 64,
    beta: int = 20,
) -> list[int]:
    """Official PyramidKV capacity schedule, including the recent window."""

    if history_tokens < 1 or target_tokens < 1 or num_layers < 1:
        raise ValueError("history, target, and layer counts must be positive")
    target_tokens = min(history_tokens, target_tokens)
    if history_tokens <= target_tokens:
        return [history_tokens] * num_layers
    recent = min(recent_window, target_tokens - 1, history_tokens - 1)
    past_average = target_tokens - recent
    if history_tokens < past_average * 2:
        return [target_tokens] * num_layers
    minimum = past_average // beta
    maximum = past_average * 2 - minimum
    if maximum >= history_tokens - recent:
        maximum = history_tokens - recent
        minimum = past_average * 2 - maximum
    step = (maximum - minimum) // max(1, num_layers - 1)
    return [
        min(history_tokens, max(1, maximum - layer * step + recent))
        for layer in range(num_layers)
    ]


def select_pyramidkv_headwise(
    scores_by_layer: Sequence[torch.Tensor],
    *,
    target_tokens: int,
    capacity_history_tokens: Optional[int] = None,
    recent_window: int = 64,
    kernel_size: int = 5,
    pooling: str = "avgpool",
    beta: int = 20,
) -> tuple[list[torch.Tensor], dict]:
    """Return official-style per-layer/per-KV-head source indices.

    Each score tensor is ``[Hkv, history_tokens]``. The observation window is
    kept verbatim; pooled headwise scores select older positions.
    """

    if not scores_by_layer:
        raise ValueError("PyramidKV requires layer scores")
    shape = tuple(scores_by_layer[0].shape)
    if len(shape) != 2 or shape[0] < 1:
        raise ValueError("PyramidKV scores must have shape [Hkv, history]")
    num_heads = shape[0]
    if any(scores.ndim != 2 or scores.shape[0] != num_heads for scores in scores_by_layer):
        raise ValueError("PyramidKV layers must have the same KV-head count")
    # A layer can have no candidate left (RACER source replacement emptied its
    # headwise history and the source exclusion removed every ordinary token);
    # like any later empty layer it keeps nothing.  Some layer must have one.
    if max(int(scores.shape[1]) for scores in scores_by_layer) < 1:
        raise ValueError("PyramidKV scores must have shape [Hkv, history]")
    if pooling not in {"avgpool", "maxpool"} or kernel_size < 1:
        raise ValueError("invalid PyramidKV pooling configuration")
    history_tokens = int(
        capacity_history_tokens
        if capacity_history_tokens is not None
        else max(scores.shape[1] for scores in scores_by_layer)
    )
    requested_target_tokens = min(history_tokens, int(target_tokens))
    nominal_target_tokens = requested_target_tokens
    budgets = pyramidkv_layer_budgets(
        history_tokens,
        nominal_target_tokens,
        len(scores_by_layer),
        recent_window=recent_window,
        beta=beta,
    )
    # The official integer funnel can round its layer mean above the nominal
    # target. Absolute serving budgets are hard admission limits, so retain the
    # same official schedule while choosing the largest nominal input whose
    # realized full-model token equivalent fits the requested bound.
    while (
        nominal_target_tokens > 1
        and sum(budgets) > requested_target_tokens * len(scores_by_layer)
    ):
        nominal_target_tokens -= 1
        budgets = pyramidkv_layer_budgets(
            history_tokens,
            nominal_target_tokens,
            len(scores_by_layer),
            recent_window=recent_window,
            beta=beta,
        )
    flat_fallback = sum(budgets) > requested_target_tokens * len(scores_by_layer)
    if flat_fallback:
        # Only a one-token target lands here: the funnel's per-layer minimum
        # is two tokens. Keep the hard bound with a flat schedule instead.
        budgets = [requested_target_tokens] * len(scores_by_layer)
    selected = []
    realized_budgets = []
    for scores, budget in zip(scores_by_layer, budgets):
        layer_history_tokens = int(scores.shape[1])
        budget = min(budget, layer_history_tokens)
        realized_budgets.append(budget)
        recent = min(recent_window, budget, layer_history_tokens)
        old_budget = budget - recent
        old_end = layer_history_tokens - recent
        suffix = torch.arange(
            old_end, layer_history_tokens, device=scores.device, dtype=torch.long
        ).expand(num_heads, -1)
        if old_budget:
            old_scores = scores[:, :old_end]
            if kernel_size > 1:
                pool = F.avg_pool1d if pooling == "avgpool" else F.max_pool1d
                old_scores = pool(
                    old_scores.unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ).squeeze(0)[..., :old_end]
            prefix = torch.topk(old_scores, k=old_budget, dim=-1).indices
            prefix = prefix.sort(dim=-1).values
            indices = torch.cat([prefix, suffix], dim=-1)
        else:
            indices = suffix
        selected.append(indices.contiguous())
    return selected, {
        "algorithm_version": "pyramidkv_official_schedule_headwise_reference_v1",
        "per_layer_budget_tokens": realized_budgets,
        "requested_target_tokens": requested_target_tokens,
        "nominal_schedule_target_tokens": nominal_target_tokens,
        "flat_schedule_fallback": flat_fallback,
        "realized_full_token_equivalent": (
            sum(realized_budgets) + len(realized_budgets) - 1
        )
        // len(realized_budgets),
        "recent_window": recent_window,
        "kernel_size": kernel_size,
        "pooling": pooling,
        "beta": beta,
        "per_head_selection": True,
        "reference_attention_backend": "torch_sdpa",
    }


def gather_reference_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    canonical_positions: Sequence[int] | torch.Tensor,
    indices: torch.Tensor,
) -> ReferenceLayerKV:
    """Gather ``[tokens,Hkv,D]`` source KV with headwise indices."""

    if key.ndim != 3 or value.shape != key.shape or indices.ndim != 2:
        raise ValueError("invalid source KV or headwise selection shape")
    tokens, heads, dim = key.shape
    if indices.shape[0] != heads or torch.any(indices < 0) or torch.any(indices >= tokens):
        raise ValueError("headwise selection is outside source KV")
    positions = torch.as_tensor(
        canonical_positions, dtype=torch.long, device=key.device
    )
    if positions.shape != (tokens,):
        raise ValueError("canonical positions must match source token count")
    source_k = key.transpose(0, 1)
    source_v = value.transpose(0, 1)
    gather = indices.unsqueeze(-1).expand(-1, -1, dim)
    layer = ReferenceLayerKV(
        key=torch.gather(source_k, 1, gather).contiguous().clone(),
        value=torch.gather(source_v, 1, gather).contiguous().clone(),
        positions=positions.expand(heads, -1).gather(1, indices).contiguous(),
    )
    layer.validate()
    return layer


def gather_reference_candidates(
    existing: Optional[ReferenceLayerKV],
    normal_key: torch.Tensor,
    normal_value: torch.Tensor,
    normal_positions: Sequence[int] | torch.Tensor,
    indices: torch.Tensor,
) -> ReferenceLayerKV:
    """Gather headwise indices over the canonical-order resident candidates."""

    key, value, candidate_positions = merge_reference_candidates(
        existing,
        normal_key,
        normal_value,
        normal_positions,
    )
    heads, candidate_tokens, dim = key.shape
    if indices.ndim != 2 or indices.shape[0] != heads:
        raise ValueError("headwise indices must have shape [Hkv,K]")
    if torch.any(indices < 0) or torch.any(indices >= candidate_tokens):
        raise ValueError("headwise selection is outside candidate KV")
    gather = indices.to(key.device).unsqueeze(-1).expand(-1, -1, dim)
    selected_key = torch.gather(key, 1, gather)
    selected_value = torch.gather(value, 1, gather)
    selected_positions = torch.gather(
        candidate_positions, 1, indices.to(candidate_positions.device)
    )
    # Selection policies return a set of resident tokens.  Store that set in
    # canonical position order so a later append can merge it without treating
    # local candidate rank as logical time.
    order = torch.argsort(selected_positions, dim=1)
    selected_gather = order.unsqueeze(-1).expand(-1, -1, dim)
    layer = ReferenceLayerKV(
        key=torch.gather(selected_key, 1, selected_gather).contiguous().clone(),
        value=torch.gather(selected_value, 1, selected_gather).contiguous().clone(),
        positions=torch.gather(selected_positions, 1, order).contiguous().clone(),
    )
    layer.validate()
    return layer


def merge_reference_candidates(
    existing: Optional[ReferenceLayerKV],
    normal_key: torch.Tensor,
    normal_value: torch.Tensor,
    normal_positions: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge external and ordinary KV into canonical order for every head.

    The external state can contain generated positions that follow an ordinary
    protected prefix.  Concatenating ``external + ordinary`` therefore does
    not define a chronological candidate axis on later turns.  All selection
    policies consume this merged axis, and the same permutation is applied to
    K, V, and absolute positions.
    """

    if normal_key.ndim != 3 or normal_value.shape != normal_key.shape:
        raise ValueError("normal candidate KV must share [tokens,Hkv,D]")
    tokens, heads, dim = normal_key.shape
    positions = torch.as_tensor(
        normal_positions, dtype=torch.long, device=normal_key.device
    )
    if positions.shape != (tokens,):
        raise ValueError("normal candidate positions must match token count")
    normal_k = normal_key.transpose(0, 1)
    normal_v = normal_value.transpose(0, 1)
    normal_pos = positions.expand(heads, -1)
    if existing is not None:
        existing.validate()
        if existing.key.shape[0] != heads or existing.key.shape[2] != dim:
            raise ValueError("existing and normal candidate shapes disagree")
        key = torch.cat([existing.key, normal_k], dim=1)
        value = torch.cat([existing.value, normal_v], dim=1)
        candidate_positions = torch.cat([existing.positions, normal_pos], dim=1)
    else:
        key, value, candidate_positions = normal_k, normal_v, normal_pos
    order = torch.argsort(candidate_positions, dim=1)
    gather = order.unsqueeze(-1).expand(-1, -1, dim)
    key = torch.gather(key, 1, gather).contiguous()
    value = torch.gather(value, 1, gather).contiguous()
    candidate_positions = torch.gather(candidate_positions, 1, order).contiguous()
    if candidate_positions.numel() and not torch.all(
        candidate_positions[:, 1:] > candidate_positions[:, :-1]
    ):
        raise ValueError(
            "reference and normal candidate positions must be disjoint per KV head"
        )
    return key, value, candidate_positions


def reference_sdpa(
    query: torch.Tensor,
    history: ReferenceLayerKV,
    normal_key: torch.Tensor,
    normal_value: torch.Tensor,
    normal_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    scale: float,
    validate_history: bool = True,
    decode_causal: bool = False,
) -> torch.Tensor:
    """Attend over headwise history plus ordinary paged KV.

    ``query`` is ``[Q,Hq,D]`` and ordinary KV is ``[N,Hkv,D]``. Returned
    output is ``[Q,Hq,D]``. Keys in both inputs are already RoPE-rotated.
    ``decode_causal`` requires one query after every supplied key position.
    """

    # Materialized serving states are validated at construction. Rechecking
    # their monotonic GPU positions on every decode token synchronizes CUDA on
    # every model layer; direct callers retain the validating default.
    if validate_history:
        history.validate()
    if query.ndim != 3 or normal_key.ndim != 3 or normal_value.shape != normal_key.shape:
        raise ValueError("invalid reference attention tensor ranks")
    q_len, q_heads, dim = query.shape
    normal_len, kv_heads, kv_dim = normal_key.shape
    if history.key.shape[0] != kv_heads or dim != kv_dim or history.key.shape[2] != dim:
        raise ValueError("reference attention head or dimension mismatch")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if normal_positions.shape != (normal_len,) or query_positions.shape != (q_len,):
        raise ValueError("reference attention position lengths mismatch")
    if q_len == 0:
        return query.new_empty((0, q_heads, dim))

    groups = q_heads // kv_heads
    if decode_causal:
        normal_k = normal_key.transpose(0, 1)
        normal_v = normal_value.transpose(0, 1)
        key = torch.cat([history.key, normal_k], dim=1)
        value = torch.cat([history.value, normal_v], dim=1)
        # A decode request has one query at the current canonical position;
        # its resident history and ordinary prefix are all earlier. The
        # caller supplies this invariant. Avoid a per-head custom mask, which
        # blocks fused CUDA SDPA, and let SDPA handle grouped KV heads without
        # copying their entire history for every query head.
        if q_len != 1:
            raise ValueError("decode reference attention requires one query")
        if query.device.type in {"cpu", "cuda"}:
            # PyTorch's grouped-query SDPA handles Hq/Hkv on CPU and CUDA.
            # Keep other backends on their established expanded-head layout.
            attention_key, attention_value = key, value
            sdpa_kwargs = {"enable_gqa": groups != 1}
        else:
            attention_key = key.repeat_interleave(groups, dim=0)
            attention_value = value.repeat_interleave(groups, dim=0)
            # Older torch_npu SDPA wrappers do not necessarily accept this
            # optional PyTorch keyword, even when its value is False.
            sdpa_kwargs = {}
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            attention_key.unsqueeze(0),
            attention_value.unsqueeze(0),
            attn_mask=None,
            dropout_p=0.0,
            scale=float(scale),
            **sdpa_kwargs,
        )
        return output.squeeze(0).transpose(0, 1).contiguous()
    # Build only one KV head's candidate row at a time. A full-history
    # repeat_interleave copies K/V once per query head, which can exceed the
    # free memory on long AppWorld extends even with a bounded causal mask.
    total_key_len = history.key.shape[1] + normal_len
    query_chunk = max(
        1,
        min(
            q_len,
            REFERENCE_SDPA_MAX_MASK_BYTES // max(1, q_heads * total_key_len),
        ),
    )
    output = torch.empty_like(query)
    for kv_head in range(kv_heads):
        head_start = kv_head * groups
        head_stop = head_start + groups
        key = history.key[kv_head]
        value = history.value[kv_head]
        key_pos = history.positions[kv_head]
        if normal_len:
            key = torch.cat([key, normal_key[:, kv_head]], dim=0)
            value = torch.cat([value, normal_value[:, kv_head]], dim=0)
            key_pos = torch.cat([key_pos, normal_positions], dim=0)
        # Treat grouped query heads as the batch axis. The expanded K/V views
        # share storage, so all heads see the same KV row without a full copy.
        grouped_key = key.unsqueeze(0).unsqueeze(0).expand(groups, 1, -1, -1)
        grouped_value = value.unsqueeze(0).unsqueeze(0).expand(groups, 1, -1, -1)
        for start in range(0, q_len, query_chunk):
            stop = min(start + query_chunk, q_len)
            q = query[start:stop, head_start:head_stop].permute(1, 0, 2).unsqueeze(1)
            mask = (
                key_pos.view(1, 1, 1, -1)
                <= query_positions[start:stop].view(1, 1, -1, 1)
            ).expand(groups, -1, -1, -1)
            attended = F.scaled_dot_product_attention(
                q, grouped_key, grouped_value,
                attn_mask=mask, dropout_p=0.0, scale=float(scale)
            )
            output[start:stop, head_start:head_stop] = (
                attended.squeeze(1).permute(1, 0, 2)
            )
    return output.contiguous()
