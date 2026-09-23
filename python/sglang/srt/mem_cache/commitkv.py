"""Tensor primitives for CommitKV's lifecycle-aware cache policy.

This module implements the method equations from CommitKV Section 3
(arXiv:2608.07855).  It deliberately does not infer agent events from token
text.  The serving layer must resolve completed tool-call and observation
messages into absolute, half-open token spans before constructing pages.

The attention window is for one model layer.  GQA keys and values are expanded
to query heads so that deletion effects are computed independently per
attention head, as required by Eqs. (6)--(7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Hashable, Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class CommitKVConfig:
    """Paper defaults from Section 4.1."""

    window_size: int = 8
    page_size: int = 16
    pending_fraction: float = 0.125
    use_threshold: float = 0.05
    dead_threshold: float = 0.01
    joint_threshold: float = 0.01
    use_percentile: float = 0.75
    dead_percentile: float = 0.25
    max_scanned_pages: int = 64
    max_pending_pages: int = 16
    checkpoint_interval: int = 128
    measurement_layer_id: int | None = None

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.page_size <= 0:
            raise ValueError("window_size and page_size must be positive")
        if not 0.0 <= self.pending_fraction <= 1.0:
            raise ValueError("pending_fraction must be in [0, 1]")
        if not 0.0 <= self.dead_threshold < self.use_threshold:
            raise ValueError("require 0 <= dead_threshold < use_threshold")
        if not 0.0 <= self.joint_threshold:
            raise ValueError("joint_threshold must be non-negative")
        if not 0.0 <= self.dead_percentile < self.use_percentile <= 1.0:
            raise ValueError(
                "require 0 <= dead_percentile < use_percentile <= 1"
            )
        if self.max_scanned_pages <= 0 or self.max_pending_pages <= 0:
            raise ValueError("page caps must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.measurement_layer_id is not None and self.measurement_layer_id < 0:
            raise ValueError("measurement_layer_id must be non-negative")


@dataclass(frozen=True)
class EventPage:
    """A contiguous, absolute token span from one completed agent event."""

    event_id: Hashable
    page_index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.page_index < 0 or self.start < 0 or self.end <= self.start:
            raise ValueError("invalid event page")

    @property
    def page_id(self) -> tuple[Hashable, int]:
        return self.event_id, self.page_index

    @property
    def token_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start


class LifecycleState(str, Enum):
    COMPLETION_CANDIDATE = "completion_candidate"
    DORMANT = "dormant"
    NEWLY_ACTIVE = "newly_active"
    STILL_ACTIVE = "still_active"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class LifecycleEvidence:
    pre_effect: float
    post_effect: float
    pre_percentile: float
    post_percentile: float
    state: LifecycleState


@dataclass(frozen=True)
class PendingCommit:
    commit_id: Hashable
    pages: tuple[EventPage, ...]
    pre_effects: Dict[tuple[Hashable, int], float]
    protected_page_ids: tuple[tuple[Hashable, int], ...]
    total_budget: int


@dataclass
class CommitKVRuntimeState:
    """Persistent lifecycle state, independent of model and scheduler code.

    The caller resolves event spans and captures one real attention layer.  No
    K/V tensors are retained here: only canonical pages, scalar pre effects,
    protected pending pages, and accepted retirement pages persist across
    calls.  ``measurement_layer_id`` is required because the paper specifies
    one attention layer but does not prescribe which layer.
    """

    config: CommitKVConfig
    pending: PendingCommit | None = None
    retired_pages: Dict[tuple[Hashable, int], EventPage] = field(default_factory=dict)
    completed_transitions: int = 0
    incomplete_transitions: int = 0

    def __post_init__(self) -> None:
        if self.config.measurement_layer_id is None:
            raise ValueError("CommitKV runtime requires measurement_layer_id")

    def record_pre(
        self,
        commit_id: Hashable,
        pages: Sequence[EventPage],
        window: "DeletionEffectWindow",
        resident_positions: Sequence[int] | torch.Tensor,
        *,
        total_budget: int,
    ) -> dict[str, object]:
        """Store pre-commit effects and choose bounded transition protection."""

        if self.pending is not None:
            raise RuntimeError("previous CommitKV transition is still pending")
        if total_budget < 1:
            raise ValueError("total_budget must be positive")
        # Pages accepted by an earlier joint retirement are gone from the cache
        # for good (Eq. 11); the serving layer re-derives the page list from
        # message spans every turn, so drop them before scanning. Otherwise a
        # retired page could be protected as pending and the checkpoint would
        # see one token as both retired and pending.
        pages = [page for page in pages if page.page_id not in self.retired_pages]
        if len(pages) > self.config.max_scanned_pages:
            raise ValueError(
                "caller-provided CommitKV scan exceeds max_scanned_pages; "
                "the paper does not specify a page truncation order"
            )
        page_ids = [page.page_id for page in pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("CommitKV page ids must be unique within a commit")
        mapped = resident_page_indices(pages, resident_positions)
        pre_effects = {
            page_id: float(window.effect(indices).item())
            for page_id, indices in mapped.items()
        }
        by_id = {page.page_id: page for page in pages}
        protected_ids, protected_local_indices = protect_pending_pages(
            mapped,
            pre_effects,
            total_budget=total_budget,
            pending_fraction=self.config.pending_fraction,
            max_pages=self.config.max_pending_pages,
        )
        self.pending = PendingCommit(
            commit_id=commit_id,
            pages=tuple(pages),
            pre_effects=pre_effects,
            protected_page_ids=tuple(protected_ids),
            total_budget=total_budget,
        )
        return {
            "commit_id": commit_id,
            "measurement_phase": "pre_commit",
            "measurement_layer_id": self.config.measurement_layer_id,
            "scanned_pages": len(pages),
            "measurable_pages": len(pre_effects),
            "protected_pending_pages": len(protected_ids),
            "protected_pending_tokens": len(protected_local_indices),
            "protected_page_ids": list(protected_ids),
            "unmeasurable_page_ids": [
                page_id for page_id in by_id if page_id not in mapped
            ],
        }

    def record_post(
        self,
        commit_id: Hashable,
        window: "DeletionEffectWindow",
        resident_positions: Sequence[int] | torch.Tensor,
    ) -> dict[str, object]:
        """Complete a paired transition and jointly accept retired pages."""

        pending = self.pending
        if pending is None or pending.commit_id != commit_id:
            raise RuntimeError("CommitKV post window does not match pending commit")
        mapped = resident_page_indices(pending.pages, resident_positions)
        missing_protected = set(pending.protected_page_ids).difference(mapped)
        if missing_protected:
            raise RuntimeError(
                "protected CommitKV page is missing from the post window: "
                f"{sorted(missing_protected, key=str)!r}"
            )
        post_effects = {
            page_id: float(window.effect(indices).item())
            for page_id, indices in mapped.items()
            if page_id in pending.pre_effects
        }
        evidence = pair_lifecycle_evidence(
            pending.pre_effects, post_effects, config=self.config
        )
        accepted, tested_effects = greedy_joint_retirement(
            evidence,
            mapped,
            window,
            joint_threshold=self.config.joint_threshold,
        )
        page_by_id = {page.page_id: page for page in pending.pages}
        for page_id in accepted:
            self.retired_pages[page_id] = page_by_id[page_id]
        self.pending = None
        self.completed_transitions += 1
        return {
            "commit_id": commit_id,
            "measurement_phase": "post_commit",
            "measurement_layer_id": self.config.measurement_layer_id,
            "eligible_pages": len(evidence),
            "lifecycle_states": {
                page_id: item.state.value for page_id, item in evidence.items()
            },
            "accepted_page_ids": list(accepted),
            "joint_test_effects": tested_effects,
            "retired_page_count": len(self.retired_pages),
        }

    def record_incomplete_post(
        self, commit_id: Hashable, *, observed_query_count: int
    ) -> dict[str, object]:
        """Close an unmeasurable next-turn window without retiring any page.

        The serving convention requires the full W generated queries in the
        immediately following turn. A later observation must not supply the
        missing queries. Once that turn ends, its temporary protection can be
        released; ordinary budget selection still applies to those pages.
        """

        pending = self.pending
        if pending is None or pending.commit_id != commit_id:
            raise RuntimeError("CommitKV incomplete post does not match pending commit")
        if not 0 <= observed_query_count < self.config.window_size:
            raise ValueError("incomplete CommitKV post requires fewer than W queries")
        self.pending = None
        self.incomplete_transitions += 1
        return {
            "commit_id": commit_id,
            "measurement_phase": "post_commit_unavailable",
            "measurement_layer_id": self.config.measurement_layer_id,
            "reason": "next_turn_ended_before_window",
            "observed_query_count": observed_query_count,
            "required_query_count": self.config.window_size,
            "incomplete_transition_policy": (
                "full_window_or_unclassified_project_convention"
            ),
            "accepted_page_ids": [],
            "released_pending_page_count": len(pending.protected_page_ids),
            "retired_page_count": len(self.retired_pages),
        }

    def checkpoint(
        self,
        baseline_indices: Iterable[int],
        resident_positions: Sequence[int] | torch.Tensor,
        *,
        target_tokens: int,
        num_layers: int,
        num_kv_heads: int,
        device: torch.device | str | None = None,
        capacity_tokens: int | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        """Build ``I_j`` from accumulated retirement and pending protection."""

        capacity = target_tokens if capacity_tokens is None else int(capacity_tokens)
        if not 0 <= capacity <= target_tokens:
            raise ValueError("CommitKV active capacity must not exceed its total budget")

        positions = (
            [int(value) for value in resident_positions.tolist()]
            if isinstance(resident_positions, torch.Tensor)
            else [int(value) for value in resident_positions]
        )
        if len(positions) != len(set(positions)):
            raise ValueError("resident positions must be unique")
        local_by_position = {
            position: index for index, position in enumerate(positions)
        }
        retired_positions = {
            position
            for page in self.retired_pages.values()
            for position in page.token_indices
        }
        retired_local = [
            local_by_position[position]
            for position in retired_positions
            if position in local_by_position
        ]

        pending_local: list[int] = []
        pending_page_count = 0
        if self.pending is not None:
            if target_tokens < self.pending.total_budget:
                raise ValueError("CommitKV total budget shrank during a transition")
            pending_by_id = {page.page_id: page for page in self.pending.pages}
            for page_id in self.pending.protected_page_ids:
                page = pending_by_id[page_id]
                try:
                    pending_local.extend(
                        local_by_position[position] for position in page.token_indices
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        "protected CommitKV page was evicted before post measurement"
                    ) from exc
                pending_page_count += 1

        selected, metadata = select_commitkv_headwise(
            baseline_indices,
            resident_token_count=len(positions),
            target_tokens=capacity,
            retired_indices=retired_local,
            pending_indices=pending_local,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            device=device,
        )
        metadata.update(
            {
                "measurement_layer_id": self.config.measurement_layer_id,
                "active_capacity_tokens": capacity,
                "window_size": self.config.window_size,
                "page_size": self.config.page_size,
                "pending_fraction": self.config.pending_fraction,
                "use_threshold": self.config.use_threshold,
                "dead_threshold": self.config.dead_threshold,
                "joint_threshold": self.config.joint_threshold,
                "use_percentile": self.config.use_percentile,
                "dead_percentile": self.config.dead_percentile,
                "max_scanned_pages": self.config.max_scanned_pages,
                "max_pending_pages": self.config.max_pending_pages,
                "checkpoint_interval": self.config.checkpoint_interval,
                "completed_transitions": self.completed_transitions,
                "incomplete_transitions": self.incomplete_transitions,
                "incomplete_transition_policy": (
                    "full_window_or_unclassified_project_convention"
                ),
                "retired_page_count": len(self.retired_pages),
                "protected_pending_page_count": pending_page_count,
                "pending_pre_total_budget_tokens": (
                    self.pending.total_budget if self.pending is not None else None
                ),
            }
        )
        return selected, metadata


@dataclass(frozen=True)
class DeletionEffectWindow:
    """One layer's per-head attention data for a CommitKV query window.

    Shapes are ``attention_weights=[Hq,Q,K]``, ``values=[Hq,K,D]``,
    ``outputs=[Hq,Q,D]``, ``query_positions=[Q]``, and
    ``key_positions=[K]``.  Token indices passed to :meth:`effect` index the
    resident K/V sequence, not absolute positions.
    """

    attention_weights: torch.Tensor
    values: torch.Tensor
    outputs: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor

    def __post_init__(self) -> None:
        weights = self.attention_weights
        values = self.values
        outputs = self.outputs
        if weights.ndim != 3 or values.ndim != 3 or outputs.ndim != 3:
            raise ValueError("attention weights, values, and outputs must be rank 3")
        heads, queries, keys = weights.shape
        if values.shape[:2] != (heads, keys):
            raise ValueError(
                "values must have shape [Hq,K,D], got "
                f"{tuple(values.shape)} for weights {tuple(weights.shape)}"
            )
        if outputs.shape != (heads, queries, values.shape[-1]):
            raise ValueError(
                "outputs must have shape [Hq,Q,D], got "
                f"{tuple(outputs.shape)}"
            )
        if tuple(self.query_positions.shape) != (queries,):
            raise ValueError("query_positions must have shape [Q]")
        if tuple(self.key_positions.shape) != (keys,):
            raise ValueError("key_positions must have shape [K]")
        if not (
            torch.isfinite(weights).all()
            and torch.isfinite(values).all()
            and torch.isfinite(outputs).all()
        ):
            raise ValueError("CommitKV attention window contains non-finite values")
        if bool((weights < 0).any()):
            raise ValueError("attention weights must be non-negative")

    def effect(
        self,
        deleted_token_indices: Sequence[int] | torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Return Eq. (7)'s maximum relative output change.

        A deletion that removes all attention mass, or changes a zero-norm
        output, has infinite effect.  This fail-closed value prevents such a
        page set from passing joint retirement.
        """

        if eps <= 0:
            raise ValueError("eps must be positive")
        indices = _unique_index_tensor(
            deleted_token_indices,
            length=self.attention_weights.shape[-1],
            device=self.attention_weights.device,
        )
        if indices.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=indices.device)

        removed_weights = self.attention_weights.float().index_select(-1, indices)
        removed_values = self.values.float().index_select(-2, indices)
        deleted_positions = self.key_positions.to(indices.device).index_select(
            0, indices
        )
        causal = deleted_positions.view(1, 1, -1) <= self.query_positions.to(
            indices.device
        ).view(1, -1, 1)
        removed_weights = removed_weights * causal
        removed_mass = removed_weights.sum(dim=-1)
        removed_output = torch.einsum(
            "hqk,hkd->hqd", removed_weights, removed_values
        )
        outputs = self.outputs.float()
        remaining_mass = 1.0 - removed_mass
        safe_mass = remaining_mass.clamp_min(eps)
        without_page = (outputs - removed_output) / safe_mass.unsqueeze(-1)
        output_norm = torch.linalg.vector_norm(outputs, dim=-1)
        change_norm = torch.linalg.vector_norm(outputs - without_page, dim=-1)
        relative = change_norm / output_norm.clamp_min(eps)

        undefined_mass = remaining_mass <= eps
        changed_zero_output = (output_norm <= eps) & (change_norm > eps)
        relative = torch.where(
            undefined_mass | changed_zero_output,
            torch.full_like(relative, float("inf")),
            relative,
        )
        relative = torch.where(
            (output_norm <= eps) & (change_norm <= eps),
            torch.zeros_like(relative),
            relative,
        )
        return relative.amax()


def build_deletion_effect_window(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    query_positions: Sequence[int] | torch.Tensor,
    key_positions: Sequence[int] | torch.Tensor,
    scale: float,
) -> DeletionEffectWindow:
    """Recompute one layer's exact causal window from rotated Q/K and V.

    Inputs use ``[Hq,Q,D]`` and ``[Hkv,K,D]`` layouts.  For GQA, each native
    KV head is repeated across its contiguous query-head group.  The returned
    tensors are sufficient for both individual and joint deletion tests.
    """

    if query_states.ndim != 3 or key_states.ndim != 3 or value_states.ndim != 3:
        raise ValueError("query, key, and value states must be rank 3")
    query_heads, query_count, head_dim = query_states.shape
    key_heads, key_count, key_dim = key_states.shape
    if value_states.shape[:2] != (key_heads, key_count):
        raise ValueError("key/value head and token dimensions must match")
    if key_dim != head_dim or value_states.shape[-1] != head_dim:
        raise ValueError("query, key, and value head widths must match")
    if query_heads <= 0 or key_heads <= 0 or query_heads % key_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query_count <= 0 or key_count <= 0:
        raise ValueError("query and key windows must be non-empty")
    if not torch.isfinite(torch.tensor(float(scale))) or float(scale) <= 0:
        raise ValueError("scale must be finite and positive")

    query_pos = torch.as_tensor(
        query_positions, dtype=torch.long, device=query_states.device
    )
    key_pos = torch.as_tensor(
        key_positions, dtype=torch.long, device=query_states.device
    )
    if tuple(query_pos.shape) != (query_count,) or tuple(key_pos.shape) != (
        key_count,
    ):
        raise ValueError("position vectors must match query and key lengths")

    groups = query_heads // key_heads
    expanded_keys = key_states.repeat_interleave(groups, dim=0)
    expanded_values = value_states.repeat_interleave(groups, dim=0)
    logits = torch.matmul(
        query_states.float(), expanded_keys.float().transpose(-2, -1)
    ) * float(scale)
    causal = key_pos.view(1, 1, -1) <= query_pos.view(1, -1, 1)
    if not bool(causal.any(dim=-1).all()):
        raise ValueError("every query requires at least one causal key")
    logits = logits.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
    outputs = torch.matmul(weights, expanded_values.float())
    return DeletionEffectWindow(
        attention_weights=weights,
        values=expanded_values,
        outputs=outputs,
        query_positions=query_pos,
        key_positions=key_pos,
    )


def partition_event_span(
    event_id: Hashable,
    start: int,
    end: int,
    *,
    page_size: int,
) -> tuple[EventPage, ...]:
    """Partition one completed event into contiguous pages of size ``<= G``."""

    if start < 0 or end <= start:
        raise ValueError("event span must be non-empty and non-negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return tuple(
        EventPage(event_id, page_index, page_start, min(page_start + page_size, end))
        for page_index, page_start in enumerate(range(start, end, page_size))
    )


def percentile_ranks(effects: Mapping[Hashable, float]) -> Dict[Hashable, float]:
    """Return deterministic empirical percentile ranks with average ties.

    The paper does not specify a tie convention.  We use the standard midrank
    empirical percentile ``(count(<x) + 0.5*count(==x)) / n``.  It preserves
    equal treatment of tied pages and keeps every rank in ``(0, 1)``.
    """

    if not effects:
        return {}
    values = torch.tensor(list(effects.values()), dtype=torch.float64)
    if torch.isnan(values).any() or bool((values < 0).any()):
        raise ValueError("deletion effects must be non-negative and not NaN")
    ranks: Dict[Hashable, float] = {}
    count = values.numel()
    for index, page_id in enumerate(effects):
        value = values[index]
        below = int((values < value).sum().item())
        equal = int((values == value).sum().item())
        ranks[page_id] = (below + 0.5 * equal) / count
    return ranks


def classify_lifecycle(
    *,
    pre_effect: float,
    post_effect: float,
    pre_percentile: float,
    post_percentile: float,
    config: CommitKVConfig,
) -> LifecycleState:
    """Classify Eq. (10) from absolute effects and percentile ranks."""

    values = (pre_effect, post_effect, pre_percentile, post_percentile)
    if any(torch.isnan(torch.tensor(float(value))) for value in values):
        raise ValueError("lifecycle evidence must not contain NaN")
    if pre_effect < 0 or post_effect < 0:
        raise ValueError("deletion effects must be non-negative")
    if not (0 <= pre_percentile <= 1 and 0 <= post_percentile <= 1):
        raise ValueError("percentiles must be in [0, 1]")

    pre_high = (
        pre_effect >= config.use_threshold
        and pre_percentile >= config.use_percentile
    )
    post_high = (
        post_effect >= config.use_threshold
        and post_percentile >= config.use_percentile
    )
    pre_low = (
        pre_effect <= config.dead_threshold
        and pre_percentile <= config.dead_percentile
    )
    post_low = (
        post_effect <= config.dead_threshold
        and post_percentile <= config.dead_percentile
    )
    if pre_high and post_low:
        return LifecycleState.COMPLETION_CANDIDATE
    if pre_low and post_low:
        return LifecycleState.DORMANT
    if pre_low and post_high:
        return LifecycleState.NEWLY_ACTIVE
    if pre_high and post_high:
        return LifecycleState.STILL_ACTIVE
    return LifecycleState.UNCERTAIN


def pair_lifecycle_evidence(
    pre_effects: Mapping[Hashable, float],
    post_effects: Mapping[Hashable, float],
    *,
    config: CommitKVConfig,
) -> Dict[Hashable, LifecycleEvidence]:
    """Pair only pages measurable in both commit windows (paper ``E_c``)."""

    eligible = [page_id for page_id in pre_effects if page_id in post_effects]
    paired_pre = {page_id: float(pre_effects[page_id]) for page_id in eligible}
    paired_post = {page_id: float(post_effects[page_id]) for page_id in eligible}
    pre_ranks = percentile_ranks(paired_pre)
    post_ranks = percentile_ranks(paired_post)
    result: Dict[Hashable, LifecycleEvidence] = {}
    for page_id in eligible:
        state = classify_lifecycle(
            pre_effect=paired_pre[page_id],
            post_effect=paired_post[page_id],
            pre_percentile=pre_ranks[page_id],
            post_percentile=post_ranks[page_id],
            config=config,
        )
        result[page_id] = LifecycleEvidence(
            paired_pre[page_id],
            paired_post[page_id],
            pre_ranks[page_id],
            post_ranks[page_id],
            state,
        )
    return result


def resident_page_indices(
    pages: Sequence[EventPage],
    resident_positions: Sequence[int] | torch.Tensor,
) -> Dict[tuple[Hashable, int], tuple[int, ...]]:
    """Map fully resident absolute event pages onto a local K/V candidate axis.

    A page missing even one token is omitted: its deletion effect cannot be
    computed in this window, so it is not a member of the paper's eligible
    set ``E_c``.  The serving layer intersects the pre- and post-window maps.
    """

    if isinstance(resident_positions, torch.Tensor):
        if resident_positions.ndim != 1:
            raise ValueError("resident_positions must be rank 1")
        positions = [int(position) for position in resident_positions.tolist()]
    else:
        positions = [int(position) for position in resident_positions]
    if len(positions) != len(set(positions)):
        raise ValueError("resident positions must be unique")
    local_by_absolute = {
        absolute_position: local_index
        for local_index, absolute_position in enumerate(positions)
    }
    result: Dict[tuple[Hashable, int], tuple[int, ...]] = {}
    for page in pages:
        local = tuple(
            local_by_absolute[position]
            for position in page.token_indices
            if position in local_by_absolute
        )
        if len(local) == len(page):
            result[page.page_id] = local
    return result


def greedy_joint_retirement(
    evidence: Mapping[Hashable, LifecycleEvidence],
    page_token_indices: Mapping[Hashable, Sequence[int]],
    post_window: DeletionEffectWindow,
    *,
    joint_threshold: float,
) -> tuple[tuple[Hashable, ...], Dict[Hashable, float]]:
    """Construct Eq. (11)'s retirement set with exact combined deletions."""

    if joint_threshold < 0:
        raise ValueError("joint_threshold must be non-negative")
    candidates = [
        page_id
        for page_id, page_evidence in evidence.items()
        if page_evidence.state == LifecycleState.COMPLETION_CANDIDATE
    ]
    for page_id in candidates:
        if page_id not in page_token_indices or not page_token_indices[page_id]:
            raise ValueError(f"missing token indices for candidate page {page_id!r}")
    candidates.sort(
        key=lambda page_id: (
            evidence[page_id].post_effect,
            -evidence[page_id].pre_effect,
        )
    )
    accepted: list[Hashable] = []
    accepted_tokens: set[int] = set()
    tested_effects: Dict[Hashable, float] = {}
    for page_id in candidates:
        proposed = accepted_tokens.union(int(i) for i in page_token_indices[page_id])
        effect = float(post_window.effect(sorted(proposed)).item())
        tested_effects[page_id] = effect
        if effect <= joint_threshold:
            accepted.append(page_id)
            accepted_tokens = proposed
    return tuple(accepted), tested_effects


def protect_pending_pages(
    page_token_indices: Mapping[Hashable, Sequence[int]],
    pre_effects: Mapping[Hashable, float],
    *,
    total_budget: int,
    pending_fraction: float,
    max_pages: int,
) -> tuple[tuple[Hashable, ...], tuple[int, ...]]:
    """Protect whole pending pages by decreasing pre-commit effect (Eq. 12)."""

    if total_budget < 0 or not 0 <= pending_fraction <= 1 or max_pages < 0:
        raise ValueError("invalid pending budget")
    pending_budget = int(total_budget * pending_fraction)
    ordered = sorted(
        pre_effects,
        key=lambda page_id: (-float(pre_effects[page_id]), str(page_id)),
    )
    selected_pages: list[Hashable] = []
    selected_tokens: set[int] = set()
    for page_id in ordered:
        effect = float(pre_effects[page_id])
        if torch.isnan(torch.tensor(effect)) or effect < 0:
            raise ValueError("pre-commit effects must be non-negative and not NaN")
        if len(selected_pages) >= max_pages:
            break
        if page_id not in page_token_indices:
            raise ValueError(f"missing token indices for pending page {page_id!r}")
        tokens = {int(i) for i in page_token_indices[page_id]}
        if not tokens or min(tokens) < 0:
            raise ValueError("pending pages must contain non-negative token indices")
        proposed = selected_tokens.union(tokens)
        if len(proposed) <= pending_budget:
            selected_pages.append(page_id)
            selected_tokens = proposed
    return tuple(selected_pages), tuple(sorted(selected_tokens))


def compose_retained_indices(
    baseline_indices: Iterable[int],
    *,
    resident_token_count: int,
    budget: int,
    retired_indices: Iterable[int] = (),
    pending_indices: Iterable[int] = (),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Apply CommitKV retirement/protection around an ordered base selector.

    CommitKV does not prescribe the base scoring rule for non-lifecycle tokens.
    ``baseline_indices`` is therefore an explicit best-to-worst order supplied
    by the serving policy.  Pending tokens are inserted first, retired tokens
    are excluded, and the returned common index vector is sorted to retain
    canonical token order.
    """

    if resident_token_count < 0 or budget < 0:
        raise ValueError("resident_token_count and budget must be non-negative")
    budget = min(budget, resident_token_count)
    retired = {int(i) for i in retired_indices}
    pending = {int(i) for i in pending_indices}
    if any(i < 0 or i >= resident_token_count for i in retired | pending):
        raise ValueError("retired/pending indices are outside the resident cache")
    if retired & pending:
        raise ValueError("a token cannot be both retired and pending")
    if len(pending) > budget:
        raise ValueError("pending tokens exceed the total cache budget")

    selected = set(pending)
    seen: set[int] = set()
    for raw_index in baseline_indices:
        if len(selected) >= budget:
            break
        index = int(raw_index)
        if index in seen:
            continue
        seen.add(index)
        if index < 0 or index >= resident_token_count:
            raise ValueError("baseline index is outside the resident cache")
        if index not in retired:
            selected.add(index)
    return torch.tensor(sorted(selected), dtype=torch.long, device=device)


def select_commitkv_headwise(
    baseline_indices: Iterable[int],
    *,
    resident_token_count: int,
    target_tokens: int,
    retired_indices: Iterable[int],
    pending_indices: Iterable[int],
    num_layers: int,
    num_kv_heads: int,
    device: torch.device | str | None = None,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """Adapt CommitKV's common Eq. (4) index set to reference-cache shapes.

    The same token indices are repeated for every layer and KV head.  This is
    intentional: unlike headwise snapshot selectors, CommitKV defines one
    retained-token set ``I_j`` that keeps K, V, and absolute positions aligned.
    """

    if num_layers <= 0 or num_kv_heads <= 0:
        raise ValueError("num_layers and num_kv_heads must be positive")
    retired_indices = tuple(int(i) for i in retired_indices)
    pending_indices = tuple(int(i) for i in pending_indices)
    retained = compose_retained_indices(
        baseline_indices,
        resident_token_count=resident_token_count,
        budget=target_tokens,
        retired_indices=retired_indices,
        pending_indices=pending_indices,
        device=device,
    )
    headwise = retained.view(1, -1).expand(num_kv_heads, -1).contiguous()
    per_layer = [headwise.clone() for _ in range(num_layers)]
    return per_layer, {
        "algorithm_version": "commitkv_arxiv_2608_07855_v1",
        "reference_attention_backend": "torch_sdpa",
        "per_head_selection": False,
        "common_retained_indices": True,
        "retained_tokens": retained.numel(),
        "retired_tokens": len(set(retired_indices)),
        "pending_tokens": len(set(pending_indices)),
    }


def apply_retained_indices(
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    indices: torch.Tensor,
    *,
    token_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one common token index set to K, V, and absolute positions."""

    if positions.ndim != 1 or indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("positions and long indices must both be rank 1")
    key_dim = token_dim % keys.ndim
    value_dim = token_dim % values.ndim
    if keys.shape[key_dim] != values.shape[value_dim] or keys.shape[key_dim] != len(
        positions
    ):
        raise ValueError("K, V, and positions must have the same token length")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= len(positions)
    ):
        raise ValueError("retained index is outside the resident cache")
    key_indices = indices.to(keys.device)
    value_indices = indices.to(values.device)
    position_indices = indices.to(positions.device)
    return (
        keys.index_select(key_dim, key_indices),
        values.index_select(value_dim, value_indices),
        positions.index_select(0, position_indices),
    )


def _unique_index_tensor(
    indices: Sequence[int] | torch.Tensor,
    *,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(indices, dtype=torch.long, device=device).reshape(-1)
    if tensor.numel() == 0:
        return tensor
    tensor = torch.unique(tensor, sorted=True)
    if int(tensor.min()) < 0 or int(tensor.max()) >= length:
        raise ValueError("deleted token index is outside the attention window")
    return tensor
