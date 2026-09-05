"""
C2KV GPU-resident LRU pool.

The K/V storage is preallocated at startup and managed by token indices, just
like SGLang's regular KV cache pool. Individual entries only retain their slot
indices and lightweight metadata, avoiding long-lived per-entry CUDA tensors.
"""

import struct
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.c2kv_semantics import compute_gist_cache_key
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

C2KV_GIST_TOKEN_BASE = 1 << 60

# One int64 position and one int64 allocator slot per token.
C2KV_METADATA_BYTES_PER_TOKEN = 2 * torch.tensor(
    [], dtype=torch.int64
).element_size()


def calculate_c2kv_pool_size(
    *,
    total_memory_bytes: int,
    pool_fraction: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
) -> Tuple[int, int]:
    """Return (max gist tokens, approximate bytes per token) for a local pool."""
    kv_bytes_per_token = (
        num_layers
        * num_kv_heads
        * (head_dim + value_head_dim)
        * torch.tensor([], dtype=dtype).element_size()
    )
    bytes_per_token = kv_bytes_per_token + C2KV_METADATA_BYTES_PER_TOKEN
    memory_budget_bytes = int(total_memory_bytes * pool_fraction)

    # MHATokenToKVPool and the position buffer reserve padded slot 0.
    padded_slot_bytes = kv_bytes_per_token + torch.tensor(
        [], dtype=torch.int64
    ).element_size()
    usable_bytes = max(memory_budget_bytes - padded_slot_bytes, 0)
    return usable_bytes // bytes_per_token, bytes_per_token


def c2kv_gist_token_ids(key_hash: str, gist_len: int) -> List[int]:
    """Deterministic synthetic token IDs for gist KV slots in the radix cache."""
    base_hash = struct.unpack(">Q", bytes.fromhex(key_hash[:16]))[0]
    base = C2KV_GIST_TOKEN_BASE + (base_hash % (1 << 59))
    return [base + j for j in range(gist_len)]


@dataclass
class C2KVEntry:
    key_hash: str
    token_indices: torch.Tensor
    gist_len: int
    original_seq_len: int
    entry_type: str = "gist"
    already_rotated: bool = False
    repair_mode: Optional[str] = None
    source_doc_index: Optional[int] = None
    repair_metadata: Optional[Dict[str, Any]] = None

    @property
    def token_len(self) -> int:
        return self.gist_len


class C2KVPool:
    def __init__(
        self,
        max_total_tokens: int,
        max_entry_tokens: Optional[int] = None,
        *,
        dtype: torch.dtype,
        num_kv_heads: int,
        head_dim: int,
        value_head_dim: int,
        num_layers: int,
        device: str,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
    ):
        self.max_total_tokens = max_total_tokens
        self.max_entry_tokens = (
            max_total_tokens if max_entry_tokens is None else max_entry_tokens
        )
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.dtype = dtype
        self.device = device
        self.start_layer = start_layer

        self.kv_cache = MHATokenToKVPool(
            size=max_total_tokens,
            page_size=1,
            dtype=dtype,
            head_num=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=value_head_dim,
            layer_num=num_layers,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=start_layer + num_layers,
            enable_alt_stream=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=max_total_tokens,
            dtype=dtype,
            device=device,
            kvcache=self.kv_cache,
            need_sort=False,
        )
        # Slot 0 is reserved consistently with MHATokenToKVPool.
        self.position_buffer = torch.empty(
            max_total_tokens + 1, dtype=torch.int64, device=device
        )

        self._current_tokens = 0
        # OrderedDict: LRU order (MRU at end)
        self._cache: OrderedDict[str, C2KVEntry] = OrderedDict()
        self._pin_counts: Counter[str] = Counter()

    @staticmethod
    def compute_hash(
        token_ids: List[int],
        *,
        compression_ratio: int,
        extractor_config: Optional[dict] = None,
    ) -> str:
        return compute_gist_cache_key(
            token_ids,
            compression_ratio,
            extractor_config,
        )

    def _validate_store_inputs(
        self,
        gist_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        gist_mask: torch.Tensor,
        gist_position_ids: torch.Tensor,
    ) -> int:
        if gist_mask.ndim != 2 or gist_mask.shape[0] != 1:
            raise ValueError(
                f"C2KV gist_mask must have shape (1, gist_len), got {gist_mask.shape}."
            )
        gist_len = gist_mask.shape[1]
        if gist_len == 0:
            raise ValueError("C2KV entry must contain at least one gist token.")
        if gist_len > self.max_entry_tokens:
            raise ValueError(
                f"C2KV entry has {gist_len} gist tokens, exceeding the per-entry "
                f"limit of {self.max_entry_tokens} tokens."
            )
        if gist_len > self.max_total_tokens:
            raise ValueError(
                f"C2KV entry has {gist_len} gist tokens, exceeding the pool "
                f"capacity of {self.max_total_tokens} tokens."
            )
        if gist_position_ids.shape != (1, gist_len):
            raise ValueError(
                "C2KV gist_position_ids shape mismatch: "
                f"{tuple(gist_position_ids.shape)} != {(1, gist_len)}."
            )
        if gist_position_ids.dtype != torch.int64:
            raise ValueError(
                "C2KV gist_position_ids must use torch.int64, got "
                f"{gist_position_ids.dtype}."
            )
        if gist_position_ids.device != self.position_buffer.device:
            raise ValueError(
                "C2KV gist_position_ids device mismatch: "
                f"{gist_position_ids.device} != {self.position_buffer.device}."
            )
        if len(gist_key_values) != self.num_layers:
            raise ValueError(
                "C2KV layer count mismatch: "
                f"{len(gist_key_values)} != {self.num_layers}."
            )

        expected_k_shape = (gist_len, self.num_kv_heads * self.head_dim)
        expected_v_shape = (gist_len, self.num_kv_heads * self.value_head_dim)
        for layer_idx, (key, value) in enumerate(gist_key_values):
            if tuple(key.shape) != expected_k_shape:
                raise ValueError(
                    f"C2KV K shape mismatch at layer {layer_idx}: "
                    f"{tuple(key.shape)} != {expected_k_shape}."
                )
            if tuple(value.shape) != expected_v_shape:
                raise ValueError(
                    f"C2KV V shape mismatch at layer {layer_idx}: "
                    f"{tuple(value.shape)} != {expected_v_shape}."
                )
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError(
                    f"C2KV K/V dtype mismatch at layer {layer_idx}: "
                    f"{key.dtype}/{value.dtype} != {self.dtype}."
                )
            if (
                key.device != self.position_buffer.device
                or value.device != self.position_buffer.device
            ):
                raise ValueError(
                    f"C2KV K/V device mismatch at layer {layer_idx}: "
                    f"{key.device}/{value.device} != {self.position_buffer.device}."
                )
        return gist_len

    def _evict_one(self, exclude_key: Optional[str] = None) -> bool:
        for key_hash, entry in list(self._cache.items()):
            if key_hash == exclude_key or self._pin_counts.get(key_hash, 0) > 0:
                continue
            del self._cache[key_hash]
            self.allocator.free(entry.token_indices)
            self._current_tokens -= entry.gist_len
            return True
        return False

    def can_allocate(self, gist_len: int, existing_key: Optional[str] = None) -> bool:
        """Return whether store() can allocate after evicting unpinned entries."""
        existing = self._cache.get(existing_key) if existing_key is not None else None
        if existing is not None:
            extra_len = max(gist_len - existing.gist_len, 0)
        else:
            extra_len = gist_len
        if extra_len <= self.allocator.available_size():
            return True

        needed = extra_len - self.allocator.available_size()
        evictable_tokens = 0
        for key_hash, entry in self._cache.items():
            if key_hash == existing_key or self._pin_counts.get(key_hash, 0) > 0:
                continue
            evictable_tokens += entry.gist_len
            if evictable_tokens >= needed:
                return True
        return False

    def _allocate_for_store(
        self, gist_len: int, existing: Optional[C2KVEntry]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Allocate/reuse slots and return (new indices, old tail to free)."""
        if existing is not None:
            reused_len = min(existing.gist_len, gist_len)
            reused = existing.token_indices[:reused_len]
            old_tail = existing.token_indices[gist_len:]
            extra_len = gist_len - reused_len
        else:
            reused = None
            old_tail = None
            extra_len = gist_len

        while self.allocator.available_size() < extra_len:
            if not self._evict_one(
                exclude_key=existing.key_hash if existing is not None else None
            ):
                raise ValueError(
                    f"C2KV pool cannot allocate {gist_len} gist token slots."
                )

        extra = self.allocator.alloc(extra_len)
        if extra_len and extra is None:
            raise ValueError(f"C2KV pool cannot allocate {gist_len} gist token slots.")
        if reused is None:
            indices = extra
        elif extra_len:
            indices = torch.cat((reused, extra))
        else:
            indices = reused
        return indices, old_tail

    def store(
        self,
        key_hash: str,
        gist_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        gist_mask: torch.Tensor,
        gist_position_ids: torch.Tensor,
        original_seq_len: int,
    ) -> C2KVEntry:
        gist_len = self._validate_store_inputs(
            gist_key_values, gist_mask, gist_position_ids
        )
        existing = self._cache.get(key_hash)
        indices, old_tail = self._allocate_for_store(gist_len, existing)

        try:
            for layer_idx, (key, value) in enumerate(gist_key_values):
                layer_id = self.start_layer + layer_idx
                self.kv_cache.get_key_buffer(layer_id)[indices] = key.view(
                    gist_len, self.num_kv_heads, self.head_dim
                )
                self.kv_cache.get_value_buffer(layer_id)[indices] = value.view(
                    gist_len, self.num_kv_heads, self.value_head_dim
                )
            self.position_buffer[indices] = gist_position_ids[0]
        except Exception:
            if existing is None:
                self.allocator.free(indices)
            elif gist_len > existing.gist_len:
                self.allocator.free(indices[existing.gist_len :])
            raise

        if existing is not None:
            self._cache.pop(key_hash)
            self._current_tokens -= existing.gist_len
        if old_tail is not None and old_tail.numel():
            self.allocator.free(old_tail)

        entry = C2KVEntry(
            key_hash=key_hash,
            token_indices=indices,
            gist_len=gist_len,
            original_seq_len=original_seq_len,
        )
        self._cache[key_hash] = entry
        self._current_tokens += gist_len
        return entry

    def store_repair(
        self,
        key_hash: str,
        key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        *,
        original_seq_len: int,
        already_rotated: bool,
        repair_mode: str,
        source_doc_index: Optional[int] = None,
        repair_metadata: Optional[Dict[str, Any]] = None,
    ) -> C2KVEntry:
        """Store raw/neutral repair KV using the same paged-token pool.

        `position_ids` are the original absolute positions of these repair KV
        tokens. If `already_rotated` is true, injection writes K directly into
        the active KV cache. Otherwise the injection path applies RoPE once at
        these absolute positions.
        """
        if position_ids.ndim != 2 or position_ids.shape[0] != 1:
            raise ValueError(
                f"repair position_ids must have shape (1, token_len), got {position_ids.shape}."
            )
        token_len = position_ids.shape[1]
        if token_len == 0:
            raise ValueError("repair entry must contain at least one token.")
        if token_len > self.max_entry_tokens or token_len > self.max_total_tokens:
            raise ValueError(
                f"repair entry has {token_len} tokens and exceeds pool limits "
                f"({self.max_entry_tokens=}, {self.max_total_tokens=})."
            )
        if position_ids.dtype != torch.int64:
            raise ValueError(
                f"repair position_ids must use torch.int64, got {position_ids.dtype}."
            )
        if position_ids.device != self.position_buffer.device:
            raise ValueError(
                f"repair position_ids device mismatch: {position_ids.device} != "
                f"{self.position_buffer.device}."
            )
        if len(key_values) != self.num_layers:
            raise ValueError(
                f"repair layer count mismatch: {len(key_values)} != {self.num_layers}."
            )

        expected_k_shape = (token_len, self.num_kv_heads * self.head_dim)
        expected_v_shape = (token_len, self.num_kv_heads * self.value_head_dim)
        for layer_idx, (key, value) in enumerate(key_values):
            if tuple(key.shape) != expected_k_shape:
                raise ValueError(
                    f"repair K shape mismatch at layer {layer_idx}: "
                    f"{tuple(key.shape)} != {expected_k_shape}."
                )
            if tuple(value.shape) != expected_v_shape:
                raise ValueError(
                    f"repair V shape mismatch at layer {layer_idx}: "
                    f"{tuple(value.shape)} != {expected_v_shape}."
                )
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError(
                    f"repair K/V dtype mismatch at layer {layer_idx}: "
                    f"{key.dtype}/{value.dtype} != {self.dtype}."
                )
            if (
                key.device != self.position_buffer.device
                or value.device != self.position_buffer.device
            ):
                raise ValueError(
                    f"repair K/V device mismatch at layer {layer_idx}: "
                    f"{key.device}/{value.device} != {self.position_buffer.device}."
                )

        existing = self._cache.get(key_hash)
        indices, old_tail = self._allocate_for_store(token_len, existing)
        try:
            for layer_idx, (key, value) in enumerate(key_values):
                layer_id = self.start_layer + layer_idx
                self.kv_cache.get_key_buffer(layer_id)[indices] = key.view(
                    token_len, self.num_kv_heads, self.head_dim
                )
                self.kv_cache.get_value_buffer(layer_id)[indices] = value.view(
                    token_len, self.num_kv_heads, self.value_head_dim
                )
            self.position_buffer[indices] = position_ids[0]
        except Exception:
            if existing is None:
                self.allocator.free(indices)
            elif token_len > existing.gist_len:
                self.allocator.free(indices[existing.gist_len :])
            raise

        if existing is not None:
            self._cache.pop(key_hash)
            self._current_tokens -= existing.gist_len
        if old_tail is not None and old_tail.numel():
            self.allocator.free(old_tail)

        entry = C2KVEntry(
            key_hash=key_hash,
            token_indices=indices,
            gist_len=token_len,
            original_seq_len=original_seq_len,
            entry_type="repair",
            already_rotated=already_rotated,
            repair_mode=repair_mode,
            source_doc_index=source_doc_index,
            repair_metadata=dict(repair_metadata or {}),
        )
        self._cache[key_hash] = entry
        self._current_tokens += token_len
        return entry

    def get(self, key_hash: str) -> Optional[C2KVEntry]:
        entry = self._cache.get(key_hash)
        if entry is None:
            return None
        self._cache.move_to_end(key_hash)
        return entry

    def pin(self, key_hash: str) -> bool:
        if key_hash not in self._cache:
            return False
        self._pin_counts[key_hash] += 1
        return True

    def pin_many(self, key_hashes: List[str]) -> bool:
        unique_keys = list(dict.fromkeys(key_hashes))
        missing = [key_hash for key_hash in unique_keys if key_hash not in self._cache]
        if missing:
            return False
        for key_hash in unique_keys:
            self._pin_counts[key_hash] += 1
        return True

    def unpin(self, key_hash: str) -> None:
        count = self._pin_counts.get(key_hash, 0)
        if count <= 1:
            self._pin_counts.pop(key_hash, None)
        else:
            self._pin_counts[key_hash] = count - 1

    def unpin_many(self, key_hashes: List[str]) -> None:
        for key_hash in dict.fromkeys(key_hashes):
            self.unpin(key_hash)

    def pinned_entries(self) -> int:
        return len(self._pin_counts)

    def get_position_ids(self, entry: C2KVEntry) -> torch.Tensor:
        return self.position_buffer[entry.token_indices]

    def get_layer_kv(
        self, entry: C2KVEntry, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_id = self.start_layer + layer_idx
        return (
            self.kv_cache.get_key_buffer(layer_id)[entry.token_indices],
            self.kv_cache.get_value_buffer(layer_id)[entry.token_indices],
        )

    def current_tokens(self) -> int:
        return self._current_tokens

    def num_entries(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self.allocator.clear()
        self._cache.clear()
        self._pin_counts.clear()
        self._current_tokens = 0
