import unittest

import torch

from sglang.srt.mem_cache.c2kv_pool import C2KVPool, calculate_c2kv_pool_size


class TestC2KVPoolSizing(unittest.TestCase):
    @staticmethod
    def create_pool(max_total_tokens, max_entry_tokens):
        return C2KVPool(
            max_total_tokens=max_total_tokens,
            max_entry_tokens=max_entry_tokens,
            dtype=torch.float32,
            num_kv_heads=2,
            head_dim=4,
            value_head_dim=4,
            num_layers=2,
            device="cpu",
        )

    def test_calculate_c2kv_pool_size(self):
        max_tokens, bytes_per_token = calculate_c2kv_pool_size(
            total_memory_bytes=80 * (1 << 30),
            pool_fraction=0.01,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            value_head_dim=128,
            dtype=torch.bfloat16,
        )

        kv_bytes_per_token = 32 * 8 * (128 + 128) * 2
        self.assertEqual(bytes_per_token, kv_bytes_per_token + 16)
        self.assertEqual(
            max_tokens,
            (int(80 * (1 << 30) * 0.01) - kv_bytes_per_token - 8)
            // bytes_per_token,
        )

    def test_reject_entry_larger_than_pool(self):
        pool = self.create_pool(max_total_tokens=1, max_entry_tokens=2)
        gist_mask = torch.ones((1, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "exceeding the pool capacity"):
            pool.store(
                key_hash="key",
                gist_key_values=[],
                gist_mask=gist_mask,
                gist_position_ids=torch.arange(2).unsqueeze(0),
                original_seq_len=8,
            )

    def test_reject_entry_larger_than_per_entry_limit(self):
        pool = self.create_pool(max_total_tokens=8, max_entry_tokens=1)
        gist_mask = torch.ones((1, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "per-entry limit"):
            pool.store(
                key_hash="key",
                gist_key_values=[],
                gist_mask=gist_mask,
                gist_position_ids=torch.arange(2).unsqueeze(0),
                original_seq_len=8,
            )

    def test_store_reads_from_preallocated_pool(self):
        pool = self.create_pool(max_total_tokens=4, max_entry_tokens=4)
        key_values = [
            (
                torch.arange(16, dtype=torch.float32).view(2, 8) + layer * 100,
                torch.arange(16, dtype=torch.float32).view(2, 8) + layer * 1000,
            )
            for layer in range(2)
        ]
        position_ids = torch.tensor([[3, 7]], dtype=torch.int64)

        entry = pool.store(
            key_hash="key",
            gist_key_values=key_values,
            gist_mask=torch.ones((1, 2), dtype=torch.bool),
            gist_position_ids=position_ids,
            original_seq_len=8,
        )

        self.assertFalse(hasattr(entry, "gist_key_values"))
        torch.testing.assert_close(pool.get_position_ids(entry), position_ids[0])
        for layer in range(2):
            key, value = pool.get_layer_kv(entry, layer)
            torch.testing.assert_close(key.view(2, 8), key_values[layer][0])
            torch.testing.assert_close(value.view(2, 8), key_values[layer][1])

    def test_lru_eviction_reuses_slots(self):
        pool = self.create_pool(max_total_tokens=2, max_entry_tokens=2)

        def store(key, offset):
            values = [
                (
                    torch.full((1, 8), offset + layer, dtype=torch.float32),
                    torch.full((1, 8), offset + layer + 10, dtype=torch.float32),
                )
                for layer in range(2)
            ]
            return pool.store(
                key_hash=key,
                gist_key_values=values,
                gist_mask=torch.ones((1, 1), dtype=torch.bool),
                gist_position_ids=torch.tensor([[offset]], dtype=torch.int64),
                original_seq_len=4,
            )

        first = store("first", 1)
        first_slot = first.token_indices.clone()
        store("second", 2)
        third = store("third", 3)

        self.assertIsNone(pool.get("first"))
        self.assertEqual(pool.num_entries(), 2)
        self.assertEqual(pool.current_tokens(), 2)
        torch.testing.assert_close(third.token_indices, first_slot)

    def test_replace_entry_reuses_and_releases_slots(self):
        pool = self.create_pool(max_total_tokens=4, max_entry_tokens=4)

        def values(gist_len, offset):
            return [
                (
                    torch.full(
                        (gist_len, 8), offset + layer, dtype=torch.float32
                    ),
                    torch.full(
                        (gist_len, 8), offset + layer + 10, dtype=torch.float32
                    ),
                )
                for layer in range(2)
            ]

        original = pool.store(
            "key",
            values(3, 1),
            torch.ones((1, 3), dtype=torch.bool),
            torch.arange(3).unsqueeze(0),
            12,
        )
        original_slots = original.token_indices.clone()
        replacement = pool.store(
            "key",
            values(1, 5),
            torch.ones((1, 1), dtype=torch.bool),
            torch.tensor([[9]], dtype=torch.int64),
            4,
        )

        torch.testing.assert_close(replacement.token_indices, original_slots[:1])
        self.assertEqual(pool.current_tokens(), 1)
        self.assertEqual(pool.allocator.available_size(), 3)
        key, value = pool.get_layer_kv(replacement, 0)
        torch.testing.assert_close(key.view(1, 8), values(1, 5)[0][0])
        torch.testing.assert_close(value.view(1, 8), values(1, 5)[0][1])

    def test_repair_entry_preserves_cache_hit_provenance(self):
        pool = self.create_pool(max_total_tokens=2, max_entry_tokens=2)
        key_values = [
            (
                torch.full((1, 8), layer + 1, dtype=torch.float32),
                torch.full((1, 8), layer + 11, dtype=torch.float32),
            )
            for layer in range(2)
        ]
        metadata = {
            "selected_relative_indices": [0],
            "cacheblend": {"chunk_count": 1, "deviation_mean": 0.25},
        }

        entry = pool.store_repair(
            key_hash="repair",
            key_values=key_values,
            position_ids=torch.tensor([[3]], dtype=torch.int64),
            original_seq_len=4,
            already_rotated=True,
            repair_mode="cacheblend",
            repair_metadata=metadata,
        )

        self.assertEqual(entry.repair_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
