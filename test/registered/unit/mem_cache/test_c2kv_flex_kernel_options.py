import unittest

import torch
from torch.nn.attention.flex_attention import create_mask

from sglang.srt.mem_cache.gist_utils import (
    C2KV_KERNEL_OPTIONS,
    GistConfig,
    get_prepare_gist_input_func,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestC2KVFlexKernelOptions(unittest.TestCase):
    def test_forward_tiles_fit_sparse_mask_blocks(self):
        prepare = get_prepare_gist_input_func(GistConfig())
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        block_mask, _, _ = prepare(input_ids, attention_mask, ratio=2)

        self.assertEqual(
            set(C2KV_KERNEL_OPTIONS),
            {"FORCE_USE_FLEX_ATTENTION", "BLOCK_M", "BLOCK_N"},
        )
        self.assertIs(C2KV_KERNEL_OPTIONS["FORCE_USE_FLEX_ATTENTION"], True)
        for option, sparse_block_size in zip(
            ("BLOCK_M", "BLOCK_N"), block_mask.BLOCK_SIZE
        ):
            tile_size = C2KV_KERNEL_OPTIONS[option]
            self.assertEqual(tile_size, 64)
            self.assertEqual(sparse_block_size % tile_size, 0)

        dense_mask = create_mask(
            block_mask.mask_mod, 1, 1, 9, 9, device="cpu"
        )[0, 0]
        expected = torch.zeros((9, 9), dtype=torch.bool)
        for query in range(6):
            expected[query, : query + 1] = True
        expected[6, [0, 1, 6]] = True
        expected[7, [0, 1, 2, 3, 6, 7]] = True
        expected[8, [0, 1, 4, 5, 6, 7, 8]] = True
        torch.testing.assert_close(dense_mask, expected)


if __name__ == "__main__":
    unittest.main()
