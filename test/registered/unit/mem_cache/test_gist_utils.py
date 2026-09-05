import unittest

import torch

from sglang.srt.mem_cache.gist_utils import (
    prepare_pic_input,
    resolve_c2kv_compression_ratio,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestPICUtils(unittest.TestCase):
    def test_full_length_pic_forces_ratio_one(self):
        self.assertEqual(
            resolve_c2kv_compression_ratio(4, full_length_pic=True), 1
        )
        self.assertEqual(
            resolve_c2kv_compression_ratio(4, full_length_pic=False), 4
        )

    def test_invalid_ratio_is_rejected_before_override(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            resolve_c2kv_compression_ratio(0, full_length_pic=True)

    def test_pic_metadata_retains_every_token(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        pic_mask, position_ids = prepare_pic_input(
            input_ids, torch.ones_like(input_ids, dtype=torch.bool)
        )

        torch.testing.assert_close(pic_mask, torch.ones_like(pic_mask))
        torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2, 3]]))

    def test_pic_rejects_padding(self):
        input_ids = torch.tensor([[1, 2, 0]])
        with self.assertRaisesRegex(ValueError, "unpadded"):
            prepare_pic_input(input_ids, torch.tensor([[True, True, False]]))

if __name__ == "__main__":
    unittest.main()
