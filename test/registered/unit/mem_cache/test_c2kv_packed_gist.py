"""CPU-only checks of packed gist geometry without importing the serving stack."""

import ast
import pathlib
import unittest
from types import SimpleNamespace


SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/mem_cache/gist_utils.py"
)
MODEL_SOURCE = SOURCE.parents[1] / "models/qwen3.py"


class Scalar:
    def __init__(self, value):
        self.value = int(value)

    def clamp(self, minimum, maximum):
        return Scalar(max(minimum, min(self.value, maximum)))

    def __int__(self):
        return self.value

    def __add__(self, other):
        return Scalar(self.value + int(other))

    def __sub__(self, other):
        return Scalar(self.value - int(other))

    def __mul__(self, other):
        return Scalar(self.value * int(other))

    def __lt__(self, other):
        return Flag(self.value < int(other))

    def __ge__(self, other):
        return Flag(self.value >= int(other))


class Flag:
    def __init__(self, value):
        self.value = bool(value)

    def __and__(self, other):
        return Flag(self.value and bool(other))

    def __or__(self, other):
        return Flag(self.value or bool(other))

    def __invert__(self):
        return Flag(not self.value)

    def __bool__(self):
        return self.value


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def unsqueeze(self, _dim):
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            assert len(index) == 2 and index[0] == slice(None)
            return Tensor(self.values[index[1]])
        if isinstance(index, slice):
            return Tensor(self.values[index])
        return self.values[int(index)]

    def squeeze(self, _dim):
        return self

    def to(self, **_kwargs):
        return self

    def narrow(self, dim, start, length):
        assert dim == 0
        return Tensor(self.values[start : start + length])

    def contiguous(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def split(self, lengths, dim):
        assert dim == 1
        output = []
        offset = 0
        for length in lengths:
            output.append(Tensor(self.values[offset : offset + length]))
            offset += length
        assert offset == len(self.values)
        return output


class Device:
    type = "cpu"


class TorchStub:
    long = "long"
    bool = "bool"

    @staticmethod
    def tensor(values, **_kwargs):
        return Tensor(values)

    @staticmethod
    def ones(shape, **_kwargs):
        return Tensor([True] * shape[-1])

    @staticmethod
    def zeros_like(tensor, **_kwargs):
        return Tensor([0] * len(tensor.values))

    @staticmethod
    def device(_value):
        return Device()

    @staticmethod
    def cat(parts, dim):
        assert dim == 1
        return Tensor([item for part in parts for item in part.values])


def load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"prepare_packed_gist_input", "apply_gist_residual_per_document"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": TorchStub}

    def create_block_mask(mask_mod, **kwargs):
        assert kwargs["B"] == 1
        return mask_mod

    namespace["create_block_mask"] = create_block_mask
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def load_model_method(namespace):
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3ForCausalLM"
    )
    method = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_gist_many"
    )
    method.decorator_list = []
    namespace["get_apply_gist_residual_func"] = lambda _cfg, _layer: (
        lambda _raw, gist, **_kwargs: gist
    )
    namespace["paper_telemetry"] = SimpleNamespace(sample=lambda *_args, **_kwargs: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODEL_SOURCE), "exec"), namespace)
    return namespace["generate_gist_many"]


class TestPackedGist(unittest.TestCase):
    def setUp(self):
        self.helpers = load_helpers()

    def prepare(self, lengths, ratio, overlap):
        ids = [list(range(100 * doc, 100 * doc + length))
               for doc, length in enumerate(lengths)]
        return self.helpers["prepare_packed_gist_input"](
            ids, ratio=ratio, gist_overlap=overlap, device="cpu"
        )

    @staticmethod
    def singleton_mask(length, ratio, overlap, query, key):
        query_raw = query < length
        key_raw = key < length
        if query_raw:
            return key_raw and query >= key
        gist_index = query - length
        if key_raw:
            return (
                gist_index * ratio - overlap <= key < (gist_index + 1) * ratio
                or key < ratio
            )
        return query >= key

    def test_mask_matches_singleton_for_unequal_lengths_and_overlap(self):
        for lengths, ratio, overlap in [([6], 4, 1), ([1, 5, 9], 4, 0), ([7, 2], 3, 2),
                                        ([4, 11, 1], 2, 5)]:
            raw, mask, gist, positions, raw_lengths, gist_lengths = self.prepare(
                lengths, ratio, overlap
            )
            self.assertEqual(raw_lengths, lengths)
            self.assertEqual(gist_lengths, [(n + ratio - 1) // ratio for n in lengths])
            self.assertEqual(len(raw.values), sum(lengths))
            self.assertEqual(len(gist.values), sum(gist_lengths))
            expected_positions = [i for n in lengths for i in range(n)] + [
                min((j + 1) * ratio - 1, n - 1)
                for n, count in zip(lengths, gist_lengths) for j in range(count)
            ]
            self.assertEqual(positions.values, expected_positions)

            raw_start = []
            gist_start = []
            offset = 0
            for length in lengths:
                raw_start.append(offset)
                offset += length
            offset = sum(lengths)
            for length in gist_lengths:
                gist_start.append(offset)
                offset += length
            for doc, (length, count) in enumerate(zip(lengths, gist_lengths)):
                indices = list(range(raw_start[doc], raw_start[doc] + length)) + list(
                    range(gist_start[doc], gist_start[doc] + count)
                )
                for q_local, q_global in enumerate(indices):
                    for k_local, k_global in enumerate(indices):
                        self.assertEqual(
                            bool(mask(0, 0, Scalar(q_global), Scalar(k_global))),
                            self.singleton_mask(
                                length, ratio, overlap, q_local, k_local
                            ),
                        )
                for other in range(len(lengths)):
                    if other == doc:
                        continue
                    other_indices = list(
                        range(raw_start[other], raw_start[other] + lengths[other])
                    ) + list(
                        range(gist_start[other], gist_start[other] + gist_lengths[other])
                    )
                    for q_global in indices:
                        for k_global in other_indices:
                            self.assertFalse(mask(0, 0, Scalar(q_global), Scalar(k_global)))
            self.assertFalse(mask(0, 0, Scalar(offset), Scalar(0)))
            self.assertFalse(mask(0, 0, Scalar(0), Scalar(offset + 128)))

    def test_residual_runs_on_each_document_slice(self):
        seen = []

        def residual(raw, gist, ratio):
            seen.append((raw.values, gist.values, ratio))
            return Tensor([value + sum(raw.values) for value in gist.values])

        result = self.helpers["apply_gist_residual_per_document"](
            Tensor([1, 3, 10, 20, 30]),
            Tensor([5, 6, 7]),
            [2, 3],
            [1, 2],
            residual,
            ratio=2,
        )
        self.assertEqual(seen, [([1, 3], [5], 2), ([10, 20, 30], [6, 7], 2)])
        self.assertEqual(result.values, [9, 66, 67])

    def test_model_return_splits_layer_kv_and_local_positions(self):
        method = load_model_method(self.helpers)
        weight = SimpleNamespace(dtype="bf16", device=Device())

        class Embed:
            def __init__(self):
                self.weight = weight

            def __call__(self, token_ids):
                return Tensor(token_ids.values)

        class Layer:
            def forward_with_gist(self, hidden, gist_mask, **kwargs):
                self.positions = kwargs["positions"].values
                self.gist_count = len(gist_mask.values)
                return hidden, (Tensor([10, 11, 12]), Tensor([20, 21, 22]))

        layer = Layer()
        instance = SimpleNamespace(
            full_length_pic=False,
            model=SimpleNamespace(embed_tokens=Embed(), layers=[layer]),
            _c2kv_gist_set=lambda _set: (
                SimpleNamespace(gist_overlap=1), Embed(), None
            ),
            generate_gist_many=None,
        )
        instance.generate_gist_many = lambda *args, **kwargs: method(
            instance, *args, **kwargs
        )
        results = instance.generate_gist_many(
            [[1, 2], [3, 4, 5]], ratio=2, projection_set="history"
        )
        self.assertEqual(layer.gist_count, 3)
        self.assertEqual(layer.positions, [0, 1, 0, 1, 2, 1, 1, 2])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0][0][0].values, [10])
        self.assertEqual(results[0][0][0][1].values, [20])
        self.assertEqual(results[1][0][0][0].values, [11, 12])
        self.assertEqual(results[1][0][0][1].values, [21, 22])
        self.assertEqual(results[0][1].values, [True])
        self.assertEqual(results[1][1].values, [True, True])
        self.assertEqual(results[0][2].values, [1])
        self.assertEqual(results[1][2].values, [1, 2])

    def test_empty_documents_and_invalid_ratio_are_rejected(self):
        for docs, ratio in [([], 2), ([[1], []], 2), ([[1]], 0)]:
            with self.assertRaises(ValueError):
                self.helpers["prepare_packed_gist_input"](
                    docs, ratio=ratio, gist_overlap=0, device="cpu"
                )


if __name__ == "__main__":
    unittest.main()
