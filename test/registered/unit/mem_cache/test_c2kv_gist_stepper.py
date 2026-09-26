"""CPU-only contract checks for incremental Qwen3 gist extraction."""

import ast
import os
import pathlib
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[4]
MODEL_SOURCE = ROOT / "python/sglang/srt/models/qwen3.py"
RUNNER_SOURCE = ROOT / "python/sglang/srt/model_executor/model_runner.py"


class Tensor:
    def __init__(self, values, *, dtype="int64", device=None):
        self.values = values
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="cpu")

    @property
    def shape(self):
        return (1, len(self.values))

    def __getitem__(self, index):
        assert isinstance(index, tuple) and index[0] == slice(None)
        return Tensor(self.values[index[1]], dtype=self.dtype, device=self.device)

    def squeeze(self, _dim):
        return self

    def contiguous(self):
        return self

    def to(self, *, dtype):
        return Tensor(self.values, dtype=dtype, device=self.device)


class TorchStub:
    Tensor = Tensor
    long = "int64"

    def __init__(self):
        self.events = []
        self.autocast_depth = 0

    def zeros(self, shape, **kwargs):
        self.events.append(("zeros", self.autocast_depth))
        return Tensor([0] * shape[1], device=kwargs["device"])

    def cat(self, tensors, *, dim):
        assert dim == 1
        self.events.append(("cat", self.autocast_depth))
        return Tensor([value for tensor in tensors for value in tensor.values])

    @contextmanager
    def autocast(self, *, device_type, dtype):
        self.events.append(("enter", device_type, dtype))
        self.autocast_depth += 1
        try:
            yield
        finally:
            self.autocast_depth -= 1
            self.events.append(("exit", self.autocast_depth))


def load_model_code(torch_stub):
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    stepper_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "C2KVGistStepper"
    )
    qwen_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3ForCausalLM"
    )
    methods = [
        node for node in qwen_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"create_gist_stepper", "generate_gist"}
    ]
    namespace = {
        "torch": torch_stub,
        "nullcontext": nullcontext,
        "os": os,
        "logger": SimpleNamespace(warning=lambda *_args: None),
        "paper_telemetry": SimpleNamespace(sample=lambda *_args, **_kwargs: None),
        "get_apply_gist_residual_func": lambda _cfg, layer: ("residual", layer),
    }
    exec(
        compile(ast.Module(body=[stepper_class, *methods], type_ignores=[]),
                str(MODEL_SOURCE), "exec"),
        namespace,
    )
    return namespace


def make_model(namespace, torch_stub, events, *, gist_dtype="bf16", layers=3):
    device = SimpleNamespace(type="cpu")

    class Embed:
        def __init__(self, dtype, label):
            self.weight = SimpleNamespace(dtype=dtype, device=device)
            self.label = label

        def __call__(self, tokens):
            events.append(("embed", self.label, torch_stub.autocast_depth))
            return Tensor([self.label] * len(tokens.values), dtype=self.weight.dtype)

    class Layer:
        def __init__(self, index):
            self.index = index

        def forward_with_gist(self, hidden_states, gist_mask, **kwargs):
            events.append((
                "layer", self.index, torch_stub.autocast_depth,
                tuple(kwargs["positions"].values),
                kwargs["attention_mask"], kwargs["apply_gist_residual"],
                kwargs["projection_set"], kwargs["ratio"],
                tuple(gist_mask.values),
            ))
            return hidden_states, (
                Tensor([f"k{self.index}"]), Tensor([f"v{self.index}"])
            )

    def prepare(_input_ids, _attention_mask, *, ratio):
        events.append(("prepare", ratio, torch_stub.autocast_depth))
        return "block-mask", Tensor([True, True]), Tensor([0, 1, 2, 3, 1, 3])

    class Qwen3ForCausalLM:
        create_gist_stepper = namespace["create_gist_stepper"]
        generate_gist = namespace["generate_gist"]

        def __init__(self):
            self.full_length_pic = False
            self.model = SimpleNamespace(
                embed_tokens=Embed("bf16", "raw"),
                gist_embed_tokens=Embed(gist_dtype, "gist"),
                layers=[Layer(i) for i in range(layers)],
            )

        def _c2kv_gist_set(self, projection_set):
            assert projection_set == "history"
            return SimpleNamespace(), self.model.gist_embed_tokens, prepare

    return Qwen3ForCausalLM(), Tensor([10, 11, 12, 13], device=device), Tensor(
        [True] * 4, device=device
    )


def result_values(result):
    kv, mask, positions = result
    return ([(k.values, v.values) for k, v in kv], mask.values, positions.values)


class TestC2KVGistStepper(unittest.TestCase):
    def test_bounded_steps_match_sync_math_and_local_autocast(self):
        for gist_dtype in ("bf16", "fp32"):
            with self.subTest(gist_dtype=gist_dtype):
                torch_stub = TorchStub()
                namespace = load_model_code(torch_stub)
                events = []
                model, ids, mask = make_model(
                    namespace, torch_stub, events, gist_dtype=gist_dtype
                )
                stepper = model.create_gist_stepper(ids, mask, ratio=2)
                self.assertEqual(events, [])
                self.assertEqual(torch_stub.events, [])
                with self.assertRaises(RuntimeError):
                    _ = stepper.result

                self.assertFalse(stepper.step())
                self.assertEqual(
                    [entry[0] for entry in events], ["prepare", "embed", "embed"]
                )
                self.assertEqual(stepper.layers_completed, 0)
                for index in range(3):
                    self.assertFalse(stepper.step())
                    self.assertEqual(stepper.layer_index, index + 1)
                    self.assertEqual(stepper.layers_completed, index + 1)
                    self.assertEqual(len(stepper.gist_key_values), index + 1)
                self.assertTrue(stepper.step())
                self.assertTrue(stepper.step())
                self.assertEqual(len([e for e in events if e[0] == "layer"]), 3)
                stepped_result = result_values(stepper.result)
                stepped_events = list(events)

                torch_stub.events.clear()
                events.clear()
                sync_result = result_values(model.generate_gist(ids, mask, ratio=2))
                self.assertEqual(stepped_result, sync_result)
                self.assertEqual(stepped_events, events)
                self.assertEqual(stepped_result[2], [1, 3])
                self.assertEqual(torch_stub.autocast_depth, 0)

                if gist_dtype == "fp32":
                    # One context per scheduled step, including finalization.
                    torch_stub.events.clear()
                    second = model.create_gist_stepper(ids, mask, ratio=2)
                    for _ in range(5):
                        second.step()
                        self.assertEqual(torch_stub.autocast_depth, 0)
                    self.assertEqual(
                        len([e for e in torch_stub.events if e[0] == "enter"]), 5
                    )

    def test_runner_factory_rejects_unsupported_modes_without_gpu_work(self):
        tree = ast.parse(RUNNER_SOURCE.read_text(encoding="utf-8"))
        runner_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
        )
        method = next(
            node for node in runner_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_c2kv_extract_stepper"
        )
        namespace = {"torch": TorchStub()}
        exec(compile(ast.Module(body=[method], type_ignores=[]),
                     str(RUNNER_SOURCE), "exec"), namespace)
        factory = namespace["create_c2kv_extract_stepper"]
        fake = SimpleNamespace(tp_size=2, model=SimpleNamespace())
        with self.assertRaisesRegex(NotImplementedError, "TP1"):
            factory(fake, None, None, 4)
        fake.tp_size = 1
        with self.assertRaisesRegex(NotImplementedError, "Qwen3ForCausalLM"):
            factory(fake, None, None, 4)
        Qwen3ForCausalLM = type("Qwen3ForCausalLM", (), {})
        fake.model = Qwen3ForCausalLM()
        fake.model.full_length_pic = True
        with self.assertRaisesRegex(NotImplementedError, "PIC"):
            factory(fake, None, None, 4)
        fake.model.full_length_pic = False
        fake.get_c2kv_compression_ratio = lambda ratio: ratio + 1
        fake.model.create_gist_stepper = lambda *args, **kwargs: (args, kwargs)
        self.assertEqual(
            factory(fake, "ids", "mask", 4),
            (("ids", "mask"), {"ratio": 5, "projection_set": "history"}),
        )


if __name__ == "__main__":
    unittest.main()
