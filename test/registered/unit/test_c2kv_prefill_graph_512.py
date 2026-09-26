"""CPU-only contract for the opt-in C2KV 512-token prefill graph gate."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SEMANTICS = ROOT / "python/sglang/srt/mem_cache/c2kv_semantics.py"
RUNNER = ROOT / "python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py"
spec = importlib.util.spec_from_file_location("c2kv_graph_512_semantics_test", SEMANTICS)
semantics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = semantics
spec.loader.exec_module(semantics)


def native_batch(**overrides):
    values = dict(
        forward_mode=SimpleNamespace(name="EXTEND"),
        batch_size=1,
        input_ids=list(range(512)),
        extend_num_tokens=512,
        capture_hidden_mode=SimpleNamespace(name="LAST"),
        c2kv_use_gist_projection=None,
        input_embeds=None,
        spec_info=None,
        is_prefill_only=False,
        c2kv_history_kv_eviction_configs=None,
        c2kv_history_kv_selection_scores=None,
        history_kv_reference_states=[None],
        history_kv_reference_configs=[None],
        history_kv_runtime_states=[None],
        history_kv_resident_positions=[[1, 2, 3]],
        c2kv_position_corrections=object(),
        return_logprob=True,
        extend_logprob_start_lens_cpu=[512],
        extend_seq_lens_cpu=[512],
        token_ids_logprobs=None,
        top_logprobs_nums=[0],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def server_args(**overrides):
    values = dict(
        enable_c2kv=True,
        disable_piecewise_cuda_graph=False,
        piecewise_cuda_graph_tokens=[512],
        piecewise_cuda_graph_compiler="eager",
        chunked_prefill_size=512,
        attention_backend="flashinfer",
        page_size=1,
        c2kv_query_proj="base",
        c2kv_gist_type="dynamic-interleave",
        c2kv_gist_param="qkv",
        c2kv_shadow_feature_layer=-2,
        enable_return_hidden_states=True,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        speculative_algorithm=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def runner_method(name):
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "PiecewiseCudaGraphRunner")
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == name)


class NoTensorTruth:
    def __bool__(self):
        raise AssertionError("Graph eligibility read a tensor truth value")


class PrefillGraph512Tests(unittest.TestCase):
    def test_native_output_only_logprob_and_corrected_positions_are_eligible(self):
        self.assertTrue(semantics.is_c2kv_prefill_graph_512_eligible(native_batch()))
        self.assertTrue(semantics.is_c2kv_prefill_graph_512_eligible(
            native_batch(return_logprob=False)))

    def test_other_shapes_modes_and_hidden_contract_fall_back(self):
        changes = (
            dict(input_ids=list(range(511)), extend_num_tokens=511),
            dict(input_ids=[1], extend_num_tokens=1),
            dict(batch_size=2),
            dict(forward_mode=SimpleNamespace(name="MIXED")),
            dict(forward_mode=SimpleNamespace(name="DECODE")),
            dict(capture_hidden_mode=SimpleNamespace(name="NULL")),
            dict(input_embeds=object()),
            dict(spec_info=object()),
            dict(is_prefill_only=True),
        )
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(semantics.is_c2kv_prefill_graph_512_eligible(
                    native_batch(**change)))

    def test_projection_and_host_side_effects_fall_back_without_tensor_truth(self):
        changes = (
            dict(c2kv_use_gist_projection=NoTensorTruth()),
            dict(c2kv_history_kv_eviction_configs=[NoTensorTruth()]),
            dict(c2kv_history_kv_selection_scores=NoTensorTruth()),
            dict(history_kv_reference_states=[NoTensorTruth()]),
            dict(history_kv_reference_configs=[NoTensorTruth()]),
            dict(history_kv_runtime_states=[NoTensorTruth()]),
        )
        for change in changes:
            with self.subTest(field=next(iter(change))):
                self.assertFalse(semantics.is_c2kv_prefill_graph_512_eligible(
                    native_batch(**change)))

    def test_prompt_logprobs_and_unhandled_top_logprobs_fall_back(self):
        changes = (
            dict(extend_logprob_start_lens_cpu=[0]),
            dict(extend_logprob_start_lens_cpu=[511]),
            dict(extend_logprob_start_lens_cpu=[NoTensorTruth()]),
            dict(top_logprobs_nums=[1]),
            dict(token_ids_logprobs=[[42]]),
        )
        for change in changes:
            with self.subTest(change=next(iter(change))):
                self.assertFalse(semantics.is_c2kv_prefill_graph_512_eligible(
                    native_batch(**change)))

    def test_setup_rejects_unsupported_mode_before_compilation(self):
        model = type("Qwen3ForCausalLM", (), {"full_length_pic": False})()
        semantics.validate_c2kv_prefill_graph_512_setup(server_args(), model, "cuda")
        for args, mdl, device in (
            (server_args(piecewise_cuda_graph_tokens=[256, 512]), model, "cuda"),
            (server_args(c2kv_query_proj="gist"), model, "cuda"),
            (server_args(enable_return_hidden_states=False), model, "cuda"),
            (server_args(tp_size=2), model, "cuda"),
            (server_args(pp_size=2), model, "cuda"),
            (server_args(dp_size=2), model, "cuda"),
            (server_args(speculative_algorithm="EAGLE"), model, "cuda"),
            (server_args(), type("Qwen3ForCausalLM", (), {"full_length_pic": True})(), "cuda"),
            (server_args(), object(), "cuda"),
            (server_args(), model, "cpu"),
        ):
            with self.subTest(args=args, model=type(mdl).__name__, device=device):
                with self.assertRaisesRegex(ValueError, "C2KV_PREFILL_GRAPH_512_UNSUPPORTED"):
                    semantics.validate_c2kv_prefill_graph_512_setup(args, mdl, device)

    def test_runner_dispatch_preserves_default_off_and_gates_opt_in(self):
        node = copy.deepcopy(runner_method("can_run"))
        node.decorator_list = []
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__",
                                 names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        scope = {
            "is_c2kv_graph_compatible": semantics.is_c2kv_graph_compatible,
            "is_c2kv_prefill_graph_512_eligible":
                semantics.is_c2kv_prefill_graph_512_eligible,
        }
        exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), scope)
        can_run = scope["can_run"]
        off = SimpleNamespace(c2kv_prefill_graph_512=False, max_num_tokens=512)
        on = SimpleNamespace(c2kv_prefill_graph_512=True, max_num_tokens=512)
        self.assertTrue(can_run(off, native_batch(input_ids=[1], extend_num_tokens=1,
                                                   return_logprob=False)))
        self.assertFalse(can_run(on, native_batch(input_ids=[1], extend_num_tokens=1,
                                                  return_logprob=False)))
        self.assertTrue(can_run(on, native_batch()))
        self.assertFalse(can_run(on, native_batch(
            c2kv_use_gist_projection=NoTensorTruth())))

    def test_runner_warmup_and_capture_use_the_opt_in_last_mode(self):
        for name in ("warmup_compile", "capture_one_batch_size"):
            method = runner_method(name)
            batches = [node for node in ast.walk(method)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name)
                       and node.func.id == "ForwardBatch"]
            self.assertEqual(len(batches), 1)
            mode = next(value for key, value in zip(batches[0].keywords,
                                                     [kw.value for kw in batches[0].keywords])
                        if key.arg == "capture_hidden_mode")
            self.assertEqual(ast.unparse(mode), "self.prefill_capture_hidden_mode")
        init_source = ast.unparse(runner_method("__init__"))
        self.assertIn("CaptureHiddenMode.LAST if self.c2kv_prefill_graph_512", init_source)
        self.assertIn("CaptureHiddenMode.NULL", init_source)
        self.assertLess(init_source.index("validate_c2kv_prefill_graph_512_setup"),
                        init_source.index("self.warmup_compile"))


if __name__ == "__main__":
    unittest.main()
