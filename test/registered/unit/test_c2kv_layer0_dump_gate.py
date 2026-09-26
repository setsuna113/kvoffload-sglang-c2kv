"""Keep optional layer-zero diagnostics out of the normal attention path."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / 'python/sglang/srt/models/qwen3.py'


def debug_preamble():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen3Attention')
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    start = next(index for index, node in enumerate(forward.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == '_c2kv_diff_path' for target in node.targets))
    end = next(index for index, node in enumerate(forward.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == '_c2kv_do_dump' for target in node.targets))
    return compile(ast.Module(body=forward.body[start:end + 1], type_ignores=[]), str(SOURCE), 'exec')


class Positions:
    def __init__(self, count=512, value=156, permit_read=False):
        self.count, self.value, self.permit_read = count, value, permit_read
        self.reads = 0

    def numel(self):
        return self.count

    def reshape(self, *_):
        return self

    def __getitem__(self, index):
        assert index == 0
        return self

    def item(self):
        self.reads += 1
        if not self.permit_read:
            raise AssertionError('Unexpected GPU scalar read')
        return self.value


class DumpGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = debug_preamble()

    def run_gate(self, *, path=None, layer=0, extend=True, count=512, value=156,
                 force=False, exists=False, prefix=(100,), permit_read=False):
        positions = Positions(count=count, value=value, permit_read=permit_read)
        env = {'C2KV_DEBUG_LAYER0_DUMP_FORCE': str(int(force))}
        if path is not None:
            env['C2KV_DEBUG_LAYER0_DUMP'] = path
        values = dict(self=SimpleNamespace(attn=SimpleNamespace(layer_id=layer)), positions=positions,
            forward_batch=SimpleNamespace(extend_prefix_lens_cpu=prefix,
                forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda: extend)))
        with patch.dict(os.environ, env, clear=True), patch('os.path.exists', return_value=exists):
            exec(self.code, {'os': os}, values)
        return values, positions.reads

    def test_disabled_dump_never_reads_position(self):
        for layer in (0, 17, 35):
            for force in (False, True):
                with self.subTest(layer=layer, force=force):
                    state, reads = self.run_gate(layer=layer, force=force)
                    self.assertFalse(state['_c2kv_do_dump'])
                    self.assertEqual(reads, 0)

    def test_ineligible_dump_never_reads_position(self):
        for overrides in ({'layer': 1}, {'extend': False}, {'count': 10},
                          {'count': 0}, {'exists': True}, {'prefix': None}):
            with self.subTest(overrides=overrides):
                state, reads = self.run_gate(path='/tmp/dump.pt', **overrides)
                self.assertFalse(state['_c2kv_do_dump'])
                self.assertEqual(reads, 0)

    def test_enabled_eligible_dump_preserves_correction(self):
        state, reads = self.run_gate(path='/tmp/dump.pt', permit_read=True)
        self.assertTrue(state['_c2kv_do_dump'])
        self.assertEqual(state['_c2kv_corr'], [56])
        self.assertEqual(reads, 1)

    def test_zero_correction_only_dumps_when_forced(self):
        for force in (False, True):
            state, reads = self.run_gate(path='/tmp/dump.pt', value=100,
                force=force, permit_read=True)
            self.assertEqual(state['_c2kv_do_dump'], force)
            self.assertEqual(state['_c2kv_corr'], [0] if force else None)
            self.assertEqual(reads, 1)


if __name__ == '__main__':
    unittest.main()
