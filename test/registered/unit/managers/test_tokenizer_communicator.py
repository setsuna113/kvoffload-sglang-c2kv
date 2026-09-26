"""CPU-only concurrency tests for the tokenizer scheduler communicator."""

import ast
import asyncio
import copy
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/tokenizer_communicator_mixin.py"
)
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
communicator_class = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "_Communicator"
)
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        communicator_class,
    ],
    type_ignores=[],
)
ast.fix_missing_locations(module)
namespace = {
    "asyncio": asyncio,
    "copy": copy,
    "deque": deque,
    "Generic": Generic,
    "T": TypeVar("T"),
}
exec(compile(module, str(SOURCE), "exec"), namespace)
Communicator = namespace["_Communicator"]


extract_method = next(
    node
    for cls in tree.body
    if isinstance(cls, ast.ClassDef) and cls.name == "TokenizerCommunicatorMixin"
    for node in cls.body
    if isinstance(node, ast.AsyncFunctionDef) and node.name == "c2kv_extract"
)
extract_module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        extract_method,
    ],
    type_ignores=[],
)
ast.fix_missing_locations(extract_module)
extract_namespace = {
    "TokenizedExtractReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
}
exec(compile(extract_module, str(SOURCE), "exec"), extract_namespace)
c2kv_extract = extract_namespace["c2kv_extract"]


class Sender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


class TestTokenizerCommunicator(unittest.IsolatedAsyncioTestCase):
    async def test_background_extraction_bypasses_batch_collector(self):
        direct_requests = []
        packed_requests = []

        async def direct(req):
            direct_requests.append(req)
            return [SimpleNamespace(success=True)]

        async def packed(req):
            packed_requests.append(req)
            return SimpleNamespace(success=True)

        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            c2kv_gist_batch_size=4,
            c2kv_extract_communicator=direct,
            c2kv_extract_batch_collector=SimpleNamespace(submit=packed),
        )
        await c2kv_extract(manager, [1], "", rid="foreground")
        await c2kv_extract(
            manager, [2], "", rid="background", background_extraction=True
        )
        self.assertEqual([req.rid for req in packed_requests], ["foreground"])
        self.assertFalse(packed_requests[0].background_extraction)
        self.assertEqual([req.rid for req in direct_requests], ["background"])
        self.assertTrue(direct_requests[0].background_extraction)

    async def test_queueing_call_keeps_fifo_and_fan_out_results(self):
        sender = Sender()
        communicator = Communicator(sender, fan_out=2)
        first = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        second = asyncio.create_task(communicator("second"))
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first"])

        communicator.handle_recv("first/0")
        self.assertFalse(first.done())
        communicator.handle_recv("first/1")
        # The newcomer runs before the queued caller wakes in the old implementation.
        third = asyncio.create_task(communicator("third"))
        self.assertEqual(await first, ["first/0", "first/1"])
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first", "second"])

        communicator.handle_recv("second/0")
        communicator.handle_recv("second/1")
        self.assertEqual(await second, ["second/0", "second/1"])
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first", "second", "third"])

        communicator.handle_recv("third/0")
        communicator.handle_recv("third/1")
        self.assertEqual(await third, ["third/0", "third/1"])

    async def test_cancelled_queued_call_does_not_send(self):
        sender = Sender()
        communicator = Communicator(sender, fan_out=1)
        first = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        cancelled = asyncio.create_task(communicator("cancelled"))
        third = asyncio.create_task(communicator("third"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled

        communicator.handle_recv("first/result")
        self.assertEqual(await first, ["first/result"])
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first", "third"])
        communicator.handle_recv("third/result")
        self.assertEqual(await third, ["third/result"])

    async def test_cancelled_in_flight_call_drains_before_next_send(self):
        sender = Sender()
        communicator = Communicator(sender, fan_out=2)
        first = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        second = asyncio.create_task(communicator("second"))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        communicator.handle_recv("first/0")
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first"])
        communicator.handle_recv("first/1")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first", "second"])
        communicator.handle_recv("second/0")
        communicator.handle_recv("second/1")
        self.assertEqual(await second, ["second/0", "second/1"])

    async def test_watching_call_collects_fan_out_and_can_repeat(self):
        sender = Sender()
        communicator = Communicator(sender, fan_out=2, mode="watching")
        first = asyncio.create_task(communicator("first"))
        await asyncio.sleep(0)
        communicator.handle_recv("first/0")
        self.assertFalse(first.done())
        communicator.handle_recv("first/1")
        self.assertEqual(await first, ["first/0", "first/1"])
        second = asyncio.create_task(communicator("second"))
        await asyncio.sleep(0)
        self.assertEqual(sender.sent, ["first", "second"])
        communicator.handle_recv("second/0")
        communicator.handle_recv("second/1")
        self.assertEqual(await second, ["second/0", "second/1"])


if __name__ == "__main__":
    unittest.main()
