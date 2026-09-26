"""CPU contracts for the optional cross-request C2KV extraction collector."""

import ast
import asyncio
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANAGERS = ROOT / "python/sglang/srt/managers"


def _collector_class():
    path = MANAGERS / "c2kv_extract_batch.py"
    spec = importlib.util.spec_from_file_location("c2kv_extract_batch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.C2KVExtractBatchCollector


Collector = _collector_class()


def _mixin_methods():
    path = MANAGERS / "tokenizer_communicator_mixin.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "TokenizerCommunicatorMixin"
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {"init_communicators", "c2kv_extract", "c2kv_bulk_cache_lookup"}
    ]
    module = ast.Module(body=methods, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "asyncio": asyncio,
        "os": os,
        "TokenizerManager": object,
        "ServerArgs": object,
        "List": list,
        "Dict": dict,
        "Any": Any,
        "Optional": Optional,
        "C2KVExtractReqOutput": object,
        "C2KVBulkCacheLookupReqOutput": object,
        "TokenizedExtractReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "C2KVBulkCacheLookupReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "C2KVExtractBatchReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "C2KVExtractBatchCollector": Collector,
        "_Communicator": lambda *_args, **_kwargs: None,
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - isolate mixin methods from optional runtime dependencies
    return namespace


def _item(rid):
    return SimpleNamespace(rid=rid, input_ids=[1, 2])


def _reply(items):
    return [
        SimpleNamespace(
            success=True,
            items=[SimpleNamespace(rid=item.rid, success=True) for item in items],
        )
    ]


def test_batch_opt_in_validates_size_and_dp(monkeypatch):
    method = _mixin_methods()["init_communicators"]

    def initialize(size, dp_size):
        monkeypatch.setenv("C2KV_GIST_BATCH_SIZE", size)
        manager = SimpleNamespace(
            send_to_scheduler=object(),
            _result_dispatcher=[],
            _get_communicator_dispatcher=list,
        )
        method(manager, SimpleNamespace(dp_size=dp_size))
        return manager

    disabled = initialize("1", 2)
    assert disabled.c2kv_extract_batch_collector is None
    assert disabled.c2kv_gist_batch_size == 1
    enabled = initialize("4", 1)
    assert enabled.c2kv_extract_batch_collector._max_size == 4
    with pytest.raises(ValueError, match="dp_size=1"):
        initialize("2", 2)
    with pytest.raises(ValueError, match="integer from 1 to 4"):
        initialize("0", 1)
    with pytest.raises(ValueError, match="integer from 1 to 4"):
        initialize("not-an-integer", 1)


@pytest.mark.asyncio
async def test_collector_batches_concurrent_requests_and_queues_during_rpc():
    sent = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def communicator(envelope):
        sent.append([item.rid for item in envelope.items])
        if len(sent) == 1:
            started.set()
            await release.wait()
        return _reply(envelope.items)

    collector = Collector(communicator, lambda **kwargs: SimpleNamespace(**kwargs), 4)
    first = [asyncio.create_task(collector.submit(_item(str(i)))) for i in range(5)]
    await started.wait()
    assert sent == [["0", "1", "2", "3"]]
    more = [asyncio.create_task(collector.submit(_item(str(i)))) for i in range(5, 8)]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*first, *more)
    assert sent == [["0", "1", "2", "3"], ["4", "5", "6", "7"]]
    assert [result.rid for result in results] == [str(i) for i in range(8)]


@pytest.mark.asyncio
async def test_collector_cancel_and_error_do_not_consume_later_replies():
    started = asyncio.Event()
    release = asyncio.Event()
    sent = []

    async def communicator(envelope):
        sent.append([item.rid for item in envelope.items])
        if len(sent) == 1:
            started.set()
            await release.wait()
            raise RuntimeError("scheduler failed")
        return _reply(envelope.items)

    collector = Collector(communicator, lambda **kwargs: SimpleNamespace(**kwargs), 2)
    first = asyncio.create_task(collector.submit(_item("cancelled")))
    second = asyncio.create_task(collector.submit(_item("failed")))
    await started.wait()
    queued = asyncio.create_task(collector.submit(_item("next")))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    with pytest.raises(RuntimeError, match="scheduler failed"):
        await second
    assert (await queued).rid == "next"
    assert sent == [["cancelled", "failed"], ["next"]]


@pytest.mark.asyncio
async def test_collector_skips_request_cancelled_before_flush():
    sent = []

    async def communicator(envelope):
        sent.append([item.rid for item in envelope.items])
        return _reply(envelope.items)

    collector = Collector(communicator, lambda **kwargs: SimpleNamespace(**kwargs), 4)
    cancelled = asyncio.create_task(collector.submit(_item("cancelled")))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert (await collector.submit(_item("later"))).rid == "later"
    assert sent == [["later"]]


@pytest.mark.asyncio
async def test_collector_rejects_malformed_reply_without_misrouting():
    calls = 0

    async def communicator(envelope):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                SimpleNamespace(
                    success=True, items=list(reversed(_reply(envelope.items)[0].items))
                )
            ]
        return _reply(envelope.items)

    collector = Collector(communicator, lambda **kwargs: SimpleNamespace(**kwargs), 2)
    results = await asyncio.gather(
        collector.submit(_item("a")),
        collector.submit(_item("b")),
        return_exceptions=True,
    )
    assert all(isinstance(result, RuntimeError) for result in results)
    assert (await collector.submit(_item("c"))).rid == "c"


@pytest.mark.asyncio
async def test_collector_preserves_each_item_failure_independently():
    async def communicator(envelope):
        return [
            SimpleNamespace(
                success=True,
                items=[
                    SimpleNamespace(
                        rid=envelope.items[0].rid,
                        success=False,
                        error="budget exhausted",
                    ),
                    SimpleNamespace(rid=envelope.items[1].rid, success=True, error=""),
                ],
            )
        ]

    collector = Collector(communicator, lambda **kwargs: SimpleNamespace(**kwargs), 2)
    failed, succeeded = await asyncio.gather(
        collector.submit(_item("failed")),
        collector.submit(_item("succeeded")),
    )
    assert failed.success is False and failed.error == "budget exhausted"
    assert succeeded.success is True and succeeded.rid == "succeeded"


@pytest.mark.asyncio
async def test_mixin_default_off_uses_singleton_communicator_without_new_attrs():
    method = _mixin_methods()["c2kv_extract"]
    sent = []

    async def communicator(req):
        sent.append(req)
        return [SimpleNamespace(success=True)]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        c2kv_extract_communicator=communicator,
    )
    result = await method(manager, [1, 2], "text", rid="single")
    assert result.success
    assert sent[0].rid == "single"
    assert sent[0].allow_cache_miss is True


@pytest.mark.asyncio
async def test_bulk_first_miss_queues_lookup_with_original_budget_and_metadata():
    method = _mixin_methods()["c2kv_bulk_cache_lookup"]
    lookups = []
    queued = []

    async def lookup(req):
        lookups.append(req)
        return [
            SimpleNamespace(
                success=True,
                hits=[],
                first_miss_index=0,
                first_miss_result=None,
            )
        ]

    async def submit(req):
        queued.append(req)
        return SimpleNamespace(
            rid=req.rid,
            success=True,
            hits=[SimpleNamespace(key_hash="hit")],
            first_miss_index=1,
            first_miss_result=SimpleNamespace(rid="miss", success=False),
        )

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        c2kv_gist_batch_size=4,
        c2kv_bulk_cache_lookup_communicator=lookup,
        c2kv_extract_batch_collector=SimpleNamespace(submit=submit),
    )
    items = [
        {
            "rid": "hit",
            "input_ids": [1],
            "compression_ratio": 4,
            "projection_set": "history",
            "allow_cache_miss": True,
        },
        {
            "rid": "miss",
            "input_ids": [2],
            "compression_ratio": 8,
            "projection_set": "tool",
            "allow_cache_miss": False,
        },
    ]
    reply = await method(
        manager,
        items,
        outer_request_id="outer",
        measurement_phase="phase",
        materialize_first_miss=True,
    )
    assert reply.first_miss_result.rid == "miss"
    assert len(reply.hits) == 1
    assert lookups == []
    assert len(queued) == 1
    assert queued[0].rid
    assert queued[0].materialize_first_miss is True
    assert [item.allow_cache_miss for item in queued[0].items] == [True, False]
    assert queued[0].items[1].compression_ratio == 8
    assert queued[0].items[1].projection_set == "tool"
    assert queued[0].items[1].c2kv_outer_request_id == "outer"
    assert queued[0].items[1].c2kv_measurement_phase == "phase"

    await method(manager, items, materialize_first_miss=False)
    assert len(queued) == 1
    assert len(lookups) == 1
    assert lookups[0].materialize_first_miss is False
    assert all(not item.allow_cache_miss for item in lookups[0].items)


@pytest.mark.asyncio
async def test_bulk_lookups_can_queue_behind_blocked_scheduler_rpc():
    method = _mixin_methods()["c2kv_bulk_cache_lookup"]
    sent = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def communicator(envelope):
        sent.append([item.items[0].rid for item in envelope.items])
        if len(sent) == 1:
            started.set()
            await release.wait()
        return [
            SimpleNamespace(
                success=True,
                items=[
                    SimpleNamespace(
                        rid=item.rid,
                        success=True,
                        hits=[],
                        first_miss_index=0,
                        first_miss_result=SimpleNamespace(rid=item.items[0].rid),
                    )
                    for item in envelope.items
                ],
            )
        ]

    collector = Collector(communicator, lambda **kwargs: SimpleNamespace(**kwargs), 4)
    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        c2kv_gist_batch_size=4,
        c2kv_bulk_cache_lookup_communicator=lambda _req: pytest.fail(
            "Bulk lookup must enter the collector before its scheduler RPC"
        ),
        c2kv_extract_batch_collector=collector,
    )

    def submit(rid):
        return asyncio.create_task(
            method(
                manager,
                [
                    {
                        "rid": rid,
                        "input_ids": [1],
                        "compression_ratio": 4,
                        "projection_set": "history",
                        "allow_cache_miss": True,
                    }
                ],
                materialize_first_miss=True,
            )
        )

    first = submit("first")
    await started.wait()
    second, third = submit("second"), submit("third")
    await asyncio.sleep(0)
    assert [item.items[0].rid for item, _ in collector._pending] == ["second", "third"]
    release.set()
    replies = await asyncio.gather(first, second, third)
    assert sent == [["first"], ["second", "third"]]
    assert [reply.first_miss_result.rid for reply in replies] == [
        "first",
        "second",
        "third",
    ]
