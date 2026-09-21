"""CPU-only tests for acknowledged request/session timeout cleanup."""

import ast
import asyncio
import functools
import logging
import math
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANAGERS = ROOT / "python/sglang/srt/managers"


def async_test(fn):
    @functools.wraps(fn)
    def run():
        return asyncio.run(fn())

    return run


def method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


abort_and_wait = method(
    MANAGERS / "tokenizer_manager.py",
    "TokenizerManager",
    "abort_request_and_wait",
    {
        "asyncio": asyncio,
        "math": math,
        "CloseSessionReqInput": lambda session_id: SimpleNamespace(
            session_id=session_id
        ),
    },
)


def make_manager(active=None, terminal=None):
    events = []

    async def close_session(obj, request=None, timeout=None):
        events.append(("close", obj.session_id, timeout))
        return True

    manager = SimpleNamespace(
        rid_to_state={} if active is None else {"rid": active},
        terminal_request_states=(
            {} if terminal is None else {"rid": terminal}
        ),
        abort_request=lambda rid: events.append(("abort", rid)),
        close_session=close_session,
    )
    return manager, events


def state(session_id="session", finish_reason=None, terminal=False):
    event = asyncio.Event()
    if terminal:
        event.set()
    item = type("State", (), {})()
    item.obj = SimpleNamespace(session_params={"id": session_id})
    item.terminal_event = event
    item.terminal_finish_reason = finish_reason
    return item


@async_test
async def test_active_abort_closes_only_after_terminal_release_event():
    active = state()
    manager, events = make_manager(active=active)
    task = asyncio.create_task(
        abort_and_wait(
            manager,
            "rid",
            session_id="session",
            close_session=True,
            timeout=1,
        )
    )
    await asyncio.sleep(0)
    assert events == [("abort", "rid")]
    assert not task.done()

    active.terminal_finish_reason = {"type": "abort", "message": "cancelled"}
    active.terminal_event.set()
    result = await task
    assert [event[0] for event in events] == ["abort", "close"]
    assert result["request_status"] == "aborted"
    assert result["session_status"] == "closed"
    assert active.obj._lifecycle_cancel_requested is True


@async_test
async def test_completed_race_uses_weak_terminal_state_and_blocks_late_commit():
    completed = state(finish_reason={"type": "stop"}, terminal=True)
    weak_states = weakref.WeakValueDictionary({"rid": completed})
    manager, events = make_manager()
    manager.terminal_request_states = weak_states

    result = await abort_and_wait(
        manager,
        "rid",
        session_id="session",
        close_session=True,
        timeout=1,
    )

    assert events[0][0] == "close"
    assert not any(event[0] == "abort" for event in events)
    assert result["request_status"] == "completed"
    assert result["session_status"] == "closed"
    assert completed.obj._lifecycle_cancel_requested is True


@async_test
async def test_unknown_request_still_closes_named_session():
    manager, events = make_manager()
    result = await abort_and_wait(
        manager,
        "rid",
        session_id="session",
        close_session=True,
        timeout=1,
    )
    assert events[0][0] == "close"
    assert result["request_status"] == "not_found"
    assert result["session_status"] == "closed"


@async_test
async def test_wrong_session_does_not_abort_or_close():
    active = state(session_id="other")
    manager, events = make_manager(active=active)
    with pytest.raises(ValueError, match="different session"):
        await abort_and_wait(
            manager,
            "rid",
            session_id="session",
            close_session=True,
            timeout=1,
        )
    assert events == []


@async_test
async def test_terminal_wait_timeout_does_not_claim_session_cleanup():
    active = state()
    manager, events = make_manager(active=active)
    with pytest.raises(asyncio.TimeoutError):
        await abort_and_wait(
            manager,
            "rid",
            session_id="session",
            close_session=True,
            timeout=0.01,
        )
    assert events == [("abort", "rid")]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_cleanup_timeout_must_be_finite_and_positive(timeout):
    manager, events = make_manager()
    with pytest.raises(ValueError, match="finite and positive"):
        asyncio.run(
            abort_and_wait(
                manager,
                "rid",
                session_id="session",
                close_session=True,
                timeout=timeout,
            )
        )
    assert events == []


class CloseSessionReqOutput:
    def __init__(self, session_id, success):
        self.session_id = session_id
        self.success = success


close_session = method(
    MANAGERS / "tokenizer_communicator_mixin.py",
    "TokenizerCommunicatorMixin",
    "close_session",
    {
        "asyncio": asyncio,
        "CloseSessionReqOutput": CloseSessionReqOutput,
    },
)
handle_close_output = method(
    MANAGERS / "tokenizer_manager.py",
    "TokenizerManager",
    "_handle_close_session_req_output",
    {"CloseSessionReqOutput": CloseSessionReqOutput},
)


@async_test
async def test_close_waiters_share_one_bounded_scheduler_ack():
    sent = []

    class Sender:
        async def send_pyobj(self, obj):
            sent.append(obj.session_id)

    manager = SimpleNamespace(
        session_close_futures={},
        send_to_scheduler=Sender(),
    )
    obj = SimpleNamespace(session_id="session")
    first = asyncio.create_task(close_session(manager, obj, timeout=1))
    await asyncio.sleep(0)
    second = asyncio.create_task(close_session(manager, obj, timeout=1))
    await asyncio.sleep(0)
    assert sent == ["session"]
    assert not first.done() and not second.done()

    handle_close_output(
        manager, CloseSessionReqOutput(session_id="session", success=True)
    )
    assert await first is True
    assert await second is True
    assert manager.session_close_futures == {}


@async_test
async def test_close_ack_wait_is_bounded():
    class Sender:
        async def send_pyobj(self, obj):
            return None

    manager = SimpleNamespace(
        session_close_futures={},
        send_to_scheduler=Sender(),
    )
    with pytest.raises(asyncio.TimeoutError):
        await close_session(
            manager, SimpleNamespace(session_id="session"), timeout=0.01
        )
    assert manager.session_close_futures == {}


@async_test
async def test_close_send_failure_removes_pending_future():
    class Sender:
        async def send_pyobj(self, obj):
            raise RuntimeError("transport failed")

    manager = SimpleNamespace(
        session_close_futures={},
        send_to_scheduler=Sender(),
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        await close_session(
            manager, SimpleNamespace(session_id="session"), timeout=1
        )
    assert manager.session_close_futures == {}


def test_scheduler_close_ack_is_created_after_session_release():
    events = []

    class Output:
        def __init__(self, session_id, success):
            events.append(("ack", session_id, success))
            self.session_id = session_id
            self.success = success

    close = method(
        MANAGERS / "scheduler.py",
        "Scheduler",
        "close_session",
        {"CloseSessionReqOutput": Output},
    )
    scheduler = SimpleNamespace(
        session_controller=SimpleNamespace(
            close=lambda obj: events.append(("release", obj.session_id)) or True
        )
    )
    result = close(scheduler, SimpleNamespace(session_id="session"))
    assert result.success is True
    assert events == [
        ("release", "session"),
        ("ack", "session", True),
    ]


def test_session_controller_close_reports_missing_session():
    close = method(
        MANAGERS / "session_controller.py",
        "SessionController",
        "close",
        {"CloseSessionReqInput": object, "logger": logging.getLogger(__name__)},
    )
    released = []
    controller = SimpleNamespace(
        sessions={"session": object()},
        _close=lambda session_id: released.append(session_id),
    )
    assert close(controller, SimpleNamespace(session_id="session")) is True
    assert released == ["session"]
    assert close(controller, SimpleNamespace(session_id="missing")) is False
