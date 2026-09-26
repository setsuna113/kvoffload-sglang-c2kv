"""Event-loop-local FIFO collector for C2KV extracts and bulk lookups."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class C2KVExtractBatchCollector:
    def __init__(self, communicator: Callable, envelope_type: Callable, max_size: int):
        if not 1 <= max_size <= 4:
            raise ValueError("C2KV gist batch size must be between 1 and 4")
        self._communicator = communicator
        self._envelope_type = envelope_type
        self._max_size = max_size
        self._pending = deque()
        self._flush_task = None

    async def submit(self, item: Any) -> Any:
        future = asyncio.get_running_loop().create_future()
        self._pending.append((item, future))
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush())
        return await future

    async def _flush(self):
        active = []
        try:
            # Let requests already scheduled in this event-loop turn join the batch.
            await asyncio.sleep(0)
            while self._pending:
                active = []
                while self._pending and len(active) < self._max_size:
                    item, future = self._pending.popleft()
                    if not future.cancelled():
                        active.append((item, future))
                if not active:
                    continue

                try:
                    replies = await self._communicator(
                        self._envelope_type(items=[item for item, _ in active])
                    )
                    if len(replies) != 1:
                        raise RuntimeError("C2KV extract batch expected one DP reply")
                    reply = replies[0]
                    if not reply.success:
                        raise RuntimeError(reply.error or "C2KV extract batch failed")
                    if len(reply.items) != len(active):
                        raise RuntimeError("C2KV extract batch reply count mismatch")
                    for (item, _), result in zip(active, reply.items):
                        result_rid = getattr(result, "rid", None)
                        if result_rid != item.rid and (
                            result_rid is not None
                            or hasattr(item, "materialize_first_miss")
                        ):
                            raise RuntimeError(
                                "C2KV extract batch reply order mismatch"
                            )
                except Exception as exc:  # noqa: BLE001 - propagate RPC failures to callers
                    for _, future in active:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    for (_, future), result in zip(active, reply.items):
                        if not future.done():
                            future.set_result(result)
                active = []
        except asyncio.CancelledError:
            pending = active + list(self._pending)
            self._pending.clear()
            for _, future in pending:
                future.cancel()
            raise
        except Exception as exc:
            logger.exception("C2KV extract batch collector stopped")
            pending = active + list(self._pending)
            self._pending.clear()
            for _, future in pending:
                if not future.done():
                    future.set_exception(exc)
        finally:
            self._flush_task = None
