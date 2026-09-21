"""Exercise the actual abort dataclass and HTTP route without a model."""
import ast
import asyncio
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, ORJSONResponse, Response
from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


@pytest.mark.parametrize("wait,timeout", [(False, False), (True, False), (True, True)])
def test_abort_http_preserves_legacy_and_acknowledged_contract(wait, timeout):
    events = []

    async def abort_wait(**kwargs):
        events.append(("wait", kwargs))
        if timeout:
            raise asyncio.TimeoutError
        return {"rid": kwargs["rid"], "session_id": kwargs["session_id"],
                "request_status": "aborted", "session_status": "closed"}

    app = FastAPI()
    app.state.openai_serving_chat = SimpleNamespace(
        release_persistent_history_session=lambda sid: events.append(("release", sid)))
    manager = SimpleNamespace(abort_request_and_wait=abort_wait,
        abort_request=lambda **kw: events.append(("legacy", kw)))
    scope = dict(globals(), _global_state=SimpleNamespace(tokenizer_manager=manager),
                 _create_error_response=lambda exc: JSONResponse({"error": str(exc)}, status_code=400))
    data_tree = ast.parse((ROOT / "managers/io_struct.py").read_text(encoding="utf-8"))
    nodes = [n for n in data_tree.body if isinstance(n, ast.ClassDef)
             and n.name in {"BaseReq", "AbortReq"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "io_struct.py", "exec"), scope)
    server_tree = ast.parse((ROOT / "entrypoints/http_server.py").read_text(encoding="utf-8"))
    route = next(n for n in server_tree.body if isinstance(n, ast.AsyncFunctionDef)
                 and n.name == "abort_request")
    route.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[route], type_ignores=[])),
                 "http_server.py", "exec"), scope)
    app.post("/abort_request")(scope["abort_request"])
    payload = {"rid": "request1"}
    if wait:
        payload.update(session_id="session1", wait_for_completion=True,
                       close_session=True, timeout=0.25)
    with TestClient(app) as client:
        response = client.post("/abort_request", json=payload)
    if not wait:
        assert response.status_code == 200 and response.content == b""
        assert events == [("legacy", {"rid": "request1", "abort_all": False})]
    else:
        assert events[0] == ("wait", {"rid": "request1", "session_id": "session1",
                                      "close_session": True, "timeout": 0.25})
        assert response.json()["rid"] == "request1"
        assert response.json()["session_id"] == "session1"
        if timeout:
            assert response.status_code == 504
            assert response.json()["request_status"] == "cleanup_timeout"
            assert len(events) == 1
        else:
            assert response.status_code == 200
            assert response.json()["session_status"] == "closed"
            assert events[-1] == ("release", "session1")
