import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/observability/paper_telemetry.py"
)
SPEC = importlib.util.spec_from_file_location("paper_telemetry_under_test", MODULE_PATH)
paper_telemetry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper_telemetry
SPEC.loader.exec_module(paper_telemetry)
_PaperTelemetry = paper_telemetry._PaperTelemetry


class _Allocator:
    size = 100

    def available_size(self):
        return 70


class _C2KVPool:
    max_total_tokens = 50

    def current_tokens(self):
        return 5


def test_request_scoped_metrics_include_generation_and_history(monkeypatch, tmp_path):
    log_path = tmp_path / "paper.jsonl"
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY_LOG", str(log_path))
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="server-1",
        outer_request_id="outer-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=100,
    )
    telemetry.set_phase("selection")
    temporary_kv = torch.zeros((2, 3), dtype=torch.float32)
    telemetry.sample(
        "selection_buffers", tensors=[temporary_kv], temporary_kv=True
    )
    req = SimpleNamespace(
        rid="server-1",
        kv_committed_len=42,
        origin_input_ids=list(range(80)),
        kv_memory_report={
            "full_equivalent_history_tokens": 60,
            "active_history_kv_tokens": 15,
            "selection_query_tokens": 7,
            "selection_query_tokens_observed": 7,
            "history_kv_runtime_status": "physical_eviction_applied",
            "history_kv_physical_eviction": {"success": True},
            "history_kv_lifecycle": {"full_history_reprefill_performed": False},
        },
    )
    telemetry.mark_generation_start(req)
    result = telemetry.finish(req=req, success=True)

    assert result["outer_request_id"] == "outer-1"
    assert result["server_request_id"] == "server-1"
    assert result["phase"] == "chat"
    assert result["generation_start"] is not None
    assert result["baseline"]["event"] == "baseline"
    assert result["peak"] is not result["baseline"]
    metrics = result["metrics"]
    assert metrics["generation_active_kv_tokens"] == 42
    assert metrics["gist_generation_duration_ns"] == 0
    assert metrics["generation_active_kv_bytes"] == 168
    assert metrics["whole_full_kv_tokens"] == 100
    assert metrics["whole_active_kv_tokens"] == 42
    assert metrics["history_full_kv_tokens"] == 60
    assert metrics["history_active_kv_tokens"] == 15
    assert metrics["temporary_extraction_recovery_peak_kv_tokens"] == 6
    assert metrics["temporary_extraction_recovery_peak_kv_bytes"] == 24
    assert metrics["full_history_reprefill"] is False
    assert metrics["selection_query_tokens_planned"] == 7
    assert metrics["selection_query_tokens_observed"] == 7
    assert metrics["history_kv_physical_eviction_success"] is True
    assert all(phase["duration_ns"] >= 0 for phase in result["phases"])
    assert {phase["name"] for phase in result["phases"]} == {
        "chat",
        "selection",
        "decode",
    }
    assert json.loads(log_path.read_text(encoding="utf-8"))["metrics"] == metrics


def test_reference_success_separates_semantic_and_storage_runtime_status(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="reference-1",
        outer_request_id="outer-reference-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=100,
    )
    req = SimpleNamespace(
        rid="reference-1",
        kv_committed_len=40,
        origin_input_ids=list(range(80)),
        kv_memory_report={
            "active_history_kv_tokens": 16,
            "history_kv_backend": "reference_attention",
            "reference_attention_backend": "torch_sdpa",
            # This is the legacy nested physical receipt emitted by CUDA v2.
            "history_kv_runtime_status": "physical_eviction_globalized",
            "history_kv_physical_eviction": {
                "success": True,
                "runtime_status": "physical_eviction_globalized",
            },
        },
    )
    telemetry.mark_generation_start(req)
    metrics = telemetry.finish(req=req, success=True)["metrics"]

    assert metrics["history_kv_backend"] == "reference_attention"
    assert metrics["history_kv_runtime_status"] == "reference_attention_ok"
    assert metrics["history_kv_storage_runtime_status"] == (
        "physical_eviction_globalized"
    )
    assert metrics["history_kv_physical_eviction_success"] is True


def test_reference_failure_runtime_status_is_not_overridden():
    req = SimpleNamespace(
        kv_memory_report={
            "history_kv_backend": "reference_attention",
            "reference_attention_backend": "torch_sdpa",
            "history_kv_runtime_status": "physical_eviction_exception",
            "history_kv_physical_eviction": {
                "success": False,
                "runtime_status": "physical_eviction_failed",
            },
        }
    )

    semantics = _PaperTelemetry._req_semantics(req)

    assert semantics["history_kv_runtime_status"] == "physical_eviction_exception"
    assert semantics["history_kv_storage_runtime_status"] is None
    assert semantics["history_kv_physical_eviction_success"] is False


def test_transformed_text_history_is_not_labeled_full(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="text-1",
        outer_request_id="outer-text-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=None,
    )
    req = SimpleNamespace(
        rid="text-1",
        kv_committed_len=30,
        origin_input_ids=list(range(30)),
        kv_memory_report={},
        c2kv_paper_history_full_kv_tokens=None,
        c2kv_paper_history_active_kv_tokens=12,
        c2kv_paper_canonical_full_source=False,
        c2kv_paper_denominator_tokenization_duration_ns=100,
    )
    telemetry.mark_generation_start(req)
    metrics = telemetry.finish(req=req, success=True)["metrics"]
    assert metrics["whole_full_kv_tokens"] is None
    assert metrics["whole_active_kv_tokens"] == 30
    assert metrics["history_full_kv_tokens"] is None
    assert metrics["history_active_kv_tokens"] == 12
    assert metrics["canonical_full_source"] is False
    assert metrics["denominator_tokenization_duration_ns"] == 100


def test_canonical_request_boundary_overrides_runtime_placeholders(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="canonical-1",
        outer_request_id="outer-canonical-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=422,
    )
    req = SimpleNamespace(
        rid="canonical-1",
        kv_committed_len=134,
        kv_memory_report={
            "full_equivalent_history_tokens": 0,
            "active_history_kv_tokens": 96,
        },
        c2kv_paper_history_full_kv_tokens=384,
        c2kv_paper_history_active_kv_tokens=None,
        c2kv_paper_canonical_full_source=True,
        c2kv_paper_whole_full_source="client_native_full_renderer",
    )
    telemetry.mark_generation_start(req)
    metrics = telemetry.finish(req=req, success=True)["metrics"]
    assert metrics["whole_full_kv_tokens"] == 422
    assert metrics["whole_full_kv_tokens_source"] == "client_native_full_renderer"
    assert metrics["history_full_kv_tokens"] == 384
    assert metrics["history_active_kv_tokens"] == 96


def test_native_tool_without_full_count_keeps_history_but_unknown_whole(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="native-tool", outer_request_id="outer-native-tool",
        phase="c2kv_native:generation", kind="generation",
    )
    req = SimpleNamespace(
        rid="native-tool", kv_committed_len=30, kv_memory_report={},
        c2kv_paper_whole_full_kv_tokens=None,
        c2kv_paper_whole_full_source="unknown_missing_client_native_full_renderer",
        c2kv_paper_history_full_kv_tokens=100,
        c2kv_paper_history_active_kv_tokens=12,
        c2kv_paper_canonical_full_source=True,
    )
    telemetry.mark_generation_start(req)
    metrics = telemetry.finish(req=req, success=True)["metrics"]
    assert metrics["whole_full_kv_tokens"] is None
    assert metrics["whole_full_kv_tokens_source"] == (
        "unknown_missing_client_native_full_renderer"
    )
    assert metrics["history_full_kv_tokens"] == 100
    assert metrics["history_active_kv_tokens"] == 12


def test_request_peak_uses_simultaneous_pool_plus_temporary_kv(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")

    class MutableAllocator:
        size = 100
        available = 80

        def available_size(self):
            return self.available

    class EmptyC2KVPool:
        max_total_tokens = 0

        def current_tokens(self):
            return 0

    allocator = MutableAllocator()
    telemetry = _PaperTelemetry()
    telemetry.configure(allocator, EmptyC2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="joint-peak",
        outer_request_id="outer-joint-peak",
        phase="extraction",
        kind="c2kv_extract",
    )

    # First sample: pool20 + temporary10 = joint30.
    telemetry.sample(
        "large_temp_small_pool",
        tensors=[torch.zeros(10, dtype=torch.float32)],
        temporary_kv=True,
    )
    # Second sample: pool39 + temporary2 = joint41.  The requested peak is
    # 41, not pooled max39 + temporary max10 = the impossible value49.
    allocator.available = 61
    telemetry.sample(
        "small_temp_large_pool",
        tensors=[torch.zeros(2, dtype=torch.float32)],
        temporary_kv=True,
    )
    result = telemetry.finish(success=True)
    metrics = result["metrics"]
    assert metrics["request_peak_resident_kv_tokens"] == 41
    assert metrics["request_peak_resident_kv_bytes"] == 164
    assert metrics["request_peak_pooled_resident_kv_tokens"] == 39
    assert metrics["temporary_extraction_recovery_peak_kv_tokens"] == 10
    assert result["peak"]["event"] == "small_temp_large_pool"
    assert result["peak"]["kv"]["simultaneous_temporary_kv_tokens"] == 2


def test_nested_tensor_storage_is_deduplicated_globally():
    base = torch.zeros(8, dtype=torch.float32)
    measured = _PaperTelemetry._tensor_bytes(
        [[base[:4]], [base[4:]], base]
    )
    assert measured["logical_bytes"] == (4 + 4 + 8) * 4
    assert measured["storage_bytes"] == base.untyped_storage().nbytes()


class _TreeCache:
    """Radix-cache stand-in whose evictable size changes between samples."""

    def __init__(self):
        self.evictable = 0
        self.protected = 0

    def evictable_size(self):
        return self.evictable

    def protected_size(self):
        return self.protected


def test_cached_evictable_kv_is_a_line_item_of_the_resident_total(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    cache = _TreeCache()
    cache.evictable = 20  # left behind by an earlier (auxiliary) request
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4, tree_cache=cache)
    telemetry.start(server_request_id="s", outer_request_id="o", phase="chat", kind="generation")
    cache.evictable = 25
    cache.protected = 10
    telemetry.sample("prefill")
    cache.evictable = 3  # evicted to make room; resident total is unchanged
    telemetry.sample("decode")
    result = telemetry.finish(req=SimpleNamespace(rid="s", kv_committed_len=30), success=True)
    metrics = result["metrics"]
    # The resident total still counts every live pool slot (30 main + 5 c2kv).
    assert metrics["request_peak_resident_kv_tokens"] == 35
    assert metrics["request_peak_resident_kv_bytes"] == 140
    # Evictable cache is reported alongside, never subtracted.
    assert metrics["baseline_cached_evictable_kv_tokens"] == 20
    assert metrics["cached_evictable_kv_peak_tokens"] == 25
    assert metrics["cached_evictable_kv_peak_bytes"] == 100
    # The resident total never rose above the baseline, so the peak is the
    # first sample at that level (the baseline), when 20 tokens were cache.
    assert metrics["request_peak_cached_evictable_kv_tokens"] == 20


def test_peak_keeps_first_sample_so_released_kv_is_not_reported_as_cache(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)

    class _GrowingAllocator:
        size = 100
        used = 30

        def available_size(self):
            return self.size - self.used

    allocator = _GrowingAllocator()
    cache = _TreeCache()
    telemetry = _PaperTelemetry()
    telemetry.configure(allocator, _C2KVPool(), bytes_per_kv_token=4, tree_cache=cache)
    cache.evictable = 20  # auxiliary call's leftover cache
    telemetry.start(server_request_id="s", outer_request_id="o", phase="chat", kind="generation")
    allocator.used = 40  # this request prefills 10 new tokens, which are protected
    cache.protected = 10
    telemetry.sample("prefill")
    # Finishing moves the request's slots into the evictable cache: same
    # resident total, but now all 30 cached tokens are evictable.
    cache.protected = 0
    cache.evictable = 30
    result = telemetry.finish(req=SimpleNamespace(rid="s", kv_committed_len=10), success=True)
    metrics = result["metrics"]
    assert metrics["request_peak_resident_kv_tokens"] == 45
    assert metrics["request_peak_cached_evictable_kv_tokens"] == 20
    assert result["peak"]["kv"]["cached_protected_kv_tokens"] == 10
    assert metrics["cached_evictable_kv_peak_tokens"] == 30


def test_uninspectable_c2kv_pool_marks_cache_breakdown_unavailable(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(server_request_id="s", outer_request_id="o", phase="chat", kind="generation")
    result = telemetry.finish(req=SimpleNamespace(rid="s", kv_committed_len=1), success=True)
    assert result["metrics"]["cached_evictable_kv_peak_tokens"] == 0
    assert result["metrics"]["request_peak_cached_evictable_kv_bytes"] == 0
    assert result["metrics"]["request_peak_resident_kv_tokens"] == 35
    assert result["metrics"]["request_peak_c2kv_cache_accounting_available"] is False


def test_c2kv_lru_entries_count_as_evictable_without_radix_cache(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)

    class _InspectableC2KVPool(_C2KVPool):
        _cache = {
            "old": SimpleNamespace(gist_len=3),
            "active": SimpleNamespace(gist_len=2),
        }
        _pin_counts = {"active": 1}

    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _InspectableC2KVPool(), bytes_per_kv_token=4)
    snapshot = telemetry._kv_snapshot()
    assert snapshot["resident_kv_tokens"] == 35
    assert snapshot["c2kv_live_kv_tokens"] == 5
    assert snapshot["c2kv_cached_evictable_kv_tokens"] == 3
    assert snapshot["c2kv_cached_pinned_kv_tokens"] == 2
    assert snapshot["cached_evictable_kv_tokens"] == 3
    assert snapshot["c2kv_cache_accounting_available"] is True

    telemetry.start(server_request_id="s", outer_request_id="o", phase="chat", kind="generation")
    result = telemetry.finish(req=SimpleNamespace(rid="s", kv_committed_len=1), success=True)
    metrics = result["metrics"]
    assert metrics["request_peak_resident_kv_tokens"] == 35
    assert metrics["request_peak_cached_evictable_kv_tokens"] == 3
    assert metrics["request_peak_c2kv_cached_evictable_kv_bytes"] == 12
    assert metrics["request_peak_c2kv_cache_accounting_available"] is True


def test_extract_wrapper_persists_cache_miss_only_gist_duration(
    monkeypatch, tmp_path
):
    log_path = tmp_path / "extract.jsonl"
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY_LOG", str(log_path))
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    monkeypatch.setattr(paper_telemetry, "start_request", telemetry.start)
    monkeypatch.setattr(paper_telemetry, "finish_request", telemetry.finish)

    class Handler:
        @paper_telemetry.measure_synchronous_request("c2kv_extract", "extraction")
        def extract(self, recv_req, output):
            return output

    request = SimpleNamespace(
        rid="extract-1",
        c2kv_outer_request_id="outer-1",
        c2kv_measurement_phase="native:extraction",
        input_ids=[1, 2, 3],
    )
    miss = SimpleNamespace(
        success=True,
        error=None,
        cache_hit=False,
        gist_generation_duration_ns=77,
        extraction_duration_ns=None,
        paper_measurement=None,
    )
    hit = SimpleNamespace(
        success=True,
        error=None,
        cache_hit=True,
        gist_generation_duration_ns=0,
        extraction_duration_ns=None,
        paper_measurement=None,
    )

    miss_result = Handler().extract(request, miss)
    hit_result = Handler().extract(request, hit)

    persisted = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert (
        miss_result.extraction_duration_ns
        == miss_result.paper_measurement["duration_ns"]
    )
    assert (
        hit_result.extraction_duration_ns
        == hit_result.paper_measurement["duration_ns"]
    )
    for report, expected_duration, expected_hit in (
        (miss_result.paper_measurement, 77, False),
        (hit_result.paper_measurement, 0, True),
        (persisted[0], 77, False),
        (persisted[1], 0, True),
    ):
        assert report["metrics"]["extraction_duration_ns"] == report["duration_ns"]
        assert report["metrics"]["gist_generation_duration_ns"] == expected_duration
        assert report["metrics"]["cache_hit"] is expected_hit


def test_reference_payload_counts_shared_session_once_and_temporary_overlap(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    layer = SimpleNamespace(key=torch.zeros(2, 3, 4), value=torch.zeros(2, 3, 4),
                            positions=torch.arange(3).expand(2, -1).clone())
    state = SimpleNamespace(layers={0: layer})
    slot = SimpleNamespace(history_kv_reference_state=state)
    cache = SimpleNamespace(slots={"s": slot})
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), None, bytes_per_kv_token=64, tree_cache=cache)
    telemetry.start(server_request_id="r", outer_request_id="r", phase="prefill", kind="generation")
    req = SimpleNamespace(rid="r", kv_committed_len=4, history_kv_reference_state=state)
    telemetry.bind_request(req)
    snap = telemetry.sample("shared")
    external = layer.key.nbytes + layer.value.nbytes + layer.positions.nbytes
    assert snap["kv"]["reference_history_resident_bytes"] == external
    assert snap["kv"]["request_resident_kv_bytes"] == 30 * 64 + external
    new_tensors = [layer.key.clone(), layer.value.clone(), layer.positions.clone()]
    peak = telemetry.sample("new_state_before_swap", tensors=new_tensors, temporary_kv=True)
    assert peak["kv"]["request_resident_kv_bytes"] == 30 * 64 + 2 * external
    telemetry.mark_generation_start(req)
    generation = telemetry._active["generation_start"]
    assert generation["request_active_kv_bytes"] == 4 * 64 + external
    assert generation["request_active_kv_tokens"] == 4 + 3
    # A new object owned by the request while the prior session state is live
    # represents two allocations and must contribute both resident payloads.
    req.history_kv_reference_state = SimpleNamespace(layers={0: SimpleNamespace(
        key=new_tensors[0], value=new_tensors[1], positions=new_tensors[2])})
    assert telemetry.sample("overlap")["kv"]["reference_history_resident_bytes"] == 2 * external
    cache.slots.clear()
    assert telemetry.sample("released")["kv"]["reference_history_resident_bytes"] == external


def test_reference_recovery_snapshot_is_resident_even_when_not_selected(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    def state():
        return SimpleNamespace(layers={0: SimpleNamespace(
            key=torch.zeros(1, 2, 4), value=torch.zeros(1, 2, 4),
            positions=torch.arange(2).view(1, 2))})
    live, baseline = state(), state()
    owner = SimpleNamespace(history_kv_reference_state=live,
                            reference_decode_baseline_state=baseline,
                            reference_decode_persistent_state=baseline)
    telemetry = _PaperTelemetry()
    telemetry.configure(None, None, bytes_per_kv_token=32,
                        tree_cache=SimpleNamespace(slots={"s": owner}))
    assert telemetry._kv_snapshot()["request_resident_kv_bytes"] == 2 * (64 + 16)
    assert telemetry._reference_payload([owner], include_snapshots=False)["bytes"] == 64 + 16


def test_generation_start_excludes_overlap_decode_reservation(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    layer = SimpleNamespace(
        key=torch.zeros(2, 3, 4), value=torch.zeros(2, 3, 4),
        positions=torch.arange(3).expand(2, -1).clone(),
    )
    state = SimpleNamespace(layers={0: layer})
    reference_bytes = layer.key.nbytes + layer.value.nbytes + layer.positions.nbytes
    measured = []
    for reserved_decode_tokens in (0, 1):
        telemetry = _PaperTelemetry()
        telemetry.configure(_Allocator(), None, bytes_per_kv_token=64)
        telemetry.start(
            server_request_id="r", outer_request_id="r",
            phase="prefill", kind="generation",
        )
        req = SimpleNamespace(
            rid="r", kv_committed_len=125 + reserved_decode_tokens,
            history_kv_reference_state=state,
        )
        telemetry.mark_generation_start(req, normal_kv_tokens=125)
        metrics = telemetry.finish(req=req, success=True)["metrics"]
        measured.append((metrics["generation_active_kv_tokens"],
                         metrics["generation_active_kv_bytes"]))
        assert req.kv_committed_len == 125 + reserved_decode_tokens
    assert measured == [(128, 125 * 64 + reference_bytes)] * 2


def test_concurrent_generation_and_extraction_keep_request_identity(monkeypatch, tmp_path):
    log_path = tmp_path / "concurrent.jsonl"
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.setenv("C2KV_PAPER_CONCURRENT", "1")
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY_LOG", str(log_path))
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    monkeypatch.setattr(paper_telemetry, "start_request", telemetry.start)
    monkeypatch.setattr(paper_telemetry, "finish_request", telemetry.finish)

    def generation(rid, tokens):
        telemetry.start(server_request_id=rid, outer_request_id=f"outer-{rid}",
                        phase="prefill", kind="generation")
        req = SimpleNamespace(rid=rid, kv_committed_len=tokens,
                              finished_reason=SimpleNamespace(to_json=lambda: {"type": "stop"}))
        telemetry.bind_request(req)
        return req

    first = generation("first", 11)
    second = generation("second", 22)
    telemetry.set_phase("selection", req=first)
    telemetry.mark_generation_start(second)
    telemetry.mark_generation_start(first)

    class Handler:
        @paper_telemetry.measure_synchronous_request("c2kv_extract", "extraction")
        def extract(self, recv_req):
            assert set(telemetry._actives) == {"first", "second", "extract"}
            telemetry.sample("extract_temporary", tensors=[torch.zeros((2, 3))],
                             temporary_kv=True)
            return SimpleNamespace(success=True, error=None, cache_hit=False,
                                   gist_generation_duration_ns=7,
                                   extraction_duration_ns=None, paper_measurement=None)

    extraction = Handler().extract(SimpleNamespace(
        rid="extract", c2kv_outer_request_id="outer-extract",
        c2kv_measurement_phase="extraction", input_ids=[1, 2, 3]))
    assert set(telemetry._actives) == {"first", "second"}
    assert extraction.paper_measurement["server_request_id"] == "extract"
    assert extraction.paper_measurement["metrics"]["gist_generation_duration_ns"] == 7
    assert extraction.paper_measurement["metrics"]["temporary_extraction_recovery_peak_kv_tokens"] == 6

    second_result = telemetry.report(second, finalize=True)
    first_result = telemetry.report(first, finalize=True)
    assert [second_result["server_request_id"], first_result["server_request_id"]] == ["second", "first"]
    assert second_result["metrics"]["generation_active_kv_tokens"] == 22
    assert first_result["metrics"]["generation_active_kv_tokens"] == 11
    assert first_result["metrics"]["temporary_extraction_recovery_peak_kv_tokens"] == 0
    assert second_result["metrics"]["temporary_extraction_recovery_peak_kv_tokens"] == 0
    assert {phase["name"] for phase in first_result["phases"]} == {"prefill", "selection", "decode"}
    assert {phase["name"] for phase in second_result["phases"]} == {"prefill", "decode"}
    assert all(row["metrics"]["memory_scope"] == "process_shared_during_overlap"
               for row in (first_result, second_result, extraction.paper_measurement))
    assert all(row["metrics"]["torch_peak_scope"] == "process_since_idle_reset"
               for row in (first_result, second_result, extraction.paper_measurement))
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert {row["server_request_id"] for row in rows} == {"first", "second", "extract"}
    assert all(row["error"] != "superseded_by_next_request" for row in rows)
    assert telemetry._actives == {}


def test_concurrent_start_does_not_reset_global_peak_for_overlapping_request(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.setenv("C2KV_PAPER_CONCURRENT", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: resets.append(1))
    telemetry = _PaperTelemetry()
    monkeypatch.setattr(telemetry, "_torch_snapshot", lambda: {
        "allocated_bytes": 10, "reserved_bytes": 20,
        "peak_allocated_bytes": 30, "peak_reserved_bytes": 40,
    })
    monkeypatch.setattr(telemetry, "_nvml_process_bytes", lambda: None)
    telemetry.start(server_request_id="a", outer_request_id="a", phase="prefill", kind="generation")
    telemetry.start(server_request_id="b", outer_request_id="b", phase="prefill", kind="generation")
    assert len(resets) == 1
    assert telemetry.finish(server_request_id="a")["metrics"]["memory_scope"] == "process_shared_during_overlap"
    telemetry.finish(server_request_id="b")
    telemetry.start(server_request_id="c", outer_request_id="c", phase="prefill", kind="generation")
    assert len(resets) == 2


def test_default_single_flight_still_supersedes_previous_request(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_CONCURRENT", raising=False)
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    telemetry.start(server_request_id="a", outer_request_id="a", phase="prefill", kind="generation")
    telemetry.start(server_request_id="b", outer_request_id="b", phase="prefill", kind="generation")
    assert telemetry.report(SimpleNamespace(rid="a"), finalize=False)["error"] == "superseded_by_next_request"
    assert list(telemetry._actives) == ["b"]


def test_raw_prefix_receipt_is_copied_into_the_request_ledger(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    req = SimpleNamespace(rid="raw-prefix", c2kv_raw_prefix_cache={
        "enabled": True, "status": "hit", "hit_tokens": 4,
        "inserted_tokens": 0, "prefix_tokens": 4, "reason": None})
    telemetry.start(server_request_id=req.rid, outer_request_id=req.rid,
                    phase="prefill", kind="generation")
    receipt = telemetry.finish(req=req)["metrics"]["c2kv_raw_prefix_cache"]
    assert receipt == req.c2kv_raw_prefix_cache
    req.c2kv_raw_prefix_cache["hit_tokens"] = 0
    assert receipt["hit_tokens"] == 4
