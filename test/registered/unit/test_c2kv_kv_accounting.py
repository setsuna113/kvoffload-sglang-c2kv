import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "c2kv_kv_accounting.py"
)
SPEC = importlib.util.spec_from_file_location("c2kv_kv_accounting", MODULE_PATH)
accounting = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(accounting)


def test_client_hint_does_not_seed_repair_runtime_counters():
    report = accounting.initialize_c2kv_kv_memory_report({
        "full_equivalent_history_tokens": 1000,
        "active_history_kv_tokens": 250,
        "active_raw_repair_tokens": 250,
        "history_kv_selected_token_count": 250,
    })
    assert report["active_history_kv_tokens"] == 0
    assert report["active_raw_repair_tokens"] == 0
    assert report["client_reported_active_counters"] == {
        "active_history_kv_tokens": 250,
        "active_raw_repair_tokens": 250,
    }

    accounting.add_c2kv_kv_memory_tokens(
        report, kind="repair", tokens=250, original_tokens=1000
    )
    assert report["active_history_kv_tokens"] == 250
    assert report["active_raw_repair_tokens"] == 250
    assert report["full_equivalent_history_tokens"] == 1000
    assert report["full_equivalent_history_tokens_source"] == "request_hint"


def test_gist_and_repair_are_each_counted_once():
    report = accounting.initialize_c2kv_kv_memory_report(None)
    accounting.add_c2kv_kv_memory_tokens(
        report, kind="gist", tokens=13, original_tokens=52
    )
    accounting.add_c2kv_kv_memory_tokens(
        report, kind="repair", tokens=51, original_tokens=51
    )
    assert report["active_history_kv_tokens"] == 64
    assert report["active_c2kv_gist_tokens"] == 13
    assert report["active_raw_repair_tokens"] == 51
    assert report["full_equivalent_history_tokens"] == 52
    assert report["full_equivalent_history_tokens_source"] == (
        "scheduler_first_injected_original_legacy"
    )

def test_full_block_uses_the_same_runtime_accounting_contract():
    report = accounting.initialize_c2kv_kv_memory_report({
        "active_history_kv_tokens": 900,
        "active_full_raw_tokens": 900,
    })
    accounting.add_c2kv_kv_memory_tokens(
        report, kind="full", tokens=900, original_tokens=900
    )
    assert report["active_history_kv_tokens"] == 900
    assert report["active_full_raw_tokens"] == 900
    assert report["full_equivalent_history_tokens_source"] == (
        "scheduler_first_injected_original_legacy"
    )
