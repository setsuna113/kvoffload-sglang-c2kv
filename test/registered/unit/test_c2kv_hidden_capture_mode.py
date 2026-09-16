"""Regression tests for the native prompt-last scheduler batch contract."""

from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)


def _batch(forward_mode, *prompt_last_flags, return_hidden_states=True):
    return ScheduleBatch(
        reqs=[
            SimpleNamespace(c2kv_prompt_last_hidden_only=flag)
            for flag in prompt_last_flags
        ],
        forward_mode=forward_mode,
        return_hidden_states=return_hidden_states,
    )


def test_prompt_last_mode_survives_overlap_schedule_batch_copy():
    batch = _batch(ForwardMode.EXTEND, True)

    assert batch.get_capture_hidden_mode() == CaptureHiddenMode.LAST
    assert batch.copy().get_capture_hidden_mode() == CaptureHiddenMode.LAST


def test_prompt_last_capture_is_disabled_during_decode():
    batch = _batch(ForwardMode.DECODE, True)

    assert batch.get_capture_hidden_mode() == CaptureHiddenMode.NULL


def test_mixed_hidden_state_consumers_keep_full_capture():
    batch = _batch(ForwardMode.EXTEND, True, False)

    assert batch.get_capture_hidden_mode() == CaptureHiddenMode.FULL


def test_speculative_capture_mode_is_preserved_without_hidden_state_returns():
    batch = _batch(ForwardMode.EXTEND, return_hidden_states=False)
    batch.spec_info = SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.LAST)

    assert batch.get_capture_hidden_mode() == CaptureHiddenMode.LAST


def test_intermediate_c2kv_round_does_not_advance_empty_output_offsets():
    req = SimpleNamespace(
        c2kv_requeued=True,
        send_token_offset=0,
        send_output_token_logprobs_offset=0,
    )
    scheduler = SimpleNamespace(
        get_load=lambda: None,
        dp_rank=0,
        send_to_detokenizer=SimpleNamespace(send_output=lambda output: None),
    )

    SchedulerOutputProcessorMixin.stream_output_generation(
        scheduler, [req], return_logprob=True
    )

    assert req.send_token_offset == 0
    assert req.send_output_token_logprobs_offset == 0
