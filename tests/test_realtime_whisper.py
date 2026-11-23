"""Pure, model-free tests for 30ms VAD segmentation and coalesced 300ms jobs."""

from voice_rag.realtime_whisper import FRAME_BYTES, Job, JobBuffer, UtteranceTracker

FRAME = b"\0" * FRAME_BYTES


def _run(tracker, voiced_sequence, start=0.0):
    jobs = []
    for i, voiced in enumerate(voiced_sequence):
        jobs.extend(tracker.feed(FRAME, voiced, start + i * 0.03))
    return jobs


def test_empty_and_quiet_audio_never_schedules_whisper():
    tracker = UtteranceTracker()
    assert _run(tracker, [False] * 100) == []
    assert tracker.flush(4.0) is None


def test_partial_requests_are_about_300ms_apart():
    tracker = UtteranceTracker(interval_ms=300, silence_ms=750)
    jobs = _run(tracker, [False] * 10 + [True] * 40)
    partials = [job for job in jobs if not job.final]
    assert len(partials) >= 3
    assert all(b.submitted_at - a.submitted_at >= 0.29 for a, b in zip(partials, partials[1:]))
    assert all(b.submitted_at - a.submitted_at <= 0.34 for a, b in zip(partials, partials[1:]))
    assert all(job.pcm for job in partials)
    assert len({job.utterance for job in jobs}) == 1


def test_silence_emits_final_and_new_utterance_starts_fresh():
    tracker = UtteranceTracker(interval_ms=300, silence_ms=750)
    jobs = _run(tracker, [True] * 35 + [False] * 27 + [True] * 36 + [False] * 27)
    final = [job for job in jobs if job.final]
    assert len(final) == 2
    assert final[0].utterance == 1
    assert final[1].utterance == 2
    assert len(final[0].pcm) >= 35 * FRAME_BYTES


def test_no_final_on_single_spurious_speech_frame():
    tracker = UtteranceTracker()
    assert _run(tracker, [False] * 9 + [True] + [False] * 35) == []


def test_flush_finalizes_without_waiting_for_silence():
    tracker = UtteranceTracker()
    _run(tracker, [True] * 12)
    job = tracker.flush(1.0)
    assert job is not None and job.final and job.utterance == 1
    assert tracker.flush(2.0) is None


def test_max_utterance_length_rolls_to_final():
    tracker = UtteranceTracker(max_utterance_s=0.6)
    jobs = _run(tracker, [True] * 25)
    assert sum(job.final for job in jobs) >= 1


def test_pending_partials_coalesce_but_finals_are_kept():
    jobs = JobBuffer()
    jobs.put(Job(1, b"a", False, 0.0))
    jobs.put(Job(1, b"ab", False, 0.3))
    assert jobs.pending() == 1
    jobs.put(Job(1, b"abc", True, 0.5))
    jobs.put(Job(2, b"def", False, 0.6))
    assert jobs.pending() == 2
    assert jobs.take() == Job(1, b"abc", True, 0.5)
    assert jobs.take() == Job(2, b"def", False, 0.6)
    jobs.close()
    assert jobs.take() is None


def test_custom_interval_validation():
    try:
        UtteranceTracker(interval_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid interval must be rejected")

def test_gpu_dependency_error_message():
    import pytest

    from voice_rag.realtime_whisper import (
        _check_gpu_runtime,
        _is_missing_gpu_runtime_error,
    )

    assert _is_missing_gpu_runtime_error("Library libcublas.so.12 is not found")
    assert _is_missing_gpu_runtime_error("Library libcudnn.so.9 is not found")
    assert not _is_missing_gpu_runtime_error("Microphone disconnected")

    with pytest.MonkeyPatch.context() as monkeypatch:
        def fail_library(name):
            raise OSError(f"missing {name}")

        monkeypatch.setattr("voice_rag.realtime_whisper.ctypes.CDLL", fail_library)
        with pytest.raises(SystemExit, match="scripts/run-whisper-cuda.sh"):
            _check_gpu_runtime()
