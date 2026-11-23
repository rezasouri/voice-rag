"""Offline Persian microphone transcription with 300 ms update requests.

Microphone -> 30ms WebRTC VAD frames -> incremental utterance snapshots -> one
nonblocking faster-whisper worker. Updates are scheduled every 300ms, but decoding
time depends on the device/model and is NOT guaranteed to finish within 300ms.
No API key or server is required.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2  # mono signed int16 PCM


@dataclass(frozen=True)
class Job:
    utterance: int
    pcm: bytes
    final: bool
    submitted_at: float


@dataclass(frozen=True)
class Result:
    utterance: int
    text: str
    final: bool
    inference_ms: int
    error: str | None = None


class UtteranceTracker:
    """Pure state machine: accurate segment boundaries with WebRTC-VAD decisions.

    First two consecutive voiced 30-ms frames activate a segment. The preceding
    300ms are retained to avoid cutting the beginning of a word.
    """

    def __init__(
        self,
        *,
        interval_ms: int = 300,
        silence_ms: int = 750,
        max_utterance_s: float = 20.0,
    ) -> None:
        if interval_ms < 100 or interval_ms > 2_000:
            raise ValueError("interval_ms must be within 100..2000")
        if silence_ms < FRAME_MS:
            raise ValueError("silence_ms must be >= 30")
        if max_utterance_s <= 0.5:
            raise ValueError("max_utterance_s must be > 0.5")
        self.interval = interval_ms / 1000.0
        self.end_frames = math.ceil(silence_ms / FRAME_MS)
        self.max_frames = math.ceil(max_utterance_s * 1000 / FRAME_MS)
        self.preroll: deque[bytes] = deque(maxlen=10)
        self.active: list[bytes] = []
        self.utterance = 0
        self.candidate_frames = 0
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_update_at = 0.0

    def feed(self, frame: bytes, voiced: bool, now: float) -> list[Job]:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"Expected {FRAME_BYTES} bytes per 30ms frame")
        if not self.active:
            self.preroll.append(frame)
            self.candidate_frames = self.candidate_frames + 1 if voiced else 0
            if self.candidate_frames < 2:
                return []
            self.utterance += 1
            self.active = list(self.preroll)
            self.preroll.clear()
            self.speech_frames = 2
            self.silence_frames = 0
            self.last_update_at = now
            self.candidate_frames = 0
            return []

        self.active.append(frame)
        if voiced:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        if self.silence_frames >= self.end_frames or len(self.active) >= self.max_frames:
            # Trim the trailing silence but keep one 30ms frame for natural word endings.
            trim = max(0, self.silence_frames - 1)
            recording = self.active[:-trim] if trim else self.active
            job = Job(self.utterance, b"".join(recording), True, now)
            # Remember the last few frames for continuity when the next phrase starts.
            self.preroll.extend(self.active[-10:])
            self.active = []
            self.silence_frames = 0
            self.speech_frames = 0
            self.candidate_frames = 0
            return [job]

        if now - self.last_update_at >= self.interval and self.speech_frames >= 2:
            self.last_update_at = now
            return [Job(self.utterance, b"".join(self.active), False, now)]
        return []

    def flush(self, now: float) -> Job | None:
        if not self.active:
            return None
        job = Job(self.utterance, b"".join(self.active), True, now)
        self.active = []
        return job


class JobBuffer:
    """One latest partial per utterance; final jobs are NEVER discarded.

    A slow model cannot create an infinite queue of obsolete partial jobs.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._jobs: deque[Job] = deque()
        self._closed = False

    def put(self, job: Job) -> None:
        with self._cv:
            if self._closed:
                raise RuntimeError("JobBuffer is closed")
            self._jobs = deque(
                j for j in self._jobs if j.final or j.utterance != job.utterance
            )
            self._jobs.append(job)
            self._cv.notify()

    def take(self) -> Job | None:
        with self._cv:
            while not self._jobs and not self._closed:
                self._cv.wait()
            return self._jobs.popleft() if self._jobs else None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return len(self._jobs)


def _decode_worker(
    model: object,
    jobs: JobBuffer,
    output: queue.Queue[Result],
) -> None:
    import numpy as np

    while (job := jobs.take()) is not None:
        started = time.monotonic()
        try:
            pcm = np.frombuffer(job.pcm, dtype="<i2").astype(np.float32) / 32768.0
            # The returned iterable is lazy; iterating actually performs inference.
            segments, _ = model.transcribe(  # type: ignore[attr-defined]
                pcm,
                language="fa",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,  # segmentation handled by WebRTC VAD above
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            output.put(
                Result(job.utterance, text, job.final, int((time.monotonic() - started) * 1000))
            )
        except Exception as exc:  # noqa: BLE001 - surface inference errors to CLI without losing mic
            output.put(
                Result(
                    job.utterance,
                    "",
                    job.final,
                    int((time.monotonic() - started) * 1000),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )


def _display(result: Result, *, finished: set[int], output_path: Path | None) -> None:
    if result.utterance in finished:
        return
    if result.error:
        print(f"\n[خطای مدل / Model error] {result.error}", file=sys.stderr, flush=True)
    if result.final:
        finished.add(result.utterance)
        print(
            f"\r\033[2K[{result.utterance}] {result.text or '(گفتاری تشخیص داده نشد)'}"
            f"  (inference={result.inference_ms}ms)",
            flush=True,
        )
        if output_path and result.text:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(result.text + "\n")
    elif result.text:
        print(f"\r\033[2K… {result.text}", end="", flush=True)


def _drain(
    responses: queue.Queue[Result],
    *,
    finished: set[int],
    output_path: Path | None,
) -> None:
    while True:
        try:
            result = responses.get_nowait()
        except queue.Empty:
            break
        _display(result, finished=finished, output_path=output_path)


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live, local Persian speech-to-text using faster-whisper; no API key",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="small", help="tiny, base, small, medium, large-v3")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-type",
        choices=("auto", "int8", "float16", "float32", "int8_float16"),
        default="auto",
    )
    parser.add_argument(
        "--interval-ms", type=int, default=300, help="Time between ASR update requests"
    )
    parser.add_argument("--silence-ms", type=int, default=750, help="Finalize after silence")
    parser.add_argument("--max-utterance-s", type=float, default=20.0)
    parser.add_argument("--vad-mode", type=int, choices=range(4), default=2)
    parser.add_argument("--input-device", default=None, help="Microphone ID or name")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--output", type=Path, help="Append finalized sentences to a text file")
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    # Import native packages only in the real CLI path; tests require no mic or model.
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "Missing audio dependency. Run: uv sync --extra mic\n"
            "On Ubuntu install: sudo apt install libportaudio2"
        ) from exc

    if args.list_devices:
        print(sd.query_devices())
        return

    try:
        tracker = UtteranceTracker(
            interval_ms=args.interval_ms,
            silence_ms=args.silence_ms,
            max_utterance_s=args.max_utterance_s,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        import webrtcvad
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Missing microphone ASR packages. Run: uv sync --extra mic") from exc

    print(f"Loading faster-whisper model={args.model}, device={args.device} …", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    vad = webrtcvad.Vad(args.vad_mode)
    frames: queue.Queue[bytes] = queue.Queue(maxsize=250)
    results: queue.Queue[Result] = queue.Queue()
    jobs = JobBuffer()
    finished: set[int] = set()
    worker = threading.Thread(
        target=_decode_worker, args=(model, jobs, results), name="whisper-worker", daemon=True
    )
    dropped = 0

    def callback(indata: object, frame_count: int, time_info: object, status: object) -> None:
        nonlocal dropped
        if status:
            dropped += 1
        if frame_count != FRAME_SAMPLES:
            dropped += 1
            return
        try:
            frames.put_nowait(bytes(indata))  # no model inference inside audio callback
        except queue.Full:
            dropped += 1

    worker.start()
    print(
        f"🎙️  Speak Persian. Update request every {args.interval_ms}ms. "
        f"End-of-utterance silence: {args.silence_ms}ms. Ctrl+C to stop.",
        flush=True,
    )
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="int16",
            device=args.input_device,
            callback=callback,
        ):
            while True:
                try:
                    frame = frames.get(timeout=0.05)
                except queue.Empty:
                    _drain(results, finished=finished, output_path=args.output)
                    continue
                try:
                    voiced = bool(vad.is_speech(frame, SAMPLE_RATE))
                except Exception as exc:
                    raise RuntimeError(f"VAD failed on microphone frame: {exc}") from exc
                for job in tracker.feed(frame, voiced, time.monotonic()):
                    jobs.put(job)
                _drain(results, finished=finished, output_path=args.output)
    except KeyboardInterrupt:
        print("\nStopping…", flush=True)
    finally:
        final = tracker.flush(time.monotonic())
        if final:
            jobs.put(final)
        jobs.close()
        worker.join(timeout=10)
        _drain(results, finished=finished, output_path=args.output)
        if dropped:
            print(f"Warning: {dropped} microphone callbacks reported overflow/drop", file=sys.stderr)
        if worker.is_alive():
            print("Warning: inference worker is still finishing; last text might be missing.")


if __name__ == "__main__":
    main()
