# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Real-inference verification for std-faster-whisper.

Runs ACTUAL faster-whisper inference (downloads the tiny model on first use) over
a real audio file: a batch transcription and a live windowed-streaming run that
feeds the audio in chunks. Prints the observed transcript, timing, and the
streaming partial/final event sequence.

Usage:
    uv run python scripts/verify_inference.py <audio_path> [model_key]

Defaults: model_key = faster-whisper/tiny. Uses device=cpu, compute_type=int8
(fast on Apple Silicon / CPU; there is no CUDA on this machine).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from standard_asr import RuntimeParams, discover_models
from standard_asr.audio_format import AudioFormat
from standard_asr.utils.audio_loader import load_audio


def _decode_to_pcm16(audio_path: str) -> tuple[bytes, float]:
    """Decode any audio file to 16 kHz mono pcm_s16le bytes (for streaming feed).

    Returns:
        ``(pcm_bytes, duration_seconds)``.
    """
    samples = np.asarray(
        load_audio(audio_path, target_sr=16000, target_channels=1), dtype=np.float32
    )
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, len(samples) / 16000.0


def run_batch(model_key: str, audio_path: str) -> None:
    """Run and print a real batch transcription."""
    print(f"\n{'=' * 70}\nBATCH  {model_key}\n{'=' * 70}")
    engine = discover_models().create(model_key, device="cpu", compute_type="int8")
    t0 = time.perf_counter()
    result = engine.transcribe(audio_path, RuntimeParams(language="en"))
    elapsed = time.perf_counter() - t0
    print(f"detected_language : {result.detected_language} (conf={result.language_confidence})")
    print(f"duration          : {result.duration:.2f}s")
    print(f"wall time         : {elapsed:.2f}s")
    print(f"segments          : {len(result.segments or [])}")
    print(f"\nTRANSCRIPT:\n{result.text.strip()}\n")
    if result.segments:
        print("First 3 segments:")
        for seg in result.segments[:3]:
            print(f"  [{seg.start:6.2f}-{seg.end:6.2f}] {seg.text.strip()}")


def run_batch_words(model_key: str, audio_path: str) -> None:
    """Run a batch transcription with word timestamps and print the first words."""
    print(f"\n{'=' * 70}\nBATCH + WORD TIMESTAMPS  {model_key}\n{'=' * 70}")
    engine = discover_models().create(model_key, device="cpu", compute_type="int8")
    result = engine.transcribe(audio_path, RuntimeParams(language="en", word_timestamps="word"))
    words = result.words or []
    print(f"word count: {len(words)}")
    for w in words[:10]:
        print(f"  [{w.start:6.2f}-{w.end:6.2f}] {w.text!r} (p={w.probability})")


async def run_streaming(model_key: str, audio_path: str) -> None:
    """Run a real windowed-streaming session, feeding the audio in chunks."""
    print(f"\n{'=' * 70}\nSTREAMING (windowed)  {model_key}\n{'=' * 70}")
    pcm, dur = _decode_to_pcm16(audio_path)
    print(f"feeding {dur:.1f}s of 16 kHz mono PCM in 1.0s chunks...\n")
    engine = discover_models().create(model_key, device="cpu", compute_type="int8")
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)

    chunk_bytes = 16000 * 2  # 1.0s of pcm_s16le
    chunks = [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]

    partials = finals = progress = 0
    t0 = time.perf_counter()
    async with engine.start_transcription(
        audio_format=fmt, params=RuntimeParams(language="en")
    ) as session:
        session.feed(chunks)
        async for event in session:
            if event.type == "partial":
                partials += 1
                print(
                    f"  partial[{event.segment_id}] (su={event.stable_until}): "
                    f"{(event.text or '').strip()[:80]}"
                )
            elif event.type == "final":
                finals += 1
                print(f"  FINAL  [{event.segment_id}]: {(event.text or '').strip()}")
            elif event.type == "progress":
                progress += 1
            elif event.type == "done":
                print("  done")
            elif event.type == "error":
                print(f"  ERROR {event.code}: {event.extra.get('detail')}")
    elapsed = time.perf_counter() - t0
    print(f"\nevents: {partials} partial, {finals} final, {progress} progress")
    print(f"wall time: {elapsed:.2f}s")
    print(f"\nREDUCED STREAM RESULT:\n{session.result().text.strip()}\n")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio_path = sys.argv[1]
    model_key = sys.argv[2] if len(sys.argv) > 2 else "faster-whisper/tiny"
    if not Path(audio_path).exists():
        print(f"audio not found: {audio_path}")
        return 2
    run_batch(model_key, audio_path)
    run_batch_words(model_key, audio_path)
    asyncio.run(run_streaming(model_key, audio_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
