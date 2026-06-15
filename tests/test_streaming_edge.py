# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Edge-case streaming tests (explicit language, word-level tail, guards)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from standard_asr import RuntimeParams, TranscriptionEvent
from standard_asr.audio_format import AudioFormat
from standard_asr.exceptions import TranscriptionError
from standard_asr.runtime_params import WordTimestampGranularity

from std_faster_whisper import TinyASR
from std_faster_whisper._streaming import FasterWhisperStreamingSession
from std_faster_whisper.engine import _prepared_to_pcm

from .conftest import FakeSegment, FakeWhisperModel, FakeWord, silent_pcm

_FMT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


async def test_streaming_explicit_language_forwarded(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # An explicit (region-tagged) language is reduced to the primary subtag and
    # forwarded to faster-whisper on every decode.
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "hola")]
    engine = TinyASR()
    async with engine.start_transcription(
        audio_format=_FMT, params=RuntimeParams(language="es-ES")
    ) as session:
        session.feed([silent_pcm(2.0) for _ in range(3)])
        async for _event in session:
            pass
    assert fake_faster_whisper.last_transcribe_kwargs["language"] == "es"


async def test_streaming_partial_tail_carries_words(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # When word timestamps are requested, the in-progress partial tail carries
    # accumulated word-level detail.
    def segments_fn(audio: Any, _kwargs: dict[str, Any]) -> list[FakeSegment]:
        secs = len(audio) / 16000.0
        words = [FakeWord(0.0, 0.5, "Hi"), FakeWord(0.5, secs, "there")]
        return [FakeSegment(0.0, secs, "Hi there", words=words)]

    fake_faster_whisper.segments_fn = segments_fn
    engine = TinyASR()
    events: list[TranscriptionEvent] = []
    async with engine.start_transcription(
        audio_format=_FMT,
        params=RuntimeParams(word_timestamps=WordTimestampGranularity.WORD),
    ) as session:
        session.feed([silent_pcm(2.0) for _ in range(3)])
        async for event in session:
            events.append(event)
    assert fake_faster_whisper.last_transcribe_kwargs["word_timestamps"] is True
    partials_with_words = [e for e in events if e.type == "partial" and e.words]
    assert partials_with_words
    assert [w.text for w in partials_with_words[-1].words] == ["Hi", "there"]


def test_append_pcm_ignores_empty_and_accumulates(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    session = FasterWhisperStreamingSession(TinyASR(), RuntimeParams())
    assert session._window_seconds() == 0.0
    session._append_pcm(b"")  # empty -> no-op
    assert session._window_seconds() == 0.0
    session._append_pcm(b"\x00")  # non-empty but < 1 sample -> decodes empty, no-op
    assert session._window_seconds() == 0.0
    session._append_pcm(silent_pcm(1.0))  # accumulate
    assert session._window_seconds() == pytest.approx(1.0)
    session._append_pcm(silent_pcm(0.5))  # concat onto existing window
    assert session._window_seconds() == pytest.approx(1.5)


def test_loading_disables_tqdm_monitor_thread(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # Loading the model must disable tqdm's monitor daemon thread so the sync
    # bridge does not leak it (a real compliance failure with the live model).
    import tqdm

    from std_faster_whisper.engine import _disable_tqdm_monitor_thread

    tqdm.tqdm.monitor_interval = 10  # simulate the default
    TinyASR().prepare()
    assert tqdm.tqdm.monitor_interval == 0
    # Idempotent: a second call is a harmless no-op.
    _disable_tqdm_monitor_thread()
    assert tqdm.tqdm.monitor_interval == 0


def test_disable_tqdm_monitor_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cosmetic suppression must never break loading -- if tqdm is missing or
    # misbehaves, the helper swallows the error.
    import sys

    from std_faster_whisper.engine import _disable_tqdm_monitor_thread

    monkeypatch.setitem(sys.modules, "tqdm", None)  # import tqdm -> raises
    _disable_tqdm_monitor_thread()  # must not raise


def test_prepared_to_pcm_requires_array() -> None:
    # Defensive guard: the streaming whole-input path expects a negotiated array.
    from standard_asr.audio_input import InputKind
    from standard_asr.engine import PreparedAudio

    prepared = PreparedAudio(kind=InputKind.ENCODED_BYTES, data=b"not-an-array")
    with pytest.raises(TranscriptionError, match="not delivered as an array"):
        _prepared_to_pcm(prepared)


def test_prepared_to_pcm_quantizes_array() -> None:
    from standard_asr.audio_input import InputKind
    from standard_asr.engine import PreparedAudio

    arr = np.array([0.0, 1.0, -1.0, 2.0, np.nan], dtype=np.float32)  # clipped + sanitized
    prepared = PreparedAudio(kind=InputKind.ARRAY, array=arr, sample_rate=16000)
    pcm = _prepared_to_pcm(prepared)
    decoded = np.frombuffer(pcm, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == 32767  # 1.0 -> max
    assert decoded[2] == -32767  # -1.0
    assert decoded[3] == 32767  # 2.0 clipped to 1.0
    assert decoded[4] == 0  # NaN -> 0
