# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure conversion helpers shared by the batch and streaming code paths.

These map faster-whisper's native objects (``Segment`` / ``Word`` /
``TranscriptionInfo``) onto Standard ASR's result models. Kept dependency-light
and side-effect-free so they are trivially unit-testable against fakes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from standard_asr.results import Segment, Word

#: 16-bit signed little-endian PCM is the canonical wire encoding (spec AI). The
#: reverse of the canonical int16->float scaling is /32768 (spec AI R4).
_PCM_SCALE = 32768.0


def pcm_s16le_to_float32(data: bytes) -> NDArray[np.float32]:
    """Decode canonical 16-bit LE PCM bytes into a float32 mono waveform.

    A trailing odd byte (a half sample split across two chunks) is dropped; the
    caller is responsible for re-joining frame boundaries before decoding if it
    needs that last sample.

    Args:
        data: Raw ``pcm_s16le`` bytes (mono).

    Returns:
        A ``float32`` array in ``[-1, 1)``; empty if ``data`` is empty.
    """
    if len(data) < 2:
        return np.zeros(0, dtype=np.float32)
    usable = len(data) - (len(data) % 2)
    # frombuffer returns a read-only view over the bytes; build an owned, writable
    # float32 array (copy=True via np.array) so callers can concatenate freely.
    samples: NDArray[np.int16] = np.frombuffer(data[:usable], dtype="<i2")
    return np.array(samples, dtype=np.float32) / _PCM_SCALE


def convert_segments(segments: Iterable[Any]) -> tuple[list[Segment], list[Word]]:
    """Convert faster-whisper segments into Standard ASR ``Segment``/``Word``.

    Word-level data is only attached when the upstream segment carries it (i.e.
    when ``word_timestamps=True`` was requested); otherwise ``words`` stays
    ``None`` so the result honors the "not requested" null semantics (spec TR.1).

    Args:
        segments: A faster-whisper segment iterable (lazily consumed -- iterating
            it drives the actual decode).

    Returns:
        A ``(segments, flattened_words)`` pair.
    """
    segment_list: list[Segment] = []
    word_list: list[Word] = []
    for segment in segments:
        words: list[Word] | None = None
        seg_words = getattr(segment, "words", None)
        if seg_words:
            words = [
                Word(
                    start=word.start,
                    end=word.end,
                    text=word.word,
                    probability=word.probability,
                )
                for word in seg_words
            ]
            word_list.extend(words)
        segment_list.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=words,
                temperature=getattr(segment, "temperature", None),
                avg_logprob=getattr(segment, "avg_logprob", None),
                compression_ratio=getattr(segment, "compression_ratio", None),
                no_speech_prob=getattr(segment, "no_speech_prob", None),
            )
        )
    return segment_list, word_list


#: Fields from faster-whisper's ``transcription_options`` that are safe to
#: surface in ``result.extra``. We deliberately exclude ``initial_prompt`` /
#: ``prefix`` / ``hotwords`` / ``suppress_tokens`` so the prompt text is not
#: echoed back (privacy) and the payload stays small enough to carry over REST.
_SAFE_OPTION_FIELDS: tuple[str, ...] = (
    "task",
    "beam_size",
    "best_of",
    "patience",
    "length_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "temperatures",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "condition_on_previous_text",
    "word_timestamps",
)


def safe_extra(info: Any) -> dict[str, Any]:
    """Build the whitelisted engine-specific ``extra`` from a ``TranscriptionInfo``.

    These are faster-whisper-private values (the decoding knobs the run used and
    the post-VAD duration) -- not standardized cross-engine metadata -- so per
    spec TR.1 they belong in ``result.extra`` (engine-specific channel), never in
    ``result.metadata``. Only small, non-sensitive options are included.

    Args:
        info: faster-whisper's ``TranscriptionInfo``.

    Returns:
        A JSON-friendly mapping for ``TranscriptionResult.extra``.
    """
    options = getattr(info, "transcription_options", None)
    safe: dict[str, Any] = {}
    if options is not None:
        for name in _SAFE_OPTION_FIELDS:
            if hasattr(options, name):
                safe[name] = getattr(options, name)
    extra: dict[str, Any] = {"transcription_options": safe}
    vad = getattr(info, "duration_after_vad", None)
    if vad is not None:
        extra["duration_after_vad"] = vad
    return extra
