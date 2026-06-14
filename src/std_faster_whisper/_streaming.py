# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Windowed streaming session for faster-whisper.

faster-whisper has **no native streaming** -- ``WhisperModel.transcribe`` is a
batch call over a whole utterance. This session synthesizes streaming output by
a **re-decode-the-window** strategy, and declares its capabilities to match
exactly what that strategy can honestly deliver (see ``_metadata.py``).

Strategy
--------
1. We consume fed PCM frames via :meth:`TranscriptionSession.audio_chunks` and
   accumulate them into one growing float32 buffer (the "window").
2. Whenever at least ``redecode_interval_s`` of *new* audio has arrived, we run
   the **whole buffer** back through ``transcribe`` (in a worker thread, so the
   event loop never blocks) and emit:
   * a ``final`` for every sentence that ends comfortably before the decode
     frontier (``settle_margin_s`` behind the last decoded timestamp) and has not
     been finalized yet -- these are stable under Whisper's local-attention
     window and won't change with a bit more trailing audio;
   * one ``partial`` carrying the *tail* (everything after the last finalized
     sentence) as the single in-progress segment.
3. On ``end_audio`` we do a last full decode and finalize the tail, then ``done``.

Honesty
-------
Whisper re-decodes the entire window each pass and may rewrite ANY earlier text,
so we set ``stable_until=0`` on every ``partial`` (``word_stability=false``) and
finalize a sentence only once it is several seconds behind the frontier. We never
emit ``supersede`` (``re_segments=false``): finalized segment ids are immutable
and the partial only ever describes the current tail. Segment ids are synthesized
deterministically (``seg-0``, ``seg-1`` ...). This is a *pragmatic* streaming
adapter for a batch engine, not a true low-latency incremental recognizer -- the
README and findings doc say so plainly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from standard_asr import RuntimeParams, TranscriptionEvent, TranscriptionSession
from standard_asr.language import effective_language, normalize_bcp47

from ._config import FasterWhisperConfig, provider_kwargs
from ._convert import convert_segments, pcm_s16le_to_float32

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import FasterWhisperASR

_LOGGER = logging.getLogger(__name__)

#: faster-whisper runs at 16 kHz mono; wire frames are negotiated to this rate.
_SAMPLE_RATE = 16000


class FasterWhisperStreamingSession(TranscriptionSession):
    """A windowed streaming session backed by ``WhisperModel.transcribe``.

    Args:
        engine: The owning engine (already model-loaded by the time
            :meth:`_produce` runs).
        gated_params: Frozen, already-gated runtime parameters (spec RT R5).
        redecode_interval_s: Minimum seconds of new audio to accumulate before
            re-decoding the window (latency vs. compute trade-off).
        settle_margin_s: A sentence is finalized only once it ends at least this
            many seconds behind the last decoded timestamp (conservative
            stability under Whisper's re-decode).
        **session_kwargs: Forwarded to :class:`TranscriptionSession` (deadlines,
            buffer sizes, ``strict_lifecycle``).
    """

    def __init__(
        self,
        engine: FasterWhisperASR,
        gated_params: RuntimeParams,
        *,
        redecode_interval_s: float = 4.0,
        settle_margin_s: float = 3.0,
        **session_kwargs: Any,
    ) -> None:
        super().__init__(**session_kwargs)
        self._engine = engine
        self._params = gated_params
        self._redecode_interval_s = redecode_interval_s
        self._settle_margin_s = settle_margin_s
        # NB: the base TranscriptionSession reserves several private attribute
        # names (``_buffer`` is its event coalescing buffer, ``_audio_queue``,
        # ``_audio_history`` ...). We deliberately name our audio window
        # ``_window`` to avoid clobbering them -- see STANDARD_ASR_FINDINGS.md.
        self._window = np.zeros(0, dtype=np.float32)
        self._language = self._resolve_language()
        # Index into the synthesized list of decoded sentences up to which we
        # have already emitted a `final`. Finalized ids are immutable.
        self._finalized_count = 0

    def _resolve_language(self) -> str | None:
        """Resolve the effective language to forward to faster-whisper.

        Returns:
            A primary-subtag language code, or ``None`` for auto-detect.
        """
        config = cast(FasterWhisperConfig, self._engine.config)
        resolved = effective_language(
            self._params.language,
            config.default_language,
            has_language_axis=True,
            runtime_override_supported=True,
        )
        if resolved and resolved != "auto":
            return normalize_bcp47(resolved).split("-", maxsplit=1)[0]
        return None

    def _decode(self, audio: NDArray[np.float32], *, want_words: bool) -> list[Any]:
        """Run a full faster-whisper decode over ``audio`` (blocking).

        Called in a worker thread via ``asyncio.to_thread``. Returns the
        materialized faster-whisper segment list (the lazy generator is consumed
        here so the inference actually runs inside the thread).

        Args:
            audio: The accumulated window as a float32 array.
            want_words: Whether to request word-level timestamps.

        Returns:
            The decoded faster-whisper segments (native objects), as a list.
        """
        model = cast(Any, self._engine.model)
        segments, _info = model.transcribe(
            audio,
            language=self._language,
            word_timestamps=want_words,
            initial_prompt=self._params.prompt,
            hotwords=" ".join(self._params.phrase_hints) if self._params.phrase_hints else None,
            **provider_kwargs(self._params.provider_params),
        )
        return list(segments)

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        """Drive the windowed re-decode loop and yield streaming events.

        Yields:
            ``final`` events for settled sentences, ``partial`` events for the
            in-progress tail, ``progress`` heartbeats carrying the audio cursor,
            and a terminal ``done``.
        """
        self._engine.ensure_loaded()
        from standard_asr.runtime_params import WordTimestampGranularity

        want_words = self._params.word_timestamps == WordTimestampGranularity.WORD
        pending = bytearray()
        # Bytes consumed since the last decode, to honor the re-decode interval.
        bytes_since_decode = 0
        bytes_per_interval = int(self._redecode_interval_s * _SAMPLE_RATE * 2)

        async for chunk in self.audio_chunks():
            pending.extend(chunk)
            bytes_since_decode += len(chunk)
            if bytes_since_decode < bytes_per_interval:
                continue
            self._append_pcm(bytes(pending))
            pending.clear()
            bytes_since_decode = 0
            for event in await self._redecode(want_words=want_words, final_pass=False):
                yield event

        # Flush any tail audio that did not reach a full interval.
        if pending:
            self._append_pcm(bytes(pending))
            pending.clear()

        # Final pass: decode whatever we have and finalize everything.
        for event in await self._redecode(want_words=want_words, final_pass=True):
            yield event
        yield TranscriptionEvent.done(audio_processed_until=self._window_seconds())

    def _append_pcm(self, data: bytes) -> None:
        """Append decoded PCM bytes to the running float32 window.

        Args:
            data: ``pcm_s16le`` mono bytes.
        """
        if not data:
            return
        samples = pcm_s16le_to_float32(data)
        if samples.size:
            self._window = np.concatenate([self._window, samples])

    def _window_seconds(self) -> float:
        """Return the duration of the accumulated window in seconds."""
        return self._window.size / _SAMPLE_RATE

    async def _redecode(self, *, want_words: bool, final_pass: bool) -> list[TranscriptionEvent]:
        """Re-decode the window and build the events for this pass.

        Args:
            want_words: Whether word-level timestamps were requested.
            final_pass: ``True`` on the post-``end_audio`` flush (finalize all
                remaining sentences regardless of the settle margin).

        Returns:
            The ordered events to yield for this pass.
        """
        cursor = self._window_seconds()
        if self._window.size == 0:
            return [TranscriptionEvent.progress(audio_processed_until=cursor)]
        try:
            raw = await asyncio.to_thread(self._decode, self._window, want_words=want_words)
        except Exception as exc:
            _LOGGER.exception("faster-whisper streaming decode failed")
            return [
                TranscriptionEvent.make_error(
                    code="engine_error",
                    recoverable=False,
                    extra={"detail": f"{type(exc).__name__}: {exc}"},
                )
            ]
        segments, _words = convert_segments(raw)
        return self._build_events(segments, cursor=cursor, final_pass=final_pass)

    def _build_events(
        self, segments: list[Any], *, cursor: float, final_pass: bool
    ) -> list[TranscriptionEvent]:
        """Turn a decoded segment list into final/partial/progress events.

        Sentences that end at least ``settle_margin_s`` behind ``cursor`` (or all
        of them on the final pass) become ``final`` events with stable ids; the
        remaining tail is one ``partial`` for the current in-progress segment.
        ``stable_until`` is always 0 (Whisper may rewrite the window).

        Args:
            segments: Standard ASR ``Segment`` objects from this decode.
            cursor: The audio time processed so far (seconds).
            final_pass: Whether to finalize all remaining sentences.

        Returns:
            The ordered events for this pass.
        """
        events: list[TranscriptionEvent] = []
        settle_before = cursor - self._settle_margin_s
        # How many sentences are settled this pass.
        settled = (
            len(segments) if final_pass else sum(1 for s in segments if s.end <= settle_before)
        )
        # Emit finals for newly settled sentences we have not finalized yet.
        for idx in range(self._finalized_count, settled):
            seg = segments[idx]
            events.append(
                TranscriptionEvent.final(
                    segment_id=f"seg-{idx}",
                    text=seg.text,
                    stable_until=0,
                    start=seg.start,
                    end=seg.end,
                    words=seg.words,
                    audio_processed_until=cursor,
                )
            )
        self._finalized_count = max(self._finalized_count, settled)

        # The tail (everything after the settled sentences) is the in-progress
        # segment, emitted as a partial unless we just finalized it.
        if not final_pass and settled < len(segments):
            tail = segments[settled:]
            tail_text = "".join(s.text for s in tail)
            tail_words: list[Any] | None = None
            for s in tail:
                if s.words:
                    tail_words = (tail_words or []) + s.words
            events.append(
                TranscriptionEvent.partial(
                    segment_id=f"seg-{settled}",
                    text=tail_text,
                    stable_until=0,
                    start=tail[0].start,
                    end=tail[-1].end,
                    words=tail_words,
                    audio_processed_until=cursor,
                )
            )
        elif not events:
            # No settled sentences and no tail to report (e.g. silence): heartbeat.
            events.append(TranscriptionEvent.progress(audio_processed_until=cursor))
        return events
