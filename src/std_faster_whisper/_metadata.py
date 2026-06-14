# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static Properties and declared Capabilities for the faster-whisper engine.

Everything here is read at the *class* level without instantiating the engine
(``standard-asr models show``, the registry, REST ``GET /v1/capabilities``), so
it MUST be honest and self-contained. Capabilities are declared **fail-closed**:
we declare only what faster-whisper genuinely delivers, in both ``batch`` and
``streaming`` modes (the streaming numbers are deliberately conservative -- see
the per-field comments and ``docs/STANDARD_ASR_FINDINGS.md``).
"""

from __future__ import annotations

from typing import Literal

from standard_asr import BaseProperties, InputKind, SampleRateRange
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsCap,
    PhraseHintsConstraints,
    PromptCap,
    PromptConstraints,
    ReconnectCap,
    StreamingCapabilities,
    StreamingGuidanceCaps,
    StreamTimestampsCap,
    WordTimestampsCap,
)

#: Whisper supports ~99 languages. We surface a representative multilingual
#: subset for the discovery UI / candidate-language validation; the model itself
#: still detects any of its languages under ``auto``. Per-preset overrides apply
#: (``*.en`` and distilled presets are English-only).
MULTILINGUAL_LANGUAGES: list[str] = [
    "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "it",
    "nl", "ar", "hi", "tr", "pl", "uk", "vi", "id", "sv", "ca",
]  # fmt: skip


class FasterWhisperProperties(BaseProperties):
    """Static metadata for the canonical multilingual faster-whisper preset.

    ``model_name`` MUST equal the entry-point key's model component so
    ``properties.model_id`` matches the registered key (compliance-enforced).
    The base describes ``faster-whisper/large-v3``; other presets subclass this
    and override ``model_name`` (+ ``selectable``/``detectable`` for English-only
    builds).

    I/O boundaries:

    * ``accepted_input = {array, encoded_file, encoded_bytes}`` -- faster-whisper's
      ``transcribe()`` accepts a decoded ``np.float32`` array, a path, or a binary
      file-like, so the standard layer passes through whichever the app provided.
      Declaring ``encoded_bytes`` lets the Web API hand in-memory uploads in
      without a temp file.
    * ``native_sample_rate = 16000``; ``accepted_sample_rates = [16000]``. Whisper
      runs at 16 kHz; the standard layer resamples anything else before us. (We
      deliberately do NOT claim ``"any"`` even though faster-whisper *can* decode
      arbitrary rates from files -- declaring the true model rate keeps the
      negotiated array contract unambiguous, and the spec's reachability
      invariant holds since native == the only accepted rate.)
    """

    engine_id: str = "faster-whisper"
    model_name: str = "large-v3"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {
        InputKind.ARRAY,
        InputKind.ENCODED_FILE,
        InputKind.ENCODED_BYTES,
    }
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [16000]
    selectable_languages: list[str] = ["auto", *MULTILINGUAL_LANGUAGES]
    detectable_languages: list[str] = list(MULTILINGUAL_LANGUAGES)
    # Streaming wire frames are canonical 16-bit PCM. We declare the encoding so
    # the standard layer fail-closed-rejects a mis-declared encoding rather than
    # mis-framing it as PCM (spec AI 3.2 wire_encodings).
    wire_encodings: list[str] | None = ["pcm_s16le"]
    description: str | None = "faster-whisper large-v3 (CTranslate2 Whisper), multilingual."


# --------------------------------------------------------------------------- #
# Guidance constraints: faster-whisper SILENTLY truncates over-budget guidance.
# `get_prompt` caps both the initial_prompt and the encoded hotwords at ~223
# tokens (max_length // 2 - 1) with no signal. Declaring conservative limits lets
# the standard layer fail-loud (strict) or truncate+diagnose (best_effort)
# BEFORE the engine eats the overflow -- exactly the silent degradation the
# guidance contract forbids (spec RT §3.3). The standard counts tokens with a
# conservative, script-aware approximation (not Whisper's BPE), so these sit
# below the ~223 hard cap with headroom: a long Latin word / URL is 1 unit here
# but several BPE tokens upstream.
# --------------------------------------------------------------------------- #
_GUIDANCE = GuidanceCaps(
    prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=200)),
    phrase_hints=PhraseHintsCap(
        supported=True,
        # Hotwords are joined into ONE string sharing the ~223-token budget; keep
        # term count and per-term length conservative so the combined set stays
        # well under it.
        constraints=PhraseHintsConstraints(max_terms=50, max_chars_per_term=40),
    ),
)

# `word_timestamps.granularities` declares which granularities the engine can
# HONESTLY DELIVER -- not which upstream switches exist. faster-whisper emits
# per-segment start/end on EVERY transcription at zero cost (default
# word_timestamps=False), so "segment" is always satisfiable and MUST be
# declared; omitting it would make the standard layer reject the cheapest,
# always-satisfiable request as a false incompatibility (spec TR.3). "word"
# flips the upstream word_timestamps=True forced-alignment pass on; "char" is
# unsupported.
_WORD_TS = WordTimestampsCap(supported=True, granularities=["word", "segment"])

DECLARED_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        word_timestamps=_WORD_TS,
        guidance=_GUIDANCE,
    ),
    # --- Streaming (windowed; faster-whisper has no native streaming) --------- #
    # We implement streaming by buffering fed PCM and periodically re-decoding
    # the accumulated audio (see _streaming.py). The capability declarations are
    # the HONEST consequence of that strategy:
    #   * streaming_input  = True  : we accept incremental PCM frames.
    #   * streaming_output = True  : we emit results before all audio arrives.
    #   * emits_partials   = True  : each re-decode emits a partial.
    #   * word_stability   = False : Whisper re-decodes the whole window and may
    #         rewrite ANY earlier text; we have no right-context guarantee, so we
    #         report stable_until=0 always (spec ST §4.2) and declare false.
    #   * re_segments      = False : we never emit `supersede`. Each partial is
    #         the single growing segment's full current text; we re-segment only
    #         the (cumulative) text in place, never retire a previously announced
    #         segment id. (We DO finalize VAD-stable older sentences as separate
    #         finalized segments, but those ids are never superseded.)
    #   * reconnect        = unsupported : a local in-process model has no remote
    #         session to reconnect (spec ST §6.3); declaring `lossy` would be
    #         dishonest -- there is no transport that drops.
    #   * finality_level   = final : a finalized segment's text won't change due
    #         to new audio, but we make NO post-processing (punctuation/ITN)
    #         immutability promise, so we MUST NOT claim `closed` (spec ST §5.3).
    #   * timestamps       = post_align : segment/word times come from Whisper's
    #         per-window decode mapped to session time, not native frame-aligned
    #         streaming timestamps.
    # Streaming guidance mirrors batch (prompt/phrase_hints honored on each
    # window); mutable_mid_stream stays false (params frozen at start, spec RT R5).
    streaming=StreamingCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        word_timestamps=_WORD_TS,
        guidance=StreamingGuidanceCaps(
            prompt=_GUIDANCE.prompt,
            phrase_hints=_GUIDANCE.phrase_hints,
        ),
        emits_partials=FlagCap(supported=True),
        re_segments=FlagCap(supported=False),
        word_stability=FlagCap(supported=False),
        reconnect=ReconnectCap(mode="unsupported"),
        finality_level=FinalityCap(mode="final"),
        timestamps=StreamTimestampsCap(mode="post_align"),
    ),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
    # faster-whisper does NOT resample arbitrary arrays itself in our contract;
    # the standard layer resamples to 16 kHz before us. Informational only
    # (spec C 3.2); accepted_sample_rates stays authoritative.
    self_resamples=FlagCap(supported=False),
)


# --------------------------------------------------------------------------- #
# Per-preset Properties subclasses. Each Whisper variant is its own entry point
# (spec IC.7) so the discovery layer can enumerate every available model. A
# preset overrides only model_name (-> matching properties.model_id), the
# language sets for English-only builds, and the human description. Everything
# else -- config, params, capabilities, the transcribe/stream pipeline -- is
# inherited unchanged.
# --------------------------------------------------------------------------- #
class TinyProperties(FasterWhisperProperties):
    """``faster-whisper/tiny`` -- smallest multilingual model (fast, low accuracy)."""

    model_name: str = "tiny"
    description: str | None = "faster-whisper tiny (smallest, fastest; for tests/smoke runs)."


class BaseModelProperties(FasterWhisperProperties):
    """``faster-whisper/base`` -- small multilingual model."""

    model_name: str = "base"
    description: str | None = "faster-whisper base (small multilingual model)."


class SmallProperties(FasterWhisperProperties):
    """``faster-whisper/small`` -- mid-size multilingual model."""

    model_name: str = "small"
    description: str | None = "faster-whisper small (mid-size multilingual model)."


class MediumProperties(FasterWhisperProperties):
    """``faster-whisper/medium`` -- large multilingual model."""

    model_name: str = "medium"
    description: str | None = "faster-whisper medium (large multilingual model)."


class LargeV3Properties(FasterWhisperProperties):
    """``faster-whisper/large-v3`` -- canonical multilingual production model."""

    # Inherits model_name = "large-v3" and the base description.


class DistilLargeV3Properties(FasterWhisperProperties):
    """``faster-whisper/distil-large-v3`` -- distilled, lower-latency, English-leaning."""

    model_name: str = "distil-large-v3"
    # distil-large-v3 is English-only. Declare it honestly so the standard layer
    # rejects a non-English `language` request instead of mis-serving it.
    selectable_languages: list[str] = ["auto", "en"]
    detectable_languages: list[str] = ["en"]
    description: str | None = (
        "faster-whisper distil-large-v3 (distilled, English-only, low latency)."
    )


class TurboProperties(FasterWhisperProperties):
    """``faster-whisper/large-v3-turbo`` -- fastest large multilingual preset."""

    model_name: str = "large-v3-turbo"
    description: str | None = "faster-whisper large-v3-turbo (fastest large multilingual preset)."
