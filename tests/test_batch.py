# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Batch-path, metadata, config, and lazy-loading tests.

Every test runs against the injected ``FakeWhisperModel``; the real
faster-whisper model is never instantiated and no weights are downloaded.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from standard_asr import RuntimeParams, StandardASR
from standard_asr.audio_input import AudioBytes, AudioPath
from standard_asr.capabilities import (
    DeclaredCapabilities,
    PhraseHintsCap,
    PromptCap,
    WordTimestampsCap,
)
from standard_asr.exceptions import (
    DiscoveryError,
    InvalidProviderParamError,
    TranscriptionError,
    UnsupportedFeatureError,
)
from standard_asr.runtime_params import ProviderParams, WordTimestampGranularity

from std_faster_whisper import (
    DistilLargeV3ASR,
    FasterWhisperASR,
    FasterWhisperConfig,
    FasterWhisperParams,
    FasterWhisperProperties,
    TinyASR,
    TurboASR,
)

from .conftest import FakeInfo, FakeSegment, FakeWhisperModel, FakeWord


def _audio(n: int = 16000) -> tuple[np.ndarray, int]:
    return (np.zeros(n, dtype=np.float32), 16000)


# --------------------------------------------------------------------------- #
# Static metadata / config / params
# --------------------------------------------------------------------------- #
def test_engine_is_standard_asr() -> None:
    assert isinstance(FasterWhisperASR(), StandardASR)


def test_class_level_metadata() -> None:
    assert isinstance(FasterWhisperASR.properties, FasterWhisperProperties)
    assert isinstance(FasterWhisperASR.declared_capabilities, DeclaredCapabilities)
    assert FasterWhisperASR.provider_params_type is FasterWhisperParams
    assert FasterWhisperASR.properties.model_id == "faster-whisper/large-v3"


def test_preset_model_ids_match_entry_point_keys() -> None:
    assert TinyASR.properties.model_id == "faster-whisper/tiny"
    assert DistilLargeV3ASR.properties.model_id == "faster-whisper/distil-large-v3"
    assert TurboASR.properties.model_id == "faster-whisper/large-v3-turbo"


def test_distil_preset_is_english_only() -> None:
    # distil-large-v3 is English-only; declared honestly so non-English requests
    # are rejected, not mis-served.
    assert DistilLargeV3ASR.properties.selectable_languages == ["auto", "en"]
    assert DistilLargeV3ASR.properties.detectable_languages == ["en"]


def test_config_defaults() -> None:
    config = FasterWhisperConfig()
    assert config.engine == "faster-whisper"
    assert config.model_path is None
    assert config.default_language == "auto"
    assert config.local_files_only is False
    assert config.hf_token is None


def test_hf_token_is_secret() -> None:
    # The HF token is a SecretStr -- masked in public dumps, never echoed.
    config = FasterWhisperConfig(hf_token="hf_supersecret")
    assert config.hf_token is not None
    assert config.hf_token.get_secret_value() == "hf_supersecret"
    assert "hf_supersecret" not in repr(config)
    assert "hf_supersecret" not in str(config.public_dump())


def test_provider_params_defaults() -> None:
    params = FasterWhisperParams()
    assert params.task == "transcribe"
    assert params.beam_size == 5
    assert params.temperature is None


def test_provider_params_swap_safety() -> None:
    # A foreign provider_params type is rejected (exact-type swap safety).
    class OtherParams(ProviderParams):
        knob: int = 1

    with pytest.raises(InvalidProviderParamError):
        FasterWhisperASR(model_path="tiny").transcribe(
            _audio(), RuntimeParams(provider_params=OtherParams())
        )


# --------------------------------------------------------------------------- #
# Lazy model loading
# --------------------------------------------------------------------------- #
def test_ensure_model_loaded_missing_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *a: object, **k: object) -> object:
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("no faster_whisper")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(DiscoveryError, match="not installed"):
        FasterWhisperASR().prepare()


def test_ensure_model_loaded_init_failure(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    fake_faster_whisper.raise_on_init = RuntimeError("weights missing")
    with pytest.raises(DiscoveryError, match="Failed to load"):
        FasterWhisperASR().prepare()


def test_prepare_loads_model_once(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    engine = FasterWhisperASR(model_path="tiny", device="cpu")
    engine.prepare()
    assert engine._model is not None
    first = engine._model
    engine.prepare()
    assert engine._model is first
    # An explicit model_path is a local override and wins over model_size.
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "tiny"


def test_preset_loads_its_model_size_by_default(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    FasterWhisperASR().prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "large-v3"
    TinyASR().prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "tiny"
    TurboASR().prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == "large-v3-turbo"


def test_hf_token_forwarded_as_plaintext_to_loader(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # The secret is materialized to plaintext ONLY at the SDK call site.
    FasterWhisperASR(model_path="tiny", hf_token="hf_abc").prepare()
    assert fake_faster_whisper.last_init_kwargs["use_auth_token"] == "hf_abc"


def test_no_hf_token_forwards_none(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    FasterWhisperASR(model_path="tiny").prepare()
    assert fake_faster_whisper.last_init_kwargs["use_auth_token"] is None


def test_download_root_disabled_forces_local_only(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("STANDARD_ASR_MODEL_DIR", "/tmp/ignored")
    engine = FasterWhisperASR(model_path="tiny", download_root="/tmp/models")
    engine.prepare()
    assert fake_faster_whisper.last_init_kwargs["local_files_only"] is True
    assert fake_faster_whisper.last_init_kwargs["download_root"] == "/tmp/models"


def test_download_root_defers_to_library_default(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)
    FasterWhisperASR(model_path="tiny").prepare()
    assert fake_faster_whisper.last_init_kwargs["download_root"] is None
    assert fake_faster_whisper.last_init_kwargs["local_files_only"] is False


def test_env_fallback_for_engine_field(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Engine-declared fields get env entries too (spec IC.4 full-table DX).
    monkeypatch.setenv("STANDARD_ASR_FASTER_WHISPER__COMPUTE_TYPE", "int8")
    FasterWhisperASR(model_path="tiny").prepare()
    assert fake_faster_whisper.last_init_kwargs["compute_type"] == "int8"


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def test_transcribe_array_basic(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "Hello world.")]
    fake_faster_whisper.info = FakeInfo(language="en")

    result = FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert result.text == "Hello world."
    assert result.detected_language == "en"
    assert result.language_confidence == pytest.approx(0.97)
    assert result.duration == pytest.approx(1.23)
    kwargs = fake_faster_whisper.last_transcribe_kwargs
    assert kwargs["language"] == "en"
    assert kwargs["word_timestamps"] is False


def test_transcribe_wraps_engine_failure_as_transcription_error(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    boom = RuntimeError("CUDA out of memory")
    fake_faster_whisper.raise_on_transcribe = boom
    with pytest.raises(TranscriptionError) as exc_info:
        FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert exc_info.value.__cause__ is boom


def test_transcribe_region_tagged_language_uses_primary_subtag(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "hi")]
    FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en-US"))
    assert fake_faster_whisper.last_transcribe_kwargs["language"] == "en"


def test_transcribe_auto_language_sends_none(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "hi")]
    FasterWhisperASR(model_path="tiny").transcribe(_audio())
    assert fake_faster_whisper.last_transcribe_kwargs["language"] is None


def test_transcribe_with_word_timestamps(fake_faster_whisper: type[FakeWhisperModel]) -> None:
    words = [FakeWord(0.0, 0.5, "Hi", 0.9), FakeWord(0.5, 1.0, "there", 0.8)]
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "Hi there", words=words)]

    result = FasterWhisperASR(model_path="tiny").transcribe(
        _audio(), RuntimeParams(language="en", word_timestamps=WordTimestampGranularity.WORD)
    )
    assert fake_faster_whisper.last_transcribe_kwargs["word_timestamps"] is True
    assert result.words is not None
    assert [w.text for w in result.words] == ["Hi", "there"]
    assert result.segments is not None
    assert result.segments[0].words is not None


def test_segment_granularity_is_declared() -> None:
    node = FasterWhisperASR.declared_capabilities.node_at("batch.word_timestamps")
    assert isinstance(node, WordTimestampsCap)
    assert set(node.granularities) == {"word", "segment"}


def test_transcribe_segment_granularity_does_not_request_words(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "Hi there")]
    result = FasterWhisperASR(model_path="tiny").transcribe(
        _audio(), RuntimeParams(language="en", word_timestamps=WordTimestampGranularity.SEGMENT)
    )
    assert fake_faster_whisper.last_transcribe_kwargs["word_timestamps"] is False
    assert result.words is None
    assert result.segments is not None
    assert result.segments[0].start == pytest.approx(0.0)


def test_transcribe_with_prompt_and_phrase_hints(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    FasterWhisperASR(model_path="tiny").transcribe(
        _audio(),
        RuntimeParams(language="en", prompt="context", phrase_hints=["Anthropic", "Claude"]),
    )
    kwargs = fake_faster_whisper.last_transcribe_kwargs
    assert kwargs["initial_prompt"] == "context"
    assert kwargs["hotwords"] == "Anthropic Claude"


def test_guidance_constraints_are_declared() -> None:
    node = FasterWhisperASR.declared_capabilities.node_at("batch.guidance.prompt")
    assert isinstance(node, PromptCap)
    assert node.constraints.max_tokens == 200
    hints = FasterWhisperASR.declared_capabilities.node_at("batch.guidance.phrase_hints")
    assert isinstance(hints, PhraseHintsCap)
    assert hints.constraints.max_terms == 50
    assert hints.constraints.max_chars_per_term == 40


def test_over_budget_prompt_fails_loud_in_strict_mode(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    long_prompt = " ".join(["word"] * 201)
    with pytest.raises(UnsupportedFeatureError, match="prompt"):
        FasterWhisperASR(model_path="tiny").transcribe(
            _audio(), RuntimeParams(language="en", prompt=long_prompt)
        )


def test_over_budget_prompt_truncated_with_diagnostic_in_best_effort(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    long_prompt = " ".join(["word"] * 201)
    result = FasterWhisperASR(model_path="tiny", strict=False).transcribe(
        _audio(), RuntimeParams(language="en", prompt=long_prompt)
    )
    forwarded = fake_faster_whisper.last_transcribe_kwargs["initial_prompt"]
    assert len(forwarded.split()) == 200
    assert any(d.code == "prompt_truncated" for d in result.diagnostics)


def test_over_limit_phrase_hints_fail_loud_in_strict_mode(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    too_many = [f"term{i}" for i in range(51)]
    with pytest.raises(UnsupportedFeatureError, match="phrase_hints"):
        FasterWhisperASR(model_path="tiny").transcribe(
            _audio(), RuntimeParams(language="en", phrase_hints=too_many)
        )


def test_transcribe_provider_params_forwarded(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    params = RuntimeParams(
        language="en",
        provider_params=FasterWhisperParams(task="translate", beam_size=3, temperature=[0.0, 0.2]),
    )
    FasterWhisperASR(model_path="tiny").transcribe(_audio(), params)
    kwargs = fake_faster_whisper.last_transcribe_kwargs
    assert kwargs["task"] == "translate"
    assert kwargs["beam_size"] == 3
    assert kwargs["temperature"] == [0.0, 0.2]


def test_transcribe_from_file_path(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "from file")]
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16, dtype=np.int16).tobytes())

    result = FasterWhisperASR(model_path="tiny").transcribe(AudioPath(wav), RuntimeParams())
    assert result.text == "from file"
    assert fake_faster_whisper.last_transcribe_kwargs["source"] == str(wav)


def test_transcribe_from_bytes_uses_binary_file_like(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "from bytes")]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16, dtype=np.int16).tobytes())

    result = FasterWhisperASR(model_path="tiny").transcribe(
        AudioBytes(buf.getvalue()), RuntimeParams()
    )
    assert result.text == "from bytes"
    assert isinstance(fake_faster_whisper.last_transcribe_kwargs["source"], io.BytesIO)


def test_transcribe_detected_language_none_when_unknown(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    fake_faster_whisper.info = FakeInfo(language=None)
    result = FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert result.detected_language is None


def test_engine_specific_data_goes_to_extra_not_metadata(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    fake_faster_whisper.segments = [FakeSegment(0.0, 1.0, "x")]
    fake_faster_whisper.info = FakeInfo(language="en", duration_after_vad=0.8)
    result = FasterWhisperASR(model_path="tiny").transcribe(_audio(), RuntimeParams(language="en"))
    assert result.metadata == {}
    assert result.extra["transcription_options"]["task"] == "transcribe"
    assert result.extra["duration_after_vad"] == pytest.approx(0.8)
