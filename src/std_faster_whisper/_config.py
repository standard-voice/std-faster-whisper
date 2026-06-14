# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Init config and provider-params models for the faster-whisper engine.

Two pydantic models:

* :class:`FasterWhisperConfig` -- init configuration (spec IC.1). Standard
  "relevant-only" axes (device, download root) come from standard mixins so the
  auto-UI renders them; engine-specific init knobs (``cpu_threads``, ``revision``)
  are declared directly. An optional Hugging Face token is a ``SecretStr`` secret
  field (spec IC.3) so it is never logged or echoed in ``/v1/models``.
* :class:`FasterWhisperParams` -- per-request decoding knobs that are
  faster-whisper-native and NOT in the portable standard set (``beam_size``,
  ``task``, VAD ...). They live in a :class:`ProviderParams` subclass (spec RT
  §3.2); passing them to a different engine raises ``InvalidProviderParamError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field, SecretStr
from standard_asr import (
    BaseConfig,
    DeviceConfigMixin,
    DownloadConfigMixin,
    LanguageConfigMixin,
    secret_field,
)
from standard_asr.runtime_params import ProviderParams


class FasterWhisperConfig(
    DeviceConfigMixin,
    DownloadConfigMixin,
    LanguageConfigMixin,
    BaseConfig[Literal["faster-whisper"]],
):
    """Init configuration for the faster-whisper engine.

    The model is selected by the entry-point preset (spec IC.7), NOT by a field
    here. ``model_path`` is only an optional *local checkpoint override* (spec
    IC.7 weights/path): point it at a converted CTranslate2 directory to load
    your own weights instead of the preset's Hub model. ``None`` (default) loads
    the preset.

    Standard axes via mixins (field present => applicable, spec IC.5):

    * ``device`` (:class:`DeviceConfigMixin`) -- "cpu" / "cuda" / "auto".
    * ``download_root`` (:class:`DownloadConfigMixin`) -- cache directory; the
      lazy loader resolves it against the spec IC.9 precedence.
    * ``default_language`` / ``default_candidate_languages``
      (:class:`LanguageConfigMixin`) -- the language axis (spec LANG R1 requires
      ``default_language`` because the engine exposes ``selectable_languages``).

    Args:
        engine: Discriminator value (entry-point-derived; never hand-written).
        model_path: Optional LOCAL checkpoint directory overriding the preset's
            model (spec IC.7 weights/path). The model is chosen by the preset,
            not by this field.
        device_index: CTranslate2 device index or list of indices.
        compute_type: CTranslate2 quantization/precision (e.g. ``"int8"``,
            ``"float16"``, ``"default"``).
        cpu_threads: CPU threads for inference (``0`` = CTranslate2 default).
        num_workers: Worker threads for parallel inference.
        local_files_only: Never download; require a cached/local model.
        revision: Optional Hugging Face model revision (branch/tag/commit).
        hf_token: Optional Hugging Face access token for gated/private model
            repositories. Secret (masked in repr / dumps / ``/v1/models``).
    """

    engine: Literal["faster-whisper"] = "faster-whisper"

    # The language axis default. faster-whisper auto-detects on `None`/`"auto"`;
    # we default to "auto" so a zero-config engine just works (spec LANG R1).
    default_language: str | None = Field(
        default="auto", description="Default language (BCP-47) or 'auto' for detection."
    )

    model_path: str | None = Field(
        default=None,
        description=(
            "Optional local checkpoint directory overriding the preset's model "
            "(spec IC.7 weights/path). The model is selected by the entry-point "
            "preset, not by this field; None loads the preset's model."
        ),
    )
    device_index: int | list[int] = Field(
        default=0, description="CTranslate2 device index/indices."
    )
    compute_type: str = Field(
        default="default",
        description="CTranslate2 quantization/precision (int8, float16, default, ...).",
    )
    cpu_threads: int = Field(default=0, ge=0, description="CPU threads (0 = CTranslate2 default).")
    num_workers: int = Field(default=1, ge=1, description="Worker threads for parallel inference.")
    local_files_only: bool = Field(default=False, description="Disable downloads when True.")
    revision: str | None = Field(default=None, description="Optional HF model revision.")
    hf_token: SecretStr | None = secret_field(
        description="Hugging Face access token for gated/private model repos (secret)."
    )


class FasterWhisperParams(ProviderParams):
    """Engine-specific decoding knobs for faster-whisper (non-portable).

    These map directly onto ``WhisperModel.transcribe`` arguments that have no
    portable standard-set equivalent. Setting any of them locks the request to
    faster-whisper: handing this object to another engine raises
    ``InvalidProviderParamError`` (spec RT §3.2, swap-safety via exact-type
    match -- so this class MUST stay a distinct terminal type).

    Args:
        task: ``"transcribe"`` (default) or ``"translate"`` (speech -> English).
            Whisper-native, so it lives here rather than the portable set;
            without it, translation would be unreachable.
        beam_size: Beam size for decoding.
        best_of: Candidates sampled when temperature > 0.
        patience: Beam-search patience factor.
        length_penalty: Exponential length-penalty constant.
        repetition_penalty: Penalty (>1) on previously generated tokens.
        no_repeat_ngram_size: Block repeats of n-grams of this size (0 = off).
        temperature: Sampling temperature(s). ``None`` -> faster-whisper default
            fallback schedule.
        compression_ratio_threshold: gzip-ratio threshold for "failed" decodes.
        log_prob_threshold: Average-logprob threshold for "failed" decodes.
        no_speech_threshold: No-speech probability threshold for silence.
        condition_on_previous_text: Feed previous output as the next window's
            prompt.
        vad_filter: Enable Silero VAD filtering.
        vad_parameters: Optional Silero VAD parameter dict.
    """

    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = Field(default=5, ge=1)
    best_of: int = Field(default=5, ge=1)
    patience: float = Field(default=1.0, gt=0.0)
    length_penalty: float = 1.0
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0)
    temperature: float | Sequence[float] | None = None
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = True
    vad_filter: bool = False
    vad_parameters: dict[str, Any] | None = None


def provider_kwargs(params: ProviderParams | None) -> dict[str, Any]:
    """Convert provider params into ``WhisperModel.transcribe`` keyword arguments.

    ``temperature`` is omitted when ``None`` so faster-whisper applies its own
    fallback schedule (a default-valued ``None`` would override it with a single
    ``None``).

    Args:
        params: The engine-specific parameters, or ``None``.

    Returns:
        Keyword arguments for ``WhisperModel.transcribe``.
    """
    if params is None:
        return {}
    fw = params if isinstance(params, FasterWhisperParams) else FasterWhisperParams()
    kwargs: dict[str, Any] = {
        "task": fw.task,
        "beam_size": fw.beam_size,
        "best_of": fw.best_of,
        "patience": fw.patience,
        "length_penalty": fw.length_penalty,
        "repetition_penalty": fw.repetition_penalty,
        "no_repeat_ngram_size": fw.no_repeat_ngram_size,
        "compression_ratio_threshold": fw.compression_ratio_threshold,
        "log_prob_threshold": fw.log_prob_threshold,
        "no_speech_threshold": fw.no_speech_threshold,
        "condition_on_previous_text": fw.condition_on_previous_text,
        "vad_filter": fw.vad_filter,
        "vad_parameters": fw.vad_parameters,
    }
    if fw.temperature is not None:
        kwargs["temperature"] = fw.temperature
    return kwargs
