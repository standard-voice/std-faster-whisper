# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Standard ASR engine class for faster-whisper.

A thin adapter over the upstream ``faster-whisper`` PyPI package
(``WhisperModel`` over CTranslate2). It subclasses :class:`EngineBase` and:

* declares :class:`~std_faster_whisper._metadata.FasterWhisperProperties` and the
  fail-closed :data:`~std_faster_whisper._metadata.DECLARED_CAPABILITIES`;
* keeps ``__init__`` pure and loads weights lazily in
  :meth:`_ensure_model_loaded` (spec IC.9);
* implements :meth:`_transcribe` (batch) and :meth:`_start_transcription`
  (windowed streaming, see :mod:`std_faster_whisper._streaming`).

Each preset (``tiny`` ... ``large-v3-turbo``) is a subclass that overrides only
:attr:`model_size` (the upstream weights id) and :attr:`properties` (for the
matching ``model_id``); the entire pipeline is inherited.
"""

from __future__ import annotations

import io
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from standard_asr import (
    RuntimeParams,
    TranscriptionResult,
    TranscriptionSession,
)
from standard_asr.audio.format import AudioFormat
from standard_asr.contract.artifacts import (
    ARTIFACT_INCOMPLETE,
    ARTIFACT_MISSING,
    ARTIFACT_READY,
    ArtifactContext,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.capabilities import DeclaredCapabilities
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    DiscoveryError,
    TranscriptionError,
)
from standard_asr.contract.language import effective_language, normalize_bcp47
from standard_asr.contract.params import ProviderParams, WordTimestampGranularity
from standard_asr.engine import (
    ArtifactDeclaration,
    BaseConfig,
    BaseProperties,
    DeclaredEngineMetadata,
    Diagnostic,
    EngineBase,
    Mode,
    PreparedAudio,
)
from standard_asr.runtime.downloads import allow_downloads, resolve_download_root

from ._artifacts import (
    HUB_ARTIFACT_ID,
    acquire,
    mel_mismatch_requirement,
    normalized_model_path,
    raise_for_gated_source,
    status_requirement,
)
from ._artifacts import (
    disable_tqdm_monitor_thread as _disable_tqdm_monitor_thread,
)
from ._config import FasterWhisperConfig, FasterWhisperParams, provider_kwargs
from ._convert import convert_segments, safe_extra
from ._metadata import (
    DECLARED_CAPABILITIES,
    BaseModelProperties,
    DistilLargeV3Properties,
    LargeV3Properties,
    MediumProperties,
    SmallProperties,
    TinyProperties,
    TurboProperties,
)
from ._streaming import FasterWhisperStreamingSession

_SAMPLE_RATE = 16000


class FasterWhisperASR(EngineBase):
    """Standard ASR adapter for the ``faster-whisper/large-v3`` preset.

    This base class IS the canonical ``large-v3`` multilingual preset. Other
    variants subclass it and override :attr:`model_size` + :attr:`properties`
    only (spec IC.7: model selection = entry-point preset, never an init
    ``model`` field).

    Args:
        **kwargs: Configuration overrides for :class:`FasterWhisperConfig`.
    """

    #: The faster-whisper model id this preset loads (passed upstream as
    #: ``model_size_or_path``). Overridden per preset; a local ``model_path``
    #: config override (spec IC.7 weights/path) still wins when set.
    model_size: ClassVar[str] = "large-v3"

    #: Preset-declared bundle files beyond the loader-universal closure. The
    #: 128-mel repos (large-v3, distil-large-v3, large-v3-turbo) ship
    #: preprocessor_config.json and need it: without the file, upstream
    #: silently builds an 80-mel feature extractor, and CTranslate2 then
    #: rejects the mismatched features at the first request (the post-load
    #: mel check turns that into an artifact error at load). The 80-mel repos
    #: do not carry the file at all (verified across every curated repo), so
    #: those presets override this with an empty tuple -- requiring it there
    #: would report a complete download as forever incomplete.
    required_bundle_files: ClassVar[tuple[str, ...]] = ("preprocessor_config.json",)

    properties: ClassVar[BaseProperties] = LargeV3Properties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = DECLARED_CAPABILITIES
    #: Static upper bounds over every supported config: the Hub presets support
    #: explicit artifact-only acquisition (upstream ``download_model``) and can
    #: also download inside ``WhisperModel(...)`` on first use. A ``model_path``
    #: instance narrows both to an externally provided requirement dynamically.
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=ArtifactDeclaration(
            acquisition_applicable=True,
            supports_explicit_acquisition=True,
            may_acquire_during_inference=True,
        )
    )
    provider_params_type: ClassVar[type[ProviderParams] | None] = FasterWhisperParams
    config_type: ClassVar[type[BaseConfig[str]] | None] = FasterWhisperConfig

    def __init__(self, **kwargs: Any) -> None:
        """Capture configuration (pure; weights load lazily, spec IC.9).

        Config is built via ``from_env``: unset fields fall back to
        ``STANDARD_ASR_FASTER_WHISPER__*`` environment variables (spec IC.4;
        double underscore between engine and field segments), explicit ``kwargs``
        win, and the HF token is wrapped in ``SecretStr`` by construction.

        Args:
            **kwargs: Configuration overrides.
        """
        self.config = FasterWhisperConfig.from_env("faster-whisper", **kwargs)
        self._model: object | None = None

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> object:
        """The loaded ``WhisperModel`` (loads it on first access).

        Returns:
            The underlying faster-whisper model instance.

        Raises:
            DiscoveryError: If the faster-whisper library is not installed.
            ArtifactUnavailableError: If required artifacts cannot resolve
                under the current policy.
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails.
            TranscriptionError: If a locally available model fails to load.
        """
        self._ensure_model_loaded()
        assert self._model is not None  # _ensure_model_loaded raises otherwise
        return self._model

    def ensure_loaded(self, *, mode: Mode = "batch") -> None:
        """Public alias for the lazy loader (used by the streaming session).

        Args:
            mode: The inference mode whose path is loading; artifact errors
                raised here carry a report resolved for this mode.

        Raises:
            DiscoveryError: If the faster-whisper library is not installed.
            ArtifactUnavailableError: If required artifacts cannot resolve
                under the current policy.
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails.
            TranscriptionError: If a locally available model fails to load.
        """
        self._ensure_model_loaded(mode=mode)

    def _ensure_model_loaded(self, *, mode: Mode = "batch") -> None:
        """Load the faster-whisper model lazily.

        The artifact guard runs first: a requirement the engine already knows
        cannot resolve under the current policy raises
        ``ArtifactUnavailableError`` before any loader call, and a failed
        allowed first-use acquisition raises ``ArtifactAcquisitionError``.
        Every other load failure keeps the batch R7 ``TranscriptionError``
        mapping (a present-but-unloadable model is an engine-execution fault,
        not a pre-inference availability state the engine can prove).

        Raises:
            DiscoveryError: If the faster-whisper library is not installed.
            ArtifactUnavailableError: If required artifacts are missing and
                cannot be acquired under the current policy.
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails.
            TranscriptionError: If a locally available model fails to load.
        """
        if self._model is not None:
            return
        # faster-whisper / huggingface_hub use tqdm for download progress bars,
        # which spawns a persistent `tqdm_monitor` DAEMON thread on first use.
        # That thread is harmless at interpreter exit, but the standard sync-bridge
        # compliance check (check_sync_bridge) flags ANY leaked background thread,
        # so an unsuppressed monitor makes a fully-correct engine fail compliance.
        # Disabling the monitor (cosmetic only) before the first tqdm instance is
        # the adapter's responsibility -- we own the lifecycle of what loading the
        # model spawns. See docs/STANDARD_ASR_FINDINGS.md.
        _disable_tqdm_monitor_thread()
        try:
            from faster_whisper import (  # pyright: ignore[reportMissingImports]
                WhisperModel,
            )
        except Exception as exc:
            raise DiscoveryError(
                "faster-whisper is not installed. Install 'std-faster-whisper' "
                "with its dependencies (pip install std-faster-whisper)."
            ) from exc

        config = cast(FasterWhisperConfig, self.config)
        local_only = config.local_files_only or not allow_downloads()
        # The artifact guard: the same cheap inspection that artifact_status()
        # reports. It decides whether a loader failure below counts as a failed
        # implicit acquisition or an engine-execution fault. The report on a
        # guard error is built from this same requirement (no second
        # inspection, no TOCTOU window) with the caller's mode.
        requirement = status_requirement(
            config, type(self).model_size, type(self).required_bundle_files
        )
        report = ArtifactReport.from_requirements(
            mode=mode, applicable=True, requirements=(requirement,)
        )
        if requirement.state != ARTIFACT_READY:
            if config.model_path is not None and requirement.state in (
                ARTIFACT_MISSING,
                ARTIFACT_INCOMPLETE,
            ):
                # The status check already ran and answered (the path is
                # absent, a file, or provably not a complete bundle); the
                # loader would only turn that knowledge into an opaque native
                # failure -- or a silent tokenizer Hub fetch.
                raise ArtifactUnavailableError(
                    f"The configured model_path {config.model_path!r} is not a "
                    f"usable CTranslate2 bundle (state: {requirement.state}).",
                    reason=cast("Any", requirement.state),
                    report=report,
                    hint="Provide the complete CTranslate2 directory or unset model_path.",
                )
            if config.model_path is None and local_only:
                raise ArtifactUnavailableError(
                    f"The {type(self).model_size} model is not fully cached "
                    f"(state: {requirement.state}) and downloads are disabled.",
                    reason="downloads_disabled",
                    report=report,
                    hint=(
                        "Enable downloads (unset local_files_only and set "
                        "STANDARD_ASR_ALLOW_DOWNLOAD=1), then run "
                        "'standard-asr pull'."
                    ),
                )
            # A model_path in state unknown is deliberately attempted: unknown
            # is not evidence of unavailability (AR.2), and the loader is the
            # authoritative check for an unrecognized local layout.
        # Spec IC.9 precedence: explicit download_root > STANDARD_ASR_MODEL_DIR >
        # library default (HF hub cache) > shared standard cache. faster-whisper
        # HAS a library default (download_root=None resolves via the HF cache),
        # so we forward the None passthrough unchanged -- forcing a concrete
        # directory would break offline loads of hub-cached models and silently
        # re-download into a second cache.
        download_root = resolve_download_root(config.download_root, has_library_default=True)
        # Model selection is by preset (the class's model_size, spec IC.7); a
        # local model_path is an optional weights/path override that wins. The
        # path uses the same canonical form status inspected -- expanding only
        # on the status side would hand a tilde path to the Hub as a repo id.
        if config.model_path is not None:
            model_source = str(normalized_model_path(config.model_path))
        else:
            model_source = type(self).model_size
        token = config.hf_token.get_secret_value() if config.hf_token is not None else None
        try:
            model = WhisperModel(
                model_size_or_path=model_source,
                device=config.device or "auto",
                device_index=(
                    config.device_indices
                    if config.device_indices is not None
                    else config.device_index
                ),
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
                download_root=None if download_root is None else str(download_root),
                local_files_only=local_only,
                revision=config.revision,
                use_auth_token=token,
            )
        except Exception as exc:
            if config.model_path is None and requirement.state != ARTIFACT_READY:
                # The loader was the allowed implicit acquisition path. An
                # access rejection carries its discovered action (the reason
                # comes from the blocker, not from which code path noticed).
                raise_for_gated_source(exc, type(self).model_size, report)
                raise ArtifactAcquisitionError(
                    f"First-use acquisition of the {type(self).model_size} "
                    f"model failed: {type(exc).__name__}.",
                    reason="failed",
                    report=report,
                    hint="Run 'standard-asr pull' to acquire it explicitly.",
                ) from exc
            raise TranscriptionError(
                f"faster-whisper failed to load the model: {type(exc).__name__}."
            ) from exc
        if config.model_path is None and requirement.state != ARTIFACT_READY:
            # The load above doubled as the implicit acquisition, and its
            # internal downloader silently falls back to the local cache when
            # the remote is unreachable -- so a load can succeed on the same
            # incomplete bundle status just reported. Verify the acquisition
            # postcondition before adopting the model. Only incomplete is
            # positive evidence against the loaded model (the same resolution
            # rule sees the directory the loader used, and it lacks closure
            # files); unknown or missing after a successful load carries no
            # such evidence (spec AR.2) -- the loader just succeeded.
            after = status_requirement(
                config, type(self).model_size, type(self).required_bundle_files
            )
            if after.state == ARTIFACT_INCOMPLETE:
                raise ArtifactAcquisitionError(
                    f"First-use acquisition of the {type(self).model_size} "
                    f"model resolved without completing the bundle (state: "
                    f"{after.state}); the source may be unreachable and the "
                    "local cache incomplete.",
                    reason="failed",
                    report=ArtifactReport.from_requirements(
                        mode=mode, applicable=True, requirements=(after,)
                    ),
                    hint="Run 'standard-asr pull' while the source is reachable.",
                )
        # The mel closure is only provable here: the loaded CT2 model
        # self-describes its n_mels, while nothing on disk names it (the CT2
        # config.json carries no feature size), so status cannot see that a
        # bundle without preprocessor_config.json makes upstream silently
        # build an 80-mel feature extractor for a 128-mel model. CT2 does
        # reject the mismatched features -- loudly, at the first request,
        # with a bare shape error (verified empirically); failing at load
        # with the artifact diagnosis is the actionable form.
        n_model = cast(Any, model).model.n_mels
        n_features = cast(Any, model).feature_extractor.mel_filters.shape[0]
        if n_model != n_features:
            if config.model_path is not None:
                hint = (
                    "Reconvert with --copy_files preprocessor_config.json "
                    "tokenizer.json, or point model_path at a matching bundle."
                )
            else:
                # A plain pull cannot repair this: the disk closure looks
                # ready, and the Hub cache never refetches a present file of
                # the resolved commit -- the executable repair is removing
                # the repository cache root (the blob store rebuilds a
                # deleted snapshot pointer without a transfer).
                hint = (
                    "Remove the model's repository cache directory and run "
                    "'standard-asr pull' again, or pin a revision whose "
                    "content matches."
                )
            # The attached report must be the one that ESTABLISHED the
            # unavailable state: the pre-load report said ready (the file
            # closure cannot see mels), so rebuild it from this discovery --
            # observed POST-load, because the loader's online resolution may
            # have moved a mutable revision past the preflight's
            # cache-resolved commit, and the action must name the snapshot
            # the mismatch actually came from.
            current = status_requirement(
                config, type(self).model_size, type(self).required_bundle_files
            )
            mismatch = mel_mismatch_requirement(
                config,
                current,
                model_n_mels=n_model,
                extractor_n_mels=n_features,
            )
            raise ArtifactUnavailableError(
                f"The loaded bundle computes {n_features}-mel features, but "
                f"the model expects {n_model}: preprocessor_config.json is "
                "missing or does not match this model.",
                reason="incomplete",
                report=ArtifactReport.from_requirements(
                    mode=mode, applicable=True, requirements=(mismatch,)
                ),
                hint=hint,
            )
        self._model = model

    def prepare(self) -> None:
        """Warm up the CTranslate2 model without transcribing (spec IC.11).

        Idempotent and synchronous. The native loader cannot separate
        acquisition from loading, so a cold cache acquires on this path too
        (under the same download policy and artifact errors as inference);
        explicit artifact-only acquisition is :meth:`acquire_artifacts`.

        Raises:
            DiscoveryError: If the faster-whisper library is not installed.
            ArtifactUnavailableError: If required artifacts are missing and
                cannot be acquired under the current policy.
            ArtifactAcquisitionError: If an allowed acquisition attempt fails.
            TranscriptionError: If a locally available model fails to load.
        """
        self._ensure_model_loaded()

    # ------------------------------------------------------------------ #
    # Inference-artifact lifecycle (protocol 1.1)
    # ------------------------------------------------------------------ #
    def _artifact_requirements(
        self,
        context: ArtifactContext,
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        """Report the recognizer bundle for the resolved config.

        Batch and streaming share the same weights, so the closure does not
        depend on the request context.

        Args:
            context: Resolved, best-effort-gated request context.

        Returns:
            One requirement: the Hub preset or the operator-provided path.
        """
        config = cast(FasterWhisperConfig, self.config)
        requirement = status_requirement(
            config, type(self).model_size, type(self).required_bundle_files
        )
        return True, (requirement,), ()

    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: ArtifactProgressCallback | None,
    ) -> None:
        """Acquire the Hub preset with the upstream artifact-only downloader.

        A ``model_path`` requirement is externally provided and never reaches
        this hook (it can never be acquired and is never a refresh target).
        A refresh carries its own re-resolution evidence: the downloader
        silently falls back to the local cache when the remote is
        unreachable, so ``acquire`` verifies the source resolution itself
        (spec AR.4).

        Args:
            context: Resolved artifact context.
            requirements: Runnable acquisition and refresh targets.
            refresh: Whether mutable targets must be re-resolved.
            progress: Serialized progress observer, if requested.

        Returns:
            None.
        """
        config = cast(FasterWhisperConfig, self.config)
        if any(item.artifact_id == HUB_ARTIFACT_ID for item in requirements):
            acquire(config, type(self).model_size, progress, refresh=refresh)

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #
    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Transcribe negotiated audio with faster-whisper.

        Args:
            prepared: Engine-ready audio (an array, a file path, or in-memory
                bytes -- one of the declared ``accepted_input`` shapes).
            params: Gated runtime parameters.

        Returns:
            A Standard ASR transcription result.

        Raises:
            ArtifactUnavailableError: If required artifacts cannot resolve
                under the current policy (an explicit R7 exemption).
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails (an explicit R7 exemption).
            TranscriptionError: If faster-whisper raises during inference. The
                batch error contract (spec RT R7) requires an engine-execution
                failure to surface as a portable ``TranscriptionError`` with the
                native exception preserved as ``__cause__``.
        """
        self._ensure_model_loaded()

        config = cast(FasterWhisperConfig, self.config)
        resolved = effective_language(
            params.language,
            config.default_language,
            has_language_axis=True,
            runtime_override_supported=True,
        )
        language = None
        if resolved and resolved != "auto":
            # The standard layer hands us the full request tag on a refinement
            # match (spec LANG R4); we reduce to Whisper's primary-subtag grain.
            language = normalize_bcp47(resolved).split("-", maxsplit=1)[0]

        source = self._source_for(prepared)
        # Only WORD requires the upstream forced-alignment pass; a SEGMENT request
        # is served by the always-present per-segment start/end and MUST NOT
        # back-fill word-level data (spec TR.3 null semantics).
        want_word_ts = params.word_timestamps == WordTimestampGranularity.WORD
        model = cast(Any, self._model)
        # faster-whisper returns a LAZY segment generator, so decode/inference
        # runs while convert_segments consumes it -- both the transcribe() call
        # and the consumption are inside the wrap (spec RT R7).
        try:
            segments, info = model.transcribe(
                source,
                language=language,
                word_timestamps=want_word_ts,
                initial_prompt=params.prompt,
                hotwords=" ".join(params.phrase_hints) if params.phrase_hints else None,
                **provider_kwargs(params.provider_params),
            )
            segment_list, word_list = convert_segments(segments)
        except Exception as exc:
            raise TranscriptionError(
                f"faster-whisper transcription failed: {type(exc).__name__}."
            ) from exc

        text = "".join(seg.text for seg in segment_list)
        detected = normalize_bcp47(info.language) if info.language else None
        return TranscriptionResult(
            text=text,
            detected_language=detected,
            language_confidence=getattr(info, "language_probability", None),
            duration=info.duration,
            segments=segment_list or None,
            words=word_list if want_word_ts else None,
            extra=safe_extra(info),
        )

    def _source_for(self, prepared: PreparedAudio) -> Any:
        """Map negotiated audio onto the source faster-whisper accepts.

        Args:
            prepared: The negotiated audio (array / bytes / path).

        Returns:
            A float32 array, an in-memory binary file-like, or a path string.
        """
        if prepared.array is not None:
            # We declare accepted_sample_rates=[16000]; the standard layer
            # negotiates to it. Assert defensively -- an off-rate array silently
            # produces wrong timings/text.
            assert prepared.sample_rate == _SAMPLE_RATE, (
                f"faster-whisper requires 16 kHz audio; got {prepared.sample_rate} Hz "
                "(audio negotiation should have resampled to 16000)."
            )
            return prepared.array
        if prepared.data is not None:
            return io.BytesIO(prepared.data)
        return prepared.path

    # ------------------------------------------------------------------ #
    # Streaming (windowed)
    # ------------------------------------------------------------------ #
    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        """Open a windowed streaming session (spec ST; see ``_streaming.py``).

        The base ``start_transcription`` template has already enforced the
        ``audio_format`` / ``audio`` exclusivity, validated the language config,
        run the fail-closed wire-format check, and gated + frozen the params
        (spec RT R5). We just build the session.

        For the whole-input path (``audio=...``) the base hands us a negotiated
        ``prepared_audio``; we pre-load it into the session's buffer so the
        OpenAI-style "submit a file, stream the result" pattern works too.

        Args:
            gated_params: Frozen, gated runtime parameters.
            audio_format: The incremental wire format, or ``None``.
            prepared_audio: The negotiated whole input, or ``None``.

        Returns:
            A :class:`FasterWhisperStreamingSession`.
        """
        session = FasterWhisperStreamingSession(self, gated_params)
        if prepared_audio is not None:
            # Whole-input path: seed the window so _produce decodes it directly.
            session.feed(_prepared_to_pcm(prepared_audio))
        return session


def _prepared_to_pcm(prepared: PreparedAudio) -> bytes:
    """Convert negotiated whole-input audio into canonical pcm_s16le bytes.

    Used only on the streaming whole-input path. The negotiated audio is already
    a 16 kHz mono float32 array (we declare ``accepted_input`` includes ``array``
    and the standard layer resamples to our native rate), so we just quantize to
    int16 LE.

    Args:
        prepared: The negotiated whole-input audio.

    Returns:
        Canonical 16-bit LE PCM bytes.

    Raises:
        TranscriptionError: If the whole input did not arrive as an array.
    """
    if prepared.array is None:
        # Defensive: array is the negotiated shape for the streaming whole-input
        # path on this engine. Bytes/path here would mean a negotiation change.
        raise TranscriptionError(
            "Streaming whole-input audio was not delivered as an array; "
            "expected a negotiated 16 kHz float32 array."
        )
    # Canonical float32 -> int16 LE quantization (spec AI R4): sanitize non-finite
    # samples, clip to [-1, 1], round-half to nearest, write little-endian int16.
    arr: NDArray[np.float32] = np.nan_to_num(
        prepared.array, nan=0.0, posinf=1.0, neginf=-1.0
    ).astype(np.float32)
    clipped: NDArray[np.float32] = arr.clip(-1.0, 1.0)
    quantized: NDArray[np.int16] = np.round(clipped * 32767.0).astype("<i2")
    return quantized.tobytes()


# --------------------------------------------------------------------------- #
# Presets. Each Whisper variant is its own entry point (spec IC.7) so the
# discovery layer can list every available model. A preset overrides only the
# model_name (for the matching properties.model_id) and the model_size (the
# upstream weights id); the config, params, capabilities, and the
# transcribe/stream pipeline are all inherited unchanged.
# --------------------------------------------------------------------------- #
class TinyASR(FasterWhisperASR):
    """The ``faster-whisper/tiny`` preset (smallest, fastest; for tests)."""

    model_size: ClassVar[str] = "tiny"
    # An 80-mel repo: no preprocessor_config.json on the Hub.
    required_bundle_files: ClassVar[tuple[str, ...]] = ()
    properties: ClassVar[BaseProperties] = TinyProperties()


class BaseASR(FasterWhisperASR):
    """The ``faster-whisper/base`` preset."""

    model_size: ClassVar[str] = "base"
    # An 80-mel repo: no preprocessor_config.json on the Hub.
    required_bundle_files: ClassVar[tuple[str, ...]] = ()
    properties: ClassVar[BaseProperties] = BaseModelProperties()


class SmallASR(FasterWhisperASR):
    """The ``faster-whisper/small`` preset."""

    model_size: ClassVar[str] = "small"
    # An 80-mel repo: no preprocessor_config.json on the Hub.
    required_bundle_files: ClassVar[tuple[str, ...]] = ()
    properties: ClassVar[BaseProperties] = SmallProperties()


class MediumASR(FasterWhisperASR):
    """The ``faster-whisper/medium`` preset."""

    model_size: ClassVar[str] = "medium"
    # An 80-mel repo: no preprocessor_config.json on the Hub.
    required_bundle_files: ClassVar[tuple[str, ...]] = ()
    properties: ClassVar[BaseProperties] = MediumProperties()


class LargeV3ASR(FasterWhisperASR):
    """The ``faster-whisper/large-v3`` preset (explicit alias of the base class)."""

    model_size: ClassVar[str] = "large-v3"
    properties: ClassVar[BaseProperties] = LargeV3Properties()


class DistilLargeV3ASR(FasterWhisperASR):
    """The ``faster-whisper/distil-large-v3`` preset (distilled, English-only)."""

    model_size: ClassVar[str] = "distil-large-v3"
    properties: ClassVar[BaseProperties] = DistilLargeV3Properties()


class TurboASR(FasterWhisperASR):
    """The ``faster-whisper/large-v3-turbo`` preset (fastest large preset)."""

    model_size: ClassVar[str] = "large-v3-turbo"
    properties: ClassVar[BaseProperties] = TurboProperties()


__all__ = [
    "BaseASR",
    "DistilLargeV3ASR",
    "FasterWhisperASR",
    "LargeV3ASR",
    "MediumASR",
    "SmallASR",
    "TinyASR",
    "TurboASR",
]
