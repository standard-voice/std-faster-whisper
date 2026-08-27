# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Inference-artifact status and acquisition for the faster-whisper engine.

The engine's one logical requirement is the CTranslate2 recognizer bundle.
Status inspection reuses the upstream cache RESOLUTION
(``download_model(..., local_files_only=True)``) and then checks completeness
itself: resolution success only proves the snapshot directory exists --
``snapshot_download`` documents that it cannot verify the files inside -- so
an interrupted download resolves but must report ``incomplete``, never
``ready``. An explicit ``model_path`` is an externally provided requirement
that the engine can inspect but never acquire.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_AUTHENTICATE,
    ARTIFACT_ACTION_OTHER,
    ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
    ARTIFACT_ACTION_REQUEST_ACCESS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_INCOMPLETE,
    ARTIFACT_MISSING,
    ARTIFACT_PROGRESS_TRANSFERRING,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ArtifactAction,
    ArtifactActionKind,
    ArtifactProgress,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.exceptions import ArtifactAcquisitionError
from standard_asr.runtime.downloads import allow_downloads, resolve_download_root

if TYPE_CHECKING:
    from ._config import FasterWhisperConfig

#: The engine's single logical requirement id for a Hub-sourced recognizer.
HUB_ARTIFACT_ID = "ct2-recognizer"
#: The requirement id when an operator-provided ``model_path`` overrides it.
LOCAL_ARTIFACT_ID = "ct2-local-path"

#: A full commit hash pins the snapshot; anything else (a branch, a tag, or the
#: default revision) is a mutable reference the Hub can re-resolve.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

#: The file WhisperModel requires at the root of a CTranslate2 directory.
_CT2_MODEL_FILE = "model.bin"
#: CTranslate2 loads without this file, but every decode reads its fields at
#: request time (suppress ids, language ids, alignment heads) and fails, so a
#: bundle lacking it cannot serve a single request (verified empirically:
#: transcribe and align both raise).
_CT2_CONFIG_FILE = "config.json"
#: Without this file, upstream falls back to Tokenizer.from_pretrained -- a
#: Hub fetch that bypasses local_files_only AND the global download toggle --
#: so a directory lacking it is not offline-ready.
_CT2_TOKENIZER_FILE = "tokenizer.json"
#: CTranslate2 refuses to load a directory without a vocabulary file
#: (vocabulary.txt in the older repos, vocabulary.json in the 128-mel ones).
_CT2_VOCABULARY_GLOB = "vocabulary.*"


def _bundle_complete(root: Path, extra_files: tuple[str, ...] = ()) -> bool:
    """Return whether a directory is a complete offline-usable CT2 bundle.

    The closure is the loader's own, established empirically: ``model.bin``
    and a vocabulary file are load requirements, ``config.json`` is read by
    every decode at request time, and a missing ``tokenizer.json`` triggers a
    silent Hub fetch past every download gate. ``preprocessor_config.json``
    is deliberately NOT here: the 80-mel repos do not carry it at all, so a
    universal requirement would report their complete downloads as forever
    incomplete -- presets that do carry it declare it via ``extra_files``.

    Args:
        root: Candidate CTranslate2 directory.
        extra_files: Preset-declared files that must also exist.

    Returns:
        ``True`` when the loader closure and every extra file are present.
    """
    required = (_CT2_MODEL_FILE, _CT2_CONFIG_FILE, _CT2_TOKENIZER_FILE, *extra_files)
    if not all((root / name).is_file() for name in required):
        return False
    return any(entry.is_file() for entry in root.glob(_CT2_VOCABULARY_GLOB))


def disable_tqdm_monitor_thread() -> None:
    """Disable tqdm's auto-spawned monitor daemon thread (idempotent, best-effort).

    Setting ``tqdm.monitor_interval = 0`` before any tqdm instance is created
    prevents the persistent ``tqdm_monitor`` thread that would otherwise leak
    past a session and trip ``check_sync_bridge``. tqdm starts the monitor in
    ``__new__``, so even upstream's disabled progress class spawns it unless
    the interval is zeroed first; the explicit acquisition path creates the
    process's first tqdm instance on a cold cache, so it must run this guard
    like the loading path does. If tqdm is somehow absent or already started
    its monitor, this is a no-op (it only ever *disables*).
    """
    try:
        import tqdm

        if tqdm.tqdm.monitor_interval != 0:
            tqdm.tqdm.monitor_interval = 0
    except Exception:
        pass


def normalized_model_path(model_path: str) -> Path:
    """Return the operator ``model_path`` in its one canonical absolute form.

    Status inspection and the loader MUST agree on this form: expanding only
    on the status side would report a tilde path as ready while the loader
    hands the raw string to the Hub as a repo id.

    Args:
        model_path: The configured local checkpoint path.

    Returns:
        The expanded, resolved path.
    """
    return Path(model_path).expanduser().resolve()


def _revision_is_pinned(revision: str | None) -> bool:
    """Return whether a configured revision is an immutable commit hash.

    Args:
        revision: Configured Hugging Face revision, or ``None``.

    Returns:
        ``True`` only for a full 40-hex commit hash.
    """
    return revision is not None and _COMMIT_SHA.fullmatch(revision) is not None


def _tree_size_bytes(root: Path) -> int | None:
    """Sum regular-file sizes under a resolved snapshot root.

    Args:
        root: Resolved artifact directory.

    Returns:
        The logical size in bytes, or ``None`` when the walk fails.
    """
    try:
        return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    except OSError:
        return None


def _resolve_cached_snapshot(
    config: FasterWhisperConfig,
    model_size: str,
    extra_files: tuple[str, ...],
) -> tuple[str, Path | None]:
    """Resolve the preset's snapshot from local caches only, then verify it.

    Resolution reuses the loader's own rule (upstream ``download_model`` with
    ``local_files_only=True``); completeness is this function's own check,
    because upstream returns an existing snapshot directory without verifying
    its contents.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.
        extra_files: Preset-declared bundle files beyond the loader closure.

    Returns:
        A ``(state, path)`` pair: ``ready`` with the snapshot, ``incomplete``
        with the partial snapshot, ``missing`` or ``unknown`` with ``None``.
    """
    try:
        from faster_whisper import download_model  # pyright: ignore[reportMissingImports]
    except Exception:
        return ARTIFACT_UNKNOWN, None

    download_root = resolve_download_root(config.download_root, has_library_default=True)
    token = config.hf_token.get_secret_value() if config.hf_token is not None else None
    try:
        resolved = cast(
            "str",
            download_model(
                model_size,
                local_files_only=True,
                cache_dir=None if download_root is None else str(download_root),
                revision=config.revision,
                use_auth_token=token,
            ),
        )
    except FileNotFoundError:
        # The documented not-in-cache outcome of an offline resolution
        # (upstream LocalEntryNotFoundError subclasses FileNotFoundError, so
        # this classification needs no fragile hub import): reliable evidence
        # of a missing snapshot. A PermissionError is deliberately NOT here.
        return ARTIFACT_MISSING, None
    except Exception:
        # Anything else (an unreadable cache, a permission failure, a stalled
        # network mount) is not evidence of absence; unknown never means
        # ready, and it never claims missing either.
        return ARTIFACT_UNKNOWN, None
    # A relative download_root yields a relative snapshot path; the report's
    # location field requires (and callers deserve) the absolute form.
    snapshot = Path(resolved).expanduser().resolve()
    if not _bundle_complete(snapshot, extra_files):
        # The snapshot directory resolved but the bundle is not complete: a
        # detectable interrupted acquisition.
        return ARTIFACT_INCOMPLETE, snapshot
    return ARTIFACT_READY, snapshot


def _local_path_requirement(config: FasterWhisperConfig) -> ArtifactRequirement:
    """Build the externally provided requirement for a ``model_path`` config.

    Preset ``extra_files`` deliberately do NOT apply here: ``model_path``
    replaces the preset's Hub bundle with an operator conversion of an
    arbitrary Whisper variant, so only the loader-universal closure is
    checkable.

    Args:
        config: Resolved engine configuration with ``model_path`` set.

    Returns:
        The single logical requirement for the operator-provided directory.
    """
    assert config.model_path is not None
    path = normalized_model_path(config.model_path)
    if not path.exists():
        state = ARTIFACT_MISSING
        message = (
            f"Provide a converted CTranslate2 model directory at {path} "
            "(the configured model_path), or unset model_path to use the "
            "preset's Hub model."
        )
    elif path.is_file():
        # Pointing model_path at a file (for example model.bin itself) would
        # make the upstream loader treat the absolute path as a Hub repo id.
        state = ARTIFACT_INCOMPLETE
        message = (
            f"The configured model_path {path} is a file; point it at the "
            "converted CTranslate2 DIRECTORY containing model.bin, "
            "config.json, tokenizer.json, and a vocabulary file."
        )
    elif _bundle_complete(path):
        state = ARTIFACT_READY
        message = None
    else:
        # The directory exists but the bundle is provably not complete: the
        # loader requires model.bin and a vocabulary file, every decode reads
        # config.json at request time, and a missing tokenizer.json triggers
        # a silent Hub fetch past every download gate. This check already ran
        # and answered, so the state is incomplete, not unknown.
        state = ARTIFACT_INCOMPLETE
        message = (
            f"The configured model_path {path} lacks part of the loadable "
            f"bundle ({_CT2_MODEL_FILE}, {_CT2_CONFIG_FILE}, "
            f"{_CT2_TOKENIZER_FILE}, and a vocabulary file); convert with "
            "--copy_files tokenizer.json, or point at the complete directory."
        )

    return ArtifactRequirement(
        artifact_id=LOCAL_ARTIFACT_ID,
        label="Operator-provided CTranslate2 model directory",
        state=state,
        required_for_inference=True,
        can_acquire_now=False,
        may_acquire_during_inference=False,
        source_is_mutable=False,
        acquisition_blocker=None if state == ARTIFACT_READY else ARTIFACT_BLOCKER_ACTION_REQUIRED,
        required_actions=()
        if message is None
        else (ArtifactAction(kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS, message=message),),
        location=path if path.exists() else None,
        size_bytes=_tree_size_bytes(path) if state == ARTIFACT_READY else None,
    )


def _hub_requirement(
    config: FasterWhisperConfig,
    model_size: str,
    extra_files: tuple[str, ...],
) -> ArtifactRequirement:
    """Build the Hub-preset requirement from a cache-only resolution.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.
        extra_files: Preset-declared bundle files beyond the loader closure.

    Returns:
        The single logical requirement for the preset's recognizer bundle.
    """
    downloads_blocked = config.local_files_only or not allow_downloads()
    state, snapshot = _resolve_cached_snapshot(config, model_size, extra_files)

    # Prefer the resolved commit over the configured reference: two engines
    # resolving the same snapshot must report the same version, and a refresh
    # of a moved branch must be observable through this field.
    artifact_version = config.revision
    if snapshot is not None and snapshot.parent.name == "snapshots":
        artifact_version = snapshot.name

    if state == ARTIFACT_READY:
        can_acquire_now = False
        blocker = None
    elif downloads_blocked:
        can_acquire_now = False
        blocker = ARTIFACT_BLOCKER_DOWNLOADS_DISABLED
    else:
        can_acquire_now = True
        blocker = None

    return ArtifactRequirement(
        artifact_id=HUB_ARTIFACT_ID,
        label=f"faster-whisper {model_size} (CTranslate2)",
        state=state,
        required_for_inference=True,
        can_acquire_now=can_acquire_now,
        may_acquire_during_inference=not downloads_blocked,
        source_is_mutable=not _revision_is_pinned(config.revision),
        acquisition_blocker=blocker,
        location=snapshot,
        size_bytes=_tree_size_bytes(snapshot) if snapshot is not None else None,
        artifact_version=artifact_version,
    )


def mel_mismatch_requirement(
    config: FasterWhisperConfig,
    base: ArtifactRequirement,
    *,
    model_n_mels: int,
    extractor_n_mels: int,
) -> ArtifactRequirement:
    """Rebuild the requirement from the load-time mel-closure discovery.

    An artifact error attaches the status report that ESTABLISHED the
    unavailable state; the pre-load, file-closure report said ready, so the
    discovery needs a fresh requirement (validated, so the per-state
    invariants hold), not the stale one. The discovery is request-scoped:
    ``artifact_status()`` stays a point-in-time disk observation (spec AR.9)
    and does not remember load-derived facts.

    Both shapes are action-blocked, never ``can_acquire_now``: the public
    acquisition path re-inspects the DISK closure, which reports the very
    files whose content is wrong as ready, so a plain pull would not run --
    and even a forced run would not repair, because the Hub cache never
    refetches a file that is already present for the resolved commit. The
    Hub action therefore names the model's repository CACHE ROOT: removing
    only the ``snapshots/<commit>`` pointer directory is not enough, the
    downloader rebuilds it from the content-addressed blob store without a
    transfer. And when a fresh download still mismatches, the source
    revision itself does not match the model -- the action says to pin one
    that does.

    Args:
        config: Resolved engine configuration.
        base: A POST-load requirement (the loader's online resolution may
            have moved a mutable revision past the preflight's cache-resolved
            commit, so a pre-load requirement can name a snapshot the
            mismatch did not come from).
        model_n_mels: The mel count the loaded model self-describes.
        extractor_n_mels: The mel count the built feature extractor computes.

    Returns:
        The requirement in state incomplete, carrying the discovery.
    """
    if config.model_path is not None:
        message = (
            f"The bundle computes {extractor_n_mels}-mel features, but the "
            f"model expects {model_n_mels}: reconvert with --copy_files "
            "preprocessor_config.json tokenizer.json, or point model_path "
            "at a matching bundle."
        )
        kind: ArtifactActionKind = ARTIFACT_ACTION_PROVIDE_ARTIFACTS
    else:
        cache_root: Path | None = None
        if base.location is not None and base.location.parent.name == "snapshots":
            cache_root = base.location.parent.parent
        target = "" if cache_root is None else f" ({cache_root})"
        message = (
            f"The cached snapshot computes {extractor_n_mels}-mel features, "
            f"but the model expects {model_n_mels}. Remove the model's "
            f"repository cache directory{target} and run 'standard-asr pull' "
            "again; removing only the snapshot folder is not enough, because "
            "the downloader rebuilds it from the blob store. If the mismatch "
            "persists, the source revision itself does not match the model; "
            "pin one that does."
        )
        kind = ARTIFACT_ACTION_OTHER
    return ArtifactRequirement(
        artifact_id=base.artifact_id,
        label=base.label,
        state=ARTIFACT_INCOMPLETE,
        required_for_inference=True,
        can_acquire_now=False,
        may_acquire_during_inference=False,
        source_is_mutable=False if config.model_path is not None else base.source_is_mutable,
        acquisition_blocker=ARTIFACT_BLOCKER_ACTION_REQUIRED,
        required_actions=(ArtifactAction(kind=kind, message=message),),
        location=base.location,
        size_bytes=base.size_bytes,
        artifact_version=None if config.model_path is not None else base.artifact_version,
    )


def status_requirement(
    config: FasterWhisperConfig,
    model_size: str,
    extra_files: tuple[str, ...] = (),
) -> ArtifactRequirement:
    """Report the engine's one logical requirement for the resolved config.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.
        extra_files: Preset-declared bundle files beyond the loader closure
            (Hub preset only; a ``model_path`` conversion is checked against
            the loader-universal closure).

    Returns:
        The requirement for either the operator path or the Hub preset.
    """
    if config.model_path is not None:
        return _local_path_requirement(config)
    return _hub_requirement(config, model_size, extra_files)


def _hub_repo_id(model_size: str) -> str:
    """Map the preset's weights id to its Hub repository, by upstream's rule.

    Mirrors ``faster_whisper.utils.download_model``: an id containing a slash
    passes through, anything else resolves through the upstream size table.
    Importing the table keeps the mapping from drifting against upstream.

    Args:
        model_size: The preset's upstream weights id.

    Returns:
        The Hugging Face repository id.
    """
    if re.match(r".*/.*", model_size):
        return model_size
    from faster_whisper.utils import (  # pyright: ignore[reportMissingImports]
        _MODELS,  # pyright: ignore[reportPrivateUsage]
    )

    return _MODELS[model_size]


def _remote_commit(config: FasterWhisperConfig, model_size: str) -> str:
    """Re-resolve the preset's revision against the source and return it.

    The positive evidence spec AR.4 requires for a refresh: the native
    downloader silently falls back to the local cache when the remote is
    unreachable, so download success can never prove that the mutable
    reference was re-resolved -- this metadata query can.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.

    Returns:
        The commit hash the source currently resolves the revision to.

    Raises:
        ArtifactAcquisitionError: If the source answered without naming a
            commit.
        Exception: Whatever the metadata query raises when the source is
            unreachable or rejects the request.
    """
    import huggingface_hub  # pyright: ignore[reportMissingModuleSource]

    token = config.hf_token.get_secret_value() if config.hf_token is not None else None
    info = huggingface_hub.HfApi(token=token).model_info(
        _hub_repo_id(model_size), revision=config.revision
    )
    sha = info.sha
    if sha is None:
        raise ArtifactAcquisitionError(
            f"The source metadata for {model_size} did not name a commit, so "
            "the refresh cannot be verified.",
            reason="failed",
        )
    return sha


def raise_for_gated_source(
    exc: BaseException,
    model_size: str,
    report: object | None = None,
) -> None:
    """Translate a gated or unauthenticated Hub rejection into actions.

    Shared by the explicit acquisition hook and the implicit first-use path:
    the reason comes from the discovered blocker, not from which code path
    noticed it.

    Args:
        exc: The native failure raised by the Hub client.
        model_size: The preset's upstream weights id, for the message.
        report: The preflight report to attach, when the caller has one.

    Returns:
        None when the failure is not an access problem.

    Raises:
        ArtifactAcquisitionError: With ``reason="action_required"`` and the
            discovered action when the source is gated or rejects the
            credentials.
    """
    try:
        from huggingface_hub.errors import (  # pyright: ignore[reportMissingModuleSource]
            GatedRepoError,
            HfHubHTTPError,
        )
    except Exception:
        return
    if isinstance(exc, GatedRepoError):
        raise ArtifactAcquisitionError(
            f"The {model_size} model repository is gated.",
            reason="action_required",
            report=cast("ArtifactReport | None", report),
            required_actions=(
                ArtifactAction(
                    kind=ARTIFACT_ACTION_REQUEST_ACCESS,
                    message=(
                        "Accept the model terms or request access on its "
                        "Hugging Face page, then configure hf_token."
                    ),
                ),
            ),
            hint="Set the hf_token config field after access is granted.",
        ) from exc
    response = getattr(exc, "response", None)
    if isinstance(exc, HfHubHTTPError) and getattr(response, "status_code", None) == 401:
        raise ArtifactAcquisitionError(
            f"The {model_size} model repository rejected the request as unauthenticated.",
            reason="action_required",
            report=cast("ArtifactReport | None", report),
            required_actions=(
                ArtifactAction(
                    kind=ARTIFACT_ACTION_AUTHENTICATE,
                    message="Configure a valid hf_token for this repository.",
                ),
            ),
            hint="Set the hf_token config field.",
        ) from exc


def acquire(
    config: FasterWhisperConfig,
    model_size: str,
    progress: ArtifactProgressCallback | None,
    *,
    refresh: bool = False,
) -> None:
    """Materialize the Hub preset with the upstream artifact-only downloader.

    The upstream helper suppresses transfer progress; an incomplete snapshot
    is completed by the same call (upstream fetches only the files that are
    absent). A refresh cannot trust that call alone: the downloader silently
    falls back to the local cache when the remote is unreachable, so a
    refresh first re-resolves the mutable revision against the source and
    then verifies that the downloaded snapshot is that resolution (spec
    AR.4). Loading stays out: this never constructs a ``WhisperModel``.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.
        progress: Optional serialized progress observer.
        refresh: Whether the mutable revision must be re-resolved.

    Returns:
        None.

    Raises:
        ArtifactAcquisitionError: With ``reason="downloads_disabled"`` when the
            engine's own offline config (or the global toggle) forbids the
            transfer -- the core template only sees the global toggle, so the
            engine flag must be enforced here; with ``reason="action_required"``
            for a gated or unauthenticated repository; with ``reason="failed"``
            when a refresh cannot prove the source was re-resolved; every
            other native failure propagates for the template to wrap as
            ``reason="failed"``.
    """
    if config.local_files_only or not allow_downloads():
        raise ArtifactAcquisitionError(
            "Acquisition needs a network transfer, and downloads are disabled for this engine.",
            reason="downloads_disabled",
            hint=(
                "Unset local_files_only and set STANDARD_ASR_ALLOW_DOWNLOAD=1 "
                "to permit the transfer."
            ),
        )
    from faster_whisper import download_model  # pyright: ignore[reportMissingImports]

    expected_commit: str | None = None
    if refresh and not _revision_is_pinned(config.revision):
        try:
            expected_commit = _remote_commit(config, model_size)
        except ArtifactAcquisitionError:
            raise
        except Exception as exc:
            raise_for_gated_source(exc, model_size)
            raise ArtifactAcquisitionError(
                f"Refresh needs to re-resolve the {model_size} source "
                f"reference, and the source metadata query failed "
                f"({type(exc).__name__}).",
                reason="failed",
                hint="Make the Hugging Face endpoint reachable, then retry.",
            ) from exc
    disable_tqdm_monitor_thread()
    if progress is not None:
        # Upstream disables its tqdm hooks, so byte totals are unavailable:
        # emit one honest indeterminate transfer event, never a fabricated
        # percentage.
        progress(
            ArtifactProgress(phase=ARTIFACT_PROGRESS_TRANSFERRING, artifact_id=HUB_ARTIFACT_ID)
        )
    download_root = resolve_download_root(config.download_root, has_library_default=True)
    token = config.hf_token.get_secret_value() if config.hf_token is not None else None
    try:
        resolved = cast(
            "str",
            download_model(
                model_size,
                local_files_only=False,
                cache_dir=None if download_root is None else str(download_root),
                revision=config.revision,
                use_auth_token=token,
            ),
        )
    except Exception as exc:
        raise_for_gated_source(exc, model_size)
        raise
    snapshot = Path(resolved).expanduser().resolve()
    # Outside the snapshots/<commit> cache layout the resolution carries no
    # comparable commit; the layout is guaranteed here because the helper is
    # only ever pointed at a Hub cache root.
    if (
        expected_commit is not None
        and snapshot.parent.name == "snapshots"
        and snapshot.name != expected_commit
    ):
        raise ArtifactAcquisitionError(
            f"Refresh resolved a {model_size} snapshot that is not the "
            "source's current commit: the source reference moved during the "
            "refresh, or the downloader fell back to the local cache without "
            "reaching the source.",
            reason="failed",
            hint="Retry while the source is reachable.",
        )
