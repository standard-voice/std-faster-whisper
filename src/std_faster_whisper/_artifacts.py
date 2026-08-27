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
    ArtifactProgress,
    ArtifactProgressCallback,
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


def normalized_model_path(config: FasterWhisperConfig) -> Path:
    """Return the operator ``model_path`` in its one canonical absolute form.

    Status inspection and the loader MUST agree on this form: expanding only
    on the status side would report a tilde path as ready while the loader
    hands the raw string to the Hub as a repo id.

    Args:
        config: Resolved engine configuration with ``model_path`` set.

    Returns:
        The expanded, resolved path.
    """
    assert config.model_path is not None
    return Path(config.model_path).expanduser().resolve()


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
    config: FasterWhisperConfig, model_size: str
) -> tuple[str, Path | None]:
    """Resolve the preset's snapshot from local caches only, then verify it.

    Resolution reuses the loader's own rule (upstream ``download_model`` with
    ``local_files_only=True``); completeness is this function's own check,
    because upstream returns an existing snapshot directory without verifying
    its contents.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.

    Returns:
        A ``(state, path)`` pair: ``ready`` with the snapshot, ``incomplete``
        with the partial snapshot, ``missing`` or ``unknown`` with ``None``.
    """
    try:
        from faster_whisper import download_model  # pyright: ignore[reportMissingImports]
        from huggingface_hub.errors import (  # pyright: ignore[reportMissingModuleSource]
            LocalEntryNotFoundError,
        )
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
    except LocalEntryNotFoundError:
        # The documented not-in-cache outcome of an offline resolution: this
        # is the one failure that is reliable evidence of a missing snapshot.
        return ARTIFACT_MISSING, None
    except Exception:
        # Anything else (an unreadable cache, a permission failure, a stalled
        # network mount) is not evidence of absence; unknown never means
        # ready, and it never claims missing either.
        return ARTIFACT_UNKNOWN, None
    # A relative download_root yields a relative snapshot path; the report's
    # location field requires (and callers deserve) the absolute form.
    snapshot = Path(resolved).expanduser().resolve()
    if not (snapshot / _CT2_MODEL_FILE).is_file():
        # The snapshot directory resolved but the recognizer file is not in
        # it: a detectable interrupted acquisition.
        return ARTIFACT_INCOMPLETE, snapshot
    return ARTIFACT_READY, snapshot


def _local_path_requirement(config: FasterWhisperConfig) -> ArtifactRequirement:
    """Build the externally provided requirement for a ``model_path`` config.

    Args:
        config: Resolved engine configuration with ``model_path`` set.

    Returns:
        The single logical requirement for the operator-provided directory.
    """
    path = normalized_model_path(config)
    if not path.exists():
        state = ARTIFACT_MISSING
        message = (
            f"Provide a converted CTranslate2 model directory at {path} "
            "(the configured model_path), or unset model_path to use the "
            "preset's Hub model."
        )
    elif (path / _CT2_MODEL_FILE).is_file():
        state = ARTIFACT_READY
        message = None
    else:
        # The path exists but lacks the recognizer file; unknown never means
        # ready, and the operator still gets a concrete next step.
        state = ARTIFACT_UNKNOWN
        message = (
            f"The configured model_path {path} exists but contains no "
            f"{_CT2_MODEL_FILE}; point it at the converted CTranslate2 "
            "directory itself."
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


def _hub_requirement(config: FasterWhisperConfig, model_size: str) -> ArtifactRequirement:
    """Build the Hub-preset requirement from a cache-only resolution.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.

    Returns:
        The single logical requirement for the preset's recognizer bundle.
    """
    downloads_blocked = config.local_files_only or not allow_downloads()
    state, snapshot = _resolve_cached_snapshot(config, model_size)

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


def status_requirement(config: FasterWhisperConfig, model_size: str) -> ArtifactRequirement:
    """Report the engine's one logical requirement for the resolved config.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.

    Returns:
        The requirement for either the operator path or the Hub preset.
    """
    if config.model_path is not None:
        return _local_path_requirement(config)
    return _hub_requirement(config, model_size)


def raise_for_gated_source(exc: BaseException, model_size: str) -> None:
    """Translate a gated or unauthenticated Hub rejection into actions.

    Shared by the explicit acquisition hook and the implicit first-use path:
    the reason comes from the discovered blocker, not from which code path
    noticed it.

    Args:
        exc: The native failure raised by the Hub client.
        model_size: The preset's upstream weights id, for the message.

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
) -> None:
    """Materialize the Hub preset with the upstream artifact-only downloader.

    The upstream helper suppresses transfer progress and re-resolves a mutable
    revision on every online call, so one code path serves both plain
    acquisition and an explicit refresh (an incomplete snapshot is completed
    the same way: upstream fetches only the files that are absent). Loading
    stays out: this never constructs a ``WhisperModel``.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.
        progress: Optional serialized progress observer.

    Returns:
        None.

    Raises:
        ArtifactAcquisitionError: With ``reason="downloads_disabled"`` when the
            engine's own offline config (or the global toggle) forbids the
            transfer -- the core template only sees the global toggle, so the
            engine flag must be enforced here; with ``reason="action_required"``
            for a gated or unauthenticated repository; every other native
            failure propagates for the template to wrap as ``reason="failed"``.
    """
    if config.local_files_only or not allow_downloads():
        raise ArtifactAcquisitionError(
            "Acquisition needs a network transfer, and downloads are disabled "
            "for this engine.",
            reason="downloads_disabled",
            hint=(
                "Unset local_files_only and set STANDARD_ASR_ALLOW_DOWNLOAD=1 "
                "to permit the transfer."
            ),
        )
    from faster_whisper import download_model  # pyright: ignore[reportMissingImports]

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
        download_model(
            model_size,
            local_files_only=False,
            cache_dir=None if download_root is None else str(download_root),
            revision=config.revision,
            use_auth_token=token,
        )
    except Exception as exc:
        raise_for_gated_source(exc, model_size)
        raise
