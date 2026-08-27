# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Inference-artifact status and acquisition for the faster-whisper engine.

The engine's one logical requirement is the CTranslate2 recognizer bundle.
Status inspection reuses the upstream cache resolution
(``download_model(..., local_files_only=True)``) so readiness is judged by the
exact rule the loader applies, never by a parallel re-implementation of the
Hugging Face cache layout. An explicit ``model_path`` is an externally
provided requirement that the engine can inspect but never acquire.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
    ARTIFACT_ACTION_REQUEST_ACCESS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_BLOCKER_UNSUPPORTED,
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


def _resolve_cached_snapshot(config: FasterWhisperConfig, model_size: str) -> Path | None:
    """Resolve the preset's snapshot from local caches only.

    This is the loader's own resolution rule (upstream ``download_model`` with
    ``local_files_only=True``), so a non-``None`` result is exactly the
    directory ``WhisperModel`` would load without a network transfer.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.

    Returns:
        The cached snapshot directory, or ``None`` when no complete local
        snapshot exists.

    Raises:
        Exception: Only when the upstream helper itself is unavailable; the
            caller maps that to an ``unknown`` state.
    """
    from faster_whisper import download_model  # pyright: ignore[reportMissingImports]

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
    except Exception:
        # local_files_only never touches the network, so a failure is reliable
        # evidence that no complete local snapshot exists for this resolution
        # (a partial download also fails here and is reported as missing).
        return None
    return Path(resolved)


def _local_path_requirement(config: FasterWhisperConfig) -> ArtifactRequirement:
    """Build the externally provided requirement for a ``model_path`` config.

    Args:
        config: Resolved engine configuration with ``model_path`` set.

    Returns:
        The single logical requirement for the operator-provided directory.
    """
    assert config.model_path is not None
    path = Path(config.model_path).expanduser().resolve()
    if not path.exists():
        state = ARTIFACT_MISSING
    elif (path / _CT2_MODEL_FILE).is_file():
        state = ARTIFACT_READY
    else:
        # The path exists but the CTranslate2 bundle shape cannot be cheaply
        # confirmed; unknown never means ready.
        state = ARTIFACT_UNKNOWN

    if state == ARTIFACT_READY:
        blocker = None
        actions: tuple[ArtifactAction, ...] = ()
    elif state == ARTIFACT_MISSING:
        blocker = ARTIFACT_BLOCKER_ACTION_REQUIRED
        actions = (
            ArtifactAction(
                kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
                message=(
                    f"Provide a converted CTranslate2 model directory at {path} "
                    "(the configured model_path), or unset model_path to use "
                    "the preset's Hub model."
                ),
            ),
        )
    else:
        blocker = ARTIFACT_BLOCKER_UNSUPPORTED
        actions = ()

    return ArtifactRequirement(
        artifact_id=LOCAL_ARTIFACT_ID,
        label="Operator-provided CTranslate2 model directory",
        state=state,
        required_for_inference=True,
        can_acquire_now=False,
        may_acquire_during_inference=False,
        source_is_mutable=False,
        acquisition_blocker=blocker,
        required_actions=actions,
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
    try:
        snapshot = _resolve_cached_snapshot(config, model_size)
    except Exception:
        # The upstream helper itself failed to import or run; the state cannot
        # be established through a cheap inspection.
        snapshot = None
        state = ARTIFACT_UNKNOWN
    else:
        state = ARTIFACT_READY if snapshot is not None else ARTIFACT_MISSING

    artifact_version = config.revision
    if snapshot is not None and artifact_version is None and snapshot.parent.name == "snapshots":
        # The Hugging Face cache stores one directory per resolved commit.
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


def acquire(
    config: FasterWhisperConfig,
    model_size: str,
    progress: ArtifactProgressCallback | None,
) -> None:
    """Materialize the Hub preset with the upstream artifact-only downloader.

    The upstream helper suppresses transfer progress and re-resolves a mutable
    revision on every online call, so one code path serves both plain
    acquisition and an explicit refresh. Loading stays out: this never
    constructs a ``WhisperModel``.

    Args:
        config: Resolved engine configuration.
        model_size: The preset's upstream weights id.
        progress: Optional serialized progress observer.

    Returns:
        None.

    Raises:
        ArtifactAcquisitionError: With ``reason="action_required"`` when the
            source reports a gated repository; every other native failure
            propagates for the template to wrap as ``reason="failed"``.
    """
    from faster_whisper import download_model  # pyright: ignore[reportMissingImports]

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
        if type(exc).__name__ == "GatedRepoError":
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
        raise
