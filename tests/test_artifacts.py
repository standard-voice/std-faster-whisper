# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Artifact-lifecycle status and acquisition (protocol 1.1).

Every test runs against the injected fakes; no weights are downloaded. The
matrix covers both configured shapes (Hub preset, operator ``model_path``),
the download-policy interactions, refresh, and the error translations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from standard_asr import (
    ARTIFACT_MISSING,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ARTIFACTS_READY,
    ARTIFACTS_UNAVAILABLE,
    ArtifactProgress,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
)
from standard_asr.engine import NO_ARTIFACT_LIFECYCLE

from std_faster_whisper import FasterWhisperASR, TinyASR

from .conftest import FakeHfApi, FakeHubCache, FakeWhisperModel

PINNED = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _downloads_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each test starts from the documented default (unset = allowed) and opts
    # into the disabled state explicitly.
    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)


def _snapshot_dir(tmp_path: Path, sha: str = PINNED) -> Path:
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"\x00" * 64)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "vocabulary.txt").write_text("<|a|>")
    return snapshot


def _ct2_dir(tmp_path: Path) -> Path:
    root = tmp_path / "local-ct2"
    root.mkdir()
    (root / "model.bin").write_bytes(b"\x00" * 16)
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    (root / "vocabulary.json").write_text("[]")
    return root


# --------------------------------------------------------------------------- #
# Static declaration
# --------------------------------------------------------------------------- #
def test_declared_metadata_upper_bounds() -> None:
    artifacts = FasterWhisperASR.declared_metadata.artifacts
    assert artifacts.applicable is True
    assert artifacts.supports_explicit_acquisition is True
    assert artifacts.may_acquire_during_inference is True
    assert artifacts != NO_ARTIFACT_LIFECYCLE


def test_protocol_version_is_1_1() -> None:
    assert FasterWhisperASR.properties.protocol_version == "1.1.0"


# --------------------------------------------------------------------------- #
# Status: Hub preset
# --------------------------------------------------------------------------- #
def test_hub_cold_cache_reports_missing_and_acquirable(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    report = TinyASR().artifact_status()
    assert report.applicable is True
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_MISSING
    assert requirement.required_for_inference is True
    assert requirement.can_acquire_now is True
    assert requirement.acquisition_blocker is None
    assert requirement.may_acquire_during_inference is True
    assert requirement.source_is_mutable is True


def test_hub_cold_cache_with_downloads_disabled(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    (requirement,) = TinyASR().artifact_status().requirements
    assert requirement.state == ARTIFACT_MISSING
    assert requirement.can_acquire_now is False
    assert requirement.acquisition_blocker == "downloads_disabled"
    assert requirement.may_acquire_during_inference is False


def test_hub_cold_cache_with_local_files_only(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    (requirement,) = TinyASR(local_files_only=True).artifact_status().requirements
    assert requirement.acquisition_blocker == "downloads_disabled"
    assert requirement.may_acquire_during_inference is False


def test_hub_ready_cache_reports_location_size_and_commit(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    snapshot = _snapshot_dir(tmp_path)
    FakeHubCache.resolved_path = str(snapshot)
    report = TinyASR().artifact_status()
    assert report.readiness == ARTIFACTS_READY
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.can_acquire_now is False
    assert requirement.acquisition_blocker is None
    assert requirement.location == snapshot
    assert requirement.size_bytes == 73
    # The commit is read off the Hugging Face snapshots/<sha> layout.
    assert requirement.artifact_version == PINNED


def test_pinned_revision_is_immutable_and_reported(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    (requirement,) = TinyASR(revision=PINNED).artifact_status().requirements
    assert requirement.source_is_mutable is False
    assert requirement.artifact_version == PINNED


def test_branch_revision_stays_mutable(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    (requirement,) = TinyASR(revision="main").artifact_status().requirements
    assert requirement.source_is_mutable is True
    assert requirement.artifact_version == "main"


def test_hub_state_unknown_when_helper_is_unavailable(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.delattr(sys.modules["faster_whisper"], "download_model")
    (requirement,) = TinyASR().artifact_status().requirements
    assert requirement.state == ARTIFACT_UNKNOWN


# --------------------------------------------------------------------------- #
# Status: operator model_path
# --------------------------------------------------------------------------- #
def test_model_path_missing_needs_provide_artifacts(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    report = TinyASR(model_path=str(tmp_path / "absent")).artifact_status()
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_MISSING
    assert requirement.can_acquire_now is False
    assert requirement.may_acquire_during_inference is False
    assert requirement.source_is_mutable is False
    assert requirement.acquisition_blocker == "action_required"
    (action,) = requirement.required_actions
    assert action.kind == "provide_artifacts"


def test_model_path_ready_directory(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    local = _ct2_dir(tmp_path)
    (requirement,) = TinyASR(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.location == local
    assert requirement.size_bytes == 22


def test_model_path_without_model_file_is_incomplete_with_guidance(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The most common operator mistake (pointing one level too high) is
    # provably not a complete bundle -- the check already ran and answered,
    # so the state is incomplete (not unknown), with a concrete next step.
    (requirement,) = TinyASR(model_path=str(tmp_path)).artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.acquisition_blocker == "action_required"
    (action,) = requirement.required_actions
    assert action.kind == "provide_artifacts"
    assert "model.bin" in action.message


def test_model_path_pointing_at_a_file_is_incomplete(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # A file path would make the upstream loader treat the absolute path as a
    # Hub repo id; status catches it with directions to the directory.
    target = tmp_path / "model.bin"
    target.write_bytes(b"\x00")
    (requirement,) = TinyASR(model_path=str(target)).artifact_status().requirements
    assert requirement.state == "incomplete"
    (action,) = requirement.required_actions
    assert "DIRECTORY" in action.message


def test_model_path_missing_tokenizer_is_incomplete(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # Without tokenizer.json upstream silently fetches it from the Hub past
    # every download gate, so the bundle is not offline-ready.
    root = tmp_path / "converted"
    root.mkdir()
    (root / "model.bin").write_bytes(b"\x00")
    (requirement,) = TinyASR(model_path=str(root)).artifact_status().requirements
    assert requirement.state == "incomplete"


def test_prepare_incomplete_model_path_is_unavailable(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The guard translates the already-answered incompleteness instead of
    # letting the loader turn it into an opaque engine fault.
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        TinyASR(model_path=str(tmp_path)).prepare()
    assert exc_info.value.reason == "incomplete"
    assert exc_info.value.report.readiness == ARTIFACTS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# Explicit acquisition and refresh
# --------------------------------------------------------------------------- #
def test_pull_acquires_and_reports_ready(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    FakeHubCache.download_target = str(_snapshot_dir(tmp_path))
    phases: list[str] = []
    report = TinyASR(hf_token="hf_abc", download_root=str(tmp_path)).acquire_artifacts(
        progress=lambda event: phases.append(event.phase)
    )
    assert report.readiness == ARTIFACTS_READY
    assert FakeHubCache.download_calls == 1
    assert FakeHubCache.last_download_kwargs["size_or_id"] == "tiny"
    assert FakeHubCache.last_download_kwargs["local_files_only"] is False
    assert FakeHubCache.last_download_kwargs["cache_dir"] == str(tmp_path)
    # The secret materializes to plaintext only at the SDK call site.
    assert FakeHubCache.last_download_kwargs["use_auth_token"] == "hf_abc"
    assert phases[0] == "resolving"
    assert "transferring" in phases
    assert phases[-1] == "finalizing"


def _gated_error() -> Exception:
    import httpx
    from huggingface_hub.errors import GatedRepoError

    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    return GatedRepoError("gated", response=response)


def _unauthorized_error() -> Exception:
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x"))
    return HfHubHTTPError("401 unauthorized", response=response)


def test_pull_gated_repo_reports_request_access(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # The REAL upstream error class: name-based matching would silently stop
    # working on a rename or subclass.
    FakeHubCache.raise_on_download = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts()
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "request_access"


def test_pull_unauthenticated_repo_reports_authenticate(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    FakeHubCache.raise_on_download = _unauthorized_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts()
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "authenticate"


def test_pull_native_failure_is_failed_with_cause(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    FakeHubCache.raise_on_download = RuntimeError("dns exploded")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts()
    assert exc_info.value.reason == "failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_pull_blocked_by_download_policy(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts()
    assert exc_info.value.reason == "downloads_disabled"
    assert FakeHubCache.download_calls == 0


def test_refresh_re_resolves_a_ready_mutable_source(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    snapshot = _snapshot_dir(tmp_path)
    FakeHubCache.resolved_path = str(snapshot)
    FakeHfApi.remote_sha = PINNED
    engine = TinyASR(hf_token="hf_abc")
    assert engine.acquire_artifacts().readiness == ARTIFACTS_READY
    assert FakeHubCache.download_calls == 0  # plain pull: ready is a no-op
    assert FakeHfApi.model_info_calls == []  # plain pull never queries the source
    engine.acquire_artifacts(refresh=True)
    assert FakeHubCache.download_calls == 1  # refresh re-resolves the branch
    # The re-resolution evidence: one metadata query against the upstream
    # size table's repo, with the engine's credentials.
    (call,) = FakeHfApi.model_info_calls
    assert call == {"repo_id": "Systran/faster-whisper-tiny", "revision": None}
    assert FakeHfApi.last_token == "hf_abc"


def test_refresh_skips_a_pinned_revision(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    report = TinyASR(revision=PINNED).acquire_artifacts(refresh=True)
    assert report.readiness == ARTIFACTS_READY
    assert FakeHubCache.download_calls == 0
    assert FakeHfApi.model_info_calls == []  # a pinned commit needs no query


def test_refresh_with_downloads_disabled_raises(
    fake_faster_whisper: type[FakeWhisperModel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "downloads_disabled"


def test_pull_on_missing_model_path_raises_action_required(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    engine = TinyASR(model_path=str(tmp_path / "absent"))
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        engine.acquire_artifacts()
    assert exc_info.value.reason == "action_required"


def test_pull_on_ready_model_path_is_a_no_op(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    engine = TinyASR(model_path=str(_ct2_dir(tmp_path)))
    report = engine.acquire_artifacts()
    assert report.readiness == ARTIFACTS_READY
    assert FakeHubCache.download_calls == 0


# --------------------------------------------------------------------------- #
# Implicit-path translation (the loading guard)
# --------------------------------------------------------------------------- #
def test_prepare_cold_cache_downloads_disabled_is_unavailable(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        TinyASR().prepare()
    assert exc_info.value.reason == "downloads_disabled"
    assert exc_info.value.report.readiness == ARTIFACTS_UNAVAILABLE


def test_prepare_missing_model_path_is_unavailable(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        TinyASR(model_path=str(tmp_path / "absent")).prepare()
    assert exc_info.value.reason == "missing"


def test_progress_events_are_validated_models() -> None:
    event = ArtifactProgress(phase="transferring", artifact_id="ct2-recognizer")
    assert event.completed_units is None and event.unit is None


def test_tree_size_returns_none_on_walk_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from std_faster_whisper import _artifacts

    def _raise(self: Path, pattern: str) -> object:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "rglob", _raise)
    assert _artifacts._tree_size_bytes(tmp_path) is None


def test_acquire_hook_ignores_a_target_set_without_the_hub_id(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # Defensive guard: the template never hands the hook a foreign target set,
    # but the hook must stay a no-op if it ever did.
    from standard_asr.contract.artifacts import ArtifactContext

    TinyASR()._acquire_artifacts(ArtifactContext(mode="batch"), (), False, None)
    assert FakeHubCache.download_calls == 0


# --------------------------------------------------------------------------- #
# Round-1 review regressions: completeness, policy, path canonicalization
# --------------------------------------------------------------------------- #
def test_partial_snapshot_reports_incomplete_and_pull_repairs(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # Upstream resolution returns an existing snapshot directory WITHOUT
    # verifying its contents, so an interrupted download resolves; readiness
    # must come from the completeness check, and pull must be able to repair.
    partial = tmp_path / "snapshots" / PINNED
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")  # model.bin never arrived
    FakeHubCache.resolved_path = str(partial)
    report = TinyASR().artifact_status()
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is True

    def _complete_download() -> None:
        (partial / "model.bin").write_bytes(b"\x00" * 64)
        (partial / "tokenizer.json").write_text("{}")
        (partial / "vocabulary.txt").write_text("<|a|>")

    FakeHubCache.on_download = _complete_download
    repaired = TinyASR().acquire_artifacts()
    assert repaired.readiness == ARTIFACTS_READY
    assert FakeHubCache.download_calls == 1


def test_empty_snapshot_reports_incomplete(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    empty = tmp_path / "snapshots" / PINNED
    empty.mkdir(parents=True)
    FakeHubCache.resolved_path = str(empty)
    (requirement,) = TinyASR().artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.size_bytes == 0


def test_refresh_respects_engine_local_files_only(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # local_files_only is the engine's own offline policy; the core template
    # only sees the global toggle, so the hook must self-gate the refresh.
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR(local_files_only=True).acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "downloads_disabled"
    assert FakeHubCache.download_calls == 0
    assert FakeHfApi.model_info_calls == []  # gated before the source query


def test_relative_download_root_yields_an_absolute_location(
    fake_faster_whisper: type[FakeWhisperModel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A relative download_root passes through resolution unchanged upstream;
    # the report's location must still be absolute (the model validator
    # rejects a relative path, which previously escaped as a raw
    # ValidationError).
    monkeypatch.chdir(tmp_path)
    snapshot = _snapshot_dir(tmp_path)
    FakeHubCache.resolved_path = str(snapshot.relative_to(tmp_path))
    (requirement,) = TinyASR(download_root="models").artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.location is not None and requirement.location.is_absolute()


def test_unreadable_cache_reports_unknown_not_missing(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # A permission failure is not evidence of absence: claiming missing would
    # tell an offline operator to enable downloads for a permissions bug.
    FakeHubCache.raise_on_resolve = PermissionError("refs unreadable")
    (requirement,) = TinyASR().artifact_status().requirements
    assert requirement.state == ARTIFACT_UNKNOWN
    assert requirement.acquisition_blocker is None  # downloads allowed: still runnable


def test_warm_cache_with_branch_revision_reports_resolved_commit(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # Two engines resolving the same snapshot must report the same version:
    # the resolved commit wins over the configured mutable reference, so a
    # refresh that moves the branch is observable through this field.
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    (requirement,) = TinyASR(revision="main").artifact_status().requirements
    assert requirement.source_is_mutable is True
    assert requirement.artifact_version == PINNED


def test_tilde_model_path_reaches_the_loader_expanded(
    fake_faster_whisper: type[FakeWhisperModel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Status and the loader must agree on the canonical path form: expanding
    # only on the status side would hand '~/x' to the Hub as a repo id.
    monkeypatch.setenv("HOME", str(tmp_path))
    local = tmp_path / "ct2"
    local.mkdir()
    (local / "model.bin").write_bytes(b"\x00")
    (local / "config.json").write_text("{}")
    (local / "tokenizer.json").write_text("{}")
    (local / "vocabulary.json").write_text("[]")
    engine = TinyASR(model_path="~/ct2")
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    engine.prepare()
    assert fake_faster_whisper.last_init_kwargs["model_size_or_path"] == str(local)


def test_transcribe_propagates_artifact_error_unwrapped(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The R7 exemption end to end: an availability failure inside the batch
    # pipeline reaches the caller as the artifact error, never wrapped into
    # TranscriptionError.
    import numpy as np

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactUnavailableError):
        TinyASR().transcribe((np.zeros(16000, dtype=np.float32), 16000))


def test_streaming_session_emits_artifact_unavailable_terminal(
    fake_faster_whisper: type[FakeWhisperModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The streaming mapping end to end: the session's producer translates the
    # availability failure into the dedicated terminal code, not engine_error.
    from standard_asr import SyncSession
    from standard_asr.audio.format import AudioFormat

    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = TinyASR()
    session = engine.start_transcription(
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    )
    with SyncSession(session) as sync:
        events = list(sync)
    (error,) = [event for event in events if event.type == "error"]
    assert error.code == "artifact_unavailable"
    assert error.recoverable is False


# --------------------------------------------------------------------------- #
# Round-3 review regressions: the load closure and refresh evidence
# --------------------------------------------------------------------------- #
def test_snapshot_missing_config_is_incomplete(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # CTranslate2 loads without config.json, but every decode reads its
    # fields at request time and fails -- a bundle lacking it cannot serve a
    # single request, so ready would be a lie.
    snapshot = _snapshot_dir(tmp_path)
    (snapshot / "config.json").unlink()
    FakeHubCache.resolved_path = str(snapshot)
    (requirement,) = TinyASR().artifact_status().requirements
    assert requirement.state == "incomplete"


def test_snapshot_missing_vocabulary_is_incomplete(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # CTranslate2 refuses to load a directory without a vocabulary file.
    snapshot = _snapshot_dir(tmp_path)
    (snapshot / "vocabulary.txt").unlink()
    FakeHubCache.resolved_path = str(snapshot)
    (requirement,) = TinyASR().artifact_status().requirements
    assert requirement.state == "incomplete"


def test_128_mel_preset_requires_preprocessor_config(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The 128-mel repos ship preprocessor_config.json and need it (without
    # the file upstream silently builds an 80-mel feature extractor); the
    # 80-mel repos do not carry it at all, so the requirement is per preset.
    snapshot = _snapshot_dir(tmp_path)
    FakeHubCache.resolved_path = str(snapshot)
    (tiny_requirement,) = TinyASR().artifact_status().requirements
    assert tiny_requirement.state == ARTIFACT_READY
    (large_requirement,) = FasterWhisperASR().artifact_status().requirements
    assert large_requirement.state == "incomplete"
    (snapshot / "preprocessor_config.json").write_text("{}")
    (large_requirement,) = FasterWhisperASR().artifact_status().requirements
    assert large_requirement.state == ARTIFACT_READY


def test_model_path_ignores_preset_bundle_files(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # model_path replaces the preset's Hub bundle with an operator conversion
    # of an arbitrary Whisper variant; only the loader-universal closure is
    # checkable there.
    local = _ct2_dir(tmp_path)
    (requirement,) = FasterWhisperASR(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_hub_repo_id_mirrors_the_upstream_rule() -> None:
    # A slashed id passes through untouched; a size resolves through the
    # genuine upstream table (registered by the fixture before the fake).
    from std_faster_whisper._artifacts import _hub_repo_id

    assert _hub_repo_id("someorg/custom-ct2") == "someorg/custom-ct2"
    assert _hub_repo_id("tiny") == "Systran/faster-whisper-tiny"


def test_implicit_load_verifies_the_acquisition_postcondition(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The loader doubles as the implicit acquisition, and its internal
    # downloader silently falls back to the local cache when the remote is
    # unreachable -- so a load can succeed on the same incomplete bundle
    # status just reported. The model must not be adopted when the bundle
    # stayed incomplete.
    partial = _snapshot_dir(tmp_path)  # complete for tiny, not for large-v3
    FakeHubCache.resolved_path = str(partial)
    engine = FasterWhisperASR()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        engine.prepare()
    assert exc_info.value.reason == "failed"
    assert "without completing" in str(exc_info.value)
    assert engine._model is None


def test_implicit_load_adopts_the_model_once_the_bundle_completes(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The happy path of the same guard: the in-loader acquisition genuinely
    # materializes the missing file, the postcondition holds, and the loaded
    # model is adopted.
    partial = _snapshot_dir(tmp_path)
    FakeHubCache.resolved_path = str(partial)
    fake_faster_whisper.on_init = lambda: (partial / "preprocessor_config.json").write_text("{}")
    engine = FasterWhisperASR()
    engine.prepare()
    assert engine._model is not None


def test_load_rejects_a_mel_mismatched_model_path_bundle(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The mel closure is only provable post-load (nothing on disk names the
    # model's mel count): a 128-mel conversion without
    # preprocessor_config.json passes the file closure, loads, and upstream
    # silently builds an 80-mel feature extractor whose output CTranslate2
    # rejects at the first request with a bare shape error (verified
    # empirically). The load-time check turns that into an actionable
    # artifact error, for ANY Whisper variant behind model_path.
    fake_faster_whisper.n_mels = 128
    engine = TinyASR(model_path=str(_ct2_dir(tmp_path)))
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        engine.prepare()
    assert exc_info.value.reason == "incomplete"
    assert "128" in str(exc_info.value)
    assert exc_info.value.hint is not None and "--copy_files" in exc_info.value.hint
    assert engine._model is None
    # The attached report must ESTABLISH the unavailable state (the error
    # contract) -- not repeat the pre-load ready observation.
    report = exc_info.value.report
    assert report is not None and report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == "incomplete"
    (action,) = requirement.required_actions
    assert action.kind == "provide_artifacts" and "128" in action.message


def test_load_rejects_a_mel_mismatched_hub_snapshot(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The same post-load authority backstops the Hub path (a present but
    # wrong preprocessor_config.json passes the file closure). The
    # requirement must NOT promise can_acquire_now: the public acquisition
    # path re-inspects the disk closure (which looks ready, so a plain pull
    # would not run), and the Hub cache never refetches a present file of
    # the resolved commit (so even a run would not repair) -- the executable
    # repair is the action's removal instruction.
    fake_faster_whisper.n_mels = 128
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    engine = TinyASR()
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        engine.prepare()
    assert exc_info.value.reason == "incomplete"
    assert exc_info.value.hint is not None and "Remove" in exc_info.value.hint
    assert engine._model is None
    report = exc_info.value.report
    assert report is not None and report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is False
    assert requirement.acquisition_blocker == "action_required"
    (action,) = requirement.required_actions
    assert action.kind == "other"
    assert "128" in action.message
    # The removal target is the repository CACHE ROOT, not the snapshot
    # pointer: the downloader rebuilds a deleted pointer from the blob store
    # without a transfer, so naming the snapshot folder would send the
    # operator through a no-op.
    assert str(tmp_path) in action.message and "blob store" in action.message


def test_mel_mismatch_report_names_the_snapshot_the_loader_used(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The loader's online resolution can move a mutable revision past the
    # preflight's cache-resolved commit; the report and action must then
    # name the snapshot the mismatch actually came from, not the stale one.
    old_sha, new_sha = "a" * 40, "b" * 40
    old = _snapshot_dir(tmp_path, sha=old_sha)
    fake_faster_whisper.n_mels = 128
    FakeHubCache.resolved_path = str(old)

    def _branch_moved_during_load() -> None:
        FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path, sha=new_sha))

    fake_faster_whisper.on_init = _branch_moved_during_load
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        TinyASR().prepare()
    assert exc_info.value.report is not None
    (requirement,) = exc_info.value.report.requirements
    assert requirement.location == tmp_path / "snapshots" / new_sha
    assert requirement.artifact_version == new_sha


def test_mel_mismatch_action_without_a_cache_layout_stays_generic() -> None:
    # A base without a snapshots/<commit> location (an unknown-state
    # observation) yields the same action minus the concrete path.
    from standard_asr.contract.artifacts import ArtifactRequirement

    from std_faster_whisper._artifacts import mel_mismatch_requirement
    from std_faster_whisper._config import FasterWhisperConfig

    base = ArtifactRequirement(
        artifact_id="ct2-recognizer",
        label="faster-whisper tiny (CTranslate2)",
        state="unknown",
        required_for_inference=True,
        can_acquire_now=True,
        may_acquire_during_inference=True,
        source_is_mutable=True,
    )
    requirement = mel_mismatch_requirement(
        FasterWhisperConfig.from_env("faster-whisper"),
        base,
        model_n_mels=128,
        extractor_n_mels=80,
    )
    (action,) = requirement.required_actions
    assert "repository cache directory and" in action.message  # no path inserted


def test_mel_mismatch_action_is_executable_end_to_end(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The repair instruction must be real: a plain pull right after the
    # mismatch is an honest no-op (status is a point-in-time DISK
    # observation, spec AR.9, and the disk closure looks ready), while
    # following the action -- removing the snapshot -- makes the same public
    # pull genuinely re-acquire.
    import shutil

    fake_faster_whisper.n_mels = 128
    snapshot = _snapshot_dir(tmp_path)
    FakeHubCache.resolved_path = str(snapshot)
    engine = TinyASR()
    with pytest.raises(ArtifactUnavailableError):
        engine.prepare()

    report = engine.acquire_artifacts()  # plain pull: disk says ready, no-op
    assert report.readiness == ARTIFACTS_READY
    assert FakeHubCache.download_calls == 0

    def _redownload() -> None:
        # The real downloader materializes the (now content-correct) bundle.
        _snapshot_dir(tmp_path / "fresh")
        FakeHubCache.resolved_path = str(tmp_path / "fresh" / "snapshots" / PINNED)

    # The action's instruction: the repository cache ROOT, not the snapshot
    # pointer (which the blob store would rebuild without a transfer).
    shutil.rmtree(tmp_path)
    FakeHubCache.on_download = _redownload
    repaired = engine.acquire_artifacts()
    assert FakeHubCache.download_calls == 1
    assert repaired.readiness == ARTIFACTS_READY


def test_refresh_fails_when_the_source_is_unreachable(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The downloader silently falls back to the local cache when the remote
    # is unreachable, so refresh must fail without re-resolution evidence
    # instead of reporting the stale cache as fresh (spec AR.4).
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.raise_on_model_info = ConnectionError("network is down")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "failed"
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert FakeHubCache.download_calls == 0  # fail fast, before any transfer


def test_refresh_detects_a_stale_cache_fallback(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # The source re-resolved to a new commit, but the download resolved the
    # old snapshot (the mid-transfer fallback): success here would claim
    # freshness the source never confirmed.
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.remote_sha = "f" * 40
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "failed"
    assert "fell back" in str(exc_info.value)


def test_refresh_gated_source_query_reports_request_access(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # An access rejection during the metadata query carries the same
    # discovered action as one during the transfer.
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.raise_on_model_info = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "request_access"


def test_refresh_fails_when_the_source_names_no_commit(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    FakeHubCache.resolved_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.remote_sha = None
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "failed"
    assert "commit" in str(exc_info.value)


def test_plain_pull_repair_needs_no_source_query(
    fake_faster_whisper: type[FakeWhisperModel], tmp_path: Path
) -> None:
    # Plain acquisition keeps its documented semantics: materialize what is
    # absent, no freshness promise, so no metadata query -- the template's
    # final status check still guards the postcondition.
    partial = tmp_path / "snapshots" / PINNED
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")
    FakeHubCache.resolved_path = str(partial)

    def _complete_download() -> None:
        (partial / "model.bin").write_bytes(b"\x00" * 64)
        (partial / "tokenizer.json").write_text("{}")
        (partial / "vocabulary.txt").write_text("<|a|>")

    FakeHubCache.on_download = _complete_download
    report = TinyASR().acquire_artifacts()
    assert report.readiness == ARTIFACTS_READY
    assert FakeHfApi.model_info_calls == []


def test_first_use_gated_repo_reports_request_access(
    fake_faster_whisper: type[FakeWhisperModel],
) -> None:
    # The reason comes from the discovered blocker, not from which code path
    # noticed it: the IMPLICIT first-use load carries the same action as pull.
    fake_faster_whisper.raise_on_init = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        TinyASR().prepare()
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "request_access"


def test_gated_translation_degrades_without_hub_errors_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from std_faster_whisper._artifacts import raise_for_gated_source

    real_import = builtins.__import__

    def _import(name: str, *a: object, **k: object) -> object:
        if name.startswith("huggingface_hub"):
            raise ImportError("no huggingface_hub")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    assert raise_for_gated_source(RuntimeError("x"), "tiny") is None
