# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point, discovery, and compliance-suite integration tests.

These confirm the plugin is discoverable and protocol-compliant the way an
application (and our own CI) sees it -- they exercise the *installed* entry
points via ``standard_asr.discover_models`` and ``standard_asr.compliance``.
"""

from __future__ import annotations

from standard_asr import discover_models
from standard_asr.audio_format import AudioFormat
from standard_asr.compliance import check_entrypoints, check_sync_bridge

from std_faster_whisper import (
    BaseASR,
    DistilLargeV3ASR,
    LargeV3ASR,
    MediumASR,
    SmallASR,
    TinyASR,
    TurboASR,
    create_base,
    create_distil_large_v3,
    create_large_v3,
    create_medium,
    create_small,
    create_tiny,
    create_turbo,
)

_EXPECTED_KEYS = {
    "faster-whisper/tiny",
    "faster-whisper/base",
    "faster-whisper/small",
    "faster-whisper/medium",
    "faster-whisper/large-v3",
    "faster-whisper/distil-large-v3",
    "faster-whisper/large-v3-turbo",
}


def test_all_presets_discovered() -> None:
    registry = discover_models()
    names = set(registry.names())
    assert names >= _EXPECTED_KEYS


def test_by_engine_lists_presets() -> None:
    registry = discover_models()
    # ModelRegistry.keys_by_engine returns the entry-point KEYS for the engine.
    assert set(registry.keys_by_engine("faster-whisper")) == _EXPECTED_KEYS


def test_registry_resolves_engine_class_without_instantiation() -> None:
    # The factory return annotation must resolve to the concrete class so the
    # registry can read class-level metadata without instantiating.
    registry = discover_models()
    engine_class = registry.engine_class("faster-whisper/tiny")
    assert engine_class is TinyASR
    assert engine_class.properties.model_id == "faster-whisper/tiny"
    # The spec exposes the parsed key components.
    spec = registry.spec("faster-whisper/tiny")
    assert spec.model_id == "faster-whisper/tiny"
    assert spec.engine_id == "faster-whisper"
    assert spec.model_name == "tiny"


def test_create_via_registry() -> None:
    engine = discover_models().create("faster-whisper/tiny")
    assert isinstance(engine, TinyASR)


def test_check_entrypoints_passes() -> None:
    report = check_entrypoints()
    assert report.passed, [f"{i.level} {i.model} {i.message}" for i in report.issues]


def test_preset_factories_return_their_classes() -> None:
    assert type(create_tiny()) is TinyASR
    assert type(create_base()) is BaseASR
    assert type(create_small()) is SmallASR
    assert type(create_medium()) is MediumASR
    assert type(create_large_v3()) is LargeV3ASR
    assert type(create_distil_large_v3()) is DistilLargeV3ASR
    assert type(create_turbo()) is TurboASR


def test_sync_bridge_no_deadlock(fake_faster_whisper: object) -> None:
    # The standard sync->async bridge must terminate without deadlock or a leaked
    # thread for our streaming session (plugin_entrypoints.md compliance surface).
    engine = TinyASR()
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    report = check_sync_bridge(lambda: engine.start_transcription(audio_format=fmt))
    assert report.passed, [i.message for i in report.issues]
