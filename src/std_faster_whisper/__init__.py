# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR engine plugin for faster-whisper.

A thin, typed adapter over the upstream ``faster-whisper`` package that makes
every Whisper preset usable by any Standard ASR-compliant application. Batch
transcription maps directly onto ``WhisperModel.transcribe``; streaming is a
windowed re-decode session honestly declared with conservative stability
semantics (see :mod:`std_faster_whisper._streaming`).

Public surface:

* Engine classes -- one per preset (:class:`FasterWhisperASR` is ``large-v3``).
* Config / params models -- :class:`FasterWhisperConfig`,
  :class:`FasterWhisperParams`.
* Properties -- :class:`FasterWhisperProperties` and per-preset subclasses.
* Entry-point factories -- ``create_tiny`` ... ``create_turbo``.
"""

from __future__ import annotations

from ._config import FasterWhisperConfig, FasterWhisperParams
from ._metadata import (
    DECLARED_CAPABILITIES,
    BaseModelProperties,
    DistilLargeV3Properties,
    FasterWhisperProperties,
    LargeV3Properties,
    MediumProperties,
    SmallProperties,
    TinyProperties,
    TurboProperties,
)
from ._streaming import FasterWhisperStreamingSession
from .engine import (
    BaseASR,
    DistilLargeV3ASR,
    FasterWhisperASR,
    LargeV3ASR,
    MediumASR,
    SmallASR,
    TinyASR,
    TurboASR,
)
from .entrypoint import (
    create_base,
    create_distil_large_v3,
    create_large_v3,
    create_medium,
    create_small,
    create_tiny,
    create_turbo,
)

__all__ = [
    "DECLARED_CAPABILITIES",
    "BaseASR",
    "BaseModelProperties",
    "DistilLargeV3ASR",
    "DistilLargeV3Properties",
    "FasterWhisperASR",
    "FasterWhisperConfig",
    "FasterWhisperParams",
    "FasterWhisperProperties",
    "FasterWhisperStreamingSession",
    "LargeV3ASR",
    "LargeV3Properties",
    "MediumASR",
    "MediumProperties",
    "SmallASR",
    "SmallProperties",
    "TinyASR",
    "TinyProperties",
    "TurboASR",
    "TurboProperties",
    "create_base",
    "create_distil_large_v3",
    "create_large_v3",
    "create_medium",
    "create_small",
    "create_tiny",
    "create_turbo",
]
