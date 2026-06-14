# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory functions for the std-faster-whisper plugin.

One factory per preset (spec IC.7: model selection = entry-point preset, never an
init ``model`` field), so ``standard-asr models list`` / the registry / a settings
UI can enumerate the available models. Each factory's return annotation is the
**concrete** preset class (NOT the ``StandardASR`` protocol) so the registry can
resolve the class -- and read its class-level ``properties`` /
``declared_capabilities`` / ``provider_params_type`` -- WITHOUT instantiating the
engine (``ModelRegistry.engine_class``; adapting_engine.md "Publish").
"""

from __future__ import annotations

from typing import Any

from .engine import (
    BaseASR,
    DistilLargeV3ASR,
    LargeV3ASR,
    MediumASR,
    SmallASR,
    TinyASR,
    TurboASR,
)


def create_tiny(**kwargs: Any) -> TinyASR:
    """Return the ``faster-whisper/tiny`` preset (smallest, fastest).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`TinyASR`.

    Returns:
        A configured tiny engine.
    """
    return TinyASR(**kwargs)


def create_base(**kwargs: Any) -> BaseASR:
    """Return the ``faster-whisper/base`` preset.

    Args:
        **kwargs: Keyword arguments forwarded to :class:`BaseASR`.

    Returns:
        A configured base engine.
    """
    return BaseASR(**kwargs)


def create_small(**kwargs: Any) -> SmallASR:
    """Return the ``faster-whisper/small`` preset.

    Args:
        **kwargs: Keyword arguments forwarded to :class:`SmallASR`.

    Returns:
        A configured small engine.
    """
    return SmallASR(**kwargs)


def create_medium(**kwargs: Any) -> MediumASR:
    """Return the ``faster-whisper/medium`` preset.

    Args:
        **kwargs: Keyword arguments forwarded to :class:`MediumASR`.

    Returns:
        A configured medium engine.
    """
    return MediumASR(**kwargs)


def create_large_v3(**kwargs: Any) -> LargeV3ASR:
    """Return the ``faster-whisper/large-v3`` preset (canonical multilingual).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`LargeV3ASR`.

    Returns:
        A configured large-v3 engine.
    """
    return LargeV3ASR(**kwargs)


def create_distil_large_v3(**kwargs: Any) -> DistilLargeV3ASR:
    """Return the ``faster-whisper/distil-large-v3`` preset (distilled, English-only).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`DistilLargeV3ASR`.

    Returns:
        A configured distil-large-v3 engine.
    """
    return DistilLargeV3ASR(**kwargs)


def create_turbo(**kwargs: Any) -> TurboASR:
    """Return the ``faster-whisper/large-v3-turbo`` preset (fastest large preset).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`TurboASR`.

    Returns:
        A configured turbo engine.
    """
    return TurboASR(**kwargs)
