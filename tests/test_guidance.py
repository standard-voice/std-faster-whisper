# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The guard against a decode that hands the guidance back (see ``_guidance``).

Written from the ways it can go wrong: the prompt or the hotwords come back as
the transcript and are reported as speech; or the guard drops real speech that
uses their words.
"""

from __future__ import annotations

from std_faster_whisper._guidance import (  # pyright: ignore[reportPrivateUsage]
    echoes_guidance,
    echoes_prompt,
)

_PROMPT = "Jezo, Quick Notes, Standard ASR, Kestrelwood library card"


def test_the_whole_prompt_is_an_echo() -> None:
    assert echoes_prompt(_PROMPT, _PROMPT)
    assert echoes_prompt(" jezo quick notes standard asr kestrelwood library card.", _PROMPT)


def test_most_of_the_prompt_is_an_echo() -> None:
    assert echoes_prompt("Jezo, Quick Notes, Standard ASR,", _PROMPT)


def test_speech_using_the_prompts_words_is_not_an_echo() -> None:
    assert not echoes_prompt("Open Jezo and renew the Kestrelwood library card.", _PROMPT)
    assert not echoes_prompt("Jezo", _PROMPT)


def test_a_term_or_two_is_too_short_to_tell() -> None:
    assert not echoes_prompt("Jezo", "Jezo")


def test_hotwords_count_as_guidance_too() -> None:
    hints = ["Jezo", "Quick Notes", "Kestrelwood"]
    assert echoes_guidance("Jezo Quick Notes Kestrelwood", None, hints)
    assert not echoes_guidance("Open Quick Notes", None, hints)
    assert not echoes_guidance("Jezo Quick Notes Kestrelwood", None, None)


def test_nothing_said_or_no_guidance_is_not_an_echo() -> None:
    assert not echoes_guidance("", _PROMPT, ["Jezo"])
    assert not echoes_guidance(_PROMPT, None, None)
    assert not echoes_guidance(_PROMPT, "", [])
