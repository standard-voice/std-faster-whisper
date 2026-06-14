<!--
SPDX-FileCopyrightText: 2026 Standard Voice Contributors
SPDX-License-Identifier: Apache-2.0
-->

# Verification

This document is a re-runnable record of how `std-faster-whisper` was verified:
exact, copy-pasteable commands, what each does, and the **actual output observed**
(pasted verbatim). It covers discovery, compliance, real batch + streaming
inference on real audio, and the unit suite.

## Environment (observed)

| Component | Value |
| --- | --- |
| Machine | Apple M5 Max, arm64, macOS 27 — **CPU/Metal, no CUDA** |
| Python | 3.12.11 (pinned via `.python-version`; system Python is 3.14, which has no CTranslate2 wheel) |
| uv | 0.11.21 |
| ffmpeg | 8.1.1 (also bundled by PyAV) |
| `standard-asr` | 0.1.0 @ git `refactor/v0.1.0-redesign` (commit `4962a74`) |
| `faster-whisper` | 1.2.1 |
| `ctranslate2` | 4.8.0 |
| `numpy` | 2.4.6 |
| Test audio | `…/standard_asr/reference/standard_asr_test_audio_english.m4a` (≈57.5s, 48 kHz stereo AAC, English) |

faster-whisper runs on CPU here via CTranslate2 with `device=cpu`,
`compute_type=int8` (fast; `int8` is auto-converted from the model's saved
float16 with a one-line CTranslate2 warning, which is expected on CPU).

## 0. Setup

```bash
cd std-faster-whisper
uv sync                       # creates .venv on Python 3.12, installs everything
```

`uv sync` resolves and installs `standard-asr` from the branch git dependency plus
`faster-whisper`, `ctranslate2`, `pyav`, etc. (54 packages).

For the CLI examples below, init config (device / compute type) is supplied via
env vars (the `transcribe` CLI takes runtime `--options`, not init config — see
finding #3). Export them once per shell:

```bash
export STANDARD_ASR_FASTER_WHISPER__DEVICE=cpu
export STANDARD_ASR_FASTER_WHISPER__COMPUTE_TYPE=int8
```

## 1. Discovery — the plugin's seven presets resolve

```bash
uv run standard-asr models list
```

Observed:

```
Discovered models:
 - faster-whisper/base             engine=faster-whisper  model=base
 - faster-whisper/distil-large-v3  engine=faster-whisper  model=distil-large-v3
 - faster-whisper/large-v3         engine=faster-whisper  model=large-v3
 - faster-whisper/large-v3-turbo   engine=faster-whisper  model=large-v3-turbo
 - faster-whisper/medium           engine=faster-whisper  model=medium
 - faster-whisper/small            engine=faster-whisper  model=small
 - faster-whisper/tiny             engine=faster-whisper  model=tiny
```

Inspect one preset's metadata, capabilities, and params schema **without loading
the model**:

```bash
uv run standard-asr models show faster-whisper/tiny
```

(Prints the entry-point coordinates plus the full canonical capabilities JSON and
the `FasterWhisperParams` JSON Schema — read instantiation-free from the class.)

## 2. Compliance — passes (entrypoints, gating, sync bridge)

Per-engine compliance (entry-point metadata + streaming param-gating probe):

```bash
uv run standard-asr compliance run faster-whisper/tiny
```

Observed:

```
[OK] Entry point compliance checks passed.
[INFO] Streaming event-sequence is not run here; cover it with
       standard_asr.compliance.check_event_sequence in your tests (see
       docs/for_asr_dev/plugin_entrypoints.md).
[OK] Compliance run passed.
```

With the sync-bridge check, which **instantiates the real model**:

```bash
uv run standard-asr compliance run --include-bridge faster-whisper/tiny
```

Observed (after the tqdm-monitor fix — see finding #2; CTranslate2 float16→float32
notice elided):

```
[OK] Entry point compliance checks passed.
[INFO] Streaming event-sequence is not run here; cover it with
       standard_asr.compliance.check_event_sequence in your tests ...
[OK] Compliance run passed.
```

> The streaming **event-sequence** contract (`check_event_sequence`) cannot be
> driven by the CLI, so it is covered in the unit suite
> (`tests/test_streaming.py::test_recorded_stream_obeys_event_sequence_contract`)
> against a recorded live stream — see §5.

Dependency doctor (read-only conflict diagnosis):

```bash
uv run standard-asr doctor
```

Observed: lists all seven presets under `[std-faster-whisper]` and
`No dependency conflicts detected.`

## 3. Real batch transcription (CLI) — correct English transcript

This downloads the `tiny` model (~75 MB) on first run, then transcribes:

```bash
uv run standard-asr transcribe faster-whisper/tiny \
    .../standard_asr/reference/standard_asr_test_audio_english.m4a \
    --options '{"language": "en"}'
```

Observed transcript (verbatim):

> This is a crazy, interesting test for testing the capabilities and initial
> prototype of standard ASR package. By doing this we are creating a sample plugin
> implementation for faster whisper and Q and 3AASR. Both are really good ASR
> engine, so we are going to try out and see if we can implement the plugin for
> these two ASRs. By doing this we will be able to understand the potential issues
> and whether our design actually working in the real, real, real, real, real
> scenario. Because we have been merely designing things for a very long time, now
> is the time to put the design into test. Complete!

Wall time ≈ 13s including the one-time model download; ≈1.1s on subsequent runs.
("Q and 3AASR" is the tiny model mishearing "Qwen3-ASR" — accuracy, not a plugin
bug; `large-v3` transcribes it correctly.)

## 4. Real batch + streaming inference (script) — partial→final demonstrated

```bash
uv run python scripts/verify_inference.py \
    .../standard_asr/reference/standard_asr_test_audio_english.m4a
```

Runs three real passes against the cached `tiny` model:

**(a) Batch** — `detected_language: en (conf=1.0)`, `duration: 57.45s`, 6 segments,
wall time ≈1.1s. First segments observed:

```
[  0.00- 10.00] This is a crazy, interesting test for testing the capabilities and initial prototype of standard ASR package.
[ 10.00- 20.00] By doing this we are creating a sample plugin implementation for faster whisper and Q and 3AASR.
[ 20.00- 29.00] Both are really good ASR engine, so we are going to try out and see if we can implement the plugin for these two ASRs.
```

**(b) Batch + word timestamps** — 108 words with timing + probability. First few:

```
[  0.00-  0.68] ' This' (p=0.627)
[  0.68-  0.98] ' is'   (p=0.991)
[  0.98-  1.20] ' a'    (p=0.993)
[  1.20-  1.54] ' crazy,' (p=0.988)
```

**(c) Streaming (windowed)** — the audio is fed in 1.0s `pcm_s16le` chunks; the
session re-decodes the growing window and emits **partials that evolve into
finals** (`stable_until=0` on every partial, segment ids `seg-0…seg-5` never
reused). Abridged observed sequence:

```
partial[seg-0] (su=0): This is a crazy interesting past for
partial[seg-0] (su=0): This is a crazy interesting test for testing the capabilities in the initial pro
FINAL  [seg-0]: This is a crazy interesting test for testing the capabilities in the initial prototype of standard ASR package.
partial[seg-1] (su=0): By doing this, we are creating a sample plugin implementation.
FINAL  [seg-1]: By doing this we are creating a sample plugin implementation for faster whisper and Q and 3AASR.
...
FINAL  [seg-4]: Because we have been merely designing things for a very long time, now it's the time to put the design into test.
FINAL  [seg-5]: Complete!
done

events: 14 partial, 6 final, 0 progress
wall time: ≈6.8s
```

The reduced stream result (`session.result().text`) matches the batch transcript.
This is a genuine live partial→final stream synthesized over a batch engine — see
the README "Streaming" section and finding notes for the windowing strategy and
its honest stability semantics.

## 5. Unit suite — 69 tests, 100% coverage, strict typing

The unit suite injects a fake `faster_whisper` module (no weights, no network) and
exercises batch, streaming (incl. the spec event-sequence contract and sync
bridge), config/secrets, params swap-safety, guidance gating, and discovery.

```bash
uv run pytest             # 69 passed; TOTAL 100% line+branch coverage
uv run ruff check .       # All checks passed!
uv run ruff format --check .
uv run pyright            # 0 errors, 0 warnings (strict mode)
```

Observed `pytest` tail:

```
Name    Stmts   Miss Branch BrPart  Cover
-----------------------------------------------
TOTAL     353      0     60      0   100%
69 passed
```

## Notes on reproducibility

- The first transcribe/script run downloads the `tiny` model to the Hugging Face
  cache; later runs are offline-capable and fast.
- `uv.lock` is committed, so `uv sync` reproduces the exact dependency set above.
- No CUDA is required or used; everything runs on CPU via CTranslate2 `int8`.
- The unit suite is hermetic (fakes the model); only §3–§4 perform real inference
  and need network access on first run.
