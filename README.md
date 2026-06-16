<!--
SPDX-FileCopyrightText: 2026 Standard Voice Contributors
SPDX-License-Identifier: Apache-2.0
-->

# std-faster-whisper

> ⚠️ **Experimental — for protocol testing.** This is an experimental Standard ASR engine plugin, published to exercise and validate the [Standard ASR](https://github.com/standard-voice/standard_asr) interface. Expect breaking changes; it is not production-ready.

A **[Standard ASR](https://github.com/standard-voice/standard_asr) engine plugin
for [faster-whisper](https://github.com/SYSTRAN/faster-whisper)**.

Install it next to `standard-asr` and every Whisper preset becomes usable by any
Standard ASR-compliant application — through the CLI, the FastAPI server, or the
Python API — with **zero application code changes**. It is a thin, typed adapter
over the upstream `faster-whisper` package (which runs Whisper on CTranslate2); it
ships **no model weights and no recognition code of its own**.

- **Batch** transcription maps directly onto `WhisperModel.transcribe`.
- **Streaming** is supported via a *windowed re-decode* session with conservative,
  honestly-declared stability semantics (faster-whisper has no native streaming —
  see [Streaming](#streaming) for exactly what is and isn't promised).
- Engine-specific decoding knobs (`beam_size`, `task=translate`, VAD, …) live in a
  typed `provider_params` escape hatch; the portable standard set (`language`,
  `word_timestamps`, `prompt`, `phrase_hints`) maps onto faster-whisper's native
  arguments.

`module: std_faster_whisper` · `distribution: std-faster-whisper` ·
`engine_id: faster-whisper` · license **Apache-2.0** (upstream faster-whisper is
MIT; see [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md)).

## Install

> **Not yet published to PyPI** — install from GitHub:

```bash
uv pip install git+https://github.com/standard-voice/std-faster-whisper
```

This installs `faster-whisper` (and CTranslate2, PyAV, …) and `standard-asr`
(pinned to the GitHub `main` branch) automatically. Once published to PyPI this
becomes `uv pip install std-faster-whisper`. For development from a checkout, use
`uv pip install -e .`.

> **Hardware.** faster-whisper runs on CPU (CTranslate2 `int8` is a good fast
> default), CUDA GPUs (`device="cuda"`, `compute_type="float16"`), and Apple
> Silicon via CPU. There is **no CUDA on Apple Silicon**; use `device="cpu"`.

## Quick start

### CLI (zero config)

```bash
# Discover installed models (this plugin registers seven presets)
standard-asr list

# Inspect a preset's metadata, capabilities, and params schema (no model load)
standard-asr show faster-whisper/tiny

# Transcribe a file
standard-asr transcribe faster-whisper/tiny path/to/audio.wav

# Verify the plugin is protocol-compliant
standard-asr compliance run faster-whisper/tiny
```

### Python — batch

```python
from standard_asr import discover_models, RuntimeParams

engine = discover_models().create("faster-whisper/tiny", device="cpu", compute_type="int8")

# Simplest form (a bare str is treated as a local path):
result = engine.transcribe("meeting.wav")
print(result.text)
print(result.detected_language, result.language_confidence)

# With portable parameters + word timestamps:
result = engine.transcribe(
    "meeting.wav",
    RuntimeParams(language="en", word_timestamps="word", prompt="Q3 budget review."),
)
for word in result.words or []:
    print(f"{word.start:6.2f}-{word.end:6.2f}  {word.text}")
```

### Python — engine-specific knobs (`provider_params`)

```python
from standard_asr import RuntimeParams
from std_faster_whisper import FasterWhisperParams

result = engine.transcribe(
    "speech_in_french.wav",
    RuntimeParams(
        provider_params=FasterWhisperParams(task="translate", beam_size=3, vad_filter=True),
    ),
)
print(result.text)  # translated to English
```

Handing `FasterWhisperParams` to a *different* engine raises
`InvalidProviderParamError` (swap safety) — that is by design.

### Python — streaming

```python
import asyncio
from standard_asr import discover_models
from standard_asr.audio_format import AudioFormat

async def main() -> None:
    engine = discover_models().create("faster-whisper/tiny", device="cpu", compute_type="int8")
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)

    async with engine.start_transcription(audio_format=fmt) as session:
        session.feed(pcm_chunk_iterator())     # yields 16 kHz mono pcm_s16le bytes
        async for event in session:
            if event.type == "partial":
                print("…", event.text)
            elif event.type == "final":
                print("✓", event.segment_id, event.text)

asyncio.run(main())
```

A synchronous bridge is provided by the standard layer (`SyncSession`) — engine
authors implement only the async path. There is also a whole-input streaming
form (`start_transcription(audio="meeting.wav")`) that decodes a complete file
and streams the result back.

## Presets

Each model is a separate **entry-point preset** (spec IC.7: model selection =
entry-point preset, never an init `model` field), so `standard-asr list`,
the registry, and any settings UI can enumerate the available models:

| Entry-point key                     | Whisper model      | Notes |
| ----------------------------------- | ------------------ | ----- |
| `faster-whisper/tiny`               | `tiny`             | Smallest/fastest. Good for tests & smoke runs. |
| `faster-whisper/base`               | `base`             | Small multilingual. |
| `faster-whisper/small`              | `small`            | Mid-size multilingual. |
| `faster-whisper/medium`             | `medium`           | Large multilingual. |
| `faster-whisper/large-v3`           | `large-v3`         | **Recommended for production** (best accuracy). |
| `faster-whisper/distil-large-v3`    | `distil-large-v3`  | Distilled, lower latency, **English-only**. |
| `faster-whisper/large-v3-turbo`     | `large-v3-turbo`   | Fastest large preset. |

faster-whisper ships ~15 sizes; this package registers a representative spread.
To add another (e.g. `large-v2`):

1. add a `Properties` subclass overriding `model_name` (in `_metadata.py`),
2. add an engine subclass overriding `model_size` (in `engine.py`),
3. add a factory (in `entrypoint.py`),
4. register the key in `pyproject.toml` under
   `[project.entry-points."standard_asr.models"]`.

`model_path` is **not** a model selector — it is an optional local CTranslate2
checkpoint path/override (spec IC.7 weights/path). The preset chooses the model.

## Configuration

Build via the registry (`discover_models().create(key, **config)`) or directly.
All fields also fall back to environment variables
`STANDARD_ASR_FASTER_WHISPER__<FIELD>` (double underscore; spec IC.4).

| Field | Default | Meaning |
| --- | --- | --- |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"`. |
| `compute_type` | `"default"` | CTranslate2 quantization (`"int8"`, `"float16"`, …). |
| `device_index` | `0` | CTranslate2 device index or list. |
| `cpu_threads` | `0` | CPU threads (`0` = CTranslate2 default). |
| `num_workers` | `1` | Parallel-inference worker threads. |
| `default_language` | `"auto"` | Language axis default (BCP-47 or `"auto"`). |
| `download_root` | `None` | Model cache dir (else HF cache; spec IC.9 precedence). |
| `local_files_only` | `False` | Never download; require a cached/local model. |
| `revision` | `None` | Hugging Face model revision. |
| `hf_token` | `None` | **Secret** HF token for gated/private repos (masked in dumps). |
| `model_path` | `None` | Optional **local** CTranslate2 checkpoint override. |

Model weights load **lazily** on first transcription. Use `engine.prepare()`
(or `standard-asr prepare …`) to pre-download/warm without transcribing.
Downloads respect `STANDARD_ASR_ALLOW_DOWNLOAD`.

## Streaming

faster-whisper has **no native streaming API**; `WhisperModel.transcribe` is a
batch call. This plugin implements streaming with a **windowed re-decode**
strategy and declares its capabilities to match *exactly* what that delivers —
nothing is faked:

- Fed PCM is accumulated into a growing window; every few seconds the **whole
  window** is re-decoded (in a worker thread, so the event loop never blocks).
- Sentences that end comfortably behind the decode frontier are emitted as
  `final` events with stable, never-reused segment ids.
- The current tail is emitted as one `partial`.
- Because Whisper re-decodes the window and may rewrite earlier text, every
  `partial` reports **`stable_until = 0`** and the engine declares
  `word_stability = false`, `re_segments = false` (it never emits `supersede`),
  `reconnect = unsupported` (a local in-process model has no transport to
  reconnect), `finality_level = final`, and `timestamps = post_align`.

This is a **pragmatic** way to get live, growing output from a batch engine — it
is not a true low-latency incremental recognizer, and the higher latency / compute
cost reflect re-decoding the window. For genuine low-latency streaming, prefer a
natively streaming engine plugin. (Capability discovery makes this honest: an
application can read `engine.supports("streaming.word_stability")` and adapt.)

## Verifying

See [`VERIFICATION.md`](VERIFICATION.md) for copy-pasteable commands that
reproduce model discovery, compliance, real batch transcription of the bundled
test audio, a live streaming run, and the unit suite — with the actual observed
output.

## License

Apache-2.0. This adapter is original work; the upstream `faster-whisper` is MIT,
CTranslate2 is MIT, and the FFmpeg libraries bundled by PyAV are LGPL. Full
dependency-license breakdown and the license-isolation rationale (Standard ASR
goal G.4.2) are in [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md).
