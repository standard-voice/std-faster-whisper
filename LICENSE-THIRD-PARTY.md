# Third-party licenses (dependency & license isolation)

`std-faster-whisper` is licensed **Apache-2.0** (see `LICENSE`). This file
documents the licenses of the libraries it depends on, so that application
developers can make an informed, license-aware choice when installing this
plugin (Standard ASR goal **G.4.2**: "dependency & license isolation —
applications choose plugins per their license and cost needs, with clear license
responsibility boundaries").

This adapter does **not** vendor, copy, or re-license any of the code below — it
declares them as ordinary runtime dependencies and they are installed and
licensed under their own terms.

## Runtime dependencies

| Package | License | Role |
| --- | --- | --- |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | **MIT** | The upstream inference engine we adapt (`WhisperModel`). |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | **MIT** | The C++ inference runtime faster-whisper builds on (pulled in transitively). |
| [standard-asr](https://github.com/standard-voice/standard_asr) | **Apache-2.0** | The protocol this plugin implements. |
| [PyAV](https://github.com/PyAV-Org/PyAV) | **BSD-3-Clause** | Bundles FFmpeg libraries for audio decoding (transitive via faster-whisper). FFmpeg itself is **LGPL-2.1+** (and, with some build options, GPL); the PyAV wheels bundle an LGPL build. |
| [tokenizers](https://github.com/huggingface/tokenizers) | **Apache-2.0** | Whisper tokenizer (transitive). |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | **Apache-2.0** | Model download/cache (transitive). |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | **MIT** | Silero VAD runtime (transitive; only used when `vad_filter=True`). |
| [numpy](https://numpy.org/) | **BSD-3-Clause** | Waveform arrays. |
| [pydantic](https://github.com/pydantic/pydantic) | **MIT** | Config / result models (transitive via standard-asr). |

## Model weights

Whisper model weights downloaded at runtime from the Hugging Face Hub are
published by their respective authors (OpenAI Whisper weights are **MIT**;
distilled variants follow their own model cards). The weights are **not**
distributed with this package — they are fetched on first use, subject to the
download policy (`STANDARD_ASR_ALLOW_DOWNLOAD`). Consult each model's card for its
exact terms.

## Note on the FFmpeg / LGPL boundary

PyAV ships FFmpeg as a dynamically linked shared library under LGPL. If your
application has constraints around LGPL, that boundary lives entirely inside the
transitive PyAV dependency of this plugin, isolated from your application code —
which is precisely the license-isolation property the plugin architecture exists
to provide. You may also avoid the audio-decode path entirely by feeding
already-decoded 16 kHz mono `float32` arrays (`AudioArray`), which this engine
accepts directly.
