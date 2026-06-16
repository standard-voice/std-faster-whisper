<!--
SPDX-FileCopyrightText: 2026 Standard Voice Contributors
SPDX-License-Identifier: Apache-2.0
-->

# Standard ASR v0.1.0 — plugin-author findings

Findings from building `std-faster-whisper` as a fully independent plugin against
`standard-asr @ main`. The protocol is in
good shape — a complete, honestly-declared **batch + streaming** engine with
discovery, compliance, CLI, doctor, real inference, 100% test coverage, and
pyright-strict typing came together in one sitting. The items below are the rough
edges I hit, ordered roughly by impact. Each has *what happened*, *why it
mattered*, and a *concrete suggestion*.

Severity legend: **[High]** blocks or silently misleads · **[Med]** real friction
· **[Low]** papercut.

---

## 1. [High] The base `TranscriptionSession` reserves private attribute names with no documented "reserved" list — a subclass silently clobbered one

**What happened.** My streaming session stored its audio window as
`self._buffer`. The base `TranscriptionSession.__init__` *also* uses
`self._buffer` for its internal event coalescing buffer (`_CoalescingBuffer`).
Because my `super().__init__()` ran first and my assignment ran second, I
overwrote the base's event buffer with a numpy array. The failure surfaced far
from the cause, deep inside the base class's iterator:

```
AttributeError: 'numpy.ndarray' object has no attribute 'get'
  .../standard_asr/streaming.py:2032  event = await asyncio.wait_for(self._buffer.get(), ...)
```

This is exactly the kind of action-at-a-distance that costs an engine author an
hour. The base reserves at least `_buffer`, `_audio_queue`, `_audio_history`,
`_audio_cursor`, `_session_started_at`, `_last_audio_activity` — none documented
as off-limits to subclasses, even though subclassing `TranscriptionSession` is
*the* extension point for streaming engines.

**Why it mattered.** The whole streaming contract is "subclass
`TranscriptionSession`, implement `_produce`". An author naturally stores state
on `self`. With no namespacing and no reserved-name list, name collisions are a
matter of when, not if, and they fail confusingly.

**Suggestion.** Pick one:
- Prefix every base-internal attribute with a class-private dunder
  (`self.__buffer`, name-mangled to `_TranscriptionSession__buffer`) so subclass
  attributes can never collide; **or**
- document a "reserved attribute names" block in `adapting_engine.md` §Streaming
  and add a `__init_subclass__`/`__setattr__` guard that raises if a subclass
  rebinds a reserved name. The second is cheaper and more discoverable.

---

## 2. [High] `check_sync_bridge` fails on faster-whisper's `tqdm_monitor` thread — a fully-correct engine fails compliance because of a transitive dependency's daemon thread

**What happened.** `standard-asr compliance run --include-bridge faster-whisper/tiny`
loads the *real* model and then runs the sync-bridge check, which failed:

```
[FAIL] sync_bridge_thread_leak: SyncSession leaked background thread(s): ['tqdm_monitor'].
```

`tqdm_monitor` is a **daemon** thread that `tqdm` (used by `huggingface_hub` for
download progress and by faster-whisper) spawns on first use and never joins. It
is not my session's thread, not bound to the event loop, and harmless at
interpreter exit — but the leak check flags *any* non-baseline thread. My own
unit `check_sync_bridge` test passed only because it uses a fake model (no tqdm).

**Why it mattered.** This is a silent trap: the engine is correct, the unit tests
are green, but real-model compliance fails for a reason that has nothing to do
with the engine's bridge behavior. Worse, the obvious reading ("my bridge leaks a
thread") sends the author hunting in the wrong place. I worked around it by
disabling the tqdm monitor in the loader (`tqdm.monitor_interval = 0`; see
`engine.py::_disable_tqdm_monitor_thread`) — but that is the *adapter* paying for
a *framework* check's overly broad definition of "leak".

**Suggestion.** Make the leak check robust to benign daemon threads from
dependencies:
- ignore threads with `daemon=True` that are not the bridge's own worker
  (the bridge knows exactly which thread it created — assert on *that* one's
  termination, not on a full `threading.enumerate()` diff); **or**
- maintain a small allowlist of known-benign thread-name patterns
  (`tqdm_monitor`, `ThreadPoolExecutor-*`); **and**
- document the gotcha + the `monitor_interval = 0` remedy in `adapting_engine.md`
  so every Whisper-family / HF-downloading plugin doesn't rediscover it.

---

## 3. [Med] The `transcribe` CLI has no way to pass init config (device, compute_type) — only runtime `--options`

**What happened.** To run real inference on CPU I needed `device="cpu",
compute_type="int8"`. The CLI is:

```
standard-asr transcribe <name> <audio> [--options JSON] [--json]
```

`--options` is *runtime params* (`RuntimeParams`), not init config. There is no
`--config`/`-c` flag and no `--set key=value`. The only way to set `device` from
the CLI is the env-var fallback:

```bash
export STANDARD_ASR_FASTER_WHISPER__DEVICE=cpu
export STANDARD_ASR_FASTER_WHISPER__COMPUTE_TYPE=int8
standard-asr transcribe faster-whisper/tiny audio.m4a --options '{"language":"en"}'
```

That works (and is good that it works!), but it's not discoverable — nothing in
`--help` hints that init config is env-only, and a new user trying
`--device cpu` gets `unrecognized arguments` with no pointer to the env route.

**Why it mattered.** "Try a model from the CLI" is the first thing both an app
dev and an end user do (G.2.2). Needing to know the exact
`STANDARD_ASR_<ENGINE>__<FIELD>` spelling to pick a device is a real barrier,
especially since the registry already has the config JSON Schema and could render
/ accept it.

**Suggestion.** Add `standard-asr transcribe ... --config '<json>'` (or repeated
`--set device=cpu --set compute_type=int8`) that feeds `registry.create(name,
**config)`. The schema is already discoverable via `config_schema`, so the CLI
could even validate it. At minimum, mention the env-var route in the
`transcribe --help` text.

---

## 4. [Med] No standard `float32 → pcm_s16le` (and reverse) helper, despite the spec pinning the exact quantization

**What happened.** For windowed streaming I had to (a) decode incoming
`pcm_s16le` wire frames to float32 and (b) on the whole-input path, quantize a
negotiated float32 array back to `pcm_s16le`. The spec (§AI R4) *normatively pins*
this conversion to the byte: clip → `round_half(x * 32767)` → little-endian
int16, with non-finite sanitization and `÷32768` on the way back. I re-implemented
it twice (`_convert.pcm_s16le_to_float32`, `engine._prepared_to_pcm`).

**Why it mattered.** This is load-bearing for cross-language wire consistency —
the spec spends a whole normative paragraph getting the rounding right precisely
so independent implementations produce identical bytes. Every streaming plugin
that bridges PCM and arrays will re-implement it, and any that uses `astype(int16)`
(truncation) instead of `round` will silently violate the pinned convention and
fail a future conformance test by ±1 LSB — the exact failure mode the spec warns
about.

**Suggestion.** Export the canonical codecs from the core:
`standard_asr.audio.pcm_s16le_to_float32(bytes) -> NDArray[float32]` and
`float32_to_pcm_s16le(NDArray) -> bytes`, implementing §AI R4 exactly (the
encoder for the array→file path already does this internally — just surface it).
Then plugins inherit correctness for free and the spec's byte-exactness is
guaranteed at one site.

---

## 5. [Med] `WhisperModel.transcribe` accepts an off-rate array and silently mis-transcribes — the protocol protects against this, but only via an author-written assert

**What happened.** faster-whisper's `transcribe(array)` assumes the array is
16 kHz; feed it 8 kHz and it produces wrong text/timings with **no error**. The
protocol *does* protect callers here: I declare `accepted_sample_rates=[16000]`
and the standard layer resamples before `_transcribe`. But the only thing
guaranteeing the negotiated array is actually 16 kHz at the call site is a
defensive `assert prepared.sample_rate == 16000` I wrote by hand. If an author
forgets that assert and the negotiation contract ever regressed, the cardinal sin
(silent wrong transcript) is back.

**Why it mattered.** It's a sharp edge precisely *because* the happy path is
silent. The protocol's promise ("the array handed to `_transcribe` is in one of
your accepted shapes at an accepted rate") is strong, but it lives in prose; the
engine can't cheaply assert it.

**Suggestion.** Have `PreparedAudio` for the `ARRAY` kind carry an invariant the
engine can trust, and/or offer a tiny `prepared.require_array(sample_rate=16000)
-> NDArray[float32]` accessor that raises a clear `AudioProcessingError` if the
negotiated rate/shape doesn't match what the engine asked for. That turns the
hand-written assert into a one-liner with a good error message.

---

## 6. [Med] `discover_models()` and the registry API have small naming surprises that cost a round-trip each

Three independent papercuts, all "guessed the obvious name, was wrong":

- **`ModelRegistry.by_engine(engine_id)` returns `list[str]` (keys), not
  `list[ModelSpec]`.** I expected spec objects and wrote `{s.model_id for s in
  by_engine(...)}`, which failed with `'str' has no attribute 'model_id'`. The
  name reads like "give me the engine's models (objects)".
- **`ModelSpec` has `.key` / `.engine_id` / `.model_name` but **no
  `.model_id`****, while `properties` has `.model_id`. Two adjacent concepts,
  two different attribute names for "the `<engine>/<model>` string"
  (`spec.key` vs `properties.model_id`). I reflexively wrote `spec.model_id`.
- **`load_audio` is `load_audio(source, target_sr=…, target_channels=…)`** and
  returns a **bare `np.ndarray`**, but I expected `target_sample_rate=` and an
  object with `.samples`/`.sample_rate` (mirroring `AudioArray`). The kwarg name
  and the bare-array return are both reasonable, just not what the surrounding
  `AudioArray` vocabulary primed me for.

**Why it mattered.** None is a blocker, but each is a "read the source to find the
real name" detour, and they undercut the "understand behavior from the code
alone" philosophy because the *names* mislead.

**Suggestion.**
- Either rename `by_engine` → `keys_by_engine`, or make it return
  `list[ModelSpec]` (and add `model_ids_by_engine` for the string list).
- Add a `ModelSpec.model_id` property aliasing `.key` so the same concept has the
  same name everywhere.
- Consider `load_audio(..., target_sample_rate=…)` as an alias, or document the
  bare-array return prominently (it's easy to assume parity with `AudioArray`).

---

## 7. [Low] `effective_language` requires every caller to pass two capability booleans it could derive itself

**What happened.** Resolving the request language is:

```python
effective_language(
    params.language, config.default_language,
    has_language_axis=True, runtime_override_supported=True,
)
```

I call this in both `_transcribe` and the streaming session. Both keyword args are
things the engine *already declared* in `properties` / `declared_capabilities` —
`has_language_axis` is "is `selectable_languages` non-trivial" and
`runtime_override_supported` is literally
`supports("<mode>.language.runtime_override")`. Passing them by hand is
boilerplate that can silently disagree with the declarations (e.g. I hardcode
`runtime_override_supported=True` in the streaming session, which is correct only
because I *also* declared it true — two sources of truth).

**Why it mattered.** It's duplicated, drift-prone state. If I flipped the
capability to false but forgot the call site, the language axis would behave
inconsistently with what the engine advertises.

**Suggestion.** `EngineBase` already exposes `_resolve_language_axis(params,
mode)` (which does derive these from the declarations and also attaches
diagnostics). Promote it / document it as *the* way engines resolve language, and
de-emphasize calling the free function `effective_language` directly from engine
code. The cookbook still calls the free function, so the recommended path is
ambiguous.

---

## 8. [Low] Streaming author has to hand-build `pcm_s16le` from the negotiated whole-input array, re-deriving the wire format

**What happened.** On the `start_transcription(audio=…)` (whole-input) path the
base hands the hook a `PreparedAudio` (a negotiated array). To reuse my
window-decode pipeline I converted that array *back* to `pcm_s16le` bytes and
`feed()` it to my own session. So the audio went array → (my) PCM → (my) float32
again. Functionally fine, but I'm round-tripping through the wire encoding inside
my own process because the streaming session's input vocabulary is PCM bytes
while the whole-input vocabulary is `PreparedAudio`.

**Why it mattered.** Minor inefficiency and extra code, and it forced finding #4
(needing the float32→PCM codec) on top of the decode side.

**Suggestion.** Allow a session to accept already-prepared audio directly (e.g. a
base `seed_prepared_audio(prepared)` that an engine can route into `_produce`
without serializing to PCM), so the whole-input path can stay in float32 when the
engine wants arrays. Not urgent — the round-trip is cheap and lossless given the
pinned quantization — but it's an avoidable two-way conversion.

---

## 9. [Low] `compliance run` is per-engine but `models list` shows seven presets — no "check them all" affordance

**What happened.** `standard-asr compliance run faster-whisper/tiny` checks one
preset. With seven presets registered I wanted "check the whole package". `run`
with no name does entrypoint-only checks; there's no `compliance run --all` that
iterates every discovered key. (My pytest covers all presets, but a publisher
doing a quick pre-release sanity check from the CLI would want it.)

**Suggestion.** `standard-asr compliance run --all` (or accept multiple names)
that runs the per-engine checks across every discovered model and summarizes.

---

## What worked notably well (worth keeping)

- **Fail-closed capabilities + honest streaming declaration.** Declaring
  `word_stability=false`, `re_segments=false`, `reconnect=unsupported`,
  `finality_level=final`, `timestamps=post_align` let me ship a *truthful*
  windowed-streaming engine over a batch model. Apps can read
  `engine.supports("streaming.word_stability")` and adapt. This is the protocol's
  honesty principle paying off concretely — I never had to fake a guarantee.
- **The guidance constraint gate** (`max_tokens` / `max_terms`) turning
  faster-whisper's *silent* prompt truncation into a loud, pre-flight strict
  error (or best-effort truncate + diagnostic) is exactly right, and trivial to
  declare.
- **`provider_params` exact-type swap safety** is a genuinely good design — the
  `check_provider_params_swap_safety` probe + `InvalidProviderParamError` caught a
  wrong-engine params object with zero billing side effects.
- **Instantiation-free discovery via the factory return annotation** is elegant:
  `models show` / `doctor` / the registry read capabilities and the params schema
  off the class without ever constructing (or authenticating) the engine.
- **`from_env` + `SecretStr` + `secret_field`** made the HF-token handling correct
  by construction — masked in `repr`/`public_dump`, materialized to plaintext only
  at the `WhisperModel(use_auth_token=…)` call site.
- **The cookbook reference plugin** was an excellent starting point — its
  batch path, guidance gating, and download-policy handling transferred almost
  verbatim. (Graduating it to its own repo mostly meant adding streaming, the
  secret token, more presets, and the independent packaging/CI.)
