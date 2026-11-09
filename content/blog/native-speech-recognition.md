+++
title = "llmman now transcribes speech natively"
description = "A Canary GGUF posted to llmman serve comes back as text. The whole pipeline, from the WAV bytes to the tokens, runs in llmman on the same ggml library as its image and video models, and matches the reference token for token."
date = 2026-11-09

[taxonomies]
tags = ["serve", "speech", "ggml"]
+++


A 10.55 s WAV goes in. One line of text comes out, 1.7 seconds later.
`llmman serve` now runs NVIDIA's Canary 1B v2 speech model itself, on
the ggml library it already loads for its image and video models.

<!-- more -->

## Try it

Until now `/v1/audio/transcriptions` went to llama-server, with a
multimodal projector in front of a language model. Canary is an
encoder-decoder model llama.cpp has no graph for, so llmman serves it
itself.

```sh
llmman serve
```

```sh
curl -s http://127.0.0.1:17434/v1/audio/transcriptions \
  -F model=hf.co/handy-computer/canary-1b-v2-gguf:Q8_0 \
  -F file=@clip.wav
{"text":"..."}
```

Any OpenAI audio API client can call it. The first request pulls the
GGUF from Hugging Face and loads it. The contract is narrow:

- **Audio:** a 16-bit mono WAV at 16 kHz, what the model was trained
  on. Anything else is refused, never resampled.
- **Endpoints:** transcription only. Chat, completions and embeddings
  are refused.
- **CLI:** `llmman run` does not transcribe yet.

## Why a pipeline of its own

When a GGUF header says Canary, the daemon spawns itself as a speech
worker, on the host or in the CPU llama.cpp container image, through
the media worker's spawn path.

A C++ port of Canary exists, but it carries a patched copy of ggml:
two copies of the library in the process tree, two versions to track.
Every operation Canary needs is already in the ggml llmman loads, from
2D convolutions and masked softmax to a sequence operator that doubles
as a depthwise convolution over time. So the graphs are Rust against that library, like the
LTX-2.3 and Qwen-Image pipelines.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400" role="img" aria-label="The speech pipeline: clip.wav goes through the WAV parser as 16 kHz samples, framing cuts 1,056 frames, the mel filterbank produces a 1,056 by 128 log-mel spectrogram, the Conformer encoder subsamples it eight times to 132 vectors of 1,024, the decoder reads them through cross-attention together with a ten-id task prompt and emits 33 token ids, and the detokenizer turns them into text" style="width:100%;height:auto;max-width:900px;margin:2rem 0;font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
<defs><marker id="sar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="var(--accent,#7aa2f7)"/></marker></defs>
<g fill="var(--muted,#8a94a8)" font-size="10" letter-spacing="1.6"><text x="250" y="56">RUST, NO WEIGHTS</text><text x="492" y="218">GGML GRAPHS OVER GGUF WEIGHTS</text><text x="250" y="218">TABLE LOOKUP</text></g>
<rect x="8" y="70" width="150" height="60" rx="8" fill="none" stroke="var(--line,#232a3a)" stroke-dasharray="4 3"/>
<g stroke="var(--line,#232a3a)" fill="var(--panel,#151a25)"><rect x="250" y="70" width="150" height="60" rx="8"/><rect x="492" y="70" width="150" height="60" rx="8"/><rect x="734" y="70" width="158" height="60" rx="8"/><rect x="250" y="230" width="150" height="60" rx="8"/><rect x="492" y="336" width="150" height="48" rx="8"/></g>
<g stroke="var(--accent,#7aa2f7)" fill="var(--panel,#151a25)"><rect x="734" y="230" width="158" height="60" rx="8"/><rect x="492" y="230" width="150" height="60" rx="8"/></g>
<rect x="8" y="230" width="150" height="60" rx="8" fill="var(--bg-soft,#11151f)" stroke="var(--accent,#7aa2f7)"/>
<g font-size="12" fill="var(--text,#d7dce6)"><text x="24" y="96">clip.wav</text><text x="266" y="96">WAV parser</text><text x="508" y="96">Framing</text><text x="750" y="96">Mel filterbank</text><text x="750" y="256">Conformer encoder</text><text x="508" y="256">Decoder</text><text x="266" y="256">Detokenizer</text><text x="24" y="256">Text</text><text x="508" y="356">Task prompt</text></g>
<g font-size="10" fill="var(--muted,#8a94a8)"><text x="24" y="114">16-bit mono, 16 kHz</text><text x="266" y="114">bounds-checked chunks</text><text x="508" y="114">pre-emphasis, Hann</text><text x="750" y="114">FFT, 128 bands, log</text><text x="750" y="274">8x subsampling, 32 blocks</text><text x="508" y="274">8 layers, cross-attention</text><text x="266" y="274">SentencePiece pieces</text><text x="24" y="274">{"text": "..."}</text><text x="508" y="372">10 ids from the GGUF</text></g>
<g fill="none" stroke="var(--accent,#7aa2f7)" stroke-width="1.2" opacity=".75" marker-end="url(#sar)"><path d="M158 100 H248"/><path d="M400 100 H490"/><path d="M642 100 H732"/><path d="M813 130 V228"/><path d="M734 260 H644"/><path d="M492 260 H402"/><path d="M250 260 H160"/><path d="M567 336 V292"/></g>
<g font-size="10" fill="var(--accent,#7aa2f7)" text-anchor="middle"><text x="204" y="92">bytes</text><text x="446" y="92">samples</text><text x="688" y="92">1,056 frames</text><text x="688" y="252">132 × 1,024</text><text x="446" y="252">33 token ids</text><text x="204" y="252">pieces</text></g>
<g font-size="10" fill="var(--muted,#8a94a8)" text-anchor="middle"><text x="446" y="118">16 kHz</text><text x="688" y="118">× 400 samples</text><text x="688" y="278">one per 80 ms</text><text x="446" y="278">greedy</text></g>
<g font-size="10" text-anchor="end"><text x="805" y="176" fill="var(--accent,#7aa2f7)">log-mel spectrogram</text><text x="805" y="190" fill="var(--muted,#8a94a8)">1,056 × 128</text></g>
</svg>

## From sound to mel bands

A WAV holds *samples*: one pressure reading every 62.5 µs. The model
works instead on *frames*, short slices over which a vowel or a
consonant is roughly stable. The recipe comes from NeMo, the NVIDIA PyTorch toolkit Canary was
trained and published with, and from the GGUF's front-end metadata:

1. **Pre-emphasis:** subtract 0.97 times the previous sample, a mild
   high-pass filter that lifts the quieter consonants.
2. **Padding:** 256 zeros on each side, half a frame, so the first frame is centred
   on the first sample.
3. **Framing:** 400 samples (25 ms) every 160 (10 ms). Frames overlap,
   so a sound cut at one edge is whole in the next.
4. **Hann window:** fade each frame's edges to zero, so the cut adds no
   energy of its own.
5. **Power spectrum:** a 512-point Fourier transform gives 257 bins per
   frame, each 31.25 Hz wide, from 0 to 8 kHz.

Those bins are evenly spaced in hertz, and hearing is not. We tell
100 Hz from 200 Hz easily and 7,000 Hz from 7,100 Hz hardly at all. The
mel scale spaces pitch the way hearing does, and a *mel band* is one
triangular filter laid out on it: narrow and dense at low frequencies,
wide and sparse at high ones. Canary uses 128. A floored logarithm and
per-band normalisation over the clip finish the front end.

The padding matters: the C++ port pads by reflection, and its notes
record a visible mel difference from that choice alone.
The stage is double precision Rust, since ggml has no short-time
Fourier transform.

## Encoder and decoder

The encoder is a FastConformer. A *Conformer* block is a Transformer
block with a 9-frame convolution between attention and feed-forward.
Attention lets every frame see every other, which captures which word a
syllable belongs to; the convolution captures the local pattern that
tells one phoneme from the next.

Attention cost grows with the square of the sequence length, so the
encoder first subsamples the mel spectrogram in time with three
stride-2 convolutions:

```text
1,056 frames -> 528 -> 264 -> 132 vectors of 1,024 values
```

One encoder pass peaks at 1.23 GB resident on Q8_0 and 2.10 GB on F16.

The decoder is an 8-layer Transformer over 16,384 SentencePiece
tokens, with *cross-attention* in every layer: self-attention looks at
the tokens written so far, cross-attention looks at the encoder's
vectors.

Canary's prompt is ten token ids from the GGUF metadata: start of
transcript, source and target language (the same, for transcription),
punctuation on, timestamps off, and a few more. A language the file
does not list is refused before any graph exists. The daemon's default
transcription prompt would be wrong here, so the worker skips it.

Decoding is greedy. On 8 CPU threads a step costs 12.1 ms on Q8_0 and
16.4 ms on F16: 0.45 s and 0.61 s for the test clip, prefill included. The worker maps
the ids to pieces through the GGUF vocabulary; punctuation and capitals
come from the model, because the prompt asked for them. Through the
daemon on a cold load, the clip answers in 3.6 s.

## Around the request

The worker applies its limits before any audio is decoded:

- At most eight requests are admitted at a time.
- The file part is capped at 4 MiB, about 131 s of audio, and the WAV
  parser checks every chunk against it, so a hostile header cannot make
  it allocate.
- One request computes at a time, behind a lock. The graphs already use
  every thread, and the lock holds memory to one encoder graph and one
  decoder session.

Audio longer than 30 s is split into equal clips and the texts
joined: a 31 s file becomes two clips of 15.5 s.

When the client disconnects, an abort flag is set, and three places
read it:

1. ggml's abort callback, which stops a running graph within about
   40 ms.
2. The decoder loop, before every token.
3. The mel front end, every 32 frames (320 ms of audio).

The third check came from a measurement: a request abandoned during
the front end held the lock for the 0.93 s of mel still to compute.
With the check, the next request takes the lock 46 ms to 339 ms after
the disconnect, in a debug build.

## Checked against NeMo

A port with a wrong padding choice still produces the right words on
a clean clip and fails quietly on the next one, so reading transcripts
tells us little. Every stage was instead compared numerically with
activations dumped from the NeMo reference on one English clip, for
both published weight files:

| Stage | Compared | F16 | Q8_0 |
|---|---|---|---|
| Mel front end | max absolute difference | 1.29e-4 | 1.29e-4 |
| Encoder blocks and output | relative rms error per stage | 4e-4 to 1.2e-3 | 3.8e-3 to 3.8e-2 |
| Decoder sublayers and logits | relative rms error | 7.7e-5 to 8.3e-4 | 1.1e-3 to 2.1e-2 |
| Prompt ids, greedy ids, text | exact match | 10/10, 33/33, equal | 10/10, 33/33, equal |

On F16 the encoder error sits near the half-precision epsilon of
4.9e-4, because the CPU matrix multiplication converts activations to
half precision before multiplying. On Q8_0 it is an order of magnitude larger, from
quantization.

## Next

- Only the CPU backend has run. Every op the graphs use is supported on
  Vulkan, but nothing has run on a GPU.
- No streaming and no timestamps.
- No overlap between long-audio clips, so a word cut at a boundary may
  come out wrong.
- No beam search or translation.

Parakeet shares the front end and the encoder, so a second speech model
means a second header value and its decoder.

## References

The work is tracked in the project's
[tracking issue](https://github.com/llmmanorg/llmman/issues/713). The model
is described on its [model card](https://huggingface.co/nvidia/canary-1b-v2),
and its reference implementation lives in [NeMo](https://github.com/NVIDIA/NeMo).
The compute library is [ggml](https://github.com/ggml-org/ggml), shared
with [llama.cpp](https://github.com/ggml-org/llama.cpp). Questions are
welcome at [github.com/llmmanorg/llmman](https://github.com/llmmanorg/llmman).
