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
as a depthwise convolution over time. So the graphs are Rust against
that library, like the LTX-2.3 and Qwen-Image pipelines.

![The speech pipeline: clip.wav goes through the WAV parser as 16 kHz samples, framing cuts 1,056 frames, the mel filterbank produces a 1,056 by 128 log-mel spectrogram, the Conformer encoder subsamples it eight times to 132 vectors of 1,024, the decoder reads them through cross-attention together with a ten-id task prompt and emits 33 token ids, and the detokenizer turns them into text](pipeline.svg)

The request passes four stages, and each one feeds the next: the front
end turns samples into mel bands, the encoder turns mel bands into audio
vectors, the decoder turns those vectors into token ids, and a table
lookup turns the ids into text.

## Front end: from samples to mel bands

A WAV holds *samples*: one pressure reading every 62.5 µs. The model
works instead on *frames*, short slices over which a vowel or a
consonant is roughly stable. The recipe comes from NeMo, the NVIDIA
PyTorch toolkit Canary was trained and published with, and from the
GGUF's front-end metadata:

1. **Pre-emphasis:** subtract 0.97 times the previous sample, a mild
   high-pass filter that lifts the quieter consonants.
2. **Padding:** 256 zeros on each side, half a frame, so the first
   frame is centred on the first sample.
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
wide and sparse at high ones. Canary uses 128, so each frame's 257 bins
become 128 numbers. A floored logarithm and per-band normalisation over
the clip finish the front end.

The padding matters: the C++ port pads by reflection, and its notes
record a visible mel difference from that choice alone. The stage is
double precision Rust, since ggml has no short-time Fourier transform.

What the front end hands on is a *log-mel spectrogram*: for the test
clip, 1,056 frames of 128 mel bands each, one frame every 10 ms.

## Encoder: from mel bands to audio vectors

The encoder turns the spectrogram into a short sequence of vectors, one
per 80 ms of audio, each describing what was said around that moment.
It works in two steps: it shrinks the sequence, then refines it.

**Shrink.** A *convolution* slides a small window along the frames and,
at each position, mixes the frames under it into one new value with
learned weights. With a stride of 2 the window moves two frames at a
time, so the output is half as long. Three of them in a row cut the
length by eight, and a final projection gives each step 1,024 values:

```text
1,056 frames x 128 bands -> 528 -> 264 -> 132 vectors of 1,024 values
```

Shorter is cheaper. The next step compares every vector with every
other, so its cost grows with the square of the length.

**Refine.** Then 32 identical blocks run over the 132 vectors, keeping
their count and size. Each block updates every vector three ways:

- **Attention:** the vector looks at all 132 and pulls in the ones that
  matter to it, such as the rest of its word or sentence.
- **Convolution:** the vector is mixed with its 4 neighbours on each
  side, 9 in all, about 0.7 s of audio. This catches local sound
  patterns, such as where one phoneme ends and the next begins.
- **Feed-forward:** a small network reworks each vector on its own,
  using what the other two gathered.

Attention plus feed-forward is a *Transformer* block, the same block
inside every LLM. Adding the convolution makes it a *Conformer* block,
and *FastConformer* is NVIDIA's version with the 8x shrink in front.
One encoder pass peaks at 1.23 GB resident on Q8_0 and 2.10 GB on F16.

## Decoder: from audio vectors to text

The decoder is a Transformer like the one in a chat model. It writes
the transcript one *token* at a time, where a token is a word or a
piece of a word from a vocabulary of 16,384. Each of its 8 layers has
two kinds of attention:

- **Self-attention** looks at the tokens written so far, so the text
  stays coherent.
- **Cross-attention** looks at the encoder's 132 vectors, so the text
  follows the audio.

It starts from a prompt of ten token ids from the GGUF metadata: start
of transcript, source and target language (the same, for
transcription), punctuation on, timestamps off, and a few more. A
language the file does not list is refused before any graph exists. The
daemon's default transcription prompt would be wrong here, so the
worker skips it.

Decoding is greedy: each step takes the most likely next token, until
the model emits end of transcript. On 8 CPU threads a step costs
12.1 ms on Q8_0 and 16.4 ms on F16. The test clip's 33 tokens take
0.45 s and 0.61 s, prefill included.

The last stage is a table lookup: the worker maps each token id to its
piece of text through the GGUF vocabulary and joins the pieces.
Punctuation and capitals come from the model, because the prompt asked
for them. Through the daemon on a cold load, the clip answers in 3.6 s.

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
