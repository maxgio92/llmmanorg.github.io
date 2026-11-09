+++
title = "llmman now transcribes speech natively"
description = "A Canary GGUF posted to llmman serve comes back as text. The whole pipeline, from the WAV bytes to the tokens, runs in llmman on the same ggml library as its image and video models, and matches the reference token for token."
date = 2026-11-09

[taxonomies]
tags = ["serve", "speech", "ggml"]
+++

A ten second WAV goes in. One line of text comes out, 1.7 seconds
later. In between, the clip was turned into 160,000 numbers, cut into
1,056 overlapping frames, read for its frequencies, folded into 128
bands shaped like human hearing, compressed to 132 vectors, and decoded
one token at a time. None of that happened in Python or in a second
binary: it ran in `llmman serve`, on the ggml library the daemon
already loads for its image and video models.

<!-- more -->

## 1. A clip goes in, text comes out

`llmman serve` has answered `/v1/audio/transcriptions` for a while, by
handing the request to llama-server with a multimodal projector in
front of a language model. Now it also serves NVIDIA's Canary 1B v2, an
encoder-decoder speech model llama.cpp has no graph for.

```sh
llmman serve
```

```sh
curl -s http://127.0.0.1:17434/v1/audio/transcriptions \
  -F model=hf.co/handy-computer/canary-1b-v2-gguf:Q8_0 \
  -F file=@clip.wav
{"text":"..."}
```

The first request pulls the GGUF from Hugging Face and loads it; the
ones after skip both. The request is the OpenAI one, so anything that
speaks the OpenAI audio API can call it. The file must be a 16-bit mono
WAV at 16 kHz, which is what the model was trained on; anything else is
refused rather than resampled. This model only transcribes: chat,
completions and embeddings are refused, and `llmman run` does not
transcribe yet.

That is the surface. The question is what happens between the WAV
bytes and the text, and how we know each step is right.

## 2. The whole pipeline at a glance

Here is the map we wanted when we started, in the order a request meets
the stages:

```text
1. Parser:        WAV bytes -> 16-bit samples, one every 62.5 microseconds
2. Mel front end: samples -> frames (25 ms every 10 ms) -> spectrum per frame -> 128 mel bands per frame
3. Encoder:       mel frames -> subsample 8x in time -> 32 Conformer blocks -> one 1,024-wide vector per 80 ms
4. Decoder:       ten-id prompt + encoder vectors -> one token per step, greedy, until end of text
5. Detokenizer:   token ids -> pieces -> text with spaces, punctuation and capitals
```

One clip moves through these steps in order, and each step shrinks
the data along the time axis: 160,000 samples become about 1,000
frames, then 132 vectors, then 33 tokens. The first two stages are
plain arithmetic in Rust with no learned weights, the next two are
ggml compute graphs over the weights in the GGUF, and the last is a
table lookup. Around all five sits a worker process that admits,
serialises and cancels requests, which section 10 covers.

A few terms recur. A **sample** is one number giving the microphone's
pressure at one instant. A **frame** is a short slice of samples, here
25 ms, over which a speech sound is roughly stable. A **bin** is one
output of the Fourier transform, a narrow band of frequency. A **mel
band** is a wider band of frequency, laid out on a scale that spaces
pitches the way hearing does. A **token** is one entry in the model's
vocabulary of 16,384 pieces of text.

## 3. Why a pipeline of its own

Every GGUF declares its architecture in its header. When the header
says Canary, the daemon spawns itself as a worker process with a speech
engine, on the host or inside the CPU llama.cpp container image, through
the same spawn path the media worker uses. The worker loads the model
once and serves the transcription route, sharing the serve loop and
the error shape with the media worker, so speech adds a pipeline and
not another server. The default transcription prompt the daemon injects
for llama-server backends is skipped: Canary carries its own, as we'll
see in section 8.

A C++ port of Canary exists, but it carries a patched copy of ggml,
which would have meant two copies of the same library in the process
tree and two versions to track. A pass over the ggml sources found that
every operation Canary needs already exists in the library llmman
loads, from 2D convolutions and masked softmax to a state-space op that
expresses a depthwise kernel. So the graphs are Rust against that
library, as the LTX-2.3 and Qwen-Image pipelines are.

## 4. From bytes to samples

WAV is a container: a header that describes the audio, then a data
chunk with the samples. The parser walks the chunks with bounds checks
and takes a byte cap from the caller, so a hostile header cannot make
it allocate. The contract is narrow on purpose: 16-bit signed integers,
one channel, 16,000 per second. Resampling inside the worker would have
been one more stage to check.

What comes out is what every speech model starts from: a sequence of
samples, one every 62.5 microseconds, each saying where the
microphone's membrane was at that instant. Ten seconds of audio is
160,000 numbers. The model never sees them directly.

## 5. From samples to frames

A speech model works on short slices of time, because the sound of a
vowel or a consonant is roughly stable for a few tens of milliseconds
and changes after that. Those slices are frames. The recipe comes from
the reference and from the front-end metadata the GGUF carries.

First, pre-emphasis: each sample has 0.97 times the previous one
subtracted from it, a mild high-pass filter that lifts the consonants,
which carry less energy than the vowels. Then 256 zeros of padding on
each side, so the first frame is centered on the first sample. Then
frames of 400 samples (25 ms) are cut starting every 160 samples
(10 ms), so each frame overlaps its neighbours by 60 percent; a sound
that falls across a cut is still whole in the frame next to it.
Finally a Hann window over each frame, a bell curve that fades the
edges to zero so the cut itself does not add energy the signal never
had.

Ten seconds of audio becomes about 1,000 frames. The 10.6 s test clip
yields 1,056. The padding is constant zeros, as in the reference; the
C++ port pads by reflection, and its own notes record a maximum mel
difference of 5.19 on one clip from that choice alone.

## 6. From frames to a spectrogram, then to mel bands

A frame is still 400 samples of pressure. What the model wants is
which frequencies are present and how strongly. Pressure over time is
what a microphone records; pitch is what a listener hears, and the
Fourier transform is the bridge between them. It asks, for a set of
evenly spaced frequencies, how much of the frame's energy rises and
falls at each one. A voice and a door slam in the same frame come out
in different places on that axis, which is how one set of numbers can
describe two concurrent sounds.

We place the 400 windowed samples in a 512-point transform (the extra
points are zeros, which lets a fast radix-2 algorithm run) and keep the
257 non-redundant outputs. Each output is a bin covering 31.25 Hz of
the range from 0 to 8,000 Hz. One power spectrum per frame, stacked,
is the spectrogram: 257 rows of frequency, one column every 10 ms.

Those bins are evenly spaced in hertz, and hearing is not. We tell 100
Hz from 200 Hz easily and 7,000 Hz from 7,100 Hz hardly at all;
perceived pitch grows roughly with the logarithm of frequency above
about 1 kHz. The mel scale is a unit of pitch built on that
observation, and a mel filterbank is a set of triangular filters laid
out evenly on it: narrow and dense at low frequencies, wide and sparse
at high ones. Canary uses 128 of them between 0 and 8,000 Hz.
Multiplying the 257-bin spectrum by the filterbank gives 128 numbers
per frame, the energy in each band as an ear would weigh it.

Two steps finish the front end. A natural logarithm with a small floor,
so silence does not become minus infinity and the range is compressed
the way loudness is perceived. Then per-band normalization over the
clip: subtract each band's mean, divide by its standard deviation. The
result is the log-mel spectrogram the model was trained on: 128
roughly zero-centered values per frame.

This stage is pure Rust in double precision, since ggml has no
short-time Fourier transform. It is deterministic whatever the thread
count, which is what makes the comparison in section 11 meaningful.

## 7. The encoder

The encoder turns 1,056 frames of 128 values into a shorter sequence of
1,024-dimensional vectors that describe what was said, without yet
deciding which words. It is a FastConformer, and the name describes its
two parts.

A Conformer block is a Transformer block with a convolution over time,
9 frames wide, between its attention and its feed-forward. Attention
lets every frame look at every other frame, which captures context
such as which word a syllable belongs to; the convolution captures the
local pattern that tells one phoneme from the next. Canary stacks 32
blocks.

The Fast part is the front. Attention cost grows with the square of the
sequence length, and 1,056 frames through 32 blocks would be slow on a
CPU. So the 128-band mel is treated as a one-channel image and
subsampled eight times in time before the first block: a 2D convolution
with stride 2, then twice a depthwise convolution with stride 2
followed by a pointwise one, then a linear layer to 1,024. Each stride
halves the frame count: 1,056 becomes 528, 264, 132. Each of the 132
vectors covers 80 ms of audio.

The output of the last block is the memory the decoder attends to: 132
rows of 1,024 numbers for the test clip. One encoder pass peaks at
1.23 GB of resident memory on the Q8_0 weights and 2.10 GB on F16.

## 8. The decoder and its prompt

The decoder is the part that looks familiar: an 8-layer pre-norm
Transformer with an output head over 16,384 SentencePiece tokens, plus
one thing an LLM does not have. In every layer, between self-attention
and the feed-forward, a cross-attention sublayer attends to the
encoder's 132 vectors. Self-attention looks at the tokens written so
far; cross-attention looks at the audio.

Canary is a prompted model, but the prompt is not words. Before any
text it reads a fixed header of ten token ids that says what to do:
start of transcript, source language, target language (the same one,
for transcription), punctuation on, no timestamps, and so on. The ids
come from the GGUF metadata, and a language the file does not list is
refused before any graph exists. This is why the daemon's default
instruction prompt would have been wrong for this model.

The loop is the one every LLM runtime knows. The keys and values over
the encoder output do not change during a request, so each layer
projects them once. The prompt goes through one prefill graph; then a
fixed-shape step graph is reused per token. Decoding is greedy. On 8
CPU threads a step costs 12.1 ms on Q8_0 and 16.4 ms on F16; the test
clip's 33 tokens take 0.45 s and 0.61 s.

## 9. Text comes out

The decoder emits SentencePiece ids. The worker drops the specials,
maps each id to its piece through the vocabulary in the GGUF, joins
the pieces and turns the word marker into a space. Punctuation and
capitalization come from the model, because the prompt asked for them.

End to end on an 8-CPU host with the Q8_0 file, the 10.6 s test clip
answers in 1.7 s through the worker and 3.6 s through the daemon on a
cold load.

## 10. The worker around it

The pipeline above is one request. The worker makes it safe to expose.

Limits apply before any audio is decoded: at most eight requests
admitted at a time, and a 4 MiB cap on the file part, which bounds the
audio to about 131 s. One request computes at a time, behind a lock:
the graphs use every thread they are given, so two at once would only
interleave, and the lock keeps memory to one encoder graph and one
decoder session.

Audio longer than 30 s is split into equal clips of at most 30 s,
transcribed in turn and joined with spaces: a 31 s file becomes two
clips of 15.5 s rather than a 30 s clip and a stub. Clips do not
overlap yet, so a word cut by a boundary may come out wrong.

Cancellation follows the client. When the client disconnects, an abort
flag is set, and three places read it: ggml's abort callback, which
stops a running graph within about 40 ms; the decoder loop, before
every token; and the mel front end, every 32 frames (320 ms of audio).
The last check came from a measurement: a request abandoned during the
front end held the lock for the 0.93 s of mel still to compute. With
the check in place, the next request takes the lock between 46 ms and
339 ms after the disconnect, in a debug build.

## 11. Checked against the reference

A speech model is forgiving: a port with a wrong padding choice still
produces the right words on a clean clip and fails quietly on the next
one, so reading the transcript tells us little. The acceptance test for
every stage was a numerical comparison with activations dumped from
the NeMo reference on one 10.6 s English clip, on both published
weight files.

| Stage | Compared | F16 | Q8_0 |
|---|---|---|---|
| Mel front end | max absolute difference | 1.29e-4 | 1.29e-4 |
| Encoder blocks and output | relative rms error per stage | 4e-4 to 1.2e-3 | 3.8e-3 to 3.8e-2 |
| Decoder sublayers and logits | relative rms error | 7.7e-5 to 8.3e-4 | 1.1e-3 to 2.1e-2 |
| Prompt ids, greedy ids, text | exact match | 10/10, 33/33, equal | 10/10, 33/33, equal |

The front end has no weights, so its numbers are the same for both
files. The weight format sets the floor for everything after it: on
F16 the encoder's relative error sits around the half-precision
epsilon of 4.9e-4, because the CPU matrix multiplication rounds
activations to half precision inside; on Q8_0 it is an order of
magnitude larger, from quantization. Relative error is the number to
watch; the exact token match says whether anyone would notice.

## Wrap up

A WAV upload now travels through a pipeline llmman owns, from samples
to frames, mel bands, encoder vectors and tokens, with a worker that
admits, serialises, clips and cancels around it. Each stage was checked
against the reference before the next one started.

Everything above runs on the CPU backend. Every op the graphs use is
supported on Vulkan, but nothing has been run on a GPU yet. There is no
streaming, no overlap between clips and no timestamps; beam search and
translation are out of scope for now. Parakeet shares the front end
and the encoder, so a second speech model means a second header value
and its decoder.

## References

The work is tracked in
[llmman#713](https://github.com/llmmanorg/llmman/issues/713). The model
is described on its [model card](https://huggingface.co/nvidia/canary-1b-v2),
and its reference implementation lives in [NeMo](https://github.com/NVIDIA/NeMo).
The compute library is [ggml](https://github.com/ggml-org/ggml), shared
with [llama.cpp](https://github.com/ggml-org/llama.cpp). Questions are
welcome at [github.com/llmmanorg/llmman](https://github.com/llmmanorg/llmman).
