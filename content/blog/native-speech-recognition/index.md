+++
title = "llmman now transcribes speech natively"
description = "A Canary GGUF posted to llmman serve comes back as text. The whole pipeline, from the WAV bytes to the tokens, runs in llmman on the same ggml library as its image and video models, and matches the reference token for token."
date = 2026-11-09

[taxonomies]
tags = ["serve", "speech", "ggml"]
+++


A 10.56 s WAV goes in, and 1.7 seconds later one line of text comes out.

Until now, llmman sent transcription to llama-server, which handles
speech by feeding audio into an existing chat model. NVIDIA's Canary 1B
v2 has its own decoder, trained only to write transcripts, and
llama.cpp has no code for it. llmman now runs Canary itself, on the ggml library it already uses for image
and video models.

<!-- more -->

## Try it

```sh
llmman serve
```

```sh
curl -s http://127.0.0.1:17434/v1/audio/transcriptions \
  -F model=hf.co/handy-computer/canary-1b-v2-gguf:Q8_0 \
  -F file=@clip.wav
{"text":"..."}
```

`llmman serve` speaks the OpenAI audio API, and the first request pulls
the model file, a GGUF, from Hugging Face in its Q8_0 (8-bit) or F16 (16-bit) version.
A request takes four fields:

- `file`: a 16-bit mono WAV at 16 kHz, the format Canary was trained on.
- `model`: the model reference.
- `language`: `en` by default.
- `response_format`: `json`, the only format.

Anything else is refused, and so are chat, completions and embeddings.
`llmman run` does not transcribe yet.

## Why a pipeline of its own

When a GGUF's architecture field reads Canary, the daemon starts a
separate speech process for it, the worker.

A model is its weights, the learned numbers in the GGUF, plus the code
that runs them. NVIDIA's code lives in NeMo, its Python toolkit for
speech, so running Canary anywhere else means a port.

A C++ port, [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
exists, but it bundles its own patched ggml, which would put two copies
of the library in one process. Every operation Canary needs
is already in the ggml llmman loads, so llmman's port is Rust on that
library, like its LTX-2.3 and Qwen-Image pipelines. The front end is
plain Rust, because ggml has no Fourier transform.

![The speech pipeline: clip.wav goes through the WAV parser as 16 kHz samples, framing cuts 1,056 frames, the mel bands step produces a 1,056 by 128 log-mel spectrogram, the Conformer encoder shrinks it eightfold to 132 vectors of 1,024, the decoder reads them through cross-attention together with a ten-id task prompt and emits 33 token ids, and the detokenizer turns them into text](pipeline.svg)

The pipeline runs four stages, in order:

1. The front end turns samples into mel bands.
2. The encoder turns mel bands into audio vectors.
3. The decoder turns those vectors into token ids.
4. A table lookup turns the ids into text.

> The numbers that follow come from one 10.56 s test clip. Frames,
> vectors and tokens grow with the length of the audio, up to a 30 s
> clip.

## Front end: from samples to mel bands

![The front end: 1, samples, one pressure reading every 62.5 microseconds; 2, cut into overlapping 25 ms frames, a new one every 10 ms; 3, each frame becomes 257 evenly spaced frequency bins, merged into 128 mel bands, narrow at low pitch and wide at high pitch; 4, the frames side by side form the log-mel spectrogram, one column per frame and one row per band](front-end.svg)

A WAV holds *samples*: one pressure reading every 62.5 µs. The model
works on *frames*, short slices over which a vowel or a
consonant is roughly stable. The recipe comes from NeMo and from the
GGUF's front-end metadata:

1. **Pre-emphasis:** subtract 0.97 times the previous sample, which
   lifts the quieter consonants.
2. **Padding:** 256 zeros on each side, so the first frame is centred
   on the first sample. NeMo pads with zeros too; transcribe.cpp mirrors
   the edge samples, and its notes record a visible change in the mel
   bands from that choice alone.
3. **Framing:** 400 samples (25 ms) every 160 (10 ms). Frames overlap,
   so a sound cut at one edge is whole in the next.
4. **Hann window:** fade each frame's edges to zero, so the cut adds no
   energy of its own.
5. **Power spectrum:** a 512-point Fourier transform splits 0 to 8 kHz
   into 257 *bins*, which are slices 31.25 Hz wide, and measures the
   frame's energy in each.

A man's and a woman's voice sit a few bins apart and sound clearly
different, yet a gap that size in the hiss of an "s" goes unheard. So
the front end groups neighbouring bins into 128 *mel bands*, frequency slices narrow
at low pitches and wide at high ones, and blends each band's energy
into one number. Each frame's 257 numbers become 128, and
the detail lost is detail we barely hear. A logarithm, which tracks
loudness the way we hear it, and a rescaling of each band to an
average of 0 and a spread of 1 over the clip finish the front end.

What the front end hands on is a *log-mel spectrogram*: for the test
clip, 1,056 frames of 128 mel bands each, one frame every 10 ms.

## Encoder: from mel bands to audio vectors

![The encoder. Shrink: the 1,056 by 128 log-mel spectrogram is halved in time three times, to 528, 264 and 132 steps, then projected to 132 vectors of 1,024 values. Refine: 32 blocks each update every vector three ways: attention looks at all 132 vectors, convolution mixes each vector with 4 neighbours on each side, about 0.7 seconds of audio, and feed-forward reworks each vector on its own](encoder.svg)

The encoder turns the spectrogram into a short sequence of vectors, one
per 80 ms of audio, each describing what was said around that moment.
It first shrinks the sequence and then refines it.

The first phase, **shrinking**, passes the frames through three
*convolutions*, one after another, each with its own weights.

A convolution reads a group of three neighbouring frames and combines
them into one new frame: it multiplies each number in the group by one
of the model's weights and adds up the results. Each group starts two
frames after the previous one, so the groups overlap: frames 1 to 3
make the first new frame, frames 3 to 5 the second, frames 5 to 7 the
third. Starting a new group every two frames gives half as many frames
out as went in.

The first convolution turns 1,056 frames into 528, the second 528 into
264, and the third 264 into 132; each also halves the 128 bands, down
to 16. A last step, a *projection*,
multiplies each of the 132 results by another set of weights to give a
vector of 1,024 values.

The second phase, **refining**, runs 32 *blocks* over the 132 vectors, one
after another. A block is a fixed sequence of steps with its own
learned weights; all 32 have the same steps and differ only in their
weights. Each block keeps the vectors' count and size, and updates every
vector three ways:

- *Attention:* the vector looks at all 132 and pulls in the ones that
  matter to it, such as the rest of its word or sentence. Its cost grows
  with the square of the vector count, so the 8x shrink makes it about
  64x cheaper.
- *Convolution:* the vector is mixed with its 4 neighbours on each
  side, 9 in all, about 0.7 s of audio. This catches local sound
  patterns, such as where one phoneme ends and the next begins.
- *Feed-forward:* a small network reworks each vector on its own,
  using what the other two gathered.

Attention plus feed-forward is a *Transformer* block, the same block
inside every LLM. Adding the convolution makes it a *Conformer* block,
and *FastConformer* is NVIDIA's version with the 8x shrink in front.
One encoder pass peaks at 1.23 GB of memory on Q8_0 and 2.10 GB on F16.

## Decoder: from audio vectors to text

![The decoder writes one token per step. The prompt of 10 ids and the words written so far sit in a row; for the next token, self-attention looks back at the tokens so far and cross-attention looks at the 132 audio vectors that cover the whole clip. The decoder picks the most likely token out of 16,384, appends it and repeats until end of text](decoder.svg)

The decoder is a Transformer like the one in a chat model. It writes
the transcript one *token* at a time, where a token is a word or a
piece of a word from a vocabulary of 16,384. Each of its 8 layers has
two kinds of attention:

- *Self-attention* looks at the tokens written so far, so the text
  stays coherent.
- *Cross-attention* looks at the encoder's 132 vectors, so the text
  follows the audio.

It starts from a prompt of ten token ids from the GGUF metadata: start
of transcript, source and target language (the same, for
transcription), punctuation on, timestamps off, and a few more. A
language the file does not list is refused before any computation
starts.

Each step takes the most likely next token, until
the model emits end of text or a length cap is reached. On 8 CPU
threads a step costs 12.1 ms on Q8_0 and 16.4 ms on F16. The test
clip's 33 tokens take 0.45 s and 0.61 s, prompt included.

Last, the worker maps each token id to its piece of text through the
GGUF vocabulary and joins the pieces. The prompt asked for
punctuation and capitals, so they are already there. The 1.7 s
covers the worker alone; through the daemon, loading the model
first, the clip answers in 3.6 s.

## Around the request

The worker applies its limits before any audio is decoded:

- At most eight requests are admitted at a time.
- The file is capped at 4 MiB, about 131 s of audio, and the WAV
  parser bounds every chunk by the bytes present, so a hostile header
  cannot make it allocate.
- One request computes at a time, behind a lock. The computation
  already uses every thread, and the lock holds memory to one encoder
  pass and one decoder session.

Audio longer than 30 s is split into equal clips, transcribed one by
one and joined. Canary was trained on utterances up to 40 s, so 30 s
clips stay within that, and a 31 s file becomes two clips of 15.5 s.

When the client disconnects, an abort flag is set, and three places
read it:

1. ggml's abort callback, which stops a running computation within
   milliseconds.
2. The decoder loop, before every token.
3. The mel front end, every 32 frames (320 ms of audio).

We added the third check after we measured a request abandoned during
the front end holding the lock for about 1.2 s of mel still to
compute, in a debug build. With the check, two debug-build runs
measured cancellation delays of 46 ms and 339 ms.

## Checked against NeMo

A port with a wrong padding choice can still write the right words on a
clean clip and fail quietly on the next one, so we compared every stage
with NeMo, on one English clip and both weight files.

We ran the clip through NeMo and saved every stage's *activations*,
the numbers one stage hands to the next: the mel spectrogram, the 1,024-value vectors after the shrink and after
several encoder blocks, and the decoder's score for every possible next
token. Then we ran the same clip through llmman and compared the two
sets of numbers position by position. Numbers like 1.29e-4 mean
1.29 × 10⁻⁴, or 0.000129. The comparison uses two measures:

- **Max absolute difference:** the largest gap between any pair.
- **Relative rms error:** the typical gap (the root mean square of the
  differences) divided by the typical size of NeMo's numbers. 1e-3
  means the two agree to about 0.1 percent.

![Typical gap between llmman and NeMo per stage, as relative rms error on a log scale from 0.001 to 10 percent. Encoder F16 0.04 to 0.12 percent, Encoder Q8_0 0.38 to 3.8 percent, Decoder F16 0.008 to 0.08 percent, Decoder Q8_0 0.11 to 2.1 percent. A dashed line marks the half-precision floor of about 0.05 percent: the F16 encoder sits on it, and Q8_0 is about ten times higher](nemo-check.svg)

| Stage | Compared | F16 | Q8_0 |
|---|---|---|---|
| Mel front end | max absolute difference | 1.29e-4 | 1.29e-4 |
| Encoder blocks and output | relative rms error per stage | 4e-4 to 1.2e-3 | 3.8e-3 to 3.8e-2 |
| Decoder layers and token scores | relative rms error | 7.7e-5 to 8.3e-4 | 1.1e-3 to 2.1e-2 |
| Prompt ids, output ids, text | exact match | 10/10, 33/33, equal | 10/10, 33/33, equal |

The front end has no weights, so both files match there. The F16 file
uses *half precision*, a 16-bit format with about three significant
digits, so each rounding can move a value by up to about 0.05%. The CPU
matrix multiplication rounds activations to half precision too, which
holds the F16 encoder near that 0.05% floor, while Q8_0's encoder error
is about ten times larger. Both files
still produce NeMo's 33 tokens and text.

## Next

- Only the CPU backend has run. Every operation llmman uses is
  supported on Vulkan, but nothing has run on a GPU.
- No streaming and no timestamps.
- No overlap between long-audio clips, so a word cut at a boundary may
  come out wrong.
- No beam search or translation.

Parakeet, NVIDIA's other speech model, shares the front end and the encoder, so a second speech model
means a second architecture value and its decoder.

## References

The work is tracked in the project's
[tracking issue](https://github.com/llmmanorg/llmman/issues/713). The model
is described on its [model card](https://huggingface.co/nvidia/canary-1b-v2),
and its reference implementation lives in [NeMo](https://github.com/NVIDIA/NeMo).
The compute library is [ggml](https://github.com/ggml-org/ggml), shared
with [llama.cpp](https://github.com/ggml-org/llama.cpp). The C++ port is
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), which also publishes the Canary GGUF
used here. Questions are
welcome at [github.com/llmmanorg/llmman](https://github.com/llmmanorg/llmman).
