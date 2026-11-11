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

Files are capped at 4 MiB, about 131 s of audio. Anything longer than
30 s is split into equal clips and the texts are joined, since Canary
was trained on utterances up to 40 s.

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
   on the first sample. NeMo pads with zeros too, and transcribe.cpp mirrors
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
the front end groups neighbouring bins into 128 *mel bands*, frequency
slices narrow at low pitches and wide at high ones, and blends each
band's energy into one number. Going from 257 numbers per frame to 128
loses detail that we barely hear. A logarithm, which tracks
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
them into one new frame. To do that, it multiplies each number in the group by one
of the model's weights and adds up the results. Each group starts two
frames after the previous one, so the groups overlap. Frames 1 to 3
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
learned weights. All 32 have the same steps and differ only in their
weights. Each block keeps the vectors' count and size, and updates every
vector three ways:

- *Attention:* each vector looks at all 132 vectors and draws on the
  ones that matter to it, such as the rest of its word or sentence.
  Because every vector looks at every other one, the work grows with the
  square of their number, so shrinking 1,056 frames to 132 vectors makes
  this step about 64 times cheaper.
- *Convolution:* the same kind of operation as in shrinking, over
  groups of 9 vectors (one vector and its 4 neighbours on each side,
  about 0.7 s of audio) that start one vector apart, so all 132 stay.
  It catches where one phoneme ends and the next begins.
- *Feed-forward:* a small network, a short chain of calculations with
  its own weights, refines each vector on its own, using what attention
  and convolution gathered.

> Attention plus feed-forward is a *Transformer* block, the same block
> inside every LLM. Adding the convolution makes it a *Conformer*
> block, and *FastConformer* is NVIDIA's version with the 8x shrink in
> front.

## Decoder: from audio vectors to text

![The decoder writes one token per step. The prompt of 10 ids and the words written so far sit in a row; for the next token, self-attention looks back at the tokens so far and cross-attention looks at the 132 audio vectors that cover the whole clip. The decoder picks the most likely token out of 16,384, appends it and repeats until end of text](decoder.svg)

The decoder is a Transformer like the one in a chat model. It writes
the transcript one *token* at a time, where a token is a word or a
piece of a word from a vocabulary of 16,384. Each of its 8 layers has
two kinds of attention:

- *Self-attention* looks at the tokens written so far, so the text
  stays coherent.
- *Cross-attention* looks at the 132 audio vectors the encoder produced,
  so the text follows the audio.

Before writing any word, the decoder reads a prompt of ten special
tokens from the GGUF metadata. Canary learned them in training as
instructions for the task, and they sit at the start of the token
sequence as if the decoder had written them, so self-attention sees
them at every step:

1. `▁`: the context slot, left empty.
2. `<|startofcontext|>`: the context marker. With the slot empty, there
   is no earlier text to continue.
3. `<|startoftranscript|>`: the transcript begins.
4. `<|emo:undefined|>`: no emotion tag.
5. `<|en|>`: the source language, here English.
6. `<|en|>`: the target language, the same for transcription.
7. `<|pnc|>`: punctuation and capitals on.
8. `<|noitn|>`: numbers stay as spoken words, such as "twenty".
9. `<|notimestamp|>`: no timestamps.
10. `<|nodiarize|>`: no speaker labels.

The first real word becomes the 11th token, after the 10 above. Each
step takes the most likely next token, until the model emits end of
text or a length cap is reached.

Last, the worker maps each token id to its piece of text through the
GGUF vocabulary and joins the pieces. The prompt asked for
punctuation and capitals, so they are already there. The 1.7 s
covers the worker alone; through the daemon, loading the model
first, the clip answers in 3.6 s.

## Checked against NeMo

A correct transcript for one clip proves little about the port. A small
mistake in one stage, such as padding the samples differently from
NeMo, can leave that transcript unchanged and still give wrong text on
another recording. So we compared the output of every stage with
NeMo's, on one English clip and with both GGUF versions of Canary,
Q8_0 and F16.

We ran the clip through NeMo and saved every stage's *activations*,
the numbers one stage hands to the next. They are the mel spectrogram,
the
1,024-value vectors after the shrink and after several encoder blocks,
and the decoder's score for every possible next token. Then we ran the
same clip through llmman and compared the two sets of numbers position
by position.

For each stage, the gap between llmman's numbers and NeMo's is a
percentage of a typical NeMo number. For instance, 0.1% means the two
agree to about one part in a thousand.

![Gap between llmman and NeMo per stage, as a percentage of a typical value, on a scale from 0.001 to 10 percent. Encoder F16 0.04 to 0.12 percent, Encoder Q8_0 0.38 to 3.8 percent, Decoder F16 0.008 to 0.08 percent, Decoder Q8_0 0.11 to 2.1 percent. A dashed line marks the half-precision floor of about 0.05 percent: the F16 encoder sits on it, and Q8_0 is about ten times higher](nemo-check.svg)

| Stage | F16 | Q8_0 |
|---|---|---|
| Mel spectrogram | at most 0.013% | at most 0.013% |
| Encoder | 0.04% to 0.12% | 0.38% to 3.8% |
| Decoder | 0.008% to 0.08% | 0.11% to 2.1% |
| Prompt, output tokens and text | identical | identical |

Q8_0 and F16 are the two published versions of Canary, with 8-bit and
16-bit weights. The front end uses no weights, so it gives the same
numbers with both. Every step from the encoder on multiplies by
weights, which F16 stores in *half precision*, a 16-bit format with
about three significant digits. That keeps its encoder near 0.05% from
NeMo, the closest the format allows, while Q8_0 keeps less detail and
lands about ten times further away. Both still produce NeMo's 33
tokens and text.

## Next

This first version transcribes audio files on the CPU. Next, we plan
to:

- **Run on GPUs:** make llmman transcribe on a GPU through Vulkan. The
  ggml library already has every operation Canary needs there, and we
  still have to test it.
- **Transcribe live audio:** transcribe while someone is still
  speaking. Today llmman waits for the whole file.
- **Add timestamps:** return the time at which each word was spoken,
  next to the text.
- **Handle long recordings better:** overlap the 30 s clips, so a word
  that falls on a cut comes out right.
- **Improve decoding:** try several candidates at each step, a method
  called beam search, for a slightly better transcript. Today each step
  keeps only the most likely token.
- **Translate:** let llmman ask Canary to translate speech into another
  language. Today it only asks it to transcribe.
- **Support Parakeet:** NVIDIA's other speech model shares Canary's
  front end and encoder, so llmman mostly needs to add its decoder.

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
