# grow-space

A from-scratch, **nano-scale (~50–60M param)** reimplementation of the **pre-training stage**
of [MAI-Thinking-1](docs/mai-thinking-1.pdf) (Microsoft AI). Faithful to the paper's
architecture and training recipe; scaled down to run on a single Colab GPU.

## What stays faithful to the paper (§2.1 / §2.6)
- Decoder-only Transformer, RMSNorm at **input + output** of each sub-layer, no biases, tied embeddings.
- Interleaved attention **5 local : 1 global** — local = sliding-window (512) + RoPE (θ=10k),
  global = **NoPE** + full causal. GQA.
- Alternating **dense-FFN / LatentMoE** blocks (first FFN dense), SwiGLU.
- LatentMoE: shared down-proj → route on original rep → **top-k softmax**, dropless; global-batch load-balance loss.
- Zero-init attention output-norm gain (Fig. 8), AdamW recipe, cosine LR, dropout 0.15, scaled init.

## Scaled-down for a ~50–60M budget
- Small **~16k BPE** tokenizer (not o200k_base's 200k).
- Few experts (**~8, top-2**), not 512. Structure faithful, counts shrunk.
- Dataset: **`DKYoon/SlimPajama-6B`** (closest open match to MAI's deduped multi-domain mix).

## Layout
```
src/
  config.py          # model + training config, param-count check
  data.py            # tokenizer + SlimPajama streaming + packing
  train.py           # AdamW / cosine / grad-clip loop  (Colab CLI entry point)
  sample.py          # generate text to eyeball progress
  model/
    attention.py     # GQA + RoPE/NoPE + sliding window + norms
    moe.py           # LatentMoE + load-balance loss
    block.py         # one transformer block (attn + FFN/MoE)
    transformer.py   # full model, interleave + tied embeddings
configs/nano.yaml    # nano hyperparameters
scripts/             # data prep, tokenizer training
tests/               # shape / sanity tests
docs/                # the paper
```

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run on Colab GPU (from terminal / Antigravity)
```bash
colab auth                       # one-time browser login (Colab Pro account)
colab run --gpu T4 src/train.py  # provision GPU, run, pull artifacts, tear down
```
