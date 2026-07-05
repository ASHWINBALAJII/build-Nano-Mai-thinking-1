"""Pre-training loop — MAI-Thinking-1 §2.6. Entry point for `colab run`.

TODO(you): AdamW(0.95, 0.925, eps 1e-8), param-group weight decay
(0.1 / 0.01 attn / 0.005 embed), grad-clip 1.0, cosine LR 2e-4 -> 2e-5 with
warmup, dropout, checkpointing, logging.
"""
