"""Transformer block — MAI-Thinking-1 §2.1.

Target: one block = Attention (local or global) + feed-forward (dense FFN or
LatentMoE). Dropout 0.15 at the sub-layer output before the residual add.

TODO(you): implement the Block (and dense SwiGLU FFN) here.
"""
