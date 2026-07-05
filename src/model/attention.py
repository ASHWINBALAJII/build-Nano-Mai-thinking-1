"""Attention block — MAI-Thinking-1 §2.1.

Target: GQA + (RoPE θ=10k on LOCAL / NoPE on GLOBAL) + sliding-window(512) vs
full-causal, RMSNorm at input AND output, output-norm gain zero-init (Fig. 8),
no biases.

TODO(you): implement RMSNorm, RoPE helpers, and the Attention module here.
"""
