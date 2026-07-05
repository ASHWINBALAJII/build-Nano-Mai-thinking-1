"""Full model — MAI-Thinking-1 §2.1.

Target: token embedding (tied to output head), stack of blocks with 5 local : 1
global attention interleave and dense/MoE FFN alternation (first FFN dense),
final RMSNorm, LM head. Scaled init: std 0.02, output projections scaled by
1/sqrt(2 * n_layers).

TODO(you): implement the Transformer here.
"""
