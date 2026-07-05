"""LatentMoE block — MAI-Thinking-1 §2.1.

Target: shared down-proj into compressed space -> route on the ORIGINAL rep ->
top-k SOFTMAX gating -> dropless grouped expert compute (SwiGLU) -> up-proj back.
Plus the global-batch load-balancing loss.

TODO(you): implement the LatentMoE module and load-balance loss here.
"""
