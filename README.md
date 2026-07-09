# build-Nano-Mai-thinking-1

A from-scratch, nano-scale reimplementation of the pre-training stage of Microsoft AI's **MAI-Thinking-1**, built to run on a single rented cloud GPU.

Faithful to the paper's architecture — GQA with local/global (5:1) attention, RoPE/NoPE, RMSNorm, SwiGLU FFN, and a LatentMoE block — but scaled down (small BPE vocab, few experts) to fit the compute budget.

**Stack:** PyTorch · SlimPajama-6B · single-file `src/train.py`. Work in progress.
