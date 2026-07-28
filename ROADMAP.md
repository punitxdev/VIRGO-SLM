# 🗺️ Virgo Project Roadmap

The development of **Virgo** is structured around iterative milestones, moving from foundational architecture implementation to specialized fine-tuning, efficiency optimizations, and multimodal extensions.

---

## 🎯 Milestone Status

- [x] **Virgo Base (120M)** — 120M Parameter decoder-only baseline trained from scratch.
- [x] **Virgo Chat** — Conversational fine-tuning pass for natural multi-turn dialogue.
- [x] **Virgo Instruct** — Instruction-following post-training (Epoch 1 & Epoch 2 alignment).
- [ ] **Virgo V2** — Architecture upgrades (FlashAttention, Grouped-Query Attention).
- [ ] **Larger Virgo Models** — Scaling up to 350M and 1B parameter variants.
- [ ] **Long Context Window** — Expanding context size from 1,024 to 8,192+ tokens via RoPE scaling.
- [ ] **Better Reasoning** — Dedicated mathematical and algorithmic synthetic data training passes.
- [ ] **Tool Use** — Function calling and API execution integration.
- [ ] **RAG (Retrieval-Augmented Generation)** — Embedded retrieval engine for external knowledge bases.
- [ ] **Virgo Multimodal** — Vision-language embedding alignment for multimodal inputs.

---

## 📅 Detailed Release Plan

```mermaid
gantt
    title Virgo SLM Development Roadmap
    dateFormat  YYYY-MM
    section Core SLM
    Virgo Base 120M           :done,    des1, 2026-01, 2026-06
    Virgo Instruct & Chat     :active,  des2, 2026-06, 2026-09
    section Scaling & Arch
    Virgo V2 (GQA & FlashAttn):         des3, 2026-09, 2026-12
    Long Context (8k+)        :         des4, 2026-11, 2027-02
    section Advanced Capabilities
    Tool Use & RAG Integration:         des5, 2027-01, 2027-05
    Multimodal Vision-Language:         des6, 2027-04, 2027-09
```

---

## 🤝 How to Influence the Roadmap

As an educational and open-source project, community input is essential! If you have suggestions for features, architecture tweaks, or benchmark suites:
- Open a feature request in [GitHub Issues](https://github.com/punitxdev/VIRGO-SLM/issues)
- Join the discussion in [Pull Requests](https://github.com/punitxdev/VIRGO-SLM/pulls)
