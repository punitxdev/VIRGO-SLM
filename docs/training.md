# 🏋️ Virgo Training & Alignment Pipeline

## Overview

The training of **Virgo SLM V1.0** consists of a multi-stage process designed to build base language fluency followed by multi-turn conversation and task-specific instruction following.

---

## 🔁 Training Stages

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│      Virgo Base        │ ───► │       Virgo Chat       │ ───► │     Virgo Instruct     │
│ Autoregressive Pretrain│      │ Multi-turn Conversation│      │  Instruct Alignment    │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

### 1. Stage 1: Virgo Base Pre-Training
- **Objective**: Autoregressive cross-entropy loss over next-token prediction:

$$\mathcal{L}(\theta) = -\sum_{i=1}^N \log P(x_i \mid x_1, x_2, \dots, x_{i-1}; \theta)$$

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, weight decay = $0.1$).
- **Learning Rate Schedule**: Warmup cosine decay with peak $L_r = 6 \times 10^{-4}$ decaying to $6 \times 10^{-5}$.
- **Gradient Clipping**: Norm clipped at $1.0$.

### 2. Stage 2: Virgo Chat Alignment
- **Objective**: Masked cross-entropy loss applied only to assistant tokens in multi-turn dialogues.
- **Context Format**: `<|system|> ... <|user|> ... <|assistant|> ...`

### 3. Stage 3: Virgo Instruct Post-Training
- **Epoch 1 & Epoch 2**: Fine-tuning pass focused on Python code syntax, Markdown structure, and direct reasoning answers.

---

## 🔒 Data Privacy & Confidentiality

> [!IMPORTANT]
> All raw pre-training corpora, instruction datasets, and synthetic dataset pipelines are unreleased and private. The datasets are not publicly released.
