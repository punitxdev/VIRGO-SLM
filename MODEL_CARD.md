# 🎴 Model Card for Virgo SLM V1.0

## Model Details

- **Model Name**: Virgo SLM V1.0
- **Model Type**: Small Language Model (SLM)
- **Architecture**: Decoder-only Transformer
- **Parameters**: 120 Million (120M)
- **Developer**: Punit Kumar Kashyap (Indian Institute of Technology Dharwad)
- **License**: MIT
- **Release Date**: July 2026
- **Repository**: [https://github.com/punitxdev/VIRGO-SLM](https://github.com/punitxdev/VIRGO-SLM)

---

## Official Model Releases (Kaggle Models)

| Model Variant | Description | Kaggle Hub Link |
| :--- | :--- | :--- |
| **Virgo Instruct** | Instruction-following model fine-tuned for prompt execution | [kaggle.com/.../virgo-instruct](https://www.kaggle.com/models/punitkashyap2007/virgo-instruct) |
| **Virgo Chat** | Multi-turn dialogue conversational alignment checkpoint | [kaggle.com/.../virgo-chat](https://www.kaggle.com/models/punitkashyap2007/virgo-chat) |
| **Virgo Align V1.0** | Preference and fine-grained alignment release | [kaggle.com/.../virgo-align-v1-0](https://www.kaggle.com/models/punitkashyap2007/virgo-align-v1-0) |
| **Virgo Base V1.0** | Raw pre-trained base model checkpoint (120M) | [kaggle.com/.../virgo-base-v1-0](https://www.kaggle.com/models/punitkashyap2007/virgo-base-v1-0) |

---

## Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Architecture** | Modern Decoder-only Transformer |
| **Parameters** | 120M (~120,458,752 non-embedding parameters) |
| **Layers (`n_layer`)** | 12 |
| **Attention Heads (`n_head`)** | 12 |
| **Embedding Dim (`d_model`)** | 768 |
| **Feed-Forward Dim (`d_ff`)** | 3,072 (SwiGLU activation) |
| **Vocabulary Size** | 45,000 (Byte-Level BPE) |
| **Context Length** | 1,024 Tokens |
| **Normalization** | RMSNorm (Pre-normalization) |
| **Position Embeddings** | RoPE (Rotary Position Embeddings) |
| **Training Objective** | Autoregressive Next-Token Prediction |

---

## Training Phases & Stages

1. **Virgo Base V1.0**: Pre-trained from scratch on general language and code domains to build initial language modeling capabilities.
2. **Virgo Chat**: Multi-turn dialogue fine-tuning stage for conversational alignment.
3. **Virgo Align V1.0**: Preference alignment pass for refined output consistency.
4. **Virgo Instruct**: Instruction-following post-training focused on task execution and prompt execution.

> [!NOTE]
> **Data Privacy Notice**: The training datasets, synthetic instruction pairs, and raw corpus used for training Virgo SLM V1.0 are unreleased and private. The datasets are not publicly released.

---

## Intended Use & Strengths

### Primary Intended Use
- **Educational & Research Exploration**: Understanding Small Language Model behaviors, memory constraints, and sampling dynamics.
- **Local Embedded Inference**: Deploying lightweight language model endpoints on standard hardware without GPU requirements.
- **Basic Code & Text Generation**: Generating algorithms, Markdown formatting, and concise text explanations.

### Key Strengths
- **Low Footprint**: Ultra-lightweight 120M parameter footprint running smoothly on CPU or single GPU.
- **Coding & Technical Concepts**: Strong performance on Python algorithmic tasks, SQL structure generation, and chemistry fundamentals.
- **Fast Generation**: High token-per-second throughput with low latency.

---

## Current Limitations & Out-of-Scope Use

### Known Limitations
- **Arithmetic & Mathematical Reasoning**: Limited precision on multi-digit mathematical calculations.
- **Strict Token Bounds**: Under tight generation limits (e.g. < 64 tokens), the model may enter repetitive loops if an `<eos>` token is not emitted early.
- **Hallucination Risk**: Like all small language models, Virgo may generate plausibly sounding but factually incorrect statements.

### Out-of-Scope Use
- Critical decision-making systems (medical diagnosis, financial trading, legal advice).
- Automated dissemination of unverified information or high-risk autonomous control systems.

---

## Ethical Considerations & Safety

Virgo SLM V1.0 has undergone basic alignment checks, but has not received extensive RLHF safety reinforcement. Deployers should implement prompt filtering and output sanitization appropriate for their application environment.
