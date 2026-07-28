# 🏗️ Virgo SLM Architecture Specification

## Overview

**Virgo SLM V1.0** is an autoregressive, decoder-only Transformer language model containing **120 Million parameters**. The model architecture is designed from first principles, incorporating state-of-the-art enhancements such as Rotary Position Embeddings (RoPE), Root Mean Square Layer Normalization (RMSNorm), and SwiGLU activation functions.

---

## 📐 Hyperparameters

```yaml
vocab_size: 45000       # Vocabulary size (Byte-Level BPE)
block_size: 1024        # Maximum context sequence length
n_layer: 12             # Number of Transformer decoder layers
n_head: 12              # Number of Attention heads
n_embd: 768             # Hidden embedding dimension
head_dim: 64            # Dimension per head (n_embd / n_head)
d_ff: 3072              # Intermediate SwiGLU feed-forward dimension
dropout: 0.0            # Training dropout rate
bias: false             # Disable bias tensors for linear projections
```

---

## 🧱 Architectural Components

### 1. Rotary Position Embeddings (RoPE)
Instead of absolute sinusoidal or learned positional embeddings, Virgo applies **Rotary Position Embeddings (RoPE)** to query and key representations. RoPE embeds relative position information directly into the attention mechanism through complex rotation matrices:

$$\mathbf{R}_{\Theta, m}^d = \mathrm{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$

### 2. Pre-Layer RMSNorm
Virgo uses **RMSNorm** (Root Mean Square Normalization) placed *before* attention and feed-forward blocks, providing numerical stability during deep training:

$$\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i, \quad \text{where } \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{d} \sum_{i=1}^d a_i^2 + \epsilon}$$

### 3. Causal Multi-Head Self-Attention
Attention queries $Q$, keys $K$, and values $V$ are computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

where $M$ is a causal lower-triangular mask preventing attention to future tokens.

### 4. SwiGLU Feed-Forward Network (FFN)
Replaces traditional ReLU/GELU MLPs with the SwiGLU gated activation:

$$\text{SwiGLU}(x) = \left(\text{Swish}(x W_g) \odot x W_u\right) W_d$$

where $\text{Swish}(z) = z \cdot \sigma(z)$.
