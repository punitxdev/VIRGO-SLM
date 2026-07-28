# ⚡ Virgo Inference & Sampling Engine

## Overview

The Virgo inference engine is optimized for low-latency generation on standard hardware using Key-Value (KV) Caching and customizable sampling controls.

---

## 🚀 Key-Value (KV) Caching

During autoregressive generation, computing key $K$ and value $V$ matrices for previous tokens at every step introduces quadratic $\mathcal{O}(N^2)$ computation. Virgo caches key and value states per layer:

$$K_{\text{cached}} = [K_{\text{past}} \,\,;\,\, K_{\text{new}}], \quad V_{\text{cached}} = [V_{\text{past}} \,\,;\,\, V_{\text{new}}]$$

This reduces token generation step complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.

---

## 🎛️ Sampling Control Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `temperature` | float | `0.0` | Logit scaling parameter. `0.0` triggers greedy decoding. |
| `top_k` | int | `1` | Restricts logit pool to Top-K probable tokens. |
| `top_p` | float | `1.0` | Nucleus sampling: truncates cumulative probability mass at `top_p`. |
| `repetition_penalty` | float | `1.10` | Penalizes logits of previously generated tokens to prevent repetition. |
| `max_tokens` | int | `128` | Upper bound limit for generated token length. |

---

## 🐍 Basic Generation Example

```python
from examples.inference import generate_text

prompt = "Explain quantum superposition in simple terms."
output = generate_text(prompt, max_tokens=128, temperature=0.7, top_p=0.9)
print(output)
```
