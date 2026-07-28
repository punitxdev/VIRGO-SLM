# 🔤 Virgo Tokenizer Specification

## Overview

Virgo utilizes a custom **45,000 Byte-Level Byte-Pair Encoding (BPE)** tokenizer trained specifically for multi-lingual natural language and code representation.

---

## ⚙️ Configuration & Special Tokens

| Token | ID | Description |
| :--- | :--- | :--- |
| `<pad>` | 0 | Padding token for batch alignment |
| `<unk>` | 1 | Unknown token placeholder |
| `<s>` | 2 | Start-of-sequence token |
| `</s>` / `<eos>` | 3 | End-of-sequence / stop token |

---

## 🎯 Design Features

1. **Byte-Level Encoding**: Handles arbitrary UTF-8 characters without incurring out-of-vocabulary (`<unk>`) token loss.
2. **Code Preserving**: Retains white-space indentation patterns (4-space tabs, single spaces, code blocks) critical for Python and Markdown execution.
3. **Compression Ratio**: Achieves an average compression ratio of ~3.6 characters per token on Python code and ~4.1 characters per token on English text.
