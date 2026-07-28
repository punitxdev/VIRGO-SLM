# Changelog

All notable changes to the **Virgo SLM** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-28

### Added
- **Official Release**: Initial public release of **Virgo SLM V1.0** (120M Parameter Small Language Model).
- **Core Architecture**: Decoder-only Transformer implementation with RMSNorm, RoPE (Rotary Position Embeddings), and SwiGLU MLP activation.
- **Custom Tokenizer**: 45,000 Byte-Level BPE tokenizer trained for high-efficiency code and natural language tokenization.
- **Inference Engine**: Fast PyTorch inference engine with KV caching, dynamic temperature, Top-K, Top-P, and Repetition Penalty controls.
- **Web UI**: Modern FastAPI + Vanilla JS dark-mode web application interface (`webapp/`).
- **Benchmark Suite**: 100-question automated evaluation pipeline comparing Virgo vs GPT-2 baselines (`benchmark/`).
- **Documentation & Model Card**: Complete open-source technical specifications, Model Card, and user guidelines.

### Changed
- Standardized project directory layout into production open-source GitHub structure (`docs/`, `examples/`, `configs/`, `scripts/`).
