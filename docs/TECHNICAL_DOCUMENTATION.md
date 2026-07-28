# 📚 Virgo Project - Technical Documentation & File Registry

This document provides a comprehensive technical overview and file registry for the **Virgo Small Language Model (SLM)** project repository, detailing all pipeline components, notebooks, trained model checkpoints, benchmark suites, instruction datasets, reports, and web application architecture.

---

## 📁 1. Notebooks Directory (`notebooks/`)

The `notebooks/` directory contains all core Jupyter notebooks responsible for dataset generation, tokenization, pre-training, instruction fine-tuning (IFT), reasoning dataset creation, and inference testing.

| File Name | Category | Description / Purpose |
| :--- | :--- | :--- |
| **`virgo-slm-v1.ipynb`** | Architecture & Pre-training | Main architecture definition and initial pre-training pipeline for the Virgo 110M parameter base model. |
| **`virgo-preprocessing.ipynb`** | Data Preprocessing | Raw text cleaning, deduplication, formatting, and preparation for model tokenization. |
| **`virgo-general-dataset-creation.ipynb`** | Dataset Pipeline | Builds the initial general pre-training dataset from diverse web and domain sources. |
| **`virgo-general-tokenize.ipynb`** | Tokenization | Converts general domain text into packed tokenized binary blobs (`virgo_chat_v1_tokenized.bin`). |
| **`virgo_genral_training.ipynb`** | Pre-training | Pre-training execution notebook for base general domain knowledge. |
| **`virgo-chat-dataset-creation-v1.ipynb`** | Dataset Pipeline | Generates conversational and instruction-response datasets for fine-tuning. |
| **`virgo-chat-dataset.ipynb`** | Dataset Pipeline | Secondary dataset processing notebook for multi-turn chat alignment. |
| **`virgo-chat-information-extraction.ipynb`**| Fine-Tuning Prep | Special dataset creation for extracting entities, dates, phone numbers, and emails. |
| **`virgo-chat-training.ipynb`** | SFT / Fine-Tuning | Supervised Fine-Tuning (SFT) pipeline for Virgo chat model checkpoints. |
| **`virgo-chat-v1-train.ipynb`** | SFT Training | Execution notebook for training initial Chat V1 checkpoints on H100/RTX GPUs. |
| **`virgo-instruction-fine-tune.ipynb`** | Instruct Tuning (IFT) | Instruction fine-tuning workflow producing `virgo_IFT_ep1.pt` and `virgo_IFT_ep2.pt`. |
| **`code-dataset-virgo-instruct.ipynb`** | Domain Fine-Tuning | Generates programming and code instruction datasets (Python, C++, JS, SQL). |
| **`virgo-v1-writing-dataset.ipynb`** | Domain Fine-Tuning | Fine-tuning dataset generation for creative writing, emails, summarization, and prose. |
| **`vrigo-v1-reasoning-dataset.ipynb`** | Domain Fine-Tuning | Dataset construction for mathematical, logical reasoning, and step-by-step chain-of-thought tasks. |
| **`virgo-inference.ipynb`** | Testing & Validation | Interactive notebook for loading checkpoints, KV-cache validation, and response generation. |
| **`virgo_base_final.pt`** | Model Artifact | Base pre-trained model checkpoint saved during notebook execution cycles. |
| **`virgo_chat_v1_tokenized.bin`** | Binary Dataset | Packed 32-bit integer token binary file for high-speed PyTorch DataLoader streaming. |

---

## 🤖 2. Trained Models Directory (`trained_models/`)

The `trained_models/` directory stores pre-trained base model weights, intermediate epoch checkpoints, GPU-specific fine-tuned checkpoints (H100, RTX), and instruction-tuned (IFT) models.

| File Name | Model Stage | Parameters / Context | Notes & Evaluation Summary |
| :--- | :--- | :--- | :--- |
| **`virgo_base_final.pt`** | Pre-trained Base | 110M Params / 1024 Context | Initial pre-trained base model prior to instruction tuning. |
| **`virgo_IFT_ep1.pt`** | Instruction Tuning | Epoch 1 IFT Checkpoint | First epoch of instruction fine-tuning on diverse prompt datasets. |
| **`virgo_IFT_ep2.pt`** | Instruction Tuning | Epoch 2 IFT Checkpoint | **Current Benchmark Model** (Achieved 50.8% overall accuracy on 100-q benchmark). |
| **`virgo_instruct.pt`** | Production Instruct | 110M Instruct Model | Standard instruction-tuned checkpoint integrated into the FastAPI backend. |
| **`virgo_instruct_v2.pt`** | Production Instruct V2| Enhanced Context / IFT | Extended context instruction model trained with expanded dataset distribution. |
| **`virgo_chat_best.pt`** | Chat SFT | Best Overall Chat Model | Top-performing overall chat checkpoint. |
| **`virgo_chat_best_h100.pt`** | Chat SFT (H100) | Epoch 1 H100 Cluster | Trained on NVIDIA H100 GPU cluster. |
| **`virgo_chat_best_h100_ep2.pt`**| Chat SFT (H100) | Epoch 2 H100 Cluster | High-throughput H100 epoch 2 checkpoint. |
| **`virgo_chat_best_h100_ep3.pt`**| Chat SFT (H100) | Epoch 3 H100 Cluster | Converged H100 epoch 3 checkpoint. |
| **`virgo_chat_best_rtx.pt`** | Chat SFT (RTX) | Best RTX 4090 Checkpoint | Fine-tuned on desktop workstation RTX GPU. |
| **`virgo_chat_best_rtx2.pt`** | Chat SFT (RTX) | RTX Epoch 2 Checkpoint | Intermediate RTX fine-tuning checkpoint. |
| **`virgo_chat_best_rtx3.pt`** | Chat SFT (RTX) | RTX Epoch 3 Checkpoint | Final converged RTX fine-tuning checkpoint. |
| **`495524322788.pdf`** | Model Spec Document | Technical PDF | Hardware allocation report and GPU training metrics log. |

---

## 📊 3. Benchmark Suite Directory (`benchmark/`)

The `benchmark/` folder contains the comparative evaluation engine and output artifacts evaluating `virgo_IFT_ep2` against `GPT-2`.

| File Name | Description |
| :--- | :--- |
| **`generate_benchmark_csv.py`** | Automated Python script executing 100 benchmark prompts across 20 categories with KV-cache optimization and multi-threading. |
| **`response.csv`** | Comprehensive CSV output containing exact prompts, Virgo generated responses, and GPT-2 baseline responses (`prompt,virgo_response,gpt_2_reponse`). |
| **`benchmark_comparison_results.json`** | Detailed JSON report with accuracy percentages per category, prompt timings, and overall scores (Virgo: 50.8%, GPT-2: 46.5%). |

---

## 📝 4. Instruction Datasets (`prompt_text/` & `datasets/`)

CSV datasets utilized for instruction fine-tuning (IFT) and task alignment across various domains.

| Directory / File | Domain / Focus | Key Details |
| :--- | :--- | :--- |
| **`prompt_text/virgo_coding_IFT.csv`** | Programming & Code | Multi-language code completion, debugging, and synthesis. |
| **`prompt_text/markdown.csv`** | Document Formatting | Markdown structure, tables, syntax highlighting, and headers. |
| **`prompt_text/reasoning.csv`** | Logic & Math | Step-by-step logic, sequence deduction, and word problems. |
| **`prompt_text/chat.csv`** | General Chat | Conversational multi-turn dialogues. |
| **`prompt_text/information_extraction.csv`**| Extraction | Named entity recognition, regex targets, phone, email, date extraction. |
| **`prompt_text/writing.csv`** | Creative & Formal | Emails, letters, poems, blog posts, and stories. |
| **`prompt_text/virgo_identity.csv`** | System Persona | Virgo model identity, creator info (Google DeepMind team inspiration), and guidelines. |
| **`prompt_text/exact_word_count.csv`** | Constrained Generation | Strict word count and sentence count constraints. |
| **`prompt_text/number_only.csv`** | Exact Output | Numeric and formula-only response validation. |
| **`prompt_text/one_sentence_ds.csv`** | Single Sentence | Short concise single-sentence answer generation. |
| **`prompt_text/one_word_ds.csv`** | Single Word | Exact one-word response tasks. |
| **`prompt_text/yes_or_no.csv`** | Binary Classification | Boolean logic and Yes/No direct answers. |
| **`datasets/virgo_chat_v1_tokenized.bin`**| Tokenized Stream | Binary token stream for base SFT. |
| **`datasets/virgo_general_dataset.zip`** | Pre-training Zip | Compressed general domain pre-training text archive. |

---

## 🌐 5. Web Application Architecture (`webapp/`)

FastAPI web application implementing the **Liquid Glassmorphism** frontend UI and high-speed PyTorch KV-Cache inference engine.

```
webapp/
├── main.py              # FastAPI server, REST endpoints (/api/chat, /api/models), dynamic checkpoint switching
├── model.py             # FastVirgoModel definition, Rotary Embeddings (RoPE), KV-Caching multi-head attention
├── inference.py         # Streaming/batch inference worker with top-k, top-p, temperature, and context memory
└── static/
    ├── index.html       # Single-page Liquid UI layout with prompt categories, model status, context toggle
    ├── style.css        # Vanilla CSS implementation of Liquid Glassmorphism design system
    ├── script.js        # Dynamic message renderer, code block copy button, context token counter
    └── logo_*.png       # Visual branding assets
```

---

## 📑 6. Technical Reports & Roadmaps (`report/`)

| File Name | Description |
| :--- | :--- |
| **`Virgo_120M_Final_Fine_Tuning_Plan.pdf`** | Comprehensive fine-tuning strategy and hyperparameter tuning plan. |
| **`Virgo_120M_Dataset_Roadmap_Current_to_300K.pdf`** | Dataset scaling plan expanding from current corpus to 300K curated instruction samples. |
| **`Virgo_Base_Pretraining_Notebook_Summary.pdf`** | Pre-training metrics, loss curves, and perplexity analysis for Virgo Base. |
| **`Virgo_Chat_Training_Summary.pdf`** | Summary of Chat SFT iterations on H100 and RTX 4090 hardware. |
| **`Virgo_General_Tokenization_Summary.pdf`** | Tokenizer vocabulary construction (45,000 vocab) and byte-pair encoding (BPE) performance. |
| **`final_report.pdf`** | Executive summary of Virgo 110M SLM development lifecycle. |
