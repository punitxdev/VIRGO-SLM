# 🤝 Contributing to Virgo SLM

Welcome! We are thrilled that you are interested in contributing to **Virgo**.

Virgo is an educational, open-source project built from scratch to understand and advance modern Small Language Models (SLMs). Whether you are a **student**, **researcher**, **beginner in Machine Learning**, or an **experienced ML engineer**, your contributions, feedback, and ideas are warmly welcomed!

---

## 🌟 Open Source Vision

Everyone is welcome to improve Virgo! You don't need to be an AI veteran to contribute. Helpful contributions include:
- Fixing bugs or optimizing PyTorch tensor operations
- Adding unit tests or benchmark prompts
- Improving technical documentation or code comments
- Sharing ideas for fine-tuning or quantization

---

## 🚀 Step-by-Step Contribution Workflow

### 1. Fork the Repository
Click the **Fork** button at the top-right corner of the GitHub page to create your personal copy of the repository.

### 2. Clone Your Fork
```bash
git clone https://github.com/YOUR-USERNAME/Virgo.git
cd Virgo
```

### 3. Create a Feature Branch
Use a descriptive name for your branch:
```bash
git checkout -b feature/add-kv-cache-optimization
# or
git checkout -b fix/tokenizer-padding-bug
```

### 4. Set Up Your Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Make Your Changes
- Write clean, documented Python code following standard `PEP 8` guidelines.
- Keep pull requests focused on a single logical change.

### 6. Commit and Push
```bash
git add .
git commit -m "feat: add KV cache optimization in inference engine"
git push origin feature/add-kv-cache-optimization
```

### 7. Open a Pull Request (PR)
Go to the original `punitxdev/VIRGO-SLM` repository on GitHub and click **Compare & pull request**. Provide a summary of your changes and why they are useful.

---

## 📋 Code Style & Quality Guidelines

- **Python**: Follow PEP 8 formatting. Use clear, self-explanatory variable names (`d_model`, `n_layers`, `num_heads`).
- **Comments**: Explain non-obvious mathematical operations or tensor matrix shapes in code comments.
- **Dependencies**: Keep external dependencies minimal so Virgo remains lightweight.

---

## 🐛 Reporting Bugs & Requesting Features

- Search existing [GitHub Issues](https://github.com/punitxdev/VIRGO-SLM/issues) to avoid duplicates.
- Provide clear steps to reproduce bugs, including your OS version, Python version, and PyTorch version.

Thank you for being part of the Virgo community! Together, let's build accessible Small Language Models!
