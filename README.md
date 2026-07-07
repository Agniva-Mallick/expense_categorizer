# 💸 Expense Categorizer Forge

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A custom, GPT-style Large Language Model (LLM) built entirely from scratch in PyTorch to solve a real-world Fintech problem: **Categorizing messy, unstructured bank transactions.**

Instead of relying on massive cloud APIs like OpenAI or Gemini, this project proves that a highly specialized, tiny LLM (only ~22M parameters) can achieve incredible accuracy on narrow classification tasks while running locally at lightning speed.

## 🚀 Features

- **Built from Scratch:** No HuggingFace `transformers` library used. The entire architecture is written in pure PyTorch.
- **Modern Transformer Architecture:** 
  - **RoPE** (Rotary Positional Embeddings)
  - **RMSNorm** (Root Mean Square Normalization)
  - **SwiGLU** Activation Functions
  - **GQA** (Grouped Query Attention)
- **Custom Tokenizer:** A bespoke Byte-Pair Encoding (BPE) tokenizer trained specifically on financial strings.
- **Synthetic Data Pipeline:** Includes a procedural generator that creates 100,000+ realistic, messy bank transactions (e.g., `AMZN Mktp US*TO382` or conversational phrases like `i spend 100 on dental clinic`) across 8 financial categories.
- **Lightning Fast:** Designed with a small context window (64 tokens) optimized for short transaction strings, allowing for rapid training on consumer GPUs (RTX 30 series GPU).

## 🧠 How it Works

Unlike standard regex or keyword-matching algorithms, this model actually learns the underlying patterns of spending. It was trained to read a transaction string and autoregressively output the correct category token.

**Categories Supported:** `Food`, `Transportation`, `Utilities`, `Entertainment`, `Shopping`, `Income`, `Housing`, `Health`.

## 💻 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy tqdm
```

### 2. Train the Model (Optional)
The project includes a synthetic data generator that will automatically build a dataset and train the model from scratch.
```bash
python train.py --preset medium
```
*Note: On an RTX 3050, the `medium` preset takes about 30-45 minutes to train 10,000 steps.*

### 3. Run Inference (CLI)
Once trained, you can test the model directly in your terminal using the `generate.py` script.

**Example 1: Raw Bank Code**
```bash
python generate.py --txn "TST* SWEETGREEN - NYC"
# Output: Food
```

**Example 2: Conversational Phrase**
```bash
python generate.py --txn "paid 100 for dental clinic"
# Output: Health
```

## 📁 Project Structure

```text
expense-categorizer-forge/
├── config.py          # Dataclass configs (Model, Train, Gen, Tokenizer)
├── dataset.py         # Synthetic transaction generator & PyTorch DataLoader
├── generate.py        # CLI Inference script (Top-K/Top-P sampling)
├── model.py           # The GPT-style Transformer architecture
├── tokenizer.py       # Custom BPE Tokenizer built from scratch
└── train.py           # The training loop with Mixed Precision (AMP)
```

## 🎓 What I Learned
Building this model from the ground up solidified my understanding of:
- The inner workings of modern Attention mechanisms and Rotary Embeddings.
- How to write and train a custom BPE tokenizer without relying on external libraries.
- Managing Out-Of-Distribution (OOD) data failures by injecting conversational noise into synthetic datasets.
- Optimizing PyTorch training loops with Automatic Mixed Precision (AMP) and Gradient Clipping.

---
*Built as a portfolio project to explore the intersection of custom AI architectures and Fintech.*
