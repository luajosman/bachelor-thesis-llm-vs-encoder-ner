# 🧠 LLM vs Encoder for Named Entity Recognition  
### Comparative Study of Encoder-Based Transformers and Large Language Models

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![GPU](https://img.shields.io/badge/GPU-Required-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 📌 Abstract

This repository contains the implementation for a Bachelor thesis investigating whether modern Large Language Models (LLMs), fine-tuned with parameter-efficient techniques such as LoRA, can achieve comparable performance to classical encoder-based transformer models (e.g., BERT) on Named Entity Recognition (NER) tasks.

The study focuses on:

- Quantitative comparison (Precision, Recall, F1-score)
- Computational cost analysis
- Training efficiency
- Practical applicability in structured prediction tasks

---

# 🎯 Research Question

> Can fine-tuned Large Language Models achieve performance comparable to encoder-based transformer models (e.g., BERT) on structured Named Entity Recognition tasks?

### Sub-questions

- How does LoRA fine-tuning influence LLM performance?
- What is the trade-off between performance and computational cost?
- Are LLMs suitable for exact token-level labeling tasks?
- Where are encoder models still superior?

---

# 🧠 Model Architectures

---

## 1️⃣ Encoder-Based Model (BERT-style)

**Architecture Type:** Bidirectional Encoder Transformer  
**Objective:** Token Classification  

### Architecture Overview

```
Input Tokens
     ↓
Embedding Layer
     ↓
Transformer Encoder Blocks (xN)
     ↓
Token Classification Head
     ↓
Label per Token (BIO format)
```

### Characteristics

- Bidirectional context
- Optimized for structured prediction
- Deterministic output
- Efficient training
- Low formatting risk

---

## 2️⃣ Large Language Model (Decoder-only, e.g., Qwen / TinyLlama)

**Architecture Type:** Autoregressive Transformer  
**Objective:** Instruction-following Generation  

### Architecture Overview

```
Instruction + Input Text
           ↓
Embedding Layer
           ↓
Decoder Transformer Blocks (xN)
           ↓
Autoregressive Generation
           ↓
JSON Output with Entity Spans
```

### Characteristics

- Generative model
- Instruction-based learning
- LoRA fine-tuning (parameter-efficient)
- Requires output parsing
- Higher computational demand
- Potential formatting errors

---

# 🔧 Fine-Tuning Strategy

## Encoder Model

- Full fine-tuning
- Cross-entropy token classification loss
- BIO tagging scheme

## LLM

- LoRA (Low-Rank Adaptation)
- Base model frozen
- Only adapter weights updated
- Instruction-based prompt format

---

# 🗂 Dataset

The dataset consists of annotated texts with labeled entities.

### Example

Text:
```
Aspirin reduces fever.
```

Entities:
- Aspirin → Chemical
- fever → Disease

---

## Encoder Format (BIO Tagging)

```
Aspirin  B-CHEM
reduces  O
fever    B-DISEASE
```

---

## LLM Instruction Format

Prompt structure:

```
### Instruction:
Extract all entities from the text.

### Input:
Aspirin reduces fever.

### Output:
```

Expected JSON output:

```json
[
  {"start": 0, "end": 7, "label": "Chemical"},
  {"start": 16, "end": 21, "label": "Disease"}
]
```

---

# 📊 Evaluation Metrics

Both approaches are evaluated using:

- Precision
- Recall
- F1-Score

### Metric Definitions

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:

- TP = True Positives  
- FP = False Positives  
- FN = False Negatives  

---

# 📈 Comparison Dimensions

| Dimension | Encoder | LLM + LoRA |
|------------|----------|-------------|
| Task Alignment | Native | Prompt-based |
| Structured Output | Yes | Generated |
| Training Speed | Fast | Slower |
| GPU Memory | Low | High |
| Determinism | High | Medium |
| Formatting Errors | None | Possible |
| Span Precision | High | Sensitive |

---

# 🏗 Project Structure

```
llm-vs-encoder-ner/
│
├── data/
│   ├── train.json
│   ├── validation.json
│   └── test.json
│
├── scripts/
│   ├── 01_train_encoder.py
│   ├── 02_eval_encoder.py
│   ├── 10_train_llm_lora.py
│   ├── 11_eval_llm.py
│   └── 20_make_plots.py
│
├── runs/
│   ├── encoder_model/
│   └── llm_lora_model/
│
├── results/
│   ├── metrics_encoder.json
│   ├── metrics_llm.json
│   └── plots/
│
└── README.md
```

---

# ⚙️ Installation

Create environment:

```bash
conda create -n thesis python=3.11
conda activate thesis
```

Install dependencies:

```bash
pip install torch transformers datasets peft accelerate scikit-learn matplotlib
```

---

# 🚀 Running Experiments

## Train Encoder

```bash
python scripts/01_train_encoder.py
```

## Evaluate Encoder

```bash
python scripts/02_eval_encoder.py
```

## Train LLM with LoRA

```bash
python scripts/10_train_llm_lora.py
```

## Evaluate LLM

```bash
python scripts/11_eval_llm.py
```

---

# 💻 Hardware Requirements

| Model | Parameters | Approx. VRAM |
|--------|-------------|---------------|
| BERT-base | ~110M | 8GB |
| Qwen-1.5B | ~1.5B | 12–16GB |
| Qwen-7B | ~7B | 24GB+ |

LoRA significantly reduces trainable parameters and memory footprint.

---

# 🔬 Expected Findings

Encoder models are expected to:

- Perform strongly on token-level NER
- Show stable F1 performance
- Require less computational cost

LLMs may:

- Struggle with exact span boundaries
- Produce formatting inconsistencies
- Excel in semantic reasoning tasks

---

# 📚 Related Work

- Devlin et al., 2019 — BERT  
- Hu et al., 2021 — LoRA  
- Instruction Tuning literature  
- Comparative studies between encoder and decoder architectures  

---

# 📌 Reproducibility

- Fixed random seeds  
- Explicit dataset splits  
- Logged hyperparameters  
- Modular training & evaluation scripts  

---

# 👨‍🎓 Author

Luaj Osman  
Bachelor Thesis – Transformer Model Comparison Study  

---

# 📜 License

This project is developed for academic research purposes.
