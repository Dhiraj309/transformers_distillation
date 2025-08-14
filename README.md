project_name: hf-distiller
license: mit
language: python
pretty_name: HF Distiller
tags:
- knowledge-distillation
- transformers
- huggingface
- pytorch
- student-model
---

# 🧪 HF Distiller — Knowledge Distillation for Hugging Face Models

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/huggingface-compatible-orange)](https://huggingface.co/)

**HF Distiller** is an **open-source toolkit** for performing **knowledge distillation** on Hugging Face Transformers models.  
It allows developers to **train smaller, faster student models** from large pre-trained teacher models while maintaining high performance.

---

## 📖 Overview

Knowledge Distillation (KD) compresses a large model into a smaller one by transferring the “knowledge” learned by the teacher to the student.  
HF Distiller wraps around Hugging Face’s `Trainer` to make KD **accessible, modular, and intuitive**.

**Key Features:**

- ✅ Load any teacher model from Hugging Face Hub
- ✅ Create smaller student models from scratch
- ✅ Supports Hugging Face tokenizers
- ✅ Seamless integration with the `datasets` library
- ✅ Transparent logging and checkpointing
- ✅ Fully compatible with PyTorch and Transformers

---
```
## 🖼 Architecture

    ┌─────────────────┐
    │   Teacher LM   │  Pretrained Hugging Face model
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Knowledge Distill│  KD loss + softened logits
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │   Student LM    │  Smaller model trained from scratch
    └─────────────────┘

```
## ⚡ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hf_distiller.git
cd hf_distiller

# Install dependencies
pip install -r requirements.txt
````

---

## 🏃 Quick Start

### Python Script Example

```python
from hf_distiller.models import load_teacher, load_student
from hf_distiller.trainer import DistillTrainer
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset

# Load teacher
teacher = load_teacher("google-bert/bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Create student model
student = load_student(
    model_name_or_path="google-bert/bert-base-uncased",
    from_scratch=True,
    n_layers=4,
    n_heads=4,
    n_embd=256,
    is_pretrained=False
)

# Sample dataset
dataset = Dataset.from_dict({"text": ["Hello world!", "AI is amazing."]})

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], max_length=128, padding=True, truncation=True)

tokenized = dataset.map(tokenize, remove_columns=["text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./student-llm",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    report_to="none"
)

# Train student with KD
trainer = DistillTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    training_args=training_args,
    kd_alpha=0.5,
    temperature=2.0,
    is_pretrained=False
)
trainer.train()
```

---

## 📂 Project Status

| Stage                | Status         |
| -------------------- | -------------- |
| Core Development     | ✅ Complete     |
| Documentation        | ✅ Complete     |
| Community Feedback   | 🚧 In Progress |
| Tutorials & Examples | 🚧 In Progress |

---

## 🤝 Collaboration

We welcome contributions from the community, including:

* Pull requests for new KD strategies
* Bug reports and feature requests
* Tutorials and example scripts
* Optimization for faster student training

🔗 **GitHub**: [yourusername/hf\_distiller](https://github.com/yourusername/hf_distiller)

---

## 📜 License

Released under the **MIT License** — free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

```

This version now mirrors the **professional HF dataset README style**:  
- Metadata block with project info  
- Badges for Python, HF, license  
- Sections for Overview, Architecture, Installation, Quick Start  
- Status table and collaboration info  

I can also create a **version with collapsible code blocks and visuals** to make it look **exactly like top-tier HF repo READMEs** if you want. Do you want me to do that?
```