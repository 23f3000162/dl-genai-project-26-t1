# dl-genai-project-26-t1

# Messy Mashup — Music Genre Classification

> Fine-tuned Audio Spectrogram Transformer for robust music genre classification under noisy, cross-song mashup conditions.

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.88%20Macro%20F1-brightgreen)](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![WandB](https://img.shields.io/badge/Tracked-W%26B-orange)](https://wandb.ai)

---

## Project Overview

This project was built for the **Messy Mashup Kaggle Competition** as part of the IIT Madras Jan 2026 Deep Learning and Gen AI course.

The task is to classify audio into one of **10 music genres** from noisy mashups where:
- Stems from **different songs** of the same genre are mixed together
- **Tempo adjustments** are applied before mixing
- **ESC-50 background noise** is added at random positions and levels

This makes the problem much harder than standard genre classification because the model must understand the overall character of a genre, not just memorise specific songs.

---

## Final Results

| Model | Type | Val F1 | Kaggle F1 |
|-------|------|--------|-----------|
| LightGBM | Classical ML | ~0.45 | — |
| XGBoost | Classical ML | ~0.48 | — |
| CNN (scratch) | Deep Learning | ~0.55 | — |
| CRNN (CNN+BiLSTM) | Deep Learning | ~0.70 | — |
| **AST Transformer** | **Pretrained** | **0.8596** | **0.88** |

---

##  Project Structure

```
messy-mashup/
│
├── notebooks/
│   ├── milestone1_eda.ipynb          # Exploratory Data Analysis
│   ├── milestone2_classical_ml.ipynb # LightGBM + XGBoost
│   ├── milestone3_cnn.ipynb          # CNN from scratch
│   ├── milestone4_crnn.ipynb         # CRNN (CNN + BiLSTM)
│   ├── milestone5_ast.ipynb          # AST Transformer fine-tuning
│   └── inference.ipynb               # Final inference + submission
│
├── report/
│   └── final_report.pdf              # Full project report
│
└── README.md
```

---

##  Models Built

### Milestone 1 — EDA
- Genre distribution analysis
- Waveform and Mel spectrogram visualization
- Stem-level analysis (drums, vocals, bass, other)
- ESC-50 noise dataset exploration
- All logged to Weights and Biases

### Milestone 2 — Classical ML
- Extracted 177-dimensional feature vectors using librosa
- Features: MFCC (80) + Chroma (24) + Spectral Contrast (7) + Mel Mean (64) + ZCR + RMS
- Trained LightGBM and XGBoost classifiers
- Compared both models on Macro F1

### Milestone 3 — CNN from Scratch
- Input: Mel spectrogram (128 bins, hop=512)
- Architecture: 4 Conv blocks → Global Average Pooling → Classifier
- Augmentation: SpecAugment + cross-song stem mixing + noise injection
- Training: AdamW + CosineAnnealingLR + label smoothing

### Milestone 4 — CRNN (CNN + BiLSTM)
- CNN extracts local spectral features
- Bidirectional LSTM captures temporal musical patterns
- 2-layer BiLSTM with 256 hidden units per direction
- Clear improvement over pure CNN for music

### Milestone 5 — AST Transformer
- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Pretrained on AudioSet (527 classes)
- Two-phase fine-tuning strategy:
  - **Phase 1**: Freeze base, train classifier head only (3 epochs, LR=3e-4)
  - **Phase 2**: Unfreeze all, full fine-tune with layer-wise LR (5 epochs)
- Layer-wise LR: base=2e-5, classifier=2e-4
- Gradient accumulation (effective batch size = 12)
- TTA at inference: 5 random crops averaged

---

##  Key Techniques Used

### Data Augmentation
- Cross-song stem mixing (matches test distribution exactly)
- Tempo stretching ±12% with 40% probability
- Random gain scaling (0.7 to 1.3)
- ESC-50 noise injection at 3–20% level (70% probability)
- SpecAugment: random frequency and time masking

### Training Tricks
- Two-phase training for stable transformer fine-tuning
- Layer-wise learning rates
- Cosine warmup scheduler
- Label smoothing = 0.1
- Gradient clipping at norm 1.0
- Early stopping with patience

### Inference
- Test Time Augmentation (TTA) with 5 random crops
- Softmax probability averaging before argmax

---

##  Experiment Tracking

All experiments tracked with **Weights and Biases**:
- Project: `23f3000162-t12026`
- Metrics logged: `train_loss`, `train_f1`, `val_f1`, `lr` per epoch
- All 5 milestones compared in one dashboard

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/23f3000162/messy-mashup.git
cd messy-mashup
```

### 2. Install dependencies
```bash
pip install torch transformers librosa lightgbm xgboost \
            torchmetrics wandb scikit-learn pandas numpy tqdm
```

### 3. Set up dataset
```
Place dataset at: /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/
```

### 4. Run milestones in order
```
Open each notebook in Kaggle and run all cells
Commit to GitHub after each milestone
```

### 5. Run inference
```
Upload best_model_phase2.pth as Kaggle dataset
Open inference.ipynb and run all cells
Download submission.csv and submit to Kaggle
```

---

## Dependencies

```
torch >= 2.0
transformers >= 4.30
librosa >= 0.10
lightgbm >= 4.0
xgboost >= 1.7
torchmetrics >= 1.0
wandb >= 0.15
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
tqdm >= 4.65
```

---

##  Why AST Works Best

The Audio Spectrogram Transformer was pretrained on **AudioSet**, which contains over 2 million audio clips from YouTube covering 527 audio event categories. This means the model already understands a huge variety of sounds before fine-tuning even starts. When you fine-tune it on music genres, it adapts very quickly because it already knows what instruments, rhythms, and audio patterns look like in a spectrogram.

A CNN trained from scratch on just 1,000 songs has to learn all of this from nothing, which is why it scores much lower.

---

##  Score Improvement Journey

```
LightGBM        →  0.45  (handcrafted features, no deep learning)
XGBoost         →  0.48  (same features, different model)
CNN             →  0.55  (learns from spectrograms directly)
CRNN            →  0.70  (adds temporal understanding via LSTM)
AST Transformer →  0.88  (pretrained on millions of audio clips)
```

---

##  What Could Improve the Score Further

- **Ensemble 3 AST models** with different seeds → expected +3 to 5%
- **Longer audio duration** (30s instead of 20s) → more musical context
- **More synthetic training samples** (10,000 per epoch)
- **Gradio/Streamlit deployment** for live demo
- **W&B Sweeps** for hyperparameter search

---

##  References

- [AST Paper — Gong et al. 2021](https://arxiv.org/abs/2104.01778)
- [ESC-50 Dataset — Piczak 2015](https://github.com/karolpiczak/ESC-50)
- [SpecAugment — Park et al. 2019](https://arxiv.org/abs/1904.08779)
- [HuggingFace AST Model Card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Weights and Biases](https://docs.wandb.ai)
- [LightGBM Paper — Ke et al. 2017](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)

---

##  Author

**Anshu Sharma**
Student ID: 23f3000162
IIT Madras — Jan 2026 Deep Learning and Gen AI Project

---

##  License

This project is for academic purposes as part of the IIT Madras curriculum.
