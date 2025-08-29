# Sentiment Analysis with BERT and ELECTRA

This repository contains a **sentiment analysis** project built on top of transformer architectures: **BERT** and **ELECTRA**. It includes reproducible notebooks for training and evaluation across two social-media datasets, along with a concise report of results.

> ✅ ELECTRA slightly outperformed BERT on our experiments and achieved perfect scores on Dataset-1. Performance dropped on the more challenging Dataset-2 (class imbalance), with ELECTRA still ahead overall.

---

## 📁 Repository Structure

```
.
├─ README.md                # Project documentation
├─ NLP_Final_Project_Report.pdf
├─ Bert.ipynb               # BERT on Dataset-1
├─ Electra.ipynb            # ELECTRA on Dataset-1
├─ BertDS_2.ipynb           # BERT on Dataset-2
└─ ElectraDS_2.ipynb        # ELECTRA on Dataset-2
```

---

## 🧠 Models

- **BERT**: `bert-base-uncased`
- **ELECTRA**: `google/electra-small-discriminator`

Both models are fine-tuned for single-sentence sentiment classification using a lightweight classification head.

### Key Training Details
- Tokenization: WordPiece (BERT), subword (ELECTRA)
- Sequence handling: lowercasing, cleaning, padding & truncation
- Loss: Cross-Entropy
- Optimizer: `AdamW` with LR scheduler
- Epochs: **5**
- Batch size: **16**

---

## 📊 Datasets

1) **Social Media Sentiments Analysis**  
   https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset

2) **Social Media Sentiment Analysis Dataset**  
   https://www.kaggle.com/datasets/abdullah0a/social-media-sentiment-analysis-dataset

> Dataset-2 is more challenging and class-imbalanced, which is reflected in the results.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- GPU recommended (CUDA)
- Core libraries: `transformers`, `torch`, `pandas`, `scikit-learn`, `numpy`, `tqdm`

```bash
pip install torch transformers pandas scikit-learn numpy tqdm
```

### Quickstart (Notebook)
1. Download/prepare the datasets (links above) into a local `data/` folder.
2. Open any notebook (e.g., `Bert.ipynb`) and set the dataset path cells.
3. Run all cells to train & evaluate the model.

### Suggested Project Script Layout (optional)
If you wish to convert the notebooks into scripts, a simple structure could be:
```
src/
  data.py          # dataset loading & preprocessing
  modeling.py      # model/optimizer/scheduler setup
  train.py         # training loop
  evaluate.py      # metrics & reports
  utils.py         # seed, logging, config
```
And run with:
```bash
python -m src.train --model bert-base-uncased --epochs 5 --batch_size 16 --dataset data/dataset1.csv
```

---

## 📈 Results Summary

### Dataset-1 (Social Media Sentiments Analysis)

| Model   | Best Val. Accuracy | Weighted F1 |
|---------|---------------------|-------------|
| BERT    | **0.9785**          | **0.9787**  |
| ELECTRA | **1.0000**          | **1.0000**  |

**Per-class (BERT)**  
- Negative: P=0.9286, R=1.0000, F1=0.9630  
- Positive: P=1.0000, R=0.9701, F1=0.9848

**Per-class (ELECTRA)**  
- Negative: P=1.0000, R=1.0000, F1=1.0000  
- Positive: P=1.0000, R=1.0000, F1=1.0000

---

### Dataset-2 (Social Media Sentiment Analysis Dataset)

| Model   | Best Val. Accuracy | Weighted F1 |
|---------|---------------------|-------------|
| BERT    | **0.3400**          | **0.3223**  |
| ELECTRA | **0.5101**          | **0.3446**  |

**Per-class (BERT)**  
- Negative: P=0.3833, R=0.4554, F1=0.4163  
- Positive: P=0.2000, R=0.1146, F1=0.1457  
- Neutral:  P=0.3600, R=0.4369, F1=0.3947

**Per-class (ELECTRA)**  
- Negative: P=0.5101, R=1.0000, F1=0.6756  
- Positive: P=0.0000, R=0.0000, F1=0.0000

> 🔎 Interpretation: Dataset-2 exhibits stronger class imbalance and higher difficulty; consider the remedies below.

---

## 🧪 Reproducing Our Setup

**Common Hyperparameters**
- epochs: `5`
- batch size: `16`
- optimizer: `AdamW`
- loss: `CrossEntropyLoss`
- learning-rate scheduling enabled

**Preprocessing**
- Lowercasing & text cleaning
- Tokenization: `BertTokenizer` / `ElectraTokenizer`
- Padding & truncation to a fixed max sequence length

---

## 🛠️ Tips for Better Performance on Dataset-2

- **Class-weighted loss** (e.g., `CrossEntropyLoss(weight=class_weights)`)
- **Resampling** (oversample minority classes / undersample majority)
- **Data augmentation** (back-translation, synonym replacement, noising)
- **Threshold tuning** for precision-recall trade-offs
- **Model scaling** (e.g., `electra-base`, `bert-base` → `bert-large`, if resources permit)

---

## 👥 Authors

- Md. Julfiqure Islam Antor — *CSE, AIUB*  
- Rafiul Hasan Shafin — *CSE, AIUB*  
- A. M. Rafinul Huq — *CSE, AIUB*

---

## 📄 Reference

For a detailed write-up of methods and results, see **`NLP_Final_Project_Report.pdf`** in this repository.

---

## 🛡️ License

MIT License — see [LICENSE](LICENSE).


---

## 📅 Project Timeline

- **February** — Project planning, dataset collection, initial literature review on BERT & ELECTRA.  
- **March** — Preprocessing pipeline implemented, tokenization strategy finalized, baseline experiments with BERT.  
- **April** — Training and fine-tuning BERT across Dataset-1 and Dataset-2, evaluation metrics integrated.  
- **May** — Experiments with ELECTRA, comparison against BERT, hyperparameter tuning.  
- **June** — Error analysis, addressing class imbalance in Dataset-2, testing augmentation strategies.  
- **July** — Final experiments, report writing, preparation of documentation and project submission.  

---
