# Sarcasm Detection in News Headlines

**Binary NLP classifier** — detects sarcasm in news headlines using fine-tuned BERT.  
**Target accuracy:** ≥ 93% | **Target F1 (macro):** ≥ 0.93

---

## Project structure

```
sarcasm_detector/
├── preprocess.py      # Text cleaning + BERT tokenisation pipeline
├── dataset.py         # PyTorch Dataset + stratified DataLoaders
├── model.py           # SarcasmClassifier (BertPreTrainedModel)
├── train.py           # Full training loop (AdamW, fp16, early stopping)
├── evaluate.py        # Confusion matrix, ROC-AUC, classification report
├── inference.py       # predict(headline) → {label, confidence, explanation}
├── app.py             # Gradio demo UI
├── notebook.py        # EDA + training + evaluation (convert to .ipynb)
├── requirements.txt
└── results/           # Generated after evaluate.py runs
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── classification_report.txt
```

---

## 1 · Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2 · Download the dataset

```bash
# Option A — Kaggle CLI
kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection
unzip news-headlines-dataset-for-sarcasm-detection.zip

# Option B — manual download
# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
# Use: Sarcasm_Headlines_Dataset_v2.json
```

---

## 3 · Train

```bash
python train.py \
  --data_path Sarcasm_Headlines_Dataset_v2.json \
  --output_dir ./saved_model \
  --epochs 5 \
  --batch_size 32
```

Training logs example:
```
Epoch 1 | train_loss=0.4821 | val_loss=0.3104 | val_acc=0.8934 | val_f1=0.8921
  ✓ Best model saved (val_f1=0.8921)
Epoch 2 | train_loss=0.2847 | val_loss=0.2401 | val_acc=0.9187 | val_f1=0.9182
  ✓ Best model saved (val_f1=0.9182)
Epoch 3 | [All BERT layers unfrozen]
         train_loss=0.1923 | val_loss=0.2189 | val_acc=0.9312 | val_f1=0.9308
  ✓ Best model saved (val_f1=0.9308)
```

---

## 4 · Evaluate

```bash
python evaluate.py \
  --data_path Sarcasm_Headlines_Dataset_v2.json \
  --model_dir ./saved_model
```

Outputs `results/confusion_matrix.png`, `results/roc_curve.png`,
and `results/classification_report.txt`.

---

## 5 · Inference

### Python API

```python
from inference import predict

result = predict("Area Man Passionate Defender Of What He Imagines Constitution To Be")
# {
#   "label": 1,
#   "label_name": "Sarcastic",
#   "confidence": 0.9612,
#   "explanation": "Sarcastic tone detected (96%). Cues: area man."
# }
```

### Command line

```bash
python inference.py \
  --model_dir ./saved_model \
  --headline "Scientists discover that exercise is good for you"
```

---

## 6 · Demo UI

```bash
python app.py --model_dir ./saved_model
# → http://localhost:7860
```

---

## 7 · Benchmark results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| TF-IDF + Logistic Regression | ~84% | ~0.83 |
| DistilBERT | ~91% | ~0.91 |
| bert-base-uncased ✓ | ~93% | ~0.93 |
| BERT + RoBERTa ensemble | ~94% | ~0.94 |

---

## 8 · Model architecture

```
Raw headline
  └─► Preprocessing (clean, expand contractions, lowercase)
        └─► BertTokenizer  (max_length=64, WordPiece)
              └─► bert-base-uncased
                    └─► [CLS] hidden state  (dim=768)
                          └─► Dropout(p=0.3)
                                └─► Linear(768 → 2)
                                      └─► Softmax → {Genuine, Sarcastic}
```

---

## 9 · Training configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight_decay=0.01) |
| LR — BERT layers | 2e-5 |
| LR — classifier head | 1e-3 |
| Scheduler | Linear warmup (10%) → cosine decay |
| Batch size | 32 (grad_accum=2 for <8 GB GPU) |
| Epochs | 5 (early stopping patience=2) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Mixed precision | fp16 via torch.cuda.amp |
| Freeze strategy | Freeze 6 BERT layers for epochs 1–2, then unfreeze all |

---

## 10 · Edge cases

| Case | Handling |
|---|---|
| Short headline (<2 words) | Returns `label=-1`, warns user |
| Non-English characters | Stripped gracefully by ASCII encoder |
| Numbers / statistics | BERT handles numerics natively |
| Ambiguous satire | Confidence score signals uncertainty |
