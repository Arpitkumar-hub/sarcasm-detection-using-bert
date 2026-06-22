"""
evaluate.py — Full evaluation of the saved model on the test set.
Produces:
  results/confusion_matrix.png
  results/roc_curve.png
  results/classification_report.txt

Usage:
    python evaluate.py --data_path Sarcasm_Headlines_Dataset_v2.json \
                       --model_dir ./saved_model
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)
import torch.nn.functional as F
from tqdm import tqdm

from dataset import get_dataloaders
from model   import SarcasmClassifier


def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            probs   = F.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds   = (probs >= 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_confusion_matrix(labels, preds, save_path: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Genuine", "Sarcastic"],
        yticklabels=["Genuine", "Sarcastic"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — sarcasm detection")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(labels, probs, save_path: str):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc         = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"BERT (AUC = {auc:.3f})", color="#534AB7", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — sarcasm detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {save_path}")


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    _, _, test_loader = get_dataloaders(args.data_path, batch_size=32)

    model = SarcasmClassifier.from_pretrained(args.model_dir)
    model.to(device)

    labels, preds, probs = collect_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, probs)
    report = classification_report(
        labels, preds, target_names=["Genuine", "Sarcastic"]
    )

    print("\n── Classification report ─────────────────────────")
    print(report)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # Save report
    with open("results/classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nAccuracy : {acc:.4f}\n")
        f.write(f"F1 macro : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc:.4f}\n")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, "results/confusion_matrix.png")
    plot_roc_curve(labels, probs, "results/roc_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--model_dir",  default="./saved_model")
    args = parser.parse_args()
    run_evaluation(args)
