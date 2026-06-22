"""
train.py — Full training loop for the Sarcasm Detection project.

Features:
  • AdamW with differential learning rates (BERT vs head)
  • Linear warmup → cosine decay scheduler
  • Mixed-precision training (fp16) via torch.cuda.amp
  • Gradient accumulation (for small-GPU setups)
  • Early stopping on validation F1
  • Best-model checkpointing via save_pretrained()

Usage:
    python train.py --data_path Sarcasm_Headlines_Dataset_v2.json \
                    --output_dir ./saved_model \
                    --epochs 5 \
                    --batch_size 32
"""

import os
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from dataset import get_dataloaders
from model   import build_model, unfreeze_all


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, labels)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="macro"),
    }


# ── Training loop ──────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_path, batch_size=args.batch_size
    )

    # Model — freeze first 6 BERT layers for warm-up
    model = build_model(freeze_layers=6)
    model.to(device)

    # Differential LR: lower for BERT body, higher for classifier head
    bert_params = [p for n, p in model.named_parameters()
                   if "classifier" not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "classifier" in n]

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": 2e-5},
            {"params": head_params, "lr": 1e-3},
        ],
        weight_decay=0.01,
    )

    total_steps   = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps  = int(total_steps * 0.10)
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler        = GradScaler(enabled=(device.type == "cuda"))

    best_f1       = 0.0
    patience_left = args.patience
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # After epoch 2 unfreeze all layers
        if epoch == 3:
            unfreeze_all(model)
            print("All BERT layers unfrozen.")

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask, token_type_ids, labels)
                loss    = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        # Checkpoint best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(args.output_dir)
            print(f"  ✓ Best model saved (val_f1={best_f1:.4f})")
            patience_left = args.patience
        else:
            patience_left -= 1
            print(f"  No improvement. Patience left: {patience_left}")
            if patience_left == 0:
                print("Early stopping triggered.")
                break

    # Final evaluation on test set
    print("\n── Test set evaluation ──────────────────────────")
    from model import SarcasmClassifier
    best_model = SarcasmClassifier.from_pretrained(args.output_dir)
    best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    print(
        f"test_loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['accuracy']:.4f} | "
        f"test_f1={test_metrics['f1']:.4f}"
    )
    return test_metrics


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--output_dir", default="./saved_model")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--grad_accum", type=int,   default=2)
    parser.add_argument("--patience",   type=int,   default=2)
    args = parser.parse_args()
    train(args)
