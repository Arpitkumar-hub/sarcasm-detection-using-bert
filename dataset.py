"""
dataset.py — PyTorch Dataset for the Sarcasm Detection project.
Loads the Kaggle "News Headlines Dataset for Sarcasm Detection"
(Misra & Arora, 2023) and returns tokenized tensors.

Dataset download:
    kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection
    unzip it → Sarcasm_Headlines_Dataset_v2.json
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess import clean_text, tokenize, tokenizer, MAX_LENGTH


# ── Dataset class ──────────────────────────────────────────────────────────

class SarcasmDataset(Dataset):
    def __init__(self, records: list[dict], max_length: int = MAX_LENGTH):
        """
        records: list of dicts with keys 'headline' and 'is_sarcastic'
        """
        self.records = records
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        text   = clean_text(row["headline"]) or "[UNK]"
        label  = int(row["is_sarcastic"])

        enc = tokenize(text, tokenizer=tokenizer, max_length=self.max_length)
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        return enc


# ── Data loading helpers ───────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    """Load the JSONL dataset file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_dataloaders(
    json_path: str,
    batch_size: int = 32,
    val_size:  float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Uses stratified splitting to preserve label balance.
    """
    records = load_json(json_path)

    labels = [r["is_sarcastic"] for r in records]

    # First split off the test set
    train_val, test_recs, lbl_tv, _ = train_test_split(
        records, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Then split train / val
    val_ratio = val_size / (1 - test_size)
    train_recs, val_recs = train_test_split(
        train_val, lbl_tv,
        test_size=val_ratio,
        stratify=lbl_tv,
        random_state=seed,
    )[::2]  # unpack only record lists (not label lists)

    # Re-split properly
    train_recs, val_recs = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=lbl_tv,
        random_state=seed,
    )

    train_ds = SarcasmDataset(train_recs)
    val_ds   = SarcasmDataset(val_recs)
    test_ds  = SarcasmDataset(test_recs)

    kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
