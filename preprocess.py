"""
preprocess.py — Text cleaning and tokenization pipeline
for the Sarcasm Detection project.
"""

import re
import contractions
from transformers import BertTokenizer

# ── Constants ──────────────────────────────────────────────────────────────
MAX_LENGTH = 64
MODEL_NAME = "bert-base-uncased"

# ── Tokenizer (shared singleton) ───────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text: str) -> str:
    """Clean a raw headline string."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Expand contractions (don't → do not)
    text = contractions.fix(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove non-ASCII characters (graceful fallback for non-English)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Remove special characters — keep letters, digits, spaces, basic punct
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str, tokenizer=tokenizer, max_length: int = MAX_LENGTH) -> dict:
    """
    Tokenize a single cleaned headline.

    Returns a dict with:
        input_ids, attention_mask, token_type_ids  (all lists of ints)
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids":      encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "token_type_ids": encoding["token_type_ids"].squeeze(0),
    }


def preprocess(text: str) -> dict:
    """Full pipeline: clean → tokenize."""
    cleaned = clean_text(text)
    if not cleaned:
        # Return zero tensors for empty / non-English inputs
        cleaned = "[UNK]"
    return tokenize(cleaned)
