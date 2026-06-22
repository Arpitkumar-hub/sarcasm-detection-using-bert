"""
inference.py — Single-headline sarcasm prediction.

API:
    from inference import predict

    result = predict("Scientists discover water is still wet")
    # {
    #   "label": 0,
    #   "confidence": 0.97,
    #   "explanation": "Straightforward factual claim. No irony detected."
    # }

Command-line usage:
    python inference.py --model_dir ./saved_model \
                        --headline "Area man passionate defender of constitution he's never read"
"""

import argparse
import json
import torch
import torch.nn.functional as F
from preprocess import preprocess, clean_text
from model import SarcasmClassifier

# ── Globals (lazy-loaded) ──────────────────────────────────────────────────
_model  = None
_device = None
MODEL_DIR = "./saved_model"

LABEL_MAP = {0: "Genuine", 1: "Sarcastic"}

# ── Explanation templates ──────────────────────────────────────────────────

SARCASTIC_CUES = [
    "irony", "obviously", "shocking", "area man", "nation", "local",
    "report", "experts", "somehow", "just", "turns out",
]

GENUINE_CUES = [
    "study", "data", "says", "announced", "president", "government",
    "million", "billion", "percent", "new", "plan", "report",
]


def _generate_explanation(headline: str, label: int, confidence: float) -> str:
    """Heuristic explanation based on label, confidence, and keyword cues."""
    h = headline.lower()

    if label == 1:
        matched = [c for c in SARCASTIC_CUES if c in h]
        if matched:
            return (
                f"Sarcastic tone detected (confidence {confidence:.0%}). "
                f"Cues: {', '.join(matched[:3])}."
            )
        return (
            f"Ironic or satirical framing detected (confidence {confidence:.0%}). "
            "Typical of satirical news sources like The Onion."
        )
    else:
        matched = [c for c in GENUINE_CUES if c in h]
        if matched:
            return (
                f"Straightforward factual claim (confidence {confidence:.0%}). "
                f"Cues: {', '.join(matched[:3])}."
            )
        return f"No irony detected (confidence {confidence:.0%}). Genuine headline."


# ── Model loader ───────────────────────────────────────────────────────────

def _load_model(model_dir: str = MODEL_DIR):
    global _model, _device
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = SarcasmClassifier.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()
    print(f"Model loaded from '{model_dir}' on {_device}.")


# ── Core predict function ──────────────────────────────────────────────────

def predict(headline: str, model_dir: str = MODEL_DIR) -> dict:
    """
    Predict whether a news headline is sarcastic.

    Args:
        headline  : raw headline string
        model_dir : path to saved model (via save_pretrained)

    Returns:
        {
            "label"      : 0 (genuine) or 1 (sarcastic),
            "label_name" : "Genuine" or "Sarcastic",
            "confidence" : float in [0, 1],
            "explanation": str
        }
    """
    _load_model(model_dir)

    # ── Edge case: empty / non-English ────────────────────────────────────
    cleaned = clean_text(headline)
    if not cleaned or len(cleaned.split()) < 2:
        return {
            "label":       -1,
            "label_name":  "Unknown",
            "confidence":  0.0,
            "explanation": "Headline too short or unrecognisable. Cannot classify.",
        }

    # ── Tokenise ──────────────────────────────────────────────────────────
    enc = preprocess(headline)
    input_ids      = enc["input_ids"].unsqueeze(0).to(_device)
    attention_mask = enc["attention_mask"].unsqueeze(0).to(_device)
    token_type_ids = enc["token_type_ids"].unsqueeze(0).to(_device)

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs    = _model(input_ids, attention_mask, token_type_ids)
        probs      = F.softmax(outputs.logits, dim=-1).squeeze(0)
        label      = int(probs.argmax().item())
        confidence = float(probs[label].item())

    explanation = _generate_explanation(headline, label, confidence)

    return {
        "label":       label,
        "label_name":  LABEL_MAP[label],
        "confidence":  round(confidence, 4),
        "explanation": explanation,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline",  required=True, help="News headline to classify")
    parser.add_argument("--model_dir", default=MODEL_DIR)
    args = parser.parse_args()

    result = predict(args.headline, args.model_dir)
    print(json.dumps(result, indent=2))
