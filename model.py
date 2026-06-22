"""
model.py — BERT-based sarcasm classifier.

Architecture:
    bert-base-uncased → [CLS] hidden state → Dropout(0.3) → Linear(768 → 2)

Usage:
    model = SarcasmClassifier()
    outputs = model(input_ids, attention_mask, token_type_ids)
    # outputs.logits → shape (batch, 2)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig


class SarcasmClassifier(BertPreTrainedModel):
    """
    Fine-tuned BERT classifier for binary sarcasm detection.
    Inherits from BertPreTrainedModel so save_pretrained() /
    from_pretrained() work out of the box.
    """

    def __init__(self, config: BertConfig, dropout: float = 0.3):
        super().__init__(config)
        self.bert    = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, 2)   # 2 classes
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.post_init()   # initialise weights, tie embeddings, etc.

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels:         torch.Tensor = None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]   # (B, 768)
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)           # (B, 2)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        # Return a simple namespace so callers can do outputs.logits / outputs.loss
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits, loss=loss)


def build_model(
    model_name: str = "bert-base-uncased",
    dropout: float = 0.3,
    freeze_layers: int = 6,
) -> SarcasmClassifier:
    """
    Load pretrained BERT weights and attach classifier head.

    Args:
        model_name    : HuggingFace model id
        dropout       : dropout probability before classifier
        freeze_layers : number of BERT encoder layers to freeze initially
                        (set to 0 to unfreeze all from the start)
    """
    model = SarcasmClassifier.from_pretrained(model_name, dropout=dropout)

    if freeze_layers > 0:
        # Freeze embeddings + first N encoder layers
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total:,} | trainable: {trainable:,}")
    return model


def unfreeze_all(model: SarcasmClassifier) -> None:
    """Unfreeze every parameter (call after warm-up phase if desired)."""
    for param in model.parameters():
        param.requires_grad = True
