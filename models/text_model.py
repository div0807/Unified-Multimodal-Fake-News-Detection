import torch.nn as nn
from transformers import BertModel


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze all except last 2 encoder layers
        for name, p in self.bert.named_parameters():
            if not any(f"encoder.layer.{i}" in name for i in [10, 11]):
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

    def forward(self, i, m):
        out = self.bert(input_ids=i, attention_mask=m)
        return self.proj(out.last_hidden_state[:, 0, :])