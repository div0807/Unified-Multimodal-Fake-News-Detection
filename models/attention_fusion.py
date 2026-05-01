import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Text attends to video, video attends to text → richer fusion"""
    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.attn_t2v = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_v2t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, t, v):
        # t, v: (B, 256) → unsqueeze → (B, 1, 256)
        t = t.unsqueeze(1)
        v = v.unsqueeze(1)

        t_attn, _ = self.attn_t2v(t, v, v)   # text queries video
        v_attn, _ = self.attn_v2t(v, t, t)   # video queries text

        t_out = self.norm1(t + t_attn).squeeze(1)
        v_out = self.norm2(v + v_attn).squeeze(1)
        return t_out, v_out


class FusionModelV2(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.cross_attn = CrossModalAttention(dim=256, heads=4, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, t, v):
        t_attn, v_attn = self.cross_attn(t, v)
        x = torch.cat([t_attn, v_attn], dim=1)   # (B, 512)
        return self.classifier(x)