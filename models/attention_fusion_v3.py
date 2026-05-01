import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Pairwise cross-modal attention: query attends to key/value."""
    def __init__(self, dim: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query, context: (B, 256) → unsqueeze → (B, 1, 256)
        q = query.unsqueeze(1)
        c = context.unsqueeze(1)
        out, _ = self.attn(q, c, c)
        return self.norm(q + out).squeeze(1)   # (B, 256)


class FusionModelV3(nn.Module):
    """
    Three-modality cross-modal attention fusion.
    Text ↔ Video ↔ Audio — each modality attends to both others,
    then all three attended features are concatenated → classifier.
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        dim = 256

        # Text attends to video and audio
        self.t2v = CrossModalAttention(dim, heads=4, dropout=0.1)
        self.t2a = CrossModalAttention(dim, heads=4, dropout=0.1)

        # Video attends to text and audio
        self.v2t = CrossModalAttention(dim, heads=4, dropout=0.1)
        self.v2a = CrossModalAttention(dim, heads=4, dropout=0.1)

        # Audio attends to text and video
        self.a2t = CrossModalAttention(dim, heads=4, dropout=0.1)
        self.a2v = CrossModalAttention(dim, heads=4, dropout=0.1)

        # Fuse attended features per modality
        self.text_fuse  = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())
        self.video_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())
        self.audio_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU())

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, t: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Each input: (B, 256)

        # Cross-attended features
        t_from_v = self.t2v(t, v)
        t_from_a = self.t2a(t, a)
        v_from_t = self.v2t(v, t)
        v_from_a = self.v2a(v, a)
        a_from_t = self.a2t(a, t)
        a_from_v = self.a2v(a, v)

        # Fuse per-modality cross-attended features
        t_rich = self.text_fuse(torch.cat([t_from_v, t_from_a], dim=1))   # (B, 256)
        v_rich = self.video_fuse(torch.cat([v_from_t, v_from_a], dim=1))  # (B, 256)
        a_rich = self.audio_fuse(torch.cat([a_from_t, a_from_v], dim=1))  # (B, 256)

        # Concatenate all three → (B, 768) → classifier
        x = torch.cat([t_rich, v_rich, a_rich], dim=1)
        return self.classifier(x)