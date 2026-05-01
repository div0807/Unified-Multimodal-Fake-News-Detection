import torch.nn as nn


class AudioModel(nn.Module):
    """
    Lightweight MLP that maps 40-dim MFCC features → 256-dim embedding.
    Kept simple intentionally: audio features are already compact (mean MFCCs),
    so a deep CNN would overfit on this small dataset.
    """
    def __init__(self, n_mfcc: int = 40):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mfcc, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 40)
        return self.proj(x)   # (B, 256)