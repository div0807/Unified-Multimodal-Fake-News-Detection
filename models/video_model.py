import torch.nn as nn
import torchvision.models as models


class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base = nn.Sequential(*list(m.children())[:-1])
        # Freeze all ResNet layers
        for p in self.base.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

    def forward(self, x):
        x = self.base(x).view(x.size(0), -1)
        return self.proj(x)