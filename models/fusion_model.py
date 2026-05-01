import torch.nn as nn
import torch

class FusionModel(nn.Module):
 def __init__(self):
  super().__init__()
  self.fc = nn.Sequential(
   nn.Linear(256+256,256),
   nn.ReLU(),
   nn.Linear(256,2)
  )

 def forward(self, t, v):
  x = torch.cat((t,v), dim=1)
  return self.fc(x)