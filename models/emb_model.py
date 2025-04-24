import torch
import torch.nn as nn
from torch.nn import functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1,**kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features = None,
        out_features = None,
        drop = 0.0,
        bias = True,
        **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)