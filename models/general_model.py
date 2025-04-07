import torch
import torch.nn as nn
from torch.nn import functional as F

class BaseModel(nn.Module):
    def __init__(self, model,num_features,head, outnorm = True,norm_layer=nn.LayerNorm,**kwargs): ##nn.BatchNorm1d
        super().__init__()
        self.model = model
        self.head = head
        if outnorm:
            self.norm_layer = norm_layer(num_features, eps=1e-6)
        else:
            self.norm_layer = nn.Identity()
        
    def forward(self, x):
        x = self.model(x)
        x = self.norm_layer(x)
        x = self.head(x)
        return x