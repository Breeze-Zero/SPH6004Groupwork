from collections import OrderedDict
from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.layers.create_act import get_act_layer
from timm.layers.create_norm import get_norm_layer


class NormClassifier(nn.Module):
    """ A Pool -> Norm -> Mlp Classifier Head for '2D' NCHW tensors
    """
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_size: Optional[int] = None,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            norm_layer: Union[str, Callable] = 'layernorm2d',
            act_layer: Union[str, Callable] = 'relu',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', linear_layer(in_features, hidden_size)),
                ('act', act_layer()),
            ]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes,hidden_features = None,**kwargs):
        """
        Args:
            in_features: 输入特征的维度
            num_classes: 类别数
        """
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features

        if hidden_features:
            self.classifiers = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_features, num_classes)
            )
        else:
            self.classifiers = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: 输入 tensor，形状为 [batch_size, in_features]
        Returns:
            logits: 输出 tensor，形状为 [batch_size, num_classes]
        """
        logits = self.classifiers(x)
        return logits

class SeparateClassifier(nn.Module):
    def __init__(self, in_features, num_classes,**kwargs):
        """
        Args:
            in_features: 输入特征的维度
            num_classes: 类别数
        """
        super(SeparateClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # 使用 ModuleList 为每个类别定义一个独立的线性分类器
        self.classifiers = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(num_classes)])
    
    def forward(self, x):
        """
        Args:
            x: 输入 tensor，形状为 [batch_size, in_features]
        Returns:
            logits: 输出 tensor，形状为 [batch_size, num_classes]
        """
        # 对每个类别分别计算对应的预测分数
        logits = [classifier(x) for classifier in self.classifiers]
        # 将每个类别的预测分数拼接，得到最终的 logits
        logits = torch.cat(logits, dim=1)
        return logits

if __name__ == "__main__":
    # 假设输入特征维度为 512，类别数为 10
    net = SeparateClassifier(in_features=512, num_classes=10)
    x = torch.randn(8, 512)  # 模拟一个 batch 大小为 8 的输入
    logits = net(x)
    print("输出形状:", logits.shape)  # 预期形状为 [8, 10]