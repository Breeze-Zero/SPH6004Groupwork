import torch
import torch.nn as nn
from models.emb_model import Mlp



class FusionStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        img_dim: int,
        text_dim: int,
        d_model: int = 512,
        num_heads: int = 8,
    ):
        """
        Args:
            num_layers: 堆叠的 block 数量
            img_dim:     输入图像特征维度
            text_dim:    输入文本特征维度
            d_model:     融合内部维度
            num_heads:   多头注意力头数
            fusion_dim:  每层输出的特征维度
        """
        super().__init__()
        self.img_proj = nn.Linear(img_dim, d_model,bias=False)
        self.text_proj = nn.Linear(text_dim, d_model,bias=False)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                ModalFusionModel(
                    img_dim=img_dim,
                    text_dim=text_dim,
                    d_model=d_model,
                    num_heads=num_heads,
                )
            )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        img_feat:  (batch, seq_len, img_dim)
        text_feat: (batch, seq_len, text_dim)
        returns:   (batch, seq_len, fusion_dim)
        """

        x_img = self.img_proj(img_feat)
        x_text = self.text_proj(text_feat)
        for block in self.blocks:
            x_img = block(x_img, x_text)  # (batch, seq_len, fusion_dim)

        return x_img


class ModalFusionModel(nn.Module):
    def __init__(self, img_dim, text_dim, d_model=256, num_heads=8):
        super(ModalFusionModel, self).__init__()

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, hidden_features=d_model*4, out_features=d_model, act_layer=nn.GELU)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, img_feat, text_feat):
        """
        input：
          img_feat：(batch, seq_len, img_dim)
          text_feat： (batch, seq_len, text_dim)
        return：
          (batch, seq_len, fusion_dim)
        """

        attn_out, _ = self.cross_attn(query=img_feat, key=text_feat, value=text_feat)

        fused = self.norm(img_feat + attn_out)
        fusion_feat = fused + self.mlp(fused)
        return self.norm1(fusion_feat)