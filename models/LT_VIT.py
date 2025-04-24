import torch
import torch.nn as nn
import torch.nn.functional as F


# class LabelGuidedTransformerBlock(nn.Module):
#     """
#     Implements
#       (1) [X_Q, X_K, X_V; Y_Q, Y_K, Y_V] = [X; Y]·W_QKV
#       (2) [X̃; Ỹ] = FFN( [ attn(X_Q,X_K,X_V);  attn(Y_Q, [X_K;Y_K], [X_V;Y_V]) ] )
#     """

#     def __init__(self, dim: int, num_heads: int,
#                  mlp_ratio: float = 4.0, dropout: float = 0.1):
#         super().__init__()
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
#         self.dim        = dim
#         self.num_heads  = num_heads
#         self.head_dim   = dim // num_heads
#         self.dropout    = dropout

#         # —— Eq. (1) 中的 W_QKV
#         self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
#         # out 投影（multi-head attention 之后）
#         self.out_proj = nn.Linear(dim, dim, bias=True)

#         # Feed‑forward network
#         hidden_dim = int(dim * mlp_ratio)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout),
#         )

#         # 层归一化
#         self.norm_x    = nn.LayerNorm(dim)
#         self.norm_y    = nn.LayerNorm(dim)
#         self.norm_ffn  = nn.LayerNorm(dim)

#     def _multi_head(self, q, k, v):
#         """
#         自己分 head，用 PyTorch 的 scaled_dot_product_attention
#         q: (B, S_q, D), k/v: (B, S_kv, D)
#         返回 (B, S_q, D)
#         """
#         B, S_q, D = q.shape
#         H, hd = self.num_heads, self.head_dim

#         # 拆成 (B, H, S, hd)
#         q = q.view(B, S_q, H, hd).transpose(1,2)      # (B, H, S_q, hd)
#         k = k.view(B, -1, H, hd).transpose(1,2)       # (B, H, S_kv, hd)
#         v = v.view(B, -1, H, hd).transpose(1,2)

#         # 合并 batch & head 维度
#         q = q.reshape(B*H, S_q, hd)
#         k = k.reshape(B*H, -1, hd)
#         v = v.reshape(B*H, -1, hd)

#         # scaled dot‑product
#         attn = F.scaled_dot_product_attention(
#             q, k, v, dropout_p=self.dropout, is_causal=False
#         )  # (B*H, S_q, hd)

#         # 拆回 (B, H, S_q, hd) → (B, S_q, D)
#         attn = attn.reshape(B, H, S_q, hd).transpose(1,2).reshape(B, S_q, D)
#         return self.out_proj(attn)

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         x: image tokens, shape (B, N, D)
#         y: label tokens, shape (B, M, D)
#         returns (x_out, y_out), 都是 (B, *, D)
#         """
#         B, N, D = x.shape
#         M = y.shape[1]

#         # —— 1) concat 并做一次线性映射得到所有 Q,K,V —— Eq. (1)
#         cat    = torch.cat([x, y], dim=1)        # (B, N+M, D)
#         qkv    = self.qkv_proj(cat)              # (B, N+M, 3D)
#         Q, K, V = qkv.chunk(3, dim=-1)           # 每个 (B, N+M, D)

#         # 拆出 image 流和 label 流的 Q/K/V
#         X_Q, X_K, X_V = Q[:, :N], K[:, :N], V[:, :N]
#         Y_Q, Y_K, Y_V = Q[:, N:], K[:, N:], V[:, N:]

#         # —— 2a) image 自注意力 —— attn(X_Q, X_K, X_V)
#         x2 = x + self._multi_head(
#             self.norm_x(X_Q),
#             self.norm_x(X_K),
#             self.norm_x(X_V),
#         )

#         # —— 2b) label 交叉注意力 —— attn(Y_Q, [X_K;Y_K], [X_V;Y_V])
#         KV_K = torch.cat([X_K, Y_K], dim=1)       # (B, N+M, D)
#         KV_V = torch.cat([X_V, Y_V], dim=1)       # (B, N+M, D)
#         y2 = y + self._multi_head(
#             self.norm_y(Y_Q),
#             self.norm_x(KV_K),  # note: 可以用同一个 norm_x 或者独立 norm_y
#             self.norm_x(KV_V),
#         )

#         # —— 3) 将两路输出 concat 再 FFN —— Eq. (2)
#         fused = torch.cat([x2, y2], dim=1)        # (B, N+M, D)
#         fused_norm = self.norm_ffn(fused)
#         ffn_out    = self.ffn(fused_norm)
#         fused2     = fused + ffn_out              # (B, N+M, D)

#         # 拆回
#         x_out = fused2[:, :N]
#         y_out = fused2[:, N:]
#         return x_out, y_out
class LabelGuidedTransformerBlock(nn.Module):
    """
    1) Patch-only self-attention (MSA)
    2) Label-only self-attention (MSA)
    3) Cross-attention: label queries attend to patch K/V → update L only
    4) FFN on concatenated tokens, then split
    """
    def __init__(self, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # 1) Norm + MSA for patch tokens
        self.norm_patch1 = nn.LayerNorm(dim)
        self.msa_patch  = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # 2) Norm + MSA for label tokens
        self.norm_label1 = nn.LayerNorm(dim)
        self.msa_label   = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # 3) Norms for cross-attn
        self.norm_cross_x = nn.LayerNorm(dim)
        self.norm_cross_L = nn.LayerNorm(dim)
        self.cross_attn   = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # 4) FFN on concat
        self.norm_ffn = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, L: torch.Tensor):
        """
        x: (B, N, D)
        L: (B, M, D)
        returns: x_out, L_out
        """
        B, N, D = x.shape
        _, M, _ = L.shape
        assert D == self.dim

        # 1) Patch MSA
        x_res = x
        x_norm = self.norm_patch1(x).transpose(0,1)    # (N, B, D)
        x_attn, _ = self.msa_patch(x_norm, x_norm, x_norm)
        x = x_res + x_attn.transpose(0,1)
        
        # 2) Label MSA
        L_res = L
        L_norm = self.norm_label1(L).transpose(0,1)    # (M, B, D)
        L_attn, _ = self.msa_label(L_norm, L_norm, L_norm)
        L = L_attn.transpose(0,1)

        # 3) Cross-attn: labels Q → patches K,V
        L_res2 = L
        # normalize
        q = self.norm_cross_L(L).transpose(0,1)        # (M, B, D)
        kv = self.norm_cross_x(x_res).transpose(0,1)       # (N, B, D)
        # query=labels, key/value=patches
        cross_out, _ = self.cross_attn(q, kv, kv)
        # update only L
        L = L_res + cross_out.transpose(0,1)


        # 4) FFN fusion
        concat = torch.cat([x, L], dim=1)   # (B, N+M, D)
        normed = self.norm_ffn(concat)
        ffned = normed + self.ffn(normed)

        x_out = ffned[:, :N]
        L_out = ffned[:, N:]

        return x_out, L_out

class LabelGuidedTransformer(nn.Module):
    """
    Stacks multiple LabelGuidedTransformerBlocks, passing the same label tokens through all blocks.
    """
    def __init__(self, num_blocks: int, 
                 num_label_tokens: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        if hasattr(self.model, 'num_features'):
            num_features = self.model.num_features
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.num_label_tokens = num_label_tokens
        dim = num_features
        self.dim = dim
        num_heads = self.dim//32
        # Single learnable set of label embeddings
        self.label_emb = nn.Parameter(torch.zeros(1, num_label_tokens, dim))
        nn.init.trunc_normal_(self.label_emb, std=0.02)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            LabelGuidedTransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.classifiers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_label_tokens)])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: img, shape (B, 3, h, w)
        Returns:
            x: updated patch tokens, shape (B, N, D)
            L: updated label tokens, shape (B, M, D)
        """
        x = self.model.forward_features(x)["x_norm_patchtokens"]
        B, N, D = x.shape
        # Expand label embeddings for batch
        L = self.label_emb.expand(B, -1, -1)

        for blk in self.blocks:
            x, L = blk(x, L)
        logits = [self.classifiers[ind](L[:,ind]) for ind in range(len(self.classifiers))]
        logits = torch.cat(logits, dim=1)

        return logits

if __name__ == "__main__":
    model = LabelGuidedTransformer(4,13)
    a = torch.randn(2,3,224,224)
    o = model(a)
    print(a.shape)

