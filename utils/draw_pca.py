import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

def load_dinov2_model(device: str = "cuda") -> torch.nn.Module:
    """
    加载 DINOv2 ViT-S/14 模型。
    """
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device).eval()
    return model

def preprocess_images(
    image_paths: List[str],
    img_size: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    读取并预处理图片，返回:
      - Tensor 格式的数据 (N,3,H,W)
      - 用于绘图的 NumPy 矩阵 (N,H,W,3)，值在 [0,255]
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(img_size + int(img_size*0.01)*10),
        T.CenterCrop(img_size),
        T.Normalize([0.5], [0.5]),
    ])
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
    batch = torch.stack(tensors).to(device)
    
    # 转回 uint8 便于可视化
    np_imgs = ((batch.cpu().numpy() * 0.5 + 0.5) * 255
               ).transpose(0, 2, 3, 1).astype(np.uint8)
    return batch, np_imgs

def extract_patch_tokens(
    model: torch.nn.Module, 
    images: torch.Tensor
) -> np.ndarray:
    """
    前向算出 patch tokens:
      返回 shape (N, patch_h*patch_w, feature_dim) 的 numpy 数组
    """
    with torch.no_grad():
        feats = model.forward_features(images)
    return feats["x_norm_patchtokens"].cpu().numpy()

def compute_fg_masks(
    x_norm_patchtokens: np.ndarray,
    patch_h: int,
    patch_w: int,
    threshold: float = 0.6
) -> List[np.ndarray]:
    """
    对所有 patch 的特征做 1D PCA，并根据阈值分出前景 mask。
    返回布尔 mask 列表，每个 mask.shape == (patch_h*patch_w,)
    """
    N = x_norm_patchtokens.shape[0]
    all_feats = x_norm_patchtokens.reshape(N * patch_h * patch_w, -1)
    
    pca1 = PCA(n_components=1)
    pc1 = pca1.fit_transform(all_feats)[:, 0]
    pc1_norm = minmax_scale(pc1)  # 归一化到 [0,1]
    pc1_images = pc1_norm.reshape(N, patch_h * patch_w)
    
    masks = [(pc1_images[i] > threshold) for i in range(N)]
    return masks

def generate_fg_pca_images(
    x_norm_patchtokens: np.ndarray,
    masks: List[np.ndarray],
    patch_h: int,
    patch_w: int
) -> List[np.ndarray]:
    """
    仅对前景 patch 做 3D PCA 重构，归一化后返回每张图的 (patch_h,patch_w,3) 矩阵。
    """
    N = x_norm_patchtokens.shape[0]
    # 堆叠所有图的前景 patch
    fg_feats = np.vstack([x_norm_patchtokens[i][masks[i], :]
                          for i in range(N)])
    pca3 = PCA(n_components=3)
    fg_trans = pca3.fit_transform(fg_feats)
    fg_scaled = minmax_scale(fg_trans)  # 归一化到 [0,1]
    
    # 按图拆分并重组
    result = []
    idx = 0
    for i in range(N):
        m = masks[i]
        cnt = m.sum()
        mat = np.zeros((patch_h * patch_w, 3), dtype=float)
        mat[m] = fg_scaled[idx:idx+cnt]
        idx += cnt
        result.append(mat.reshape(patch_h, patch_w, 3))
    return result

def main(
    image_paths: List[str],
    img_size: int = 448,
    threshold: float = 0.6,
    device: str = "cuda"
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    输入:
      - image_paths: 本地图片文件路径列表
      - img_size: 输入到模型的正方形边长 (必须能被14整除)
      - threshold: 前景分割阈值
      - device: "cuda" 或 "cpu"
    返回:
      - 原始图片矩阵 (N,H,W,3)
      - 对应的前景 PCA 可视化矩阵列表，每个 (patch_h,patch_w,3)
    """
    assert img_size % 14 == 0, "img_size 必须被 14 整除"
    patch_h = patch_w = img_size // 14

    model = load_dinov2_model(device)
    imgs_tensor, imgs_plot = preprocess_images(image_paths, img_size, device)
    x_norm = extract_patch_tokens(model, imgs_tensor)
    masks = compute_fg_masks(x_norm, patch_h, patch_w, threshold)
    fg_pca_imgs = generate_fg_pca_images(x_norm, masks, patch_h, patch_w)
    
    return imgs_plot, fg_pca_imgs

if __name__ == "__main__":
    # Example
    img_list = [f"images/{i}.jpg" for i in range(1, 5)]
    origs, fg_pcas = main(img_list, img_size=448, threshold=0.6, device="cuda")
    
    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    for idx, fg in enumerate(fg_pcas):
        plt.imsave(f"outputs/fg_pca_{idx+1}.png", fg)