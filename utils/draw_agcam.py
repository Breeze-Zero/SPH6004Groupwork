import torch
import torch.nn as nn
from einops.layers.torch import Reduce, Rearrange
import sys
sys.path.append('/home/users/nus/e1373616/.cache/torch/hub/facebookresearch_dinov2_main')
from dinov2.layers import MemEffAttention  # DINOv2 自注意力模块
import os
sys.path.append(os.getcwd())
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  
import cv2
class LabelGuidedAGCAM:
    """
    AG-CAM Explainer for your LabelGuidedTransformer:
      - 在内部 backbone（dinov2_vitb14_reg）上的 MemEffAttention 打钩子
      - 针对某个 label token 的 logit 反向，生成 patch-level CAM
    """
    def __init__(self,
                 lg_model: nn.Module,
                 head_fusion: str = 'sum',
                 layer_fusion: str = 'sum'):
        """
        Args:
            lg_model: 你的 LabelGuidedTransformer 实例
            head_fusion/layer_fusion: 'sum' 或 'mean'
        """
        self.lg_model    = lg_model.eval()
        self.backbone    = lg_model.model        # dinov2_vitb14_reg
        self.head_fusion = head_fusion
        self.layer_fusion= layer_fusion

        self.attn_list, self.grad_list = [], []
        # 在 backbone 里所有 MemEffAttention 上打钩子
        for m in self.backbone.modules():
            if isinstance(m, MemEffAttention):
                m.register_forward_hook(self._hook_attn)
                m.register_full_backward_hook(self._hook_grad)

    def _hook_attn(self, module, inp, out):
        # out: [B, H, N, N]，取 CLS→patch
        self.attn_list.append(out[:,:,0:1,1:])  # [B,H,1,P]

    def _hook_grad(self, module, grad_in, grad_out):
        # grad_out[0] 跟 out 同 shape
        self.grad_list.append(grad_out[0][:,:,0:1,1:])  # [B,H,1,P]

    def generate(self,
                 img: torch.Tensor,
                 label_idx: int = None):
        """
        Args:
            img: [1,3,H,W]（已归一化）
            label_idx: 要可视化的 label token 下标；若 None 用预测的那个
        Returns:
            pred_label: int
            cam: [1,1,ph,pw] CAM map
        """
        # 清空上次 hooks 收集的数据
        self.attn_list.clear()
        self.grad_list.clear()

        # 1) 前向拿 logits
        logits = self.lg_model(img)            # [1, M]
        pred   = logits.argmax(dim=1)          # [1]
        idx    = label_idx if label_idx is not None else pred[0].item()

        # 2) 选定 logit 反向
        score = logits[0, idx]
        self.backbone.zero_grad()              # 只要 backbone 的 grad hook
        score.backward()

        # 3) 拼接所有层、所有头的 attn & grad
        A = torch.cat(self.attn_list, dim=0)   # [L, B, H, 1, P]
        G = torch.cat(self.grad_list, dim=0)   # [L, B, H, 1, P]

        # 4) 只留正梯度 & sigmoid(attn)
        G = torch.relu(G)
        A = torch.sigmoid(A)
        M = G * A                              # [L,B,H,1,P]

        # 5) 聚合 heads & layers → [B, P]
        M = M.squeeze(3)      # [L,B,H,P]
        M = M.permute(1,0,2,3) # [B,L,H,P]
        M = Reduce('b l h p -> b l p', reduction=self.head_fusion)(M)
        M = Reduce('b l p -> b p',     reduction=self.layer_fusion)(M)

        # 6) reshape 到 patch-grid
        ph = pw = img.shape[-1] // self.backbone.patch_size
        cam = M.view(1,1,ph,pw)   # [1,1,ph,pw]

        return idx, cam

class Dino2AGCAM:
    """AG-CAM for DINOv2 Vision Transformer."""
    def __init__(self, model: nn.Module,
                 head_fusion: str = 'sum',
                 layer_fusion: str = 'sum'):
        """
        Args:
            model: 已加载的 DINOv2 VisionTransformer（DinoVisionTransformer 实例）。
            head_fusion: 多头融合方式，'sum' 或 'mean'。
            layer_fusion: 多层融合方式，'sum' 或 'mean'。
        """
        self.model = model.eval()
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        # 遍历所有子模块，找到 MemEffAttention，注册钩子
        for mod in self.model.modules():
            if isinstance(mod, MemEffAttention):
                mod.register_forward_hook(self._hook_attn)
                mod.register_full_backward_hook(self._hook_grad)

    def _hook_attn(self, module, input, output):
        # output shape: [B, num_heads, N, N]（N = 1 + num_patches + register_tokens）
        # 取 CLS token 对所有 patch 的注意力（第 0 行，跳过自己）
        att = output[:, :, 0:1, 1:]  # [B, H, 1, P]
        self.attn_matrix.append(att)

    def _hook_grad(self, module, grad_input, grad_output):
        # grad_output[0] 与前向输出同形
        grad = grad_output[0][:, :, 0:1, 1:]  # [B, H, 1, P]
        self.grad_attn.append(grad)

    @torch.no_grad()
    def _forward(self, x: torch.Tensor):
        # 利用 forward_features 只跑到最后 norm 前，可确保 hook 收集到所有层
        return self.model.forward_features(x)['x_prenorm']

    def generate(self, x: torch.Tensor, cls_idx: int = None):
        """
        Args:
            x: 输入图像 tensor，形状 [1,3,H,W]，需归一化
            cls_idx: 指定类别索引，否则用模型预测的 argmax
        Returns:
            pred: 预测标签
            cam: [1, 1, H_patch, W_patch] 的热力图张量
        """
        self.attn_matrix.clear()
        self.grad_attn.clear()

        # 前向
        feats = self._forward(x)            # 触发所有 forward hook
        out = self.model.head(self.model.norm(feats[:, 0]))  # cls token 分类头
        pred = out.argmax(dim=1)

        # 选定类别反向
        idx = cls_idx if cls_idx is not None else pred[0].item()
        loss = out[0, idx]
        self.model.zero_grad()
        loss.backward()

        # 把所有层、所有头拼到一起
        attn = torch.cat(self.attn_matrix, dim=0)   # [L, B, H, 1, P]
        grad = torch.cat(self.grad_attn, dim=0)     # [L, B, H, 1, P]

        # 仅保留正梯度，sigmoid 归一化注意力
        grad = torch.relu(grad)
        attn = torch.sigmoid(attn)
        mask = grad * attn                          # [L, B, H, 1, P]

        # 去掉那一维 1，把头和层聚合
        mask = mask.squeeze(3)                      # [L, B, H, P]
        mask = mask.permute(1, 0, 2)  # [B, L, H, P]
        mask = Reduce('b l h p -> b l p', reduction=self.head_fusion)(mask)
        mask = Reduce('b l p -> b p',     reduction=self.layer_fusion)(mask)

        # 还原成 patch 网格
        _, P = mask.shape[-2:]
        ph = pw = x.shape[-1] // self.model.patch_size
        cam = mask.reshape(1, 1, ph, pw)            # [B,1,ph,pw]
        return pred, cam

# —— 使用示例 ——  
if __name__ == '__main__':
    from PIL import Image
    import torchvision
    import h5py
    import numpy as np
    from collections import OrderedDict
    idx = 3000 #36038 #35960
    from models.LT_VIT import *
    from models.classifiers import *
    from models.general_model import BaseModel
    pathologies = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    # model = LabelGuidedTransformer(4,13).cuda()
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    if hasattr(backbone, 'num_features'):
        num_features = backbone.num_features
    else:
        raise ValueError(f"model {name} has no 'num_features'")
    head = Classifier(in_features=num_features, num_classes=13)
    model = BaseModel(model = backbone, num_features = num_features, head = head, outnorm = False).cuda()

    ckpt = torch.load('ckpt/Dinov2-best_metric_model.ckpt', map_location='cuda')

    state_dict = ckpt["state_dict"]
    # 去掉前缀，只保留真正网络需要的键
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean_state_dict[k.replace("model.", "", 1)] = v
    model.load_state_dict(clean_state_dict, strict=True)

    val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    with h5py.File('dataset/emb_test_data_v2.h5', 'r') as hf:
        ind = hf['X_text_tag'][:]
        file_names = hf['file_name'][ind]
        labels = hf['y'][ind]
        text_emb = hf['X_text'][ind]
        labels[labels != 1] = 0
    
    with h5py.File('dataset/img_test_data.h5', 'r') as hf:
        img = hf['images'][idx]
        inp = val_transforms(torchvision.transforms.ToPILImage()(img))
    indices = np.where(labels[idx] == 1)[0]
    print(indices)
    for i in indices:
        print(pathologies[i])
    inp = inp.unsqueeze(0).cuda()  # [1,3,224,224]
    # explainer = LabelGuidedAGCAM(model)
    # # 3. 生成 CAM
    # pred, cam = explainer.generate(inp)
    # print(f'Predicted class: {pred.item()} — CAM size: {cam.shape}')

    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import numpy as np

    # ------------ 1. 准备模型与输入 ------------
    model.eval()
    input_tensor = inp  # [1,3,224,224]，已归一化至 ImageNet 标准

    # 恢复到 [0,1] RGB numpy
    img_np = x = cv2.resize(img, (224, 224))
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # ------------ 2. 前向预测并取 top-K 标签 ------------
    with torch.no_grad():
        logits = model(input_tensor)            # [1, num_labels]
        probs  = torch.sigmoid(logits)[0]       # 多标签用 sigmoid
    probs_np = probs.cpu().numpy()
    topk_idx = probs_np.argsort()[-4:][::-1]   # 取前4大概率标签

    def reshape_transform(tensor, height=16, width=16, n_reg=model.model.num_register_tokens):
        # tensor: [B, 1+H*W+n_reg, C]
        # 跳过 class token 及所有 register tokens
        result = tensor[:, n_reg+1:].reshape(
            tensor.size(0), height, width, tensor.size(2)
        )
        # 同样转置到 (B, C, H, W)
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    # ------------ 3. 配置 CAM 对象 ------------
    # 选最后一层 transformer block 的 MLP 作为 target layer
    target_layer = model.model.blocks[-1].norm1
    campp = GradCAMPlusPlus(model=model,
                            reshape_transform=reshape_transform,
                            target_layers=[target_layer])

    # ------------ 4. 生成独立子图 ------------
    fig, axes = plt.subplots(1, len(topk_idx)+1, figsize=(4*(len(topk_idx)+1),4))

    for i, cls_idx in enumerate(topk_idx):
        targets = [ClassifierOutputTarget(cls_idx)]
        grayscale_cam = campp(input_tensor=input_tensor,eigen_smooth=True,
                            targets=targets)[0]
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        axes[i].imshow(overlay)
        axes[i].set_title(f"{pathologies[cls_idx]}\nProb={probs_np[cls_idx]:.2f}")
        axes[i].axis('off')

    # ------------ 5. 加权融合 CAM ------------
    # 对所有标签计算 CAM 并加权求和
    all_cams = []
    for cls_idx in range(len(probs_np)):
        targets = [ClassifierOutputTarget(cls_idx)]
        cam_i = campp(input_tensor=input_tensor,eigen_smooth=True,
                    targets=targets)[0]
        all_cams.append(cam_i * probs_np[cls_idx])
    fusion_cam = np.sum(all_cams, axis=0)
    fusion_cam = (fusion_cam - fusion_cam.min()) / (fusion_cam.max() - fusion_cam.min())
    fusion_overlay = show_cam_on_image(img_np, fusion_cam, use_rgb=True)

    axes[-1].imshow(fusion_overlay)
    axes[-1].set_title("Weighted Fusion CAM")
    axes[-1].axis('off')

    # ------------ 6. 保存 ------------
    plt.tight_layout()
    plt.savefig("multi_label_CAM_figure.png", dpi=300, bbox_inches="tight")
    plt.show()