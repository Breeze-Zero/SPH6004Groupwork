import os
import glob
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 结果文件夹路径
model_dict = {'ResNet-152':'result/test_resnet152.npz',
            'EfficientNet-B7':'result/test_tf_efficientnet_b7.ns_jft_in1k.npz',
            'ConvNextv2-Base':'result/test_convnextv2_base.fcmae.npz',
            'ViT-Base':'result/test_vit_base_patch8_224.augreg2_in21k_ft_in1k.npz',
            'Swin-Base':'result/test_swin_base_patch4_window7_224.ms_in22k.npz',
            'VisionLSTM-Base':'result/test_VisionLSTM.npz'
}

plt.figure(figsize=(6, 6))
for model_name, path in model_dict.items():
    data = np.load(path)
    y_score = data["preds"]
    y_true = data["labels"]
    n_classes = y_true.shape[1]
    # 1) 为每个类别计算 ROC
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])

    # 2) 收集所有类别的假阳性率点，并取唯一值
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 3) 在这些统一的 fpr 点上插值每个类别的 tpr，然后求平均
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    # 4) 计算 macro-average AUC
    macro_auc = auc(all_fpr, mean_tpr)

    # 5) 绘制该模型的 macro ROC
    plt.plot(
        all_fpr,
        mean_tpr,
        linewidth=2,
        label=f"{model_name} (macro-AUC = {macro_auc:.3f})"
    )

# 绘制对角线参考
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")

# 美化
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Macro-average ROC Curves Across Models", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存高分辨率图像
plt.savefig("macro_roc_curves.png", dpi=300, bbox_inches="tight")
plt.show()