import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

# --------- 配置 ----------
# 替换为你的最佳模型 npz 文件路径
npz_path = "result/test_LTViT.npz"

# --------- 数据加载 ----------
data = np.load(npz_path)
y_true = data["labels"]  # 形状 (N_samples, N_classes)
y_scores = data["preds"]  # 同上

n_classes = y_true.shape[1]
class_names = [
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

# --------- 1. Macro Precision-Recall 曲线 ----------
# 计算各类 PR 曲线
fpr_list = []
pr_curves = {}
all_recall = np.unique(
    np.concatenate([
        precision_recall_curve(y_true[:, i], y_scores[:, i])[1]
        for i in range(n_classes)
    ])
)

mean_precision = np.zeros_like(all_recall)
for i in range(n_classes):
    precision_i, recall_i, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
    pr_curves[i] = (precision_i, recall_i)
    mean_precision += np.interp(all_recall, recall_i[::-1], precision_i[::-1])
mean_precision /= n_classes

# 计算 macro 和 micro 平均 AP
macro_ap = average_precision_score(y_true, y_scores, average="macro")
precision_m, recall_m, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
micro_ap = average_precision_score(y_true, y_scores, average="micro")

plt.figure(figsize=(6, 6))
plt.plot(all_recall, mean_precision,
         label=f"Macro-average PR (AP = {macro_ap:.3f})", linewidth=2)
plt.plot(recall_m, precision_m,
         label=f"Micro-average PR (AP = {micro_ap:.3f})", linewidth=2, linestyle="--")

plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Macro vs. Micro Precision-Recall Curve", fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("macro_micro_pr_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# --------- 2. 各类别 ROC AUC & PR AUC 对比柱状图 ----------
roc_aucs = []
pr_aucs = []
for i in range(n_classes):
    roc_aucs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
    pr_aucs.append(average_precision_score(y_true[:, i], y_scores[:, i]))

x = np.arange(n_classes)
width = 0.4

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, roc_aucs, width, label="ROC AUC",color='#99b9e9')
bars2 = ax.bar(x + width/2, pr_aucs, width, label="PR AUC",color='#f9d580')

# 标注数值
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}", 
                xy=(bar.get_x() + bar.get_width() / 2, height-0.001),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("AUC", fontsize=12,weight='bold')
ax.set_title("Per-class ROC AUC and PR AUC Comparison", fontsize=14,weight='bold')
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("per_class_auc_pr_auc.png", dpi=400, bbox_inches="tight")
plt.show()