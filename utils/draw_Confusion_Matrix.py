import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

# Load your .npz data
data = np.load('result/test_LTViT.npz')
y_true = data["labels"]  # 形状 (N_samples, N_classes)
y_pred = data["preds"]  # 同上
threshold = 0.5
y_pred = (y_pred >= threshold).astype(int)

# Compute multilabel confusion matrices
cm = multilabel_confusion_matrix(y_true, y_pred)

# Summarize into (labels, 4) array: TN, FP, FN, TP
summary = np.array([[m[0,0], m[0,1], m[1,0], m[1,1]] for m in cm])

labels = [f'Label {i+1}' for i in range(cm.shape[0])]
columns = ['TN', 'FP', 'FN', 'TP']

fig, ax = plt.subplots()
im = ax.imshow(summary, aspect='auto')
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(columns)
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Annotate each cell
for i in range(summary.shape[0]):
    for j in range(summary.shape[1]):
        ax.text(j, i, summary[i, j], ha='center', va='center')

ax.set_title('Multilabel Confusion Matrix Heatmap')
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=400, bbox_inches="tight")
plt.show()