import h5py
import numpy as np
with h5py.File('dataset/emb_train_data_v2.h5', 'r') as hf:
    ind = hf['X_text_tag'][:]
    file_names = hf['file_name'][ind]
    labels = hf['y'][ind]
    text_emb = hf['X_text'][ind]
    labels[labels != 1] = 0
for i in range(len(labels)):
    indices = np.where(labels[i] == 1)[0]
    if len(indices)>2:
        print(i)