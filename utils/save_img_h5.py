import h5py
import cv2
from tqdm import tqdm
import os
def load_img(embedding_path):
    x = cv2.imread(embedding_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (256, 256))
    return x

with h5py.File("dataset/emb_train_data_v2.h5", 'r') as hf:
    ind = hf['X_text_tag'][:]
    file_names = hf['file_name'][ind]
num_samples = len(file_names)
with h5py.File("dataset/img_train_data.h5", 'w') as hf:
    ds_images = hf.create_dataset('images', shape=(num_samples, 256, 256, 3))
    ds_names = hf.create_dataset('file_names', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
    
    for idx, file_name in enumerate(tqdm(file_names)):
        name_str = file_name.decode('utf-8')
        img_path = os.path.join("../physionet.org/files/mimic-cxr-jpg/2.1.0/files", name_str + ".jpg")
        img = load_img(img_path)
        ds_images[idx] = img
        ds_names[idx] = name_str


with h5py.File("dataset/emb_val_data_v2.h5", 'r') as hf:
    ind = hf['X_text_tag'][:]
    file_names = hf['file_name'][ind]
num_samples = len(file_names)
with h5py.File("dataset/img_val_data.h5", 'w') as hf:
    ds_images = hf.create_dataset('images', shape=(num_samples, 256, 256, 3))
    ds_names = hf.create_dataset('file_names', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
    
    for idx, file_name in enumerate(tqdm(file_names)):
        name_str = file_name.decode('utf-8')
        img_path = os.path.join("../physionet.org/files/mimic-cxr-jpg/2.1.0/files", name_str + ".jpg")
        img = load_img(img_path)
        ds_images[idx] = img
        ds_names[idx] = name_str

with h5py.File("dataset/emb_test_data_v2.h5", 'r') as hf:
    ind = hf['X_text_tag'][:]
    file_names = hf['file_name'][ind]
num_samples = len(file_names)
with h5py.File("dataset/img_test_data.h5", 'w') as hf:
    ds_images = hf.create_dataset('images', shape=(num_samples, 256, 256, 3))
    ds_names = hf.create_dataset('file_names', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
    
    for idx, file_name in enumerate(tqdm(file_names)):
        name_str = file_name.decode('utf-8')
        img_path = os.path.join("../physionet.org/files/mimic-cxr-jpg/2.1.0/files", name_str + ".jpg")
        img = load_img(img_path)
        ds_images[idx] = img
        ds_names[idx] = name_str