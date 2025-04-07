import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageEmbDataset(Dataset):
    def __init__(self, h5_file_path):

        with h5py.File(h5_file_path, 'r') as hf:
            ind = hf['X_text_tag'][:]
            self.X = hf['X'][ind] 
            self.file_names = hf['file_name'][ind]
            self.labels = hf['y'][ind]
            self.labels = np.where(self.labels == 1, 1, 0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'input': torch.tensor(self.X[idx]),
            'label': torch.tensor(self.labels[idx],dtype=torch.int32),
            'file_name': self.file_names[idx].decode('utf-8')
        }
        return sample


class TextEmbDataset(Dataset):

    def __init__(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as hf:
            ind = hf['X_text_tag'][:]
            self.X_text = hf['X_text'][ind]
            self.file_names = hf['file_name'][ind]
            self.labels = hf['y'][ind]
            self.labels = np.where(self.labels == 1, 1, 0)

    def __len__(self):
        return len(self.X_text)

    def __getitem__(self, idx):
        sample = {
            'input': torch.tensor(self.X_text[idx]),
            'label': torch.tensor(self.labels[idx],dtype=torch.int32),
            'file_name': self.file_names[idx].decode('utf-8')
        }
        return sample


class ImageTextEmbDataset(Dataset):
    def __init__(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as hf:
            ind = hf['X_text_tag'][:]
            self.X = hf['X'][ind]
            self.X_text = hf['X_text'][ind]
            self.file_names = hf['file_name'][ind]
            self.labels = hf['y'][ind]
            self.labels = np.where(self.labels == 1, 1, 0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'input': torch.tensor(self.X[idx]),
            'input_text': torch.tensor(self.X_text[idx]),
            'label': torch.tensor(self.labels[idx],dtype=torch.int32),
            'file_name': self.file_names[idx].decode('utf-8')
        }
        return sample


if __name__ == "__main__":
    print(len(ImageEmbDataset('dataset/emb_train_data_v2.h5')))
    print(len(ImageEmbDataset('dataset/emb_val_data_v2.h5')))
    print(len(ImageEmbDataset('dataset/emb_test_data_v2.h5')))