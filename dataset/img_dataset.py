import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os.path
import random
import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import torchvision
from PIL import Image
import h5py



def load_img(embedding_path):
    # x = cv2.imread(embedding_path)
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # x = cv2.resize(x, (256, 256))
    # x = torchvision.transforms.ToPILImage()(x)
    return Image.open(embedding_path).convert('RGB')


class MIMIC_Img_Dataset(Dataset):

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

    split_ratio = [0.8, 0.1, 0.1]

    def __init__(
        self,
        embedpath,
        img_h5_path,
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    ):
        super().__init__()
        self.img_h5_path = img_h5_path
        self.transforms = transforms
        with h5py.File(embedpath, 'r') as hf:
            ind = hf['X_text_tag'][:]
            self.file_names = hf['file_name'][ind]
            self.labels = hf['y'][ind]
            self.text_emb = hf['X_text'][ind]
            self.labels[self.labels != 1] = 0
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["label"] = torch.tensor(self.labels[idx],dtype=torch.int32)
        sample['file_name'] = self.file_names[idx].decode('utf-8')
        with h5py.File(self.img_h5_path, 'r') as hf:
            sample["input"] = self.transforms(torchvision.transforms.ToPILImage()(hf['images'][idx]))
        
        return sample


if __name__ == "__main__":
    embedpath = "dataset/emb_train_data_v2.h5"
    img_h5_path = "dataset/img_train_data.h5"
    dataset = MIMIC_Img_Dataset(embedpath,img_h5_path,transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]))
    print(dataset[1000])
    print(dataset[1000]["input"].shape)
