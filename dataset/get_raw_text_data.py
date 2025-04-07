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
import tensorflow as tf
from tqdm import tqdm
from text_processing import ReportProcessor


def load_embedding(embedding_path):
    raw_dataset = tf.data.TFRecordDataset([embedding_path])
    for raw_record in raw_dataset.take(1):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      embedding_feature = example.features.feature['embedding']
      embedding_values = embedding_feature.float_list.value
    return torch.tensor(embedding_values)

IMAGE_IDS_TO_IGNORE = {
    "0518c887-b80608ca-830de2d5-89acf0e2-bd3ec900",
    "03b2e67c-70631ff8-685825fb-6c989456-621ca64d",
    "786d69d0-08d16a2c-dd260165-682e66e9-acf7e942",
    "1d0bafd0-72c92e4c-addb1c57-40008638-b9ec8584",
    "f55a5fe2-395fc452-4e6b63d9-3341534a-ebb882d5",
    "14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179",
    "9c42d877-dfa63a03-a1f2eb8c-127c60c3-b20b7e01",
    "996fb121-fab58dd2-7521fd7e-f9f3133c-bc202556",
    "56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557",
    "93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6",
    "f57b4a53-5fecd631-2fe14e8a-f4780ee0-b8471007",
    "d496943d-153ec9a5-c6dfe4c0-4fb9e57f-675596eb",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50m",
    "422689b1-40e06ae8-d6151ff3-2780c186-6bd67271",
    "8385a8ad-ad5e02a8-8e1fa7f3-d822c648-2a41a205",
    "e180a7b6-684946d6-fe1782de-45ed1033-1a6f8a51",
    "f5f82c2f-e99a7a06-6ecc9991-072adb2f-497dae52",
    "6d54a492-7aade003-a238dc5c-019ccdd2-05661649",
    "2b5edbbf-116df0e3-d0fea755-fabd7b85-cbb19d84",
    "db9511e3-ee0359ab-489c3556-4a9b2277-c0bf0369",
    "87495016-a6efd89e-a3697ec7-89a81d53-627a2e13",
    "810a8e3b-2cf85e71-7ed0b3d3-531b6b68-24a5ca89",
    "a9f0620b-6e256cbd-a7f66357-2fe78c8a-49caac26",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50",
}

class MIMIC_Embed_Dataset(Dataset):

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
        csvpath,
        metacsvpath,
        splitcsvpath,
        views=["PA"],
        data_aug=None,
        seed=0,
        unique_patients=True,
        mode=["train", "validate", "test"][0],
    ):

        super().__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = sorted(self.pathologies)
        self.report_processor = ReportProcessor()
        self.mode = mode
        self.embedpath = embedpath
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)
        # self.splitcsv = pd.read_csv(splitcsvpath)
        self.csv = self.csv.set_index(["subject_id", "study_id"])
        self.metacsv = self.metacsv.set_index(["subject_id", "study_id"])
        # self.splitcsv = self.splitcsv.set_index(["subject_id", "study_id"])
        # self.csv = self.csv.join(self.splitcsv['split'])
        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        n_row = self.csv.shape[0]

        # spit data to one of train valid test
        if self.mode == "train":
            self.csv = self.csv[: int(n_row * self.split_ratio[0])]
            # self.csv = self.csv[self.csv['split'] == 'train']
        elif self.mode == "validate":
            self.csv = self.csv[
                int(n_row * self.split_ratio[0]) : int(
                    n_row * (self.split_ratio[0] + self.split_ratio[1])
                )
            ]
            # self.csv = self.csv[self.csv['split'] == 'validate']
        elif self.mode == "test":
            self.csv = self.csv[-int(n_row * self.split_ratio[-1]) :]
            # self.csv = self.csv[self.csv['split'] == 'test']
        else:
            raise ValueError(
                f"attr:mode has to be one of [train, valid, test] but your input is {self.mode}"
            )

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views,
        )

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
        
        #data_aug
        embed_file = os.path.join(
            self.embedpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".tfrecord",
        )
        sample["embedding"] = load_embedding(embed_file)
        #sample["embedding"] = embed_file
        sample['file_name'] = os.path.join("p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id)
        report_subpath = f"p{subjectid[:2]}/p{subjectid}/s{studyid}.txt"
        report_path = os.path.join('../physionet.org/files/mimic-cxr-jpg/2.1.0/files', report_subpath)
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                full_report = f.read()
            processed_report = self.report_processor(full_report, study=f's{studyid}')
            if processed_report is not None:
                sample['report'] = ' '.join(processed_report)
            else:
                sample['report'] = ''
        else:
            sample['report'] = ''

        return sample


if __name__ == "__main__":
    embedpath = "../physionet.org/files/image-embeddings-mimic-cxr/1.0/files"
    csvpath = "../physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
    metacsvpath = "../physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
    splitcsvpath = '../physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz'
    dataset = MIMIC_Embed_Dataset(embedpath,csvpath,metacsvpath,splitcsvpath,mode = "train")
    print(dataset[1000])

    # # save .h5
    import h5py

    # 创建HDF5文件
    with h5py.File('dataset/text_train_data.h5', 'w') as hf:
        str_dt = h5py.string_dtype(encoding='utf-8')
        ds_file_name = hf.create_dataset('file_name', shape=(len(dataset),), dtype=str_dt)
        ds_texts = hf.create_dataset('texts', shape=(len(dataset),), dtype=str_dt)
        ds_y = hf.create_dataset('y', shape=(len(dataset), *dataset[0]['lab'].shape), dtype='float32')
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            ds_file_name[idx] = data['file_name']
            ds_texts[idx] = data['report']
            ds_y[idx] = data['lab']
       
    dataset = MIMIC_Embed_Dataset(embedpath,csvpath,metacsvpath,splitcsvpath,mode = "validate")
    with h5py.File('dataset/text_val_data.h5', 'w') as hf:
        str_dt = h5py.string_dtype(encoding='utf-8')
        ds_file_name = hf.create_dataset('file_name', shape=(len(dataset),), dtype=str_dt)
        ds_texts = hf.create_dataset('texts', shape=(len(dataset),), dtype=str_dt)
        ds_y = hf.create_dataset('y', shape=(len(dataset), *dataset[0]['lab'].shape), dtype='float32')
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            ds_file_name[idx] = data['file_name']
            ds_texts[idx] = data['report']
            ds_y[idx] = data['lab']

    dataset = MIMIC_Embed_Dataset(embedpath,csvpath,metacsvpath,splitcsvpath,mode = "test")
    with h5py.File('dataset/text_test_data.h5', 'w') as hf:
        str_dt = h5py.string_dtype(encoding='utf-8')
        ds_file_name = hf.create_dataset('file_name', shape=(len(dataset),), dtype=str_dt)
        ds_texts = hf.create_dataset('texts', shape=(len(dataset),), dtype=str_dt)
        ds_y = hf.create_dataset('y', shape=(len(dataset), *dataset[0]['lab'].shape), dtype='float32')
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            ds_file_name[idx] = data['file_name']
            ds_texts[idx] = data['report']
            ds_y[idx] = data['lab']