import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from utils.losses import BCEWithLogitsLoss,ReweightedBCELoss,FocalLoss
from torchmetrics.classification import MultilabelAUROC,MultilabelAccuracy
from pytorch_lightning.callbacks import ModelSummary,ModelCheckpoint,LearningRateMonitor,EarlyStopping
import argparse
import yaml
from dataset.create_dataset import create_dataset
from models.create_model import create_model
from torchvision.transforms import v2
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import roc_auc_score
from utils.load_config import load_config,print_config
from train import TorchModule



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train with pytorch-lightning')
    parser.add_argument('--config',default='configs/config.yaml',type=str)
    args = parser.parse_args()


    config = load_config(args.config)
    print_config(config)

    pl.seed_everything(42)
    NUM_WORKERS = config['dataloader']['num_workers']
    BATCH_SZIE = config['dataloader']['batch_size']
    model_name = config['model']['name']

    dataset_cfg = config['dataset']
    dataset_train, dataset_val, dataset_test = create_dataset(**dataset_cfg)

    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SZIE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=True, prefetch_factor=2)

    model = TorchModule(model_name, config['model']['params'], lr=config['optimizer']['lr'],label_smoothing=config['train']['label_smoothing'],use_mixup=config['train']['use_mixup'])

    trainer = pl.Trainer(
        max_epochs=config['train']['epochs'],
        accelerator="auto",
        precision= '16-mixed',
        devices=1,
        num_sanity_val_steps=2,
    )

    trainer.test(model, dataloaders=test_dataloader,ckpt_path=os.path.join("./ckpt/",f"{model_name}-best_metric_model.ckpt"))