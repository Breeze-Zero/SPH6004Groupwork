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
NUM_CLASSES = 13

class TorchModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, lr=1e-4,label_smoothing=0.0,use_mixup = False):
        """TorchModule.

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: AdamW, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.

        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, **model_hparams) #torch.compile(create_model(model_name, **model_hparams))
        # Create loss module
        self.loss_module = nn.BCEWithLogitsLoss()#BCEWithLogitsLoss(label_smoothing)
        self.aucmetric = MultilabelAUROC(num_labels=NUM_CLASSES)
        self.multi_aucmetric = MultilabelAUROC(num_labels=NUM_CLASSES, average=None)
        self.accmetric = MultilabelAccuracy(num_labels=NUM_CLASSES)
        self.label_tag = [
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

        self.use_mixup = use_mixup
        if self.use_mixup:
            cutmix = v2.CutMix(num_classes=NUM_CLASSES)
            mixup = v2.MixUp(num_classes=NUM_CLASSES)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


    def on_validation_start(self):
        self.val_preds = []
        self.val_labels = []

    def forward(self, inp):
        return self.model(inp)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,self.parameters()), self.hparams.lr, weight_decay=2e-5)

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.trainer.max_epochs,
            cycle_mul=1.,
            lr_min=1e-5,
            warmup_lr_init=1e-5,
            warmup_t=10,
            cycle_limit=1,
            t_in_epochs=True,
        )
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        labels = batch['label']
        inp = batch['input']
        if self.use_mixup:
            inp, labels = self.cutmix_or_mixup(inp, labels)

        if 'input_text' in batch:
            inp_text = batch['input_text']
            preds = self.model(inp,inp_text)
        else:
            preds = self.model(inp)
        loss = self.loss_module(preds, labels.float())
    
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        inp = batch['input']
        if 'input_text' in batch:
            inp_text = batch['input_text']
            preds = self.model(inp,inp_text)
        else:
            preds = self.model(inp)
        preds = torch.sigmoid(preds)
        # self.aucmetric.update(preds, labels)
        # self.accmetric.update(preds, labels)
        # auc = self.aucmetric(preds, labels)
        acc = self.accmetric(preds, labels)
        # self.log("val_auc", auc, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # 保存原始logits用于更新权重
        self.val_preds.append(preds.detach().cpu().numpy())
        self.val_labels.append(labels.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        # auc = self.aucmetric.compute()
        # acc = self.accmetric.compute()
        # self.log("val_auc", auc)
        # self.log("val_acc", acc)
        # self.aucmetric.reset()
        # self.accmetric.reset()
        # 合并所有验证集预测结果
        # all_preds = torch.cat(self.val_preds)
        # all_labels = torch.cat(self.val_labels)

        all_preds = np.concatenate(self.val_preds, axis=0)
        all_labels = np.concatenate(self.val_labels, axis=0)
        
        try:
            overall_auc = roc_auc_score(all_labels, all_preds, average="macro")
        except ValueError as e:
            self.log("val_auc", float('nan'), prog_bar=True)
            print(f"Error computing overall AUC: {e}")
        else:
            self.log("val_auc", overall_auc, prog_bar=True)
        
        # # 更新损失函数权重
        # self.loss_module.update_weights(all_preds, all_labels)
        self.val_preds.clear()
        self.val_labels.clear()

    def on_test_start(self):
        # 初始化测试时的预测结果和标签列表
        self.test_preds = []
        self.test_labels = []
    def test_step(self, batch, batch_idx):
        labels = batch['label']
        inp = batch['input']
        if 'input_text' in batch:
            inp_text = batch['input_text']
            preds = self.model(inp,inp_text)
        else:
            preds = self.model(inp)

        preds = torch.sigmoid(preds)


        self.test_preds.append(preds.detach().cpu().numpy())
        self.test_labels.append(labels.detach().cpu().numpy())

        acc = self.accmetric(preds, labels)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
    
    def on_test_epoch_end(self):

        all_preds = np.concatenate(self.test_preds, axis=0)
        all_labels = np.concatenate(self.test_labels, axis=0)
        
        try:
            overall_auc = roc_auc_score(all_labels, all_preds, average="macro")
        except ValueError as e:
            self.log("test_auc", float('nan'))
            print(f"Error computing overall AUC: {e}")
        else:
            self.log("test_auc", overall_auc)
        
        num_classes = all_preds.shape[1]
        for i in range(num_classes):
            try:
                auc_per_class = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except ValueError as e:
                auc_per_class = float('nan')
                print(f"Error computing AUC for class {i}: {e}")
            self.log(f"{self.label_tag[i]}", auc_per_class)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train with pytorch-lightning')
    parser.add_argument('--config',default='configs/config.yaml',type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42)
    NUM_WORKERS = config['dataloader']['num_workers']
    BATCH_SZIE = config['dataloader']['batch_size']
    model_name = config['model']['name']

    dataset_cfg = config['dataset']
    dataset_train, dataset_val, dataset_test = create_dataset(**dataset_cfg)

    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SZIE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SZIE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SZIE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=True)

    model = TorchModule(model_name, config['model']['params'], lr=config['optimizer']['lr'],label_smoothing=config['train']['label_smoothing'],use_mixup=config['train']['use_mixup'])
    best_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_auc",
            mode="max",
            dirpath="./ckpt/",
            filename=f"{model_name}-best_metric_model",
        )
    trainer = pl.Trainer(
        max_epochs=config['train']['epochs'],
        accelerator="auto",
        precision= '16-mixed',
        devices=1,
        num_sanity_val_steps=2,
        logger=WandbLogger(project="SPH6004",name=model_name),
        callbacks=[LearningRateMonitor(logging_interval="step"),best_checkpoint_callback,ModelSummary(max_depth=3),EarlyStopping('val_auc',mode='max', patience=10)],
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader,ckpt_path=os.path.join("./ckpt/",f"{model_name}-best_metric_model.ckpt"))