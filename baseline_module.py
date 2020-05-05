import os

import pytorch_lightning as pl
import torch
import wandb
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.wrn import WideResNet
from lib.randaugment import RandAugment
from lib.data import split_cifar, CIFAR_STATS
from lib.core import *


class Baseline(pl.LightningModule):
    "Base class for experiments setting up optimizers and dataloaders from config"
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        seed_all(self.hparams.seed)
        self.model = WideResNet(num_groups=3, N=4, num_classes=6, k=self.hparams.width)
        self.loss = LabelSmoothingLoss(6, self.hparams.smoothing)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_n):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        with torch.no_grad():
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
            lr, mom = extract_opt_stats(self.trainer)

        log = {"train_loss": loss, "train_acc": acc, "lr": lr, "mom": mom}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_n):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        log = {k: torch.stack([o[k] for o in outputs]).mean() for k in keys}
        return {"val_loss": log["val_loss"], "val_acc": log["val_acc"], "log": log}

    def configure_optimizers(self):
        "We use one-cycle scheduling policy in all experiments with AdamW optimizer as most reliable"
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr, total_steps=self.hparams.trainer.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def prepare_data(self):
        "Prepare supervised and unsupervised datasets from cifar"
        n = self.hparams.randaug_n
        m = self.hparams.randaug_m
        train_tfm = Compose([RandAugment(n, m), ToTensor(), Normalize(*CIFAR_STATS)])
        valid_tfm = Compose([ToTensor(), Normalize(*CIFAR_STATS)])

        train = CIFAR10(self.hparams.data_path, True, download=True)
        valid = CIFAR10(self.hparams.data_path, False)

        seed_all(self.hparams.seed)
        train_ds_l, train_ds_u, valid_ds = split_cifar(
            train, valid,
            n_labeled=self.hparams.n_labeled,
            n_overlap=self.hparams.n_overlap,
            labeled_tfm=train_tfm,
            unlabeled_tfm=train_tfm,
            valid_tfm=valid_tfm
        )

        self.train_ds = train_ds_l
        self.valid_ds = valid_ds

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        # NOTE trainer uses min(max_epochs, max_steps) to stop, we don't want that
        self.trainer.max_epochs = self.trainer.max_steps // len(train_loader)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size*2, shuffle=False, drop_last=False, num_workers=os.cpu_count())


@config(config_path="conf/baseline.yaml")
def train(hparams):
    print(hparams.pretty())

    if hparams.lr_find:
        print("Running lr finder")
        model = Baseline(hparams)
        trainer = pl.Trainer(**hparams.trainer)

        lr_find = trainer.lr_find(model, max_lr=10)
        plot_lr_find(lr_find.results)
        exit(0)

    wandb_logger = pl.loggers.WandbLogger(name=hparams.name, project=hparams.project, version=hparams.version, offline=hparams.offline)
    checkpoint_path = os.path.join(wandb_logger.experiment.dir, "checkpoints", "{epoch}-{val_acc:.2f}")
    checkpoints = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_acc", period=hparams.checkpoint_period)

    model = Baseline(hparams)
    trainer = pl.Trainer(logger=wandb_logger, checkpoint_callback=checkpoints, **hparams.trainer)
    trainer.fit(model)

if __name__ == "__main__":
    train()
