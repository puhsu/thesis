import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomRotation

from lib.wrn import WideResNet
from lib.randaugment import RandAugment
from lib.sketchaug import SketchDeformation, ExpandChannels
from lib.data import *
from lib.core import *


class Baseline(pl.LightningModule):
    "Base class for experiments setting up optimizers and dataloaders from config"
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        seed_all(self.hparams.seed)
        n_classes = 6 if self.hparams.dataset == "cifar" else 10

        self.model = WideResNet(num_groups=3, N=4, num_classes=n_classes, k=self.hparams.width)
        self.loss = nn.CrossEntropyLoss()

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
        dataset_path = self.hparams.dataset_path
        n_labeled = self.hparams.n_labeled
        n_overlap = self.hparams.n_overlap
        seed = self.hparams.seed

        if self.hparams.dataset == "cifar":
            n = self.hparams.randaug_n
            m = self.hparams.randaug_m
            if self.hparams.strong_tfm:
                train_tfm = Compose([RandAugment(n, m), ToTensor(), Normalize(*CIFAR_STATS)])
            else:
                train_tfm = Compose([RandomCrop(32, 4, padding_mode="reflect"), RandomHorizontalFlip(), ToTensor(), Normalize(*CIFAR_STATS)])
            valid_tfm = Compose([ToTensor(), Normalize(*CIFAR_STATS)])
            sup_ds, unsup_ds = Cifar.uda_ds(dataset_path, n_labeled, n_overlap, train_tfm, seed=seed)
            val_ds = Cifar.val_ds(dataset_path, valid_tfm)

        if self.hparams.dataset == "quickdraw":
            train_tfm = Compose([ExpandChannels, SketchDeformation, RandomHorizontalFlip(), RandomRotation(30), RandomCrop(128, 18), ToTensor()])
            valid_tfm = Compose([ExpandChannels, ToTensor()])
            sup_ds, unsup_ds = QuickDraw.uda_ds(dataset_path, n_labeled, n_overlap, train_tfm, seed=seed)
            val_ds = QuickDraw.val_ds(dataset_path, valid_tfm)

        self.train_ds = sup_ds
        self.valid_ds = val_ds
        print("Loaded {} train examples and {} validation examples".format(len(self.train_ds), len(self.valid_ds)))


    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        # NOTE trainer uses min(max_epochs, max_steps) to stop, we don't want that
        if not self.hparams.lr_find:
            self.trainer.max_epochs = self.trainer.max_steps // len(train_loader)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size*2, shuffle=False, drop_last=False, num_workers=os.cpu_count())


@config(config_path="conf/baseline_cifar.yaml")
def train(hparams):
    print(hparams.pretty())

    if hparams.lr_find:
        print("Running lr finder")
        model = Baseline(hparams)
        trainer = pl.Trainer(**hparams.trainer)

        lr_find = trainer.lr_find(model, min_lr=1e-7, max_lr=10)
        plot_lr_find(lr_find.results)
        exit(0)

    if hparams.wandb:
        print("Using wandb logger")
        import wandb
        logger = pl.loggers.WandbLogger(name=hparams.name, project=hparams.project, version=hparams.version, offline=hparams.offline)
        checkpoint_path = os.path.join(wandb_logger.experiment.dir, "checkpoints", "{epoch}-{val_acc:.2f}")
        checkpoints = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_acc", period=hparams.trainer.check_val_every_n_epoch)
    else:
        from lib.checkpoint import ModelCheckpoint

        print("Using tensorboard logger")
        logger = pl.loggers.TensorBoardLogger(save_dir=os.environ["LOGS_PATH"], name=hparams.name)
        checkpoint_path = os.path.join(os.environ["SNAPSHOT_PATH"], "checkpoint.pth")
        checkpoints = ModelCheckpoint(checkpoint_path, monitor="val_acc", period=hparams.trainer.check_val_every_n_epoch)

        if os.path.isfile(checkpoint_path):
            print("Resuming from latest checkpoint")
            hparams.trainer.resume_from_checkpoint = checkpoint_path

    model = Baseline(hparams)
    trainer = pl.Trainer(logger=logger, checkpoint_callback=checkpoints, **hparams.trainer)
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(os.environ["SNAPSHOT_PATH"], "final.pth"))

if __name__ == "__main__":
    train()
