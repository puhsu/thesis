import os
import hydra
import wandb

import pytorch_lightning as pl
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.wrn import WideResNet
from lib.randaugment import RandAugment
from lib.data import split_cifar, CIFAR_STATS
from lib.core import LabelSmoothingLoss, seed_all, extract_opt_stats, config


class Baseline(pl.LightningModule):
    "Base class for experiments setting up optimizers and dataloaders from config"
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        seed_all(self.cfg.seed)
        self.model = WideResNet(num_groups=4, N=3, num_classes=6, k=self.cfg.width)
        self.loss = LabelSmoothingLoss(6, self.cfg.smoothing)

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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.lr, total_steps=self.cfg.trainer.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def prepare_data(self):
        "Prepare supervised and unsupervised datasets from cifar"
        train_tfm = Compose([RandAugment(), ToTensor(), Normalize(*CIFAR_STATS)])
        valid_tfm = Compose([ToTensor(), Normalize(*CIFAR_STATS)])

        train = CIFAR10(self.cfg.data_path, True, download=True)
        valid = CIFAR10(self.cfg.data_path, False)

        seed_all(self.cfg.seed)
        train_ds_l, train_ds_u, valid_ds = split_cifar(
            train, valid,
            n_labeled=self.cfg.n_labeled,
            n_overlap=self.cfg.n_overlap,
            labeled_tfm=train_tfm,
            unlabeled_tfm=train_tfm,
            valid_tfm=valid_tfm
        )

        self.train_ds = train_ds_l
        self.valid_ds = valid_ds

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        # NOTE trainer uses min(max_epochs, max_steps) to stop, we don't want that
        self.trainer.max_epochs = self.trainer.max_steps // len(train_loader)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.cfg.batch_size*2, shuffle=False, drop_last=False, num_workers=os.cpu_count())


@hydra.main(config_path="conf/baseline.yaml", strict=False)
def train(cfg):
    print(cfg.pretty())

    if cfg.lr_find:
        # For plotting in iterm
        os.environ["MPLBACKEND"] = "module://itermplot"
        os.environ["ITERMPLOT"] = "rv"

        print("Running lr finder")
        model = Baseline(cfg)
        trainer = pl.Trainer(**cfg.trainer)

        lr_find = trainer.lr_find(model, max_lr=10)
        lr_find.plot(suggest=True, show=True)
        print("Suggested learning rate:", lr_find.suggestion())
        exit(0)

    logger = pl.loggers.WandbLogger(name=cfg.name, project=cfg.project, version=cfg.version)
    checkpoint_path = os.path.join(logger.experiment.dir, "checkpoints", "{epoch}-{val_acc:.2f}")
    checkpoints = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_acc", period=cfg.checkpoint_period)

    model = Baseline(cfg)
    trainer = pl.Trainer(logger=logger, checkpoint_callback=checkpoints, **cfg.trainer)
    trainer.fit(model)

if __name__ == "__main__":
    train()
