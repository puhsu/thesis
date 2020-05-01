import os

import wandb
import hydra

import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch_optimizer as optim

from torchvision.datasets import CIFAR10
from torchvision import transforms

from lib.data import split_cifar, get_image_classes, ImageDataset, CIFAR_STATS
from lib.wrn import WideResNet
from lib.randaugment import RandAugment
from lib.core import seed_all


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SupervisedClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Ensure same initialization
        seed_all(self.cfg.model.seed)
        self.wrn = WideResNet(4, 3, 6, k=2)
        self.loss = LabelSmoothingLoss(6, smoothing=self.cfg.model.label_smoothing)

    def forward(self, x):
        return self.wrn(x)

    def training_step(self, batch, batch_n):
        x, y = batch
        y_hat = self.wrn(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        log = {"train_loss": loss.item(), "train_acc": acc.item(), "lr": self.trainer}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_n):
        x, y = batch
        y_hat = self.wrn(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()
        val_acc_mean = torch.stack([o["val_acc"] for o in outputs]).mean()
        log = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        return {"val_loss": val_loss_mean, "log": log}

    def configure_optimizers(self):
        optimizer = optim.Ranger(self.wrn.parameters(), lr=self.cfg.model.max_lr, weight_decay=self.cfg.model.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.model.max_lr,
            total_steps=self.cfg.trainer.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # Data preparation below

    def prepare_data(self):
        train = CIFAR10(self.cfg.data.path, True, download=True)
        val = CIFAR10(self.cfg.data.path, False, download=True)

        seed_all(self.cfg.data.seed)
        classes, _ = get_image_classes(self.cfg.data.n_overlap, train.class_to_idx)
        train_images, train_targets, *_ = split_cifar(train.data, train.targets, classes, [], n_labeled=self.cfg.data.n_labeled)
        val_images, val_targets, *_ = split_cifar(val.data, val.targets, classes, [], n_labeled=1000)

        train_tfm = transforms.Compose([RandAugment(), transforms.ToTensor(), transforms.Normalize(*CIFAR_STATS)])
        val_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*CIFAR_STATS)])
        self.train_ds = ImageDataset(train_images, train_targets, tfm=train_tfm)
        self.val_ds = ImageDataset(val_images, val_targets, tfm=val_tfm)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.cfg.model.batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.cfg.model.batch_size*2, shuffle=False, num_workers=os.cpu_count())


@hydra.main(config_path="conf/baseline.yaml", strict=False)
def train(cfg):
    print(cfg.pretty())

    if cfg.run.lr_find:
        # Works from iterm with itermplot
        model = SupervisedClassifier(cfg)
        trainer = pl.Trainer()
        lr_find = trainer.lr_find(model, max_lr=10)
        lr_find.plot(suggest=True, show=True)

    logger = pl.loggers.WandbLogger(
        name=cfg.run.name,
        project=cfg.run.project,
        version=cfg.run.version,
        save_dir=os.getcwd()
    )
    path = logger.experiment.dir
    ckpt = None

    if cfg.run.version and cfg.run.restore_from:
        ckpt = wandb.restore(cfg.restore_from, root=os.path.join(path,"checkpoints"))
        print(f"Resuming training from checkpoint {ckpt.name}")

    model = SupervisedClassifier(cfg)
    model.prepare_data()

    checkpoints = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(path, "checkpoints", "{epoch}"),
        period=cfg.checkpoint_period,
    )

    cfg.trainer.max_epochs = cfg.trainer.max_steps // len(model.train_dataloader())

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoints,
        resume_from_checkpoint=ckpt and ckpt.name,
        **cfg.trainer
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
