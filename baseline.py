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
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Ensure same initialization
        seed_all(self.hparams.model.seed)
        self.wrn = WideResNet(4, 3, 6, k=10)
        self.loss = LabelSmoothingLoss(6, smoothing=self.hparams.model.label_smoothing)

    def forward(self, x):
        return self.wrn(x)

    def training_step(self, batch, batch_n):
        x, y = batch
        y_hat = self.wrn(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        # Extract optimizer params
        opt = self.trainer.optimizers[0]
        lr = opt.param_groups[0]['lr']
        momentum = opt.param_groups[0]['betas'][0]

        log = {"train_loss": loss.item(), "train_acc": acc.item(), "lr": lr, "momentum": momentum}
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
        optimizer = torch.optim.AdamW(self.wrn.parameters(), lr=self.hparams.model.max_lr, weight_decay=self.hparams.model.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.model.max_lr,
            total_steps=self.hparams.trainer.max_steps,
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
        train = CIFAR10(self.hparams.data.path, True, download=True)
        val = CIFAR10(self.hparams.data.path, False, download=True)

        seed_all(self.hparams.data.seed)
        classes, _ = get_image_classes(self.hparams.data.n_overlap, train.class_to_idx)
        train_images, train_targets, *_ = split_cifar(train.data, train.targets, classes, [], n_labeled=self.hparams.data.n_labeled)
        val_images, val_targets, *_ = split_cifar(val.data, val.targets, classes, [], n_labeled=1000)

        train_tfm = transforms.Compose([RandAugment(), transforms.ToTensor(), transforms.Normalize(*CIFAR_STATS)])
        val_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*CIFAR_STATS)])
        self.train_ds = ImageDataset(train_images, train_targets, tfm=train_tfm)
        self.val_ds = ImageDataset(val_images, val_targets, tfm=val_tfm)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.hparams.model.batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.hparams.model.batch_size*2, shuffle=False, num_workers=os.cpu_count())


@hydra.main(config_path="conf/baseline.yaml", strict=False)
def train(hparams):
    print(hparams.pretty())

    if hparams.run.lr_find:
        # Works from iterm with itermplot
        model = SupervisedClassifier(hparams)
        trainer = pl.Trainer(**hparams.trainer)
        lr_find = trainer.lr_find(model, max_lr=10)
        lr_find.plot(suggest=True, show=True)

    logger = pl.loggers.WandbLogger(
        name=hparams.run.name,
        project=hparams.run.project,
        version=hparams.run.version,
        save_dir=os.getcwd()
    )
    path = logger.experiment.dir
    ckpt = None

    if hparams.run.version and hparams.run.restore_file and hparams.run.restore_run:
        ckpt = wandb.restore(
            name=hparams.run.restore_file,
            run_path=hparams.run.restore_run,
            root=os.path.join(path,"checkpoints")
        )
        print(f"Resuming training from checkpoint {ckpt.name}")

    model = SupervisedClassifier(hparams)
    model.prepare_data()

    checkpoints = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(path, "checkpoints", "{epoch}"),
        period=hparams.run.checkpoint_period,
    )

    hparams.trainer.max_epochs = hparams.trainer.max_steps // len(model.train_dataloader())

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoints,
        resume_from_checkpoint=ckpt and ckpt.name,
        **hparams.trainer
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
