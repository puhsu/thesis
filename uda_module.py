import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomRotation

from lib.wrn import WideResNet
from lib.randaugment import RandAugment
from lib.sketchaug import SketchDeformation
from lib.data import *
from lib.core import *


class UDA(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        seed_all(self.hparams.seed)
        inp_nf = 3 if self.hparams.dataset == "cifar" else 1
        n_classes = 6 if self.hparams.dataset == "cifar" else 10

        self.model = WideResNet(num_groups=3, N=4, num_classes=n_classes, k=self.hparams.width, inp_nf=inp_nf)
        self.xent = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_n):
        "Unsupervised data augmentation loss computation"
        (x, y), (x_ori, x_aug) = batch

        sup_logits = self.forward(x)
        l_loss = self.xent(sup_logits, y)
        aug_logits = self.forward(x_aug)

        with torch.no_grad():
            ori_logits = self.forward(x_ori) / self.hparams.uda_softmax_temp
            sup_prob = F.softmax(sup_logits, dim=-1)
            ori_prob = F.softmax(ori_logits, dim=-1)
            aug_prob = F.softmax(aug_logits, dim=-1)

            acc = (sup_logits.argmax(dim=-1) == y).float().mean()
            lr, mom = extract_opt_stats(self.trainer)

        sup_max_prob = sup_prob.max(dim=-1).values
        ori_max_prob = ori_prob.max(dim=-1).values
        aug_max_prob = aug_prob.max(dim=-1).values

        u_loss = F.kl_div(F.log_softmax(aug_logits, dim=-1), ori_prob, reduction="none").sum(dim=-1)
        u_mask = (ori_max_prob > self.hparams.uda_confidence_threshold).float()
        u_loss = (u_loss * u_mask).sum() / (u_mask.sum() + 1e-6)

        if self.hparams.uda_tsa:
            # anneal supervised loss
            if self.hparams.lr_find:
                t = 0
            else:
                t = self.trainer.global_step / self.trainer.max_steps
            tsa_conf_threshold = t * (1 - 1/6) + 1/6
            l_mask = (sup_max_prob > tsa_conf_threshold).float()
            l_loss = (l_loss * l_mask).sum() / l_mask.sum()
        else:
            l_loss = l_loss.mean()

        loss = l_loss + self.hparams.uda_loss_weight * u_loss

        log = {
            "train_full_loss": loss,
            "train_acc": acc,
            "train_xent": l_loss,
            "train_kl": u_loss,
            "sup_max_prob": sup_max_prob.mean(),
            "unsup_ori_max_prob": ori_max_prob.mean(),
            "unsup_aug_max_prob": aug_max_prob.mean(),
            "confident_ratio": u_mask.mean(),
            "lr": lr,
            "mom": mom,
        }
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_n):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.xent(y_hat, y).mean()
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return {"val_xent": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        log = {k: torch.stack([o[k] for o in outputs]).mean() for k in keys}
        return {"val_xent": log["val_xent"], "val_acc": log["val_acc"], "log": log}

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

            uda_tfm = Compose([RandAugment(n, m), ToTensor(), Normalize(*CIFAR_STATS)])
            sup_tfm = Compose([RandomCrop(32, 4, padding_mode="reflect"), RandomHorizontalFlip(), ToTensor(), Normalize(*CIFAR_STATS)])
            val_tfm = Compose([ToTensor(), Normalize(*CIFAR_STATS)])
            sup_ds, unsup_ds = Cifar.uda_ds(dataset_path, n_labeled, n_overlap, sup_tfm, uda_tfm, seed=seed)
            val_ds = Cifar.val_ds(dataset_path, val_tfm)

        if self.hparams.dataset == "quickdraw":
            uda_tfm = Compose([SketchDeformation, RandomHorizontalFlip(), RandomRotation(30), RandomCrop(128, 18), ToTensor()])
            sup_tfm = Compose([RandomCrop(128, 9), RandomHorizontalFlip(), ToTensor()])
            val_tfm = ToTensor()
            sup_ds, unsup_ds = QuickDraw.uda_ds(dataset_path, n_labeled, n_overlap, sup_tfm, uda_tfm, seed=seed)
            val_ds = QuickDraw.val_ds(dataset_path, val_tfm)

        self.train_ds_sup = sup_ds
        self.train_ds_unsup = unsup_ds
        self.valid_ds = val_ds

    def train_dataloader(self):
        train_loader_sup = DataLoader(self.train_ds_sup, batch_size=self.hparams.batch_size_l, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        train_loader_unsup = DataLoader(self.train_ds_unsup, batch_size=self.hparams.batch_size_u, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        train_loader = ConcatDataLoader(train_loader_sup, train_loader_unsup)

        # NOTE trainer uses min(max_epochs, max_steps) to stop, we don't want that
        if not self.hparams.lr_find:
            self.trainer.max_epochs = self.trainer.max_steps // len(train_loader)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size_l*2, shuffle=False, drop_last=False, num_workers=os.cpu_count())


@config(config_path="conf/uda.yaml")
def train(hparams):
    print(hparams.pretty())

    if hparams.lr_find:
        print("Running lr finder")
        model = UDA(hparams)
        trainer = pl.Trainer(**hparams.trainer)

        lr_find = trainer.lr_find(model, min_lr=1e-7, max_lr=10)
        plot_lr_find(lr_find.results)
        exit(0)

    wandb_logger = pl.loggers.WandbLogger(name=hparams.name, project=hparams.project, version=hparams.version, offline=hparams.offline)
    checkpoint_path = os.path.join(wandb_logger.experiment.dir, "checkpoints", "{epoch}-{val_acc:.2f}")
    checkpoints = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_acc", period=hparams.checkpoint_period)

    model = UDA(hparams)
    trainer = pl.Trainer(logger=wandb_logger, checkpoint_callback=checkpoints, **hparams.trainer)
    trainer.fit(model)

if __name__ == "__main__":
    train()
