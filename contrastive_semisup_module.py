import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomRotation

from lib.wrn_contrastive import WideResNet
from lib.randaugment import RandAugment
from lib.sketchaug import SketchDeformation, ExpandChannels
from lib.data import *
from lib.core import *


def momentum_update(model_q, model_k, m=0.999):
    "model_k = m * model_k + (1 - m) model_q"
    for p1, p2 in zip(model_q.parameters(), model_k.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def enqueue(queue, k):
    return torch.cat([queue, k], dim=0)


def dequeue(queue, max_len):
    if queue.shape[0] >= max_len:
        return queue[-max_len:]  # queue follows FIFO
    else:
        return queue

class MomentumContrast(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        seed_all(self.hparams.seed)
        self.model_q = WideResNet(num_groups=3, N=4, k=self.hparams.width)
        self.model_k = WideResNet(num_groups=3, N=4, k=self.hparams.width)
        momentum_update(self.model_q, self.model_k, 0)
        for param in self.model_k.parameters():
            param.requires_grad = False

        n_classes = 6 if self.hparams.dataset == "cifar" else 10
        self.classifier = nn.Linear(128, n_classes)
        self.xent = nn.CrossEntropyLoss()
        if hparams.trainer.resume_from_checkpoint is not None:
            self.register_buffer("queue", torch.randn(hparams.queue_size, 128, requires_grad=False))
        else:
            self.register_buffer("queue", torch.randn(100, 128, requires_grad=False))

    def forward(self, x):
        return self.model_q(x)

    def training_step(self, batch, batch_n):
        "Momentum contrast"
        (x,y), (x_q, x_k) = batch
        bs = x_q.size(0)
        qs = self.queue.size(0)

        q = self.model_q(x_q)
        q = F.normalize(q, p=2)

        momentum_update(self.model_q, self.model_k)
        k = self.model_k(x_k).detach()
        k = F.normalize(k, p=2)

        pos = torch.bmm(q.view(bs, 1, -1), k.view(bs, -1, 1)).view(bs, -1)
        neg = torch.mm(q, self.queue.T)

        logits = torch.cat([pos, neg], dim=-1) / self.hparams.temperature
        labels = torch.zeros(bs, device=logits.device, dtype=torch.long)   # positive pair goes first
        contrastive_loss = self.xent(logits, labels)

        # Combine contrastive loss with standard classification
        # using same backbone
        y_hat = self.classifier(self.model_q.features(x))
        xent = self.xent(y_hat, y)
        loss = xent + 0.5 * contrastive_loss

        contrastive_pred = logits[:, 0].mean()

        lr, mom = extract_opt_stats(self.trainer)
        log = {"train_loss": loss, "train_xent": xent, "contrastive_loss": loss, "contrastive_pred": contrastive_pred,  "lr": lr, "mom": mom}

        self.queue = enqueue(self.queue, k)
        self.queue = dequeue(self.queue, max_len=self.hparams.queue_size)

        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_n):
        "Contrastive loss on validation"
        x, y = batch
        y_hat = self.classifier(self.model_q.features(x))
        loss = self.xent(y_hat, y).mean()
        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        return {"val_xent": loss, "val_acc", acc}

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        log = {k: torch.stack([o[k] for o in outputs]).mean() for k in keys}
        return {"val_loss": log["val_loss"], "log": log}

    def configure_optimizers(self):
        "We use one-cycle scheduling policy in all experiments with AdamW optimizer as most reliable"
        optimizer = torch.optim.AdamW(self.model_q.parameters(), lr=self.hparams.lr)
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
            train_tfm = Compose([RandAugment(n, m), ToTensor(), Normalize(*CIFAR_STATS)])
            val_tfm   = Compose([ToTensor(), Normalize(*CIFAR_STATS)])

            sup_ds, unsup_ds = Cifar.uda_ds(dataset_path, n_labeled, n_overlap, sup_transform=train_tfm, uda_transform=train_tfm, seed=seed)
            val_ds = Cifar.val_ds(dataset_path, val_tfm)

        if self.hparams.dataset == "quickdraw":
            train_tfm = Compose([ExpandChannels, SketchDeformation, RandomHorizontalFlip(), RandomRotation(30), RandomCrop(128, 18), ToTensor()])
            val_tfm   = Compose([ExpandChannels, ToTensor()])

            sup_ds, unsup_ds = QuickDraw.uda_ds(dataset_path, n_labeled, n_overlap, sup_transform=train_tfm, uda_transform=train_tfm, seed=seed)
            val_ds = QuickDraw.val_ds(dataset_path, val_tfm)

        self.sup_ds = sup_ds
        self.unsup_ds = unsup_ds
        self.valid_ds = val_ds

    def train_dataloader(self):
        train_loader_sup = DataLoader(self.sup_ds, batch_size=self.hparams.batch_size_l, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        train_loader_unsup = DataLoader(self.unsup_ds, batch_size=self.hparams.batch_size_u, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        train_loader = ConcatDataLoader(train_loader_sup, train_loader_unsup)

        # NOTE trainer uses min(max_epochs, max_steps) to stop, we don't want that
        if not self.hparams.lr_find:
            self.trainer.max_epochs = self.trainer.max_steps // len(train_loader)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size_l*2, shuffle=False, drop_last=False, num_workers=os.cpu_count())

    def on_epoch_end(self):
        self.trainer.save_checkpoint(os.path.join(os.environ.get("SNAPSHOT_PATH"), "checkpoint.ckpt"))

@config(config_path="conf/contrastive_semisup_cifar.yaml")
def train(hparams):
    print(hparams.pretty())

    if hparams.lr_find:
        print("Running lr finder")
        model = MomentumContrast(hparams)
        trainer = pl.Trainer(**hparams.trainer)

        lr_find = trainer.lr_find(model, min_lr=1e-7, max_lr=10, num_training=500)
        plot_lr_find(lr_find.results)
        exit(0)

    if hparams.wandb:
        print("Using wandb logger")
        import wandb
        logger = pl.loggers.WandbLogger(name=hparams.name, project=hparams.project, version=hparams.version, offline=hparams.offline)
        checkpoint_path = os.path.join(logger.experiment.dir, "checkpoints", "{epoch}-{val_acc:.2f}")
        checkpoints = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", period=hparams.trainer.check_val_every_n_epoch)
    else:
        from lib.checkpoint import ModelCheckpoint

        print("Using tensorboard logger")
        logger = pl.loggers.TensorBoardLogger(save_dir=os.environ["LOGS_PATH"], name=hparams.name)
        checkpoint_path = os.path.join(os.environ["SNAPSHOT_PATH"], "checkpoint.ckpt")
        if os.path.isfile(checkpoint_path):
            print("Resuming from latest checkpoint")
            hparams.trainer.resume_from_checkpoint = checkpoint_path

    model = MomentumContrast(hparams)
    trainer = pl.Trainer(logger=logger, **hparams.trainer)
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(os.environ.get("SNAPSHOT_PATH", "."), "final.ckpt"))

if __name__ == "__main__":
    train()
