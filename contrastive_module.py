import copy
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomRotation

from lib.wrn_contrastive import WideResNet
from lib.randaugment import RandAugment
from lib.sketchaug import SketchDeformation, ExpandChannels
from lib.data import *
from lib.core import *


def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def momentum_update(model_q, model_k, beta=0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
    model_k.load_state_dict(param_k)


class MomentumContrast(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        seed_all(self.hparams.seed)
        self.model_q = WideResNet(num_groups=3, N=4, k=self.hparams.width)


    def forward(self, x):
        return self.model_q(x)

    def on_train_start(self):
        "Initialize queue and momentum encoder"
        device = next(self.model_q.parameters()).device
        print(device)
        self.model_k = copy.deepcopy(self.model_q)
        self.queue = torch.randn(self.hparams.queue_size, 128, device=device)
        self.xent = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_n):
        "Momentum contrast"
        x_q, x_k = batch
        bs = x_q.size(0)
        qs = self.queue.size(0)

        q = self.model_q(x_q)
        q = F.normalize(q, p=2)

        with torch.no_grad():
            momentum_update(self.model_q, self.model_k)
            k = self.model_k(x_k)
            k = F.normalize(k, p=2)

        pos = torch.bmm(q.view(bs, 1, -1), k.view(bs, -1, 1)).view(bs, -1)
        neg = torch.mm(q, self.queue.T)

        logits = torch.cat([pos, neg], dim=-1)
        labels = torch.zeros(bs, device=logits.device, dtype=torch.long)   # positive pair goes first
        loss = self.xent(logits / self.hparams.temperature, labels)

        lr, mom = extract_opt_stats(self.trainer)
        log = {"train_loss": loss, "lr": lr, "mom": mom}

        with torch.no_grad():
            self.queue = queue_data(self.queue, k)
            self.queue = dequeue_data(self.queue, K=self.hparams.queue_size)

        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_n):
        "Contrastive loss on validation"
        x_q, x_k = batch
        bs = x_q.size(0)
        qs = self.queue.size(0)

        q = self.model_q(x_q)
        k = self.model_k(x_k)

        q = F.normalize(q, p=2)
        k = F.normalize(k, p=2)

        pos = torch.bmm(q.view(bs, 1, -1), k.view(bs, -1, 1)).view(bs, -1)
        neg = torch.mm(q, self.queue.T)

        logits = torch.cat([pos, neg], dim=-1)
        labels = torch.zeros(bs, device=logits.device, dtype=torch.long)   # positive pair goes first
        loss = self.xent(logits / self.hparams.temperature, labels)

        return {"val_loss": loss}

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
            val_tfm = Compose([ToTensor(), Normalize(*CIFAR_STATS)])

            sup_ds, unsup_ds = Cifar.uda_ds(dataset_path, n_labeled, n_overlap, seed=seed)
            val_ds = Cifar.val_ds(dataset_path, val_tfm)
            val_ds = Cifar(val_ds.data, transform0=train_tfm, transform1=train_tfm, mode=Mode.UNSUP)
            train_ds = Cifar(sup_ds.data + unsup_ds.data, transform0=train_tfm, transform1=train_tfm, mode=Mode.UNSUP)

        if self.hparams.dataset == "quickdraw":
            train_tfm = Compose([ExpandChannels, SketchDeformation, RandomHorizontalFlip(), RandomRotation(30), RandomCrop(128, 18), ToTensor()])
            val_tfm   = Compose([ExpandChannels, ToTensor()])

            sup_ds, unsup_ds = QuickDraw.uda_ds(dataset_path, n_labeled, n_overlap, seed=seed)
            train_ds = QuickDraw(sup_ds.data + unsup_ds.data, transform0=train_tfm, transform1=train_tfm, mode=Mode.UNSUP)
            val_ds = QuickDraw.val_ds(dataset_path, val_tfm)
            val_ds = QuickDraw(val_ds.data, transform0=train_tfm, transform1=train_tfm, mode=Mode.UNSUP)

        self.train_ds = train_ds
        self.valid_ds = val_ds

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())
        # NOTE trainer uses min(max_epochs, max_steps) to stop, we don't want that
        if not self.hparams.lr_find:
            self.trainer.max_epochs = self.trainer.max_steps // len(train_loader)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size*2, shuffle=False, drop_last=False, num_workers=os.cpu_count())


@config(config_path="conf/contrastive_cifar.yaml")
def train(hparams):
    print(hparams.pretty())

    if hparams.lr_find:
        print("Running lr finder")
        model = UDA(hparams)
        trainer = pl.Trainer(**hparams.trainer)

        lr_find = trainer.lr_find(model, min_lr=1e-7, max_lr=10)
        plot_lr_find(lr_find.results)
        exit(0)

    if hparams.wandb:
        print("Using wandb logger")
        import wandb
        logger = pl.loggers.WandbLogger(name=hparams.name, project=hparams.project, version=hparams.version, offline=hparams.offline)
        checkpoint_path = os.path.join(wandb_logger.experiment.dir, "checkpoints", "{epoch}-{val_acc:.2f}")
        checkpoints = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", period=hparams.trainer.check_val_every_n_epoch)
    else:
        from lib.checkpoint import ModelCheckpoint

        print("Using tensorboard logger")
        logger = pl.loggers.TensorBoardLogger(save_dir=os.environ["LOGS_PATH"], name=hparams.name)
        checkpoint_path = os.path.join(os.environ["SNAPSHOT_PATH"], "checkpoint")
        checkpoints = ModelCheckpoint(checkpoint_path, monitor="val_loss", period=hparams.trainer.check_val_every_n_epoch)

        if os.path.isfile(checkpoint_path):
            print("Resuming from latest checkpoint")
            hparams.trainer.resume_from_checkpoint = checkpoint_path + ".ckpt"

    model = MomentumContrast(hparams)
    trainer = pl.Trainer(logger=logger, checkpoint_callback=checkpoints, **hparams.trainer)
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(os.environ["SNAPSHOT_PATH"], "final.ckpt"))

if __name__ == "__main__":
    train()
