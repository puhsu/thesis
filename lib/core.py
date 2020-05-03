__all__ = ["seed_all"]

import argparse
import random
import os

import omegaconf
import torch
import torch.nn as nn
import numpy as np


def seed_all(seed=69):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_opt_stats(trainer):
    "Return pytorch optimizer lr and momentum"
    opt = trainer.optimizers[0]
    lr = opt.param_groups[0]['lr']
    momentum = opt.param_groups[0]['betas'][0]
    return lr, momentum


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


def config(config_path):
    "Configuration decorator, combines argparse with yaml config"

    parser = argparse.ArgumentParser(add_help=False, description="Trainer launcher")
    parser.add_argument("--help, -h", action="store_true", help="Script help")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
