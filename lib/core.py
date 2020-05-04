__all__ = ["seed_all", "config", "extract_opt_stats", "LabelSmoothingLoss", "Flatten", "plot_lr_find"]

import argparse
import random
import functools
import os

import omegaconf
import yaml
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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def plot_lr_find(results):
    # Setup pyplot to plot to console if itermplot is available
    os.environ["MPLBACKEND"] = "module://itermplot"
    os.environ["ITERMPLOT"] = "rv"

    import matplotlib.pyplot as plt

    lr = results["lr"]
    loss = results["loss"]

    min_loss = np.argmin(loss)
    max_slope = np.argmin(np.gradient(loss))

    min_idx = np.searchsorted(lr, lr[min_loss]/10)

    plt.figure(figsize=(10,5))
    plt.semilogx(lr, loss)
    plt.scatter(lr[min_idx], loss[min_idx], label=f"lr={lr[min_loss]/10:.4f}", c="red", zorder=10)
    plt.scatter(lr[max_slope], loss[max_slope], label=f"lr={lr[max_slope]:.4f}", c="green", zorder=10)
    plt.legend()
    plt.show()


def config(config_path):
    "Configuration decorator, combines argparse with yaml config"

    parser = argparse.ArgumentParser(add_help=False, description="Trainer launcher")
    parser.add_argument("--help", "-h", action="store_true", help="Script help")
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    with open(config_path) as f:
        cfg = omegaconf.DictConfig(yaml.safe_load(f))

    if args.help:
        print("Config:", end="\n\n")
        print(cfg.pretty(), end="\n")
        print("Provide any key=value arguments to override config values (use dots for.nested=overrides)")
        exit(0)

    cfg.merge_with_dotlist(args.overrides)

    def config_merge_wrapper(train_function):
        @functools.wraps(train_function)
        def decorated():
            train_function(cfg)

        return decorated

    return config_merge_wrapper


