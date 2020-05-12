__all__ = ["seed_all", "config", "extract_opt_stats", "LabelSmoothingLoss", "Flatten", "plot_lr_find"]

import argparse
import random
import functools
import os

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
    plt.scatter(lr[min_idx], loss[min_idx], label="lr={:.6f}".format(lr[min_loss]/10), c="red", zorder=10)
    plt.scatter(lr[max_slope], loss[max_slope], label="lr={:.6f}".format(lr[max_slope]), c="green", zorder=10)
    plt.legend()
    plt.show()


class DotDict(dict):
    "A dictionary that supports dot notation"
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def merge_with_dotlist(self, dotlist):
        for arg in dotlist:
            if not isinstance(arg, str):
                raise ValueError("Invalid item in dotlist")

            idx = arg.find("=")
            if idx == -1:
                key = arg
                value = None
            else:
                key = arg[0:idx]
                value = arg[idx + 1 :]
                value = yaml.load(value, Loader=yaml.SafeLoader)
                self.update(key, value)

    def update(self, key, value):
        "Update from key.subkey=value notation"
        split = key.split(".")
        root = self
        for k in split[:-1]:
            next_root = self.get(k, None)
            if next_root is None:
                root[k] = DotDict()
            root = root[k]

        last = split[-1]
        root[last] = value

    @staticmethod
    def to_dict(container):
        "Convert to python dict"
        if not isinstance(container, dict):
            return container

        retdict = {}
        for key, value in container.items():
            retdict[key] = DotDict.to_dict(value)
        return retdict

    def pretty(self):
        "Pretty print config"
        return yaml.dump(DotDict.to_dict(self), Dumper=yaml.SafeDumper)


def config(config_path):
    "Configuration decorator, combines argparse with yaml config"
    def config_merge_wrapper(train_function):

        @functools.wraps(train_function)
        def decorated():
            parser = argparse.ArgumentParser(add_help=False, description="Trainer launcher")
            parser.add_argument("--help", "-h", action="store_true", help="Script help")
            parser.add_argument("--config", type=str, default=config_path)
            parser.add_argument("overrides", nargs="*")
            args = parser.parse_args()

            with open(args.config) as f:
                cfg = DotDict(yaml.safe_load(f))

            if args.help:
                print("Config:", end="\n\n")
                print(cfg.pretty(), end="\n")
                print("Provide any key=value arguments to override config values (use dots for.nested=overrides)")
                exit(0)

            cfg.merge_with_dotlist(args.overrides)
            train_function(cfg)

        return decorated

    return config_merge_wrapper


