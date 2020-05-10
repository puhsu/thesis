__all__ = ["QuickDraw", "Cifar", "CIFAR_STATS", "ConcatDataLoader"]

import os
import tarfile
import torch
import torchvision
import requests
import numpy as np
import enum

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from tqdm.autonotebook import tqdm

from PIL import Image
from collections import Counter
from pathlib import Path

from .core import seed_all


CIFAR_STATS = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])


class Mode(enum.Enum):
    SUP = 1
    UNSUP = 2


def get_image_files(path):
    "Recursively get all images from `path`"
    path = Path(path)
    res = []
    for p,d,f in os.walk(path):
        p = Path(p)
        res += [p/f for f in f if f.endswith(".png")]
    return res


class ConcatDataLoader:
    "Iterate multiple dataloaders (expects loaders with shuffle=True)"
    def __init__(self, *loaders):
        self.loaders = loaders

    def __iter__(self):
        for output in zip(*self.loaders):
            yield output

    def __len__(self):
        return min(len(l) for l in self.loaders)


class Cifar(Dataset):
    "CIFAR-10 wrapper with factory methods"

    sup_classes = ["bird", "cat", "deer", "dog", "frog", "horse"]
    other_classes = ["airplane", "automobile", "ship", "truck"]

    def __init__(self, data, targets=None, transform0=None, transform1=None, mode=Mode.UNSUP):
        self.mode = mode
        self.data = data
        self.targets = targets
        self.transform0 = transform0 or Compose([ToTensor(), Normalize(*CIFAR_STATS)])
        self.transform1 = transform1 or Compose([ToTensor(), Normalize(*CIFAR_STATS)])

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])

        if self.mode == Mode.SUP:
            return self.transform0(img), self.targets[i]
        if self.mode == Mode.UNSUP:
            return self.transform0(img), self.transform1(img)

    def __len__(self):
        return len(self.data)


    @classmethod
    def uda_ds(cls, path="data", n_labeled=400, n_overlap=4, sup_transform=None, uda_transform=None, seed=69):
        seed_all(seed)
        cifar = CIFAR10(path, train=True, download=True)

        sup_classes = cls.sup_classes
        other_classes = cls.other_classes

        unsup_classes = (
            list(np.random.choice(sup_classes, size=n_overlap, replace=False)) +
            list(np.random.choice(other_classes, size=4-n_overlap, replace=False))
        )

        sup_classes   = [cifar.class_to_idx[c] for c in sup_classes]
        unsup_classes = [cifar.class_to_idx[c] for c in unsup_classes]
        old_to_new = {cifar.class_to_idx[c]: i for i, c in enumerate(cls.sup_classes)}

        sup_data = []
        sup_targets = []
        unsup_data = []
        counts = Counter()

        for i in np.random.permutation(len(cifar.targets)):
            x, y = cifar.data[i], cifar.targets[i]
            counts[y] += 1
            if y in sup_classes and counts[y] <= n_labeled:
                sup_data.append(x)
                sup_targets.append(old_to_new[y])
            elif y in unsup_classes and counts[y] > n_labeled:
                unsup_data.append(x)

        sup_ds = cls(sup_data, sup_targets, sup_transform, mode=Mode.SUP)
        unsup_ds = cls(unsup_data, None, sup_transform, uda_transform, mode=Mode.UNSUP)
        return sup_ds, unsup_ds

    @classmethod
    def val_ds(cls, path="cifar", transform=None):
        "Creates cifar validation set with sup classes only"

        cifar = CIFAR10(path, train=False, download=True)
        old_to_new = {cifar.class_to_idx[c]: i for i, c in enumerate(cls.sup_classes)}

        data = []
        targets = []

        for x, y in zip(cifar.data, cifar.targets):
            if cifar.classes[y] in cls.sup_classes:
                data.append(x)
                targets.append(old_to_new[y])

        return cls(data, targets, transform, mode=Mode.SUP)




class QuickDraw(Dataset):
    "QuickDraw dataset with factory methods"

    sup_classes   = ["octopus", "pig", "bird", "cat", "bee", "crab", "squirrel", "face", "anvil", "flower"]
    other_classes = ["eyeglasses", "bowtie", "bus", "syringe", "birthday cake", "basketball",
                     "bicycle", "laptop", "clock", "guitar"]

    @staticmethod
    def download(path, file_id="1urjhim8CmqgcyaJfHjO83VyzbBbRqIBA"):
        "Downloads prepared quickdraw dataset from google-drive"

        path = Path(path)
        if (path/"quickdraw").exists():
            print("Already downloaded")
            return

        URL = "https://docs.google.com/uc?export=download"

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response):
            CHUNK_SIZE = 32768
            with open(path/"quickdraw.tar.gz", "wb") as f:
                for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response)

        with tarfile.open(path/"quickdraw.tar.gz") as tar:
            tar.extractall(path)


    def __init__(self, data, targets=None, transform0=None, transform1=None, mode=Mode.UNSUP):
        """Default constructor for 2 different modes:
           For supervised mode transforms should be transform0
           For unsupervised mode transforms should be both transform0 and transform1 (as in uda or moco)
        """
        self.mode = mode
        self.data = data
        self.targets = targets
        self.transform0 = transform0 or ToTensor()
        self.transform1 = transform1 or ToTensor()

    def __getitem__(self, i):
        img = Image.open(self.data[i])

        if self.mode == Mode.SUP:
            return self.transform0(img), self.targets[i]
        if self.mode == Mode.UNSUP:
            return self.transform0(img), self.transform1(img)

    def __len__(self):
        return len(self.data)

    @classmethod
    def uda_ds(cls, path="data", n_labeled=1000, n_overlap=10, sup_transform=None, uda_transform=None, seed=69):
        "Creates labeled and unlabeled datasets for semi-supervised learning with uda"
        path = Path(path)
        cls.download(path)
        seed_all(seed)

        sup_classes = cls.sup_classes
        other_classes = cls.other_classes

        unsup_classes = (
            list(np.random.choice(sup_classes, size=n_overlap, replace=False)) +
            list(np.random.choice(other_classes, size=10-n_overlap, replace=False))
        )

        data = get_image_files(path/"quickdraw"/"train")
        counts = Counter()

        sup_data = []
        sup_targets = []
        unsup_data = []

        for f in np.random.permutation(data):
            c = f.parent.name
            counts[c] += 1
            if c in sup_classes and counts[c] <= n_labeled:
                sup_data.append(f)
                sup_targets.append(sup_classes.index(c))
            elif c in unsup_classes and counts[c] > n_labeled:
                unsup_data.append(f)

        sup_ds = cls(sup_data, sup_targets, sup_transform, mode=Mode.SUP)
        unsup_ds = cls(unsup_data, None, sup_transform, uda_transform, mode=Mode.UNSUP)

        return sup_ds, unsup_ds

    @classmethod
    def val_ds(cls, path="data", transform=None):
        "Creates validation dataset (only sup classes)"
        path = Path(path)
        cls.download(path)
        sup_classes = cls.sup_classes

        data = [f for f in get_image_files(path/"quickdraw"/"valid") if f.parent.name in sup_classes]
        targets = [cls.sup_classes.index(f.parent.name) for f in data]

        return cls(data, targets, transform, mode=Mode.SUP)
