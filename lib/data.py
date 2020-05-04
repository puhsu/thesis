__all__ = [
    "split_cifar", "SupervisedDataset", "UnsupervisedDataset", "CIFAR_STATS",
    "ConcatDataLoader"
]

import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from collections import Counter

CIFAR_STATS = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])


class SupervisedDataset(Dataset):
    "Supervised dataset from numpy arrays"
    def __init__(self, images, targets, transform):
        self.images = [Image.fromarray(img) for img in images]
        self.targets = targets
        self.transform = transform

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]

    def __len__(self):
        return len(self.images)


class UnsupervisedDataset(Dataset):
    "UDA specific dataset. Loads two versions of same image with 2 transforms"
    def __init__(self, images, transform0, transform1):
        self.images = [Image.fromarray(img) for img in images]
        self.transform0 = transform0
        self.transform1 = transform1

    def __getitem__(self, i):
        img = self.images[i]
        return self.transform0(img), self.transform1(img)

    def __len__(self):
        return len(self.images)


class ConcatDataLoader:
    "Iterate multiple dataloaders (expects loaders with shuffle=True)"
    def __init__(self, *loaders):
        self.loaders = loaders

    def __iter__(self):
        for output in zip(*self.loaders):
            yield output

    def __len__(self):
        return min(len(l) for l in self.loaders)


def get_classes(overlap, class_to_idx):
    "Split cifar classes with overlap \in [0, 4]"

    assert 0 <= overlap <= 4
    animals = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    objects = ['airplane', 'automobile', 'ship', 'truck']

    l_classes = animals

    ul_classes = (
        list(np.random.choice(animals, size=overlap, replace=False)) +
        list(np.random.choice(objects, size=4-overlap, replace=False))
    )

    l_classes = [class_to_idx[c] for c in l_classes]
    ul_classes = [class_to_idx[c] for c in ul_classes]

    return l_classes, ul_classes


def split_cifar(
    train, valid, n_labeled, n_overlap,
    labeled_tfm, unlabeled_tfm, valid_tfm
):
    "Split cifar for semi-supervised classification"
    assert 0 < n_labeled <= 5000
    l_count = Counter()
    l_classes, u_classes = get_classes(n_overlap, train.class_to_idx)
    train_l, valid_l, train_u = [], [], []

    for i in np.random.permutation(len(train)):
        _, y = train[i]
        l_count[y] += 1

        if y in l_classes and l_count[y] <= n_labeled:
            train_l.append(i)
        elif y in u_classes and l_count[y] > n_labeled:
            train_u.append(i)

    for i in np.random.permutation(len(valid)):
        _, y = valid[i]
        if y in l_classes:
            valid_l.append(i)

    # map indices to [0, 6]
    old_to_new = {l: i for i, l in enumerate(l_classes)}
    train_targets_l = [old_to_new[train.targets[i]] for i in train_l]
    valid_targets_l = [old_to_new[valid.targets[i]] for i in valid_l]

    train_ds_l = SupervisedDataset(train.data[train_l], train_targets_l, labeled_tfm)
    train_ds_u = UnsupervisedDataset(train.data[train_u], labeled_tfm, unlabeled_tfm)
    valid_ds = SupervisedDataset(valid.data[valid_l], valid_targets_l, valid_tfm)

    return train_ds_l, train_ds_u, valid_ds
