__all__ = ["get_image_classes", "split_cifar", "ImageDataset"]

import torch
import numpy as np
import torchvision.transforms.functional as T
from PIL import Image

from collections import Counter

CIFAR_STATS = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])

def get_image_classes(overlap, class_to_idx):
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


def split_cifar(images, targets, l_classes, ul_classes, n_labeled=400, to_pil=True):
    "Split cifar for semi-supervised classification"
    assert 0 < n_labeled <= 5000
    l_count = Counter()

    l_images, l_targets = [], []
    ul_images, ul_targets = [], []

    for img, t in zip(images, targets):
        l_count[t] += 1
        if to_pil:
            img = Image.fromarray(img)
        if t in l_classes and l_count[t] <= n_labeled:
            l_images.append(img)
            l_targets.append(t)
        elif t in ul_classes and l_count[t] > n_labeled:
            ul_images.append(img)
            ul_targets.append(t)

    # map indices to [0, 6]
    old_to_new = {l: i for i, l in enumerate(l_classes)}
    l_targets = [old_to_new[l] for l in l_targets]

    return l_images, l_targets, ul_images, ul_targets


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets=None, tfm=None):
        self.images = images
        self.targets = targets
        self.transform = T.to_tensor if tfm is None else tfm

    def __getitem__(self, i):
        if self.targets is None:
            return self.transform(self.images[i])
        return self.transform(self.images[i]), self.targets[i]

    def __len__(self):
        return len(self.images)
