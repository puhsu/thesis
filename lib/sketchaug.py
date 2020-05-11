import random
import numpy as np
from PIL import Image


def get_new_idx(x, lam, a, b):
    g = (2 * b - 2 * a) * x + 2 * a
    f = x + 0.5 * lam * x * (np.sin(g) - np.sin(2 * b))
    return f


def dfx(x, lam, a, b):
    g = (2 * b - 2 * a) * x + 2 * a
    df = 1 + 0.5 * lam * (np.sin(g) - np.sin(2 * b)) + lam * x * (b - a) * np.cos(g)
    return df


def deform_x(img_arr, lam, a, b):
    height, width = img_arr.shape
    idxs = np.arange(width) / float(width - 1)
    new_idxs = get_new_idx(idxs, lam, a, b)
    new_idxs = np.floor(new_idxs * (width - 1)).astype(int)
    lost_idxs = set(range(width)) - set(new_idxs)

    new_img = np.zeros_like(img_arr, dtype=np.uint8)
    for i, new_idx in enumerate(new_idxs):
        if new_idx > width or new_idx < 0: continue
        new_img[:, new_idx] |= img_arr[:, i]

    for lost_idx in lost_idxs:
        new_img[:, lost_idx] |= new_img[:, lost_idx - 1]
    return new_img


def deform_y(img_arr, lam, a, b):
    height, width = img_arr.shape
    idxs = np.arange(height) / float(height - 1)
    new_idxs = get_new_idx(idxs, lam, a, b)
    new_idxs = np.floor(new_idxs * (height - 1)).astype(int)
    lost_idxs = set(range(height)) - set(new_idxs)

    new_img = np.zeros_like(img_arr, dtype=np.uint8)
    for i, new_idx in enumerate(new_idxs):
        if new_idx > height or new_idx < 0: continue
        new_img[new_idx] |= img_arr[i]

    for lost_idx in lost_idxs:
        new_img[lost_idx] |= new_img[lost_idx - 1]

    return new_img


def SketchDeformation(img):
    "Nonlinear sketch transformation (strong augmentation)"

    params = [
        (0, 0, 0),
        (2.2, 0, 1),
        (-2, 0, 1),
        (2.3, 0, 0.5),
        (-3, 0, 0.5),
        (4.5, 0.5, 1),
        (-7, 0.5, 1),
    ]

    img = np.array(img)
    img = deform_x(img, *random.choice(params))
    img = deform_y(img, *random.choice(params))

    return Image.fromarray(img)
