__all__ = ["RandAugment"]

import random
import cv2
import numpy as np

from PIL import ImageOps, ImageEnhance, ImageFilter, Image
from albumentations import Cutout
from albumentations.augmentations.functional import shift_scale_rotate

PARAMETER_MAX=10


def float_parameter(level, maxval):
    "Helper function to scale `val` between 0 and maxval."
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    "Helper function to scale `val` between 0 and maxval."
    return int(level * maxval / PARAMETER_MAX)


# Geometric transformations

def identity(img, level):
    return img

def flip_lr(img, level):
    "Left right flip"
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_ud(img, level):
    "Up down flip"
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def rotate(img, level):
    "Rotate for 30 degrees max"
    degrees = int_parameter(level, 30)
    if random.random() < 0.5:
        degrees = -degrees
    return Image.fromarray(shift_scale_rotate(np.array(img), degrees, 1.0, 0, 0))

def scale(img, level):
    "Scale image with level. Zoom in/out at random"
    v = float_parameter(level, 1.0)
    if random.random() < 0.5:
        v = -v * 0.4
    return Image.fromarray(shift_scale_rotate(np.array(img), 0, v + 1.0, 0, 0))

def shift(img, level):
    "Do shift with level strength in random directions"
    s = int_parameter(level, 10)
    do_x, do_y = random.choice([(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1)])
    return Image.fromarray(shift_scale_rotate(np.array(img), 0, 1.0, do_x * s, do_y * s))

def cutout(img, level):
    "Cutout `level` blocks from image"
    level = int_parameter(level, 10)
    aug = Cutout(num_holes=level, always_apply=True)
    return Image.fromarray(aug(image=np.array(img))["image"])

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.linalg.inv(A.T * A) * A.T @ B
    return np.r_[np.array(res).flatten(), 1].reshape(3,3)

def perspective(img, level):
    "Perspective transformation"
    w, h = img.size
    y, x = float_parameter(level, 8), float_parameter(level, 8)
    if random.random() < 0.5: y = -y
    if random.random() < 0.5: x = -x

    # Compute perspective warp coordinates
    orig =  [(0, 0), (h, 0), (h, w), (0, w)]
    persp = [(0-y, 0-x), (h+y, 0+x), (h+y, w+x), (0-y, w-x)]
    coefs = find_coeffs(orig, persp)

    img = np.array(img).astype(np.float32) / 255.
    persp_img = cv2.warpPerspective(
        img,
        coefs,
        (h,w),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101
    )
    return Image.fromarray((persp_img * 255.).astype(np.uint8))


# Color transforms
def invert(img, level):
    "Negative image"
    return ImageOps.invert(img)

def equalize(img, level):
    "Equalize image histogram"
    return ImageOps.equalize(img)

def posterize(img, level):
    "Control bits count used to store colors"
    bits = 5 - int_parameter(level, 4)
    return ImageOps.posterize(img, bits)

def contrast(img, level):
    "Change contrast with param in [0.5, 1.0, 3.0]"
    v = float_parameter(level, 2.0)
    if random.random() < 0.5:
        v = -v / 4.0
    return ImageEnhance.Contrast(img).enhance(v + 1.0)

def color(img, level):
    "Change color with param in [0, 1.0, 3.0]"
    v = float_parameter(level, 2.0)
    if random.random() < 0.5:
        v = -v / 2.0
    return ImageEnhance.Color(img).enhance(v + 1.0)

def brightness(img, level):
    "Controll brightness with param in [1/3, 2.0]"
    v = float_parameter(level, 1.0)
    if random.random() < 0.5:
        v = -v / 1.5
    return ImageEnhance.Brightness(img).enhance(v + 1.0)

def sharpness(img, level):
    "Controll sharpness with param in [0, 4]"
    v = float_parameter(level, 3.0)
    if random.random() < 0.5:
        v = -v / 3.0
    return ImageEnhance.Sharpness(img).enhance(v + 1.0)


class RandAugment:
    "RandAugment augmentation policy operating on PIL images"
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.transforms = [
            identity,
            flip_lr,
            flip_ud,
            rotate,
            scale,
            shift,
            cutout,
            perspective,
            invert,
            equalize,
            posterize,
            contrast,
            color,
            brightness,
            sharpness,
        ]

    def __call__(self, x):
        transforms = np.random.choice(self.transforms, self.n)

        for op in transforms:
            x = op(x, self.m)

        return x
