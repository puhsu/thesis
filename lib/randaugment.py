__all__ = ["RandAugment"]

import random
import cv2
import numpy as np

from PIL import ImageOps, ImageEnhance, ImageFilter, Image


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

def shift_scale_rotate(img, angle, scale, dx, dy):
    "Affine transformation"
    img = np.array(img)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    return Image.fromarray(cv2.warpAffine(img, M=matrix, dsize=(width,height), borderMode=cv2.BORDER_REFLECT101))

def rotate(img, level):
    "Rotate for 30 degrees max"
    degrees = int_parameter(level, 30)
    if random.random() < 0.5:
        degrees = -degrees
    return shift_scale_rotate(img, degrees, 1, 0, 0)

def scale(img, level):
    "Scale image with level. Zoom in/out at random"
    v = float_parameter(level, 1.0)
    if random.random() < 0.5:
        v = -v * 0.4
    return shift_scale_rotate(img, 0, v + 1.0, 0, 0)

def shift(img, level):
    "Do shift with level strength in random directions"
    s = int_parameter(level, 10)
    do_x, do_y = random.choice([(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1)])
    return shift_scale_rotate(img, 0, 1.0, do_x * s, do_y * s)

class Cutout:
    "Cutout multiple holes"
    def __init__(self, n_holes, height=8, width=8):
        self.n_holes = n_holes
        self.hole_height = height
        self.hole_width  = width

    def __call__(self, img):
        img = np.array(img)
        height, width = img.shape[:2]

        holes = []

        for i in range(self.n_holes):
            y1 = random.randint(0, height - self.hole_height)
            x1 = random.randint(0, width - self.hole_width)
            y2 = y1 + self.hole_height
            x2 = x1 + self.hole_width
            holes.append((x1, y1, x2, y2))

        img = img.copy()
        for x1, y1, x2, y2 in holes:
            img[y1:y2, x1:x2] = 0

        return Image.fromarray(img)

def cutout(img, level):
    "Cutout `level` blocks from image"
    level = int_parameter(level, 10)
    aug = Cutout(n_holes=level)
    return aug(img)

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
    def __init__(self, n=3, m=9, uda=False):
        self.n, self.m = n, m
        self.uda = uda

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

        ops = []
        self.policies = []
        for t in self.transforms:
            for m in range(1, 11):
                ops += [(t, 0.5, m)]

        for op1 in ops:
            for op2 in ops:
                self.policies += [[op1, op2]]

    def __call__(self, x):
        if self.uda:
            policy = random.choice(self.policies)
            for op, p, m in policy:
                if random.random() < p:
                    x = op(x, m)
            return x

        transforms = np.random.choice(self.transforms, self.n)
        for op in transforms:
            x = op(x, self.m)

        return x
