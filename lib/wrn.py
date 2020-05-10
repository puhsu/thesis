__all__ = ["WideResNet"]

# https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/models/wideresnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        return

    if hasattr(m, 'weight'):
        nn.init.kaiming_uniform_(m.weight)
    if hasattr(m, 'bias') and m.bias:
        nn.init.zeros_(m.bias)


def conv_2d(ni, nf, ks, stride):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)

def bn(ni, init_zero=False):
    m = nn.BatchNorm2d(ni)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

def bn_relu_conv(ni, nf, ks, stride, init_zero=False):
    bn_initzero = bn(ni, init_zero=init_zero)
    return nn.Sequential(bn_initzero, nn.ReLU(inplace=True), conv_2d(ni, nf, ks, stride))

def identity(x):
    return x

def make_group(N, ni, nf, block, stride, drop_p):
    return [block(ni if i == 0 else nf, nf, stride if i == 0 else 1, drop_p) for i in range(N)]


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class BasicBlock(nn.Module):
    def __init__(self, ni, nf, stride, drop_p=0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, 3, stride)
        self.conv2 = bn_relu_conv(nf, nf, 3, 1)
        self.drop = nn.Dropout(drop_p, inplace=True) if drop_p else None
        self.shortcut = conv_2d(ni, nf, 1, stride) if ni != nf else identity

    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        if self.drop:
            x = self.drop(x)
        x = self.conv2(x)
        return x.add_(r)


class WideResNet(nn.Module):
    def __init__(self, num_groups, N, num_classes, k=1, drop_p=0.0, start_nf=16, inp_nf=3):
        super().__init__()
        n_channels = [start_nf]
        for i in range(num_groups):
            n_channels.append(start_nf*(2**i)*k)

        layers = [conv_2d(inp_nf, n_channels[0], 3, 1)]
        for i in range(num_groups):
            layers += make_group(N, n_channels[i], n_channels[i+1], BasicBlock, (1 if i==0 else 2), drop_p)

        layers += [nn.AdaptiveAvgPool2d(1), bn_relu_conv(n_channels[-1], num_classes, 1, 1), Flatten()]
        self.features = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.features(x)
