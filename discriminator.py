import torch.nn as nn
import torch
from snlayer import SpectralNorm
from model import BasicBlock
import numpy as np


class BasicBlock_Discrim(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, nobn=False):
        super(BasicBlock_Discrim, self).__init__()
        self.downsample = downsample
        self.nobn = nobn

        self.conv1 = SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=False)
        if self.downsample:
            self.conv2 = nn.Sequential(nn.AvgPool2d(2, 2), SpectralNorm(
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)))
        else:
            self.conv2 = SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes or self.downsample:
            if self.downsample:
                self.skip = nn.Sequential(nn.AvgPool2d(2, 2), SpectralNorm(nn.Conv2d(inplanes, planes, 1, 1)))
            else:
                self.skip = SpectralNorm(nn.Conv2d(inplanes, planes, 1, 1, 0))
        else:
            self.skip = None
        self.stride = stride

    def forward(self, x):
        residual = x
        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        out = self.conv1(out)
        if not self.nobn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            residual = self.skip(x)
        out += residual
        return out


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        assert input_size in [16, 64]
        self.blocks = [128, 128, 256, 256, 512, 512]
        pool_start = len(self.blocks) - 4 if input_size == 64 else len(self.blocks) - 2
        self.out_layer = nn.Sequential(
            SpectralNorm(nn.Linear(16 * self.blocks[-1], self.blocks[-1])),
            nn.ReLU(),
            SpectralNorm(nn.Linear(self.blocks[-1], 1))
        )

        rbs = []
        in_feat = 3
        for i in range(len(self.blocks)):
            b_down = bool(i >= pool_start)
            rbs.append(BasicBlock_Discrim(in_feat, self.blocks[i], downsample=b_down, nobn=True))
            in_feat = self.blocks[i]
        self.residual_blocks = nn.Sequential(*rbs)

    def forward(self, x):
        out = self.residual_blocks(x)
        out = out.view(-1, 16 * self.blocks[-1])
        out = self.out_layer(out)
        return out