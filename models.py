import collections as col

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import VIEWS


class AllViewsGaussianNoise(nn.Module):
    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        return {
            VIEWS.L_CC: self.single_add_gaussian_noise(x[VIEWS.L_CC]),
            VIEWS.L_MLO: self.single_add_gaussian_noise(x[VIEWS.L_MLO]),
            VIEWS.R_CC: self.single_add_gaussian_noise(x[VIEWS.R_CC]),
            VIEWS.R_MLO: self.single_add_gaussian_noise(x[VIEWS.R_MLO]),
        }

    def single_add_gaussian_noise(self, single_view):
        if not self.gaussian_noise_std or not self.training:
            return single_view
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class FourViewEfficientNet(nn.Module):
    def __init__(self, input_channels, bN="b0"):
        super(FourViewEfficientNet, self).__init__()

        from efficientnet_pytorch import EfficientNet
        from efficientnet_pytorch import utils as en_utils


        self.net = EfficientNet.from_name('efficientnet-%s' % bN)
        if input_channels != 3:
            print("EfficientNet: replace first conv")
            self.net._conv_stem = en_utils.Conv2dDynamicSamePadding(input_channels, 32, kernel_size=3, stride=2, bias=False)
        else:
            print("EfficientNet: keep first conv")
        self._fc = nn.Sequential()

        self.mlo = self.net
        self.cc = self.net
        self.model_dict = {}
        self.model_dict[VIEWS.L_CC] = self.l_cc = self.cc
        self.model_dict[VIEWS.L_MLO] = self.l_mlo = self.mlo
        self.model_dict[VIEWS.R_CC] = self.r_cc = self.cc
        self.model_dict[VIEWS.R_MLO] = self.r_mlo = self.mlo

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }

        return h_dict

    def single_forward(self, single_x, view):
        return self.model_dict[view](single_x)


class FourViewEfficientNetOnly(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FourViewEfficientNetOnly, self).__init__()

        self.four_view_net = FourViewEfficientNet(input_channels)
        self.fc = nn.Linear(1000, output_channels)

        self.all_views_gaussian_noise_layer = AllViewsGaussianNoise(0.01)


    def forward(self, x):
        h = self.all_views_gaussian_noise_layer(x)
        h = self.four_view_net(h)

        hh = torch.stack([h[VIEWS.L_CC], h[VIEWS.R_CC], h[VIEWS.L_MLO], h[VIEWS.R_MLO]], -1)
        n, c, _ = hh.size()
        hh = hh.view(n, c, -1).mean(-1)
        hh = self.fc(hh)

        return hh
