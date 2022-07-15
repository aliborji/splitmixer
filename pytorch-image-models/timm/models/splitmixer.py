from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from .helpers import build_model_with_cfg, checkpoint_seq

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PartialChannelMixer(nn.Module):
    def __init__(self, dim, is_odd, ratio):
        super().__init__()
        self.dim = dim
        self.partial_c = int(dim *ratio)
        self.mixer = nn.Conv2d(self.partial_c, self.partial_c, kernel_size=1)
        self.is_odd = is_odd
    
    def forward(self, x):
        if self.is_odd == 0:
            idx = self.partial_c
            return torch.cat((self.mixer(x[:, :idx]), x[:, idx:]), dim=1)
        else:
            idx = self.dim - self.partial_c
            return torch.cat((x[:, :idx], self.mixer(x[:, idx:])), dim=1)


def SplitMixerI(dim, depth, kernel_size=5, patch_size=2, n_classes=10,
                ratio=2/3, img_size=32, **kwargs):
    ''' Partial overlap model, one segment per iteration '''
    n_patch = img_size // patch_size
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        # nn.LayerNorm([dim, n_patch, n_patch]),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, (1,kernel_size), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    # nn.LayerNorm([dim, n_patch, n_patch]),
                    nn.Conv2d(dim, dim, (kernel_size,1), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    # nn.LayerNorm([dim, n_patch, n_patch]),
                )),
                PartialChannelMixer(dim, i % 2, ratio),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                # nn.LayerNorm([dim, n_patch, n_patch]),
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

def SplitMixerI_channel_only(dim, depth, kernel_size=5, patch_size=2,
                             n_classes=10, ratio=2/3, **kwargs):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, (kernel_size,kernel_size), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                PartialChannelMixer(dim, i % 2, ratio),
                nn.GELU(),
                nn.BatchNorm2d(dim),
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

# ------------------------------------------------------------------------------

class ChannelBin(nn.Module):
    def __init__(self, dim, remainder, n_part=3):
        super().__init__()
        self.dim = dim
        self.remainder = remainder
        self.n_part = n_part
        self.bin_dim = int(dim / n_part)
        self.c = dim - self.bin_dim * (n_part - 1) if (
            remainder == n_part - 1) else self.bin_dim
        self.mixer = nn.Conv2d(self.c, self.c, kernel_size=1)
    
    def forward(self, x):
        start = self.remainder * self.bin_dim
        end = self.dim if (self.remainder == self.n_part - 1) else (
            (self.remainder + 1) * self.bin_dim)
        return torch.cat((x[:, :start], self.mixer(x[:, start : end]),
                          x[:, end:]), dim=1)


def SplitMixerII(dim, depth, kernel_size=5, patch_size=2, n_classes=10,
                 n_part=3, **kwargs):
    ''' Non overlapping; In each iteration only one segment is convolved '''
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, (1,kernel_size), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim, (kernel_size,1), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                ChannelBin(dim, i % n_part, n_part),
                nn.GELU(),
                nn.BatchNorm2d(dim),
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

#-------------------------------------------------------------------------------

class ChannelPatch(nn.Module):
    ''' Non-overlap; Same layer '''
    def __init__(self, dim, n_part):
        super().__init__()
        assert dim % n_part == 0, (
            f'dim {dim} need to be divisible by n_part {n_part}')
        self.dim = dim
        self.n_part = n_part
        self.c = dim // n_part
        self.mixer = nn.Conv2d(self.c, self.c, kernel_size=1)
    
    def forward(self, x):
        c = self.c
        x = [self.mixer(x[:, c * i : c * (i + 1)]) for i in range(self.n_part)]
        return torch.cat(x, dim=1)


def SplitMixerIII(dim, depth, kernel_size=5, patch_size=2, n_classes=10,
                  n_part=3, **kwargs):
    ''' Non overlapping; In each iteration all segments are convolved;
        parameteres are shared across segments '''
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, (1,kernel_size), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim, (kernel_size,1), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                ChannelPatch(dim, n_part),
                nn.GELU(),
                nn.BatchNorm2d(dim),
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

#-------------------------------------------------------------------------------

class ChannelPatchUnshared(nn.Module):
    ''' Non-overlap; Same layer, multiple convs '''
    def __init__(self, dim, n_part):
        super().__init__()
        self.dim = dim
        self.n_part = n_part
        c = dim // n_part
        last_c = dim - c * (n_part - 1)
        self.mixer = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=1) for _ in range(n_part - 1)] + (
                [nn.Conv2d(last_c, last_c, kernel_size=1)]))
        self.c, self.last_c = c, last_c
   
    def forward(self, x):
        c, last_c = self.c, self.last_c
        x = [self.mixer[i](x[:, c * i : c * (i + 1)]) for i in (
            range(self.n_part - 1))] + [self.mixer[-1](x[:, -last_c:])]
        return torch.cat(x, dim=1)


def SplitMixerIV(dim, depth, kernel_size=5, patch_size=2, n_classes=10,
                 n_part=3, **kwargs):
    ''' Non overlapping; In each iteration all segments are convolved;
        no parameter sharing across segments '''
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, (1,kernel_size), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim, (kernel_size,1), groups=dim,
                              padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                ChannelPatchUnshared(dim, n_part),
                nn.GELU(),
                nn.BatchNorm2d(dim),
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

#-------------------------------------------------------------------------------

# CIFAR-10
_cfg_cifar10 = {
    'url': '',
    'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': None,
    'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010),
    'classifier': 'head'
}

@register_model
def splitmixeri_256_8_cifar10(pretrained=False, **kwargs):
    model = SplitMixerI(256, 8, kernel_size=5, patch_size=2, n_classes=10, ratio=2/3)
    model.default_cfg = _cfg_cifar10
    return model

@register_model
def splitmixerii_256_8_cifar10(pretrained=False, **kwargs):
    model = SplitMixerII(256, 8, kernel_size=5, patch_size=2, n_classes=10, n_part=3)
    model.default_cfg = _cfg_cifar10
    return model


# CIFAR-100
_cfg_cifar100 = {
    'url': '',
    'num_classes': 100, 'input_size': (3, 32, 32), 'pool_size': None,
    'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761),
    'classifier': 'head'
}

@register_model
def splitmixeri_256_8_cifar100(pretrained=False, **kwargs):
    model = SplitMixerI(256, 8, kernel_size=5, patch_size=2, n_classes=100, ratio=2/3)
    model.default_cfg = _cfg_cifar100
    return model

@register_model
def splitmixerii_256_8_cifar100(pretrained=False, **kwargs):
    model = SplitMixerII(256, 8, kernel_size=5, patch_size=2, n_classes=100, n_part=3)
    model.default_cfg = _cfg_cifar100
    return model


# ImageNet
_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
    'classifier': 'head'
}

@register_model
def splitmixeri_1536_20(pretrained=False, **kwargs):
    model = SplitMixerI(1536, 20, kernel_size=9, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeri_768_32(pretrained=False, **kwargs):
    model = SplitMixerI(768, 32, kernel_size=7, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeri_channel_only_1536_20(pretrained=False, **kwargs):
    model = SplitMixerI_channel_only(1536, 20, kernel_size=9, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeri_channel_only_768_32(pretrained=False, **kwargs):
    model = SplitMixerI_channel_only(768, 32, kernel_size=7, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixerii_1536_20(pretrained=False, **kwargs):
    model = SplitMixerII(1536, 20, kernel_size=9, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixerii_768_32(pretrained=False, **kwargs):
    model = SplitMixerII(768, 32, kernel_size=7, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeriii_1536_20(pretrained=False, **kwargs):
    model = SplitMixerIII(1536, 20, kernel_size=9, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeriii_768_32(pretrained=False, **kwargs):
    model = SplitMixerIII(768, 32, kernel_size=7, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeriv_1536_20(pretrained=False, **kwargs):
    model = SplitMixerIV(1536, 20, kernel_size=9, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model

@register_model
def splitmixeriv_768_32(pretrained=False, **kwargs):
    model = SplitMixerIV(768, 32, kernel_size=7, patch_size=7, n_classes=1000, **kwargs)
    model.default_cfg = _cfg
    return model
