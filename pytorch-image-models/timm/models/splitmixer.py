""" SplitMixer
original github: https://github.com/aliborji/splitmixer

"""

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from .helpers import build_model_with_cfg, checkpoint_seq
from .layers import SelectAdaptivePool2d

import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList


# --------------------------- Building Blocks ----------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ChannelMixerI(nn.Module):
    ''' Partial overlap; In each block only one segment is convolved. '''
    def __init__(self, hdim, is_odd=0, ratio=2/3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.partial_c = int(hdim *ratio)
        self.mixer = nn.Conv2d(self.partial_c, self.partial_c, kernel_size=1)
        self.is_odd = is_odd
    
    def forward(self, x):
        if self.is_odd == 0:
            idx = self.partial_c
            return torch.cat((self.mixer(x[:, :idx]), x[:, idx:]), dim=1)
        else:
            idx = self.hdim - self.partial_c
            return torch.cat((x[:, :idx], self.mixer(x[:, idx:])), dim=1)


class ChannelMixerII(nn.Module):
    ''' No overlap; In each block only one segment is convolved. '''
    def __init__(self, hdim, remainder=0, num_segments=3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.remainder = remainder
        self.num_segments = num_segments
        self.bin_dim = int(hdim / num_segments)
        self.c = hdim - self.bin_dim * (num_segments - 1) if (
            remainder == num_segments - 1) else self.bin_dim
        self.mixer = nn.Conv2d(self.c, self.c, kernel_size=1)
    
    def forward(self, x):
        start = self.remainder * self.bin_dim
        end = self.hdim if (self.remainder == self.num_segments - 1) else (
            (self.remainder + 1) * self.bin_dim)
        return torch.cat((x[:, :start], self.mixer(x[:, start : end]),
                          x[:, end:]), dim=1)


class ChannelMixerIII(nn.Module):
    ''' No overlap; In each block all segments are convolved;
        Parameters are shared across segments. '''
    def __init__(self, hdim, num_segments=3, **kwargs):
        super().__init__()
        assert hdim % num_segments == 0, (
            f'hdim {hdim} need to be divisible by num_segments {num_segments}')
        self.hdim = hdim
        self.num_segments = num_segments
        self.c = hdim // num_segments
        self.mixer = nn.Conv2d(self.c, self.c, kernel_size=1)
    
    def forward(self, x):
        c = self.c
        x = [self.mixer(x[:, c * i : c * (i + 1)]) for i in range(self.num_segments)]
        return torch.cat(x, dim=1)


class ChannelMixerIV(nn.Module):
    ''' No overlap; In each block all segments are convolved;
        No parameter sharing across segments. '''
    def __init__(self, hdim, num_segments=3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.num_segments = num_segments
        c = hdim // num_segments
        last_c = hdim - c * (num_segments - 1)
        self.mixer = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=1) for _ in range(num_segments - 1)
            ] + ([nn.Conv2d(last_c, last_c, kernel_size=1)]))
        self.c, self.last_c = c, last_c
   
    def forward(self, x):
        c, last_c = self.c, self.last_c
        x = [self.mixer[i](x[:, c * i : c * (i + 1)]) for i in (
            range(self.num_segments - 1))] + [self.mixer[-1](x[:, -last_c:])]
        return torch.cat(x, dim=1)


class ChannelMixerV(nn.Module):
    ''' Partial overlap; In each block all segments are convolved;
        No parameter sharing across segments. '''
    def __init__(self, hdim, ratio=2/3, **kwargs):
        super().__init__()
        self.hdim = hdim
        self.c = int(hdim *ratio)
        self.mixer1 = nn.Conv2d(self.c, self.c, kernel_size=1)
        self.mixer2 = nn.Conv2d(self.c, self.c, kernel_size=1)

    def forward(self, x):
        c, hdim = self.c, self.hdim
        x = torch.cat((self.mixer1(x[:, :c]), x[:, c:]), dim=1)
        return torch.cat((x[:, :(hdim - c)], self.mixer2(x[:, (hdim - c):])), dim=1)

# ------------------------------ Main Model ------------------------------------

class SplitMixer(nn.Module):
    def __init__(self, hdim, num_blocks, kernel_size=5, patch_size=2,
                 num_classes=10, ratio=2/3, num_segments=2, img_size=32,
                 mixer_setting='I', spatial_trick=True, channel_trick=True,
                 act_layer=nn.GELU, global_pool='avg', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = hdim
        self.grad_checkpointing = False

        self.patch_emb = nn.Sequential(
            nn.Conv2d(3, hdim, kernel_size=patch_size, stride=patch_size),
            act_layer(),
            nn.BatchNorm2d(hdim),
            )

        self.mixer_blocks = ModuleList()

        for i in range(num_blocks):
            spatial_mixer = Residual(nn.Sequential(
                nn.Conv2d(hdim, hdim, (1, kernel_size), groups=hdim, padding="same"),
                act_layer(),
                nn.BatchNorm2d(hdim),
                nn.Conv2d(hdim, hdim, (kernel_size, 1), groups=hdim, padding="same"),
                act_layer(),
                nn.BatchNorm2d(hdim),
                )) if spatial_trick else Residual(nn.Sequential(
                    nn.Conv2d(hdim, hdim, kernel_size, groups=hdim, padding="same"),
                    act_layer(),
                    nn.BatchNorm2d(hdim),
                    ))

            self.mixer_blocks.append(spatial_mixer)

            if channel_trick:
                mixer_args = {
                    'hdim': hdim, 'is_odd': i % 2, 'ratio': ratio,
                    'remainder': i % num_segments, 'num_segments': num_segments,
                    }
                channel_mixer = nn.Sequential(
                    globals()[f'ChannelMixer{mixer_setting}'](**mixer_args),
                    act_layer(),
                    nn.BatchNorm2d(hdim),
                    )
            else:
                channel_mixer = nn.Sequential(
                    nn.Conv2d(hdim, hdim, kernel_size=1),
                    act_layer(),
                    nn.BatchNorm2d(hdim)
                    )
            self.mixer_blocks.append(channel_mixer)

        self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(hdim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^patch_emb', blocks=r'^mixer_blocks\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_emb(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for mixer_block in self.mixer_blocks:
                x = checkpoint_seq(mixer_block, x)
        else:
            for mixer_block in self.mixer_blocks:
                x = mixer_block(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pooling(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

#-------------------------------------------------------------------------------

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head', 'first_conv': 'patch_emb.0', **kwargs
        }

default_cfgs = {'splitmixer_1536_20': _cfg(), 'splitmixer_768_32': _cfg(),}


def _create_splitmixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(SplitMixer, variant, pretrained, **kwargs)


@register_model
def splitmixer_1536_20(pretrained=False, **kwargs):
    model_args = dict(hdim=1536, num_blocks=20, kernel_size=9, patch_size=7, **kwargs)
    return _create_splitmixer('splitmixer_1536_20', pretrained, **model_args)


@register_model
def splitmixer_768_32(pretrained=False, **kwargs):
    model_args = dict(hdim=768, num_blocks=32, kernel_size=7, patch_size=7, act_layer=nn.ReLU, **kwargs)
    return _create_splitmixer('splitmixer_768_32', pretrained, **model_args)


@register_model
def splitmixer_1024_20_ks9_p14(pretrained=False, **kwargs):
    model_args = dict(hdim=1024, num_blocks=20, kernel_size=9, patch_size=14, **kwargs)
    return _create_splitmixer('splitmixer_1024_20_ks9_p14', pretrained, **model_args)
