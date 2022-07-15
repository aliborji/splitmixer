import torch

import sys
sys.path.append('./pytorch-image-models/')
from timm.models.convmixer import ConvMixer
from timm.models.splitmixer import SplitMixerI, SplitMixerII, SplitMixerIII, SplitMixerIV

device = 'cuda'

args_hdim = 256
args_depth = 8
args_psize = 7
args_conv_ks = 7
num_classes = 102

I_ratios = [2/3, 3/5, 4/7, 5/9, 6/11]
II_n_part = [2, 3, 4, 5, 6]
III_n_part = [2, 4, 8]
IV_n_part = [2, 3, 4, 5]

labels = ['convmixer']
models = [ConvMixer(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes)]

models += [SplitMixerI(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes, ratio=r) for r in I_ratios]
labels += [f'splitmixerI-{r}' for r in I_ratios]

models += [SplitMixerII(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes, n_part=_p) for _p in II_n_part]
labels += [f'splitmixerII-{_p}' for _p in II_n_part]

models += [SplitMixerIII(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes, n_part=_p) for _p in III_n_part]
labels += [f'splitmixerIII-{_p}' for _p in III_n_part]

models += [SplitMixerIV(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes, n_part=_p) for _p in IV_n_part]
labels += [f'splitmixerIV-{_p}' for _p in IV_n_part]

batch_size = 64
dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float).to(device)
repetitions=100

for i in range(len(models)):
    print('\n', labels[i])
    model = models[i].to(device)
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    Throughput =   (repetitions * batch_size)/total_time
    print('Throughput:',Throughput)
