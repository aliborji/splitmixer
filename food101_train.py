import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import time
import numpy as np

import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import sys
sys.path.append('./pytorch-image-models/')
from timm.models.convmixer import ConvMixer
from timm.models.splitmixer import SplitMixerI, SplitMixerII, SplitMixerIII, SplitMixerIV

device = 'cuda'

args_name = 'SplitMixer'
args_batch_size = 128#64
args_scale = 1 #0.75
args_reprob = 0 #.25
args_ra_m = 12 #8
args_ra_n = 2#1
args_jitter = 0 #.1
args_hdim = 256
args_depth = 8
args_psize = 7
args_conv_ks = 7
args_wd = 0.005 #.01
args_clip_norm = True
args_epochs = 100
args_lr_max = 0.01
args_workers = 2
num_classes = 101


mean = {
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar100': (0.5071, 0.4867, 0.4408),
'flower': (0.507, 0.487, 0.441),
'food': (0.485, 0.456, 0.406),
}

std = {
'cifar10': (0.2023, 0.1994, 0.2010),
'cifar100': (0.2675, 0.2565, 0.2761),
'flower': (0.267, 0.256, 0.276),
'food': (0.229, 0.224, 0.225)
}

mean, std = mean['food'], std['food']

train_transform = transforms.Compose([
    transforms.Resize((230,230)),                                              
    transforms.RandomResizedCrop(224, scale=(args_scale, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=args_ra_n, magnitude=args_ra_m),
    transforms.ColorJitter(args_jitter, args_jitter, args_jitter),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=args_reprob)
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),                                                                          
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = torchvision.datasets.Food101(root='./data', split='train',
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args_batch_size,
                                          shuffle=True, num_workers=args_workers)

testset = torchvision.datasets.Food101(root='./data', split='test',
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args_batch_size,
                                         shuffle=False, num_workers=args_workers)

# model = ConvMixer(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes).to(device)
model = SplitMixerI(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes, ratio=2./3.).to(device)
# model = SplitMixerIV(args_hdim, args_depth, patch_size=args_psize, kernel_size=args_conv_ks, n_classes=num_classes, n_part=2).to(device)


model = nn.DataParallel(model).cuda()
print(model)


lr_schedule = lambda t: np.interp([t], [0, args_epochs*2//5, args_epochs*4//5, args_epochs], 
                                  [0, args_lr_max, args_lr_max/20.0, 0])[0]

opt = optim.AdamW(model.parameters(), lr=args_lr_max, weight_decay=args_wd)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


for epoch in range(args_epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        lr = lr_schedule(epoch + (i + 1)/len(trainloader))
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        if args_clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        
    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)   

    print(f'[{args_name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')
