## 라이브러리
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from torchvision import transforms, datasets

## 학습 파라미터 설정
lr = 1e-3
batch_size = 4
n_epochs = 100

data_dir = "./datasets"
ckpt_dir = "./checkpoint"
log_dir = "./log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def concat(unpool, enc):
    # 1. unpool feature를 padding하는 방법
    # diffY = enc.size()[2] - unpool.size()[2]
    # diffX = enc.size()[3] - unpool.size()[3]
    #
    # unpool = F.pad(unpool, [diffY//2, diffY - diffY//2,
    #                             diffX//2, diffX - diffX//2])
    
    # 2. enc feature를 crop하는 방법
    print(enc.shape)
    enc = torchvision.transforms.CenterCrop(size=unpool.shape[2:])(enc)
    print(enc.shape)

    result = torch.cat([unpool, enc], dim=1)  # dim - 0: batch, 1: channel, 2: height, 3: width

    return result

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=1024)

        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=512)

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=256)

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=128)

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
        self.fc = CBR2d(in_channels=64, out_channels=2, kernel_size=1, padding=0, bias=True)

    def forward(self, x):

        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)

        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = concat(unpool4, enc4_2)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = concat(unpool3, enc3_2)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = concat(unpool2, enc2_2)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = concat(unpool1, enc1_2)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        # print(f'input: {x.shape}')
        # print(f'enc1_1: {enc1_1.shape}')
        # print(f'enc1_2: {enc1_2.shape}')
        # print(f'pool1: {pool1.shape}')
        # print(f'enc2_1: {enc2_1.shape}')
        # print(f'enc2_2: {enc2_2.shape}')
        # print(f'pool2: {pool2.shape}')
        # print(f'enc3_1: {enc3_1.shape}')
        # print(f'enc3_2: {enc3_2.shape}')
        # print(f'pool3: {pool3.shape}')
        # print(f'enc4_1: {enc4_1.shape}')
        # print(f'enc4_2: {enc4_2.shape}')
        # print(f'pool4: {pool4.shape}')
        # print(f'enc5_1: {enc5_1.shape}')
        # print(f'dec5_1: {dec5_1.shape}')
        # print(f'unpool4: {unpool4.shape}')
        # print(f'cat4: {cat4.shape}')
        # print(f'dec4_2: {dec4_2.shape}')
        # print(f'dec4_1: {dec4_1.shape}')
        # print(f'unpool3: {unpool3.shape}')
        # print(f'cat3: {cat3.shape}')
        # print(f'dec3_2: {dec3_2.shape}')
        # print(f'dec3_1: {dec3_1.shape}')
        # print(f'unpool2: {unpool2.shape}')
        # print(f'cat2: {cat2.shape}')
        # print(f'dec2_2: {dec2_2.shape}')
        # print(f'dec2_1: {dec4_1.shape}')
        # print(f'unpool1: {unpool1.shape}')
        # print(f'cat1: {cat1.shape}')
        # print(f'dec1_2: {dec1_2.shape}')
        # print(f'dec1_1: {dec1_1.shape}')
        # print(f'result: {x.shape}')

        return x

net = UNet()
rand = torch.rand((1, 1, 572, 572))

result = net(rand)


# 데이터 로더 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label':label}

        if self.transform:
            data = self.transform(data)

        return data

## Transform 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input'],

        label = label.transpose((2, 0, 1)).astype(np.float32)  # numpy: HWC, torch tensor: CHW
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input'],

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input'],

        if np.random.rand() > 0.5:
            label = np.fliplr(label)  # flip left right
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)  # flip up down
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

##
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    RandomFlip(),
    ToTensor(),
])
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)

##
data = dataset_train.__getitem__(0)

input = data['input']
label = data['label']

##
plt.subplot(121)
plt.imshow(input.squeeze())

plt.subplot(122)
plt.imshow(label.squeeze())

plt.show()