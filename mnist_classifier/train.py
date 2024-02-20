## 딥러닝 할껀데, 실습만 합니다. - 001 MNIST classifier

# 라이브러리
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

## 트레이닝에 필요한 파라미터 설정
lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = './checkpoint'
log_dir = './log'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

## 네트워크 구축
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        # input size: 28x28x1
        x = self.conv1(x)  # 28x28x1 -> 24x24x10
        x = self.pool1(x)  # 24x24x10 -> 12x12x10
        x = self.relu1(x)  # 12x12x10 -> 12x12x10

        x = self.conv2(x)  # 12x12x10 -> 8x8x20
        x = self.drop2(x)  # 8x8x20 -> 8x8x20
        x = self.pool2(x)  # 8x8x20 -> 4x4x20
        x = self.relu2(x)  # 4x4x20 -> 4x4x20

        x = x.view(-1, 320)  # 4x4x20 -> 1x320

        x = self.fc1(x)  # 1x320 -> 1x50
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)  # 1x50 -> 1x10

        return x

## 네트워크를 저장하거나 불러오는 함수
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, os.path.join(ckpt_dir, f'model_epoch_{epoch}'))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim

## MNIST 데이터 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

dataset = datasets.MNIST(download=True, root='./', train=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

num_data = len(loader.dataset)
num_batch = np.ceil(num_data / batch_size)

## 네트워크 설정 및 필요한 손실함수 구현
net = Net().to(device)
params = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)

writer = SummaryWriter(log_dir=log_dir)

## 트레이닝 시작하기
start = time.time()
for epoch in range(1, num_epoch+1):
    net.train()

    loss_arr = []
    acc_arr = []

    for batch, (input, label) in enumerate(loader, 1):
        input = input.to(device)
        label = label.to(device)

        output = net(input)
        pred = fn_pred(output)

        optim.zero_grad()

        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)

        loss.backward()
        optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        print(f'TRAIN: EPOCH: {epoch}/{num_epoch}, BATCH: {batch}/{num_batch}, loss: {np.mean(loss_arr)}, acc: {np.mean(acc_arr)}')

    writer.add_scalar('loss', np.mean(loss_arr), epoch)
    writer.add_scalar('acc', np.mean(acc_arr), epoch)

    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

# print elapsed time
end = time.time()
elapsed = end - start
print(elapsed)

writer.close()