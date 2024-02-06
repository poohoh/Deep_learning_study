import os

import numpy as np
import torch
import torchvision.models
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms
transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# STL-10 dataset
data_root = './data'
os.makedirs(data_root, exist_ok=True)
train_dataset = torchvision.datasets.STL10(root=data_root, transform=transforms, split='train', download=True)
test_dataset = torchvision.datasets.STL10(root=data_root, transform=transforms, split='test', download=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# VGGNet pre-trained model
model = torchvision.models.vgg16(pretrained=True).to(device)
# summary(model, (3, 224, 224), device=device.type)

# change classifier
model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True).to(device)


# inference on STL10
accuracy = 0.0
for data, label in tqdm.tqdm(test_loader):
    # to device
    data = data.to(device)
    label = label.to(device)

    # prediction
    pred = model(data).argmax(dim=1)
    metric_batch = pred.eq(label.view_as(pred)).sum().item()
    accuracy += metric_batch

accuracy = accuracy / float(len(test_loader.dataset))
print(f'accuracy before fine tuning: {accuracy}')

# # hyperparameter
# epochs = 40
# criterion = nn.CrossEntropyLoss(reduction='sum')
# opt = torch.optim.Adam(params=model.classifier[6].parameters(), lr=0.001)
# lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)
#
# # training
# for epoch in range(epochs):
#     for data, label in tqdm.tqdm(train_loader):