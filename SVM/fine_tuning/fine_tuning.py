import os

import matplotlib.pyplot as plt
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
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)

def calculate_loss_accuracy(loader, criterion):
    running_loss = 0.0
    running_accuracy = 0.0
    len_data = len(loader.dataset)

    model.eval()
    with torch.no_grad():
        for data, label in tqdm.tqdm(loader):
            # to device
            data = data.to(device)
            label = label.to(device)

            # prediction
            output = model(data)
            pred = output.argmax(dim=1)
            metric_batch = pred.eq(label.view_as(pred)).sum().item()

            # loss and metric
            running_loss += criterion(output, label).item()
            running_accuracy += metric_batch

        loss = running_loss / float(len_data)
        accuracy = running_accuracy / float(len_data)

    return loss, accuracy

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
criterion = nn.CrossEntropyLoss(reduction='sum')
loss, accuracy = calculate_loss_accuracy(test_loader, criterion)
print(f'before fine tuning - loss: {loss}, accuracy: {accuracy}')

# hyperparameter
epochs = 40
criterion = nn.CrossEntropyLoss(reduction='sum')
opt = torch.optim.Adam(params=model.classifier[6].parameters(), lr=0.01)
lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)

# training
model.train()
loss_history = {
    'train': [],
    'val': [],
}
metric_history = {
    'train': [],
    'val': [],
}
best_loss = float('inf')
for epoch in range(epochs):
    running_loss = 0.0
    running_metric = 0.0

    for data, label in tqdm.tqdm(train_loader):
        # to device
        data = data.to(device)
        label = label.to(device)

        # loss and accuracy
        output = model(data)
        loss = criterion(output, label)
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        running_metric += pred.eq(label.view_as(pred)).sum().item()

        # update
        opt.zero_grad()
        loss.backward()
        opt.step()

    train_loss = running_loss / float(len(train_loader.dataset))
    train_metric = running_metric / float(len(train_loader.dataset))

    # lr schedule
    lr_scheduler.step()

    # save loss and metric history
    loss_history['train'].append(train_loss)
    metric_history['train'].append(train_metric)

    val_loss, val_metric = calculate_loss_accuracy(test_loader, criterion)
    loss_history['val'].append(val_loss)
    metric_history['val'].append(val_metric)

    print(f'[epoch {epoch}/{epochs}]')
    print(f'train loss: {train_loss}, train_metric: {train_metric}, val_loss: {val_loss}, val_metric: {val_metric}')

    # save loss graph
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(epoch+1), loss_history['train'], label='train loss')
    plt.plot(range(epoch+1), loss_history['val'], label='val loss')

    plt.subplot(1,2,2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(epoch+1), metric_history['train'], label='train metric')
    plt.plot(range(epoch+1), metric_history['val'], label='val metric')
    plt.savefig('./train_result.png')

    # save best model
    if val_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_epoch{epoch}.pt'))
        best_loss = val_loss