import copy
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import transforms
import tqdm

from vggnet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# define model
model = VGG_16(in_channels=3, num_classes=1000, init_weights=True).to(device)
# summary(model, input_size=(3, 224, 224), device=device.type)

criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.01)

# scheduler
lr_scheduler = StepLR(opt, step_size=30, gamma=0.1)

# load dataset
data_path = './data'
os.makedirs(data_path, exist_ok=True)
train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
val_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

# images only
train_imgs = [item[0] for item in train_dataset]  # item[0], item[1]은 각각 image, label
train_imgs = torch.stack(train_imgs, dim=0).numpy()
val_imgs = [item[0] for item in val_dataset]
val_imgs = torch.stack(val_imgs, dim=0).numpy()

print(len(train_imgs))
print(len(val_imgs))

# calculate mean and std
train_meanR = train_imgs[:, 0, :, :].mean()
train_meanG = train_imgs[:, 1, :, :].mean()
train_meanB = train_imgs[:, 2, :, :].mean()
train_stdR = train_imgs[:, 0, :, :].std()
train_stdG = train_imgs[:, 1, :, :].std()
train_stdB = train_imgs[:, 2, :, :].std()

val_meanR = val_imgs[:, 0, :, :].mean()
val_meanG = val_imgs[:, 1, :, :].mean()
val_meanB = val_imgs[:, 2, :, :].mean()
val_stdR = val_imgs[:, 0, :, :].std()
val_stdG = val_imgs[:, 1, :, :].std()
val_stdB = val_imgs[:, 2, :, :].std()

print(train_meanR, train_meanG, train_meanB)
print(train_stdR, train_stdG, train_stdB)

# transforms
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([val_meanR, val_meanG, val_meanB], [val_stdR, val_stdG, val_stdB])
])

train_dataset.transform = train_transforms
val_dataset.transform = val_transforms

# subset_indices = list(range(1000))
# subset_dataset = Subset(train_dataset, subset_indices)
# subset_val_indices = list(range(100))
# subset_val_dataset = Subset(val_dataset, subset_val_indices)

train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=16, shuffle=True)

params_train = {
    'num_epochs': 100,
    'optimizer': opt,
    'loss_func': criterion,
    'train_dl': train_dl,
    'val_dl': val_dl,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
}

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    # get output class
    # pred = output.argmax(dim=1, keepdim=True)
    pred = output

    # compare output class with target class
    corrects = pred.eq(target.view_as(pred)).sum().item()

    return corrects

def loss_batch(criterion, output, target, opt=None):
    # get loss
    loss = criterion(output, target)

    # get performance metric
    metric_b = metrics_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

def loss_epoch(model, criterion, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in tqdm.tqdm(dataset_dl):
        # batch to device
        xb = xb.to(device)
        yb = yb.float().to(device)

        # five crop: bs, crops, channel, h, w
        bs, c, h, w = xb.size()
        output_ = model(xb.view(-1, c, h, w))
        output = output_.view(bs, -1).mean(1)

        # get loss per batch
        loss_b, metric_b = loss_batch(criterion, output, yb, opt)

        # update running loss
        running_loss += loss_b

        # update running metric
        if metric_b is not None:
            running_metric += metric_b

        # break the loop in case of sanity check
        if sanity_check is True:
            break

    # average loss
    loss = running_loss / float(len_data)

    # average metric value
    metric = running_metric / float(len_data)

    return loss, metric

def train_val(model, params):
    # extract model parameters
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']

    # history of loss values in each epoch
    loss_history = {
        'train': [],
        'val': [],
    }

    # history of metric values in each epoch
    metric_history = {
        'train': [],
        'val': [],
    }

    # initialize best loss to a large value
    best_loss = float('inf')

    # main loop
    for epoch in range(num_epochs):
        # check 1 epoch start time
        start_time = time.time()

        # get current learning rate
        current_lr = get_lr(opt)
        print(f'Epoch: {epoch}/{num_epochs-1}, current lr: {current_lr}')

        # train model on training dataset
        model.train()
        train_loss, train_metric = loss_epoch(model, criterion, train_dl, sanity_check, opt)

        # collect loss and metric for training dataset
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        # evaluate model on validation dataset
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, criterion, val_dl, sanity_check)

        # save best loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # save model
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./models/epoch_{epoch}.pt')

        # collect loss and metric for validation dataset
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        # learning rate schedule
        lr_scheduler.step()

        print(f'train loss: {train_loss}, val loss: {val_loss}, accuracy: {100*val_metric}, time: {time.time() - start_time}')
        print('-'*20)

        # save loss graph
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(range(epoch+1), loss_history['train'])
        plt.subplot(1,2,2)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(range(epoch+1), metric_history['val'])
        plt.savefig('./result.png')

    return model, loss_history, metric_history

def create_folder(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        print('Error')

# learning rate
current_lr = get_lr(opt)
print(f'current lr: {current_lr}')

# create weight folder
create_folder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)