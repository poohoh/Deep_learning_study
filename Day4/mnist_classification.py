import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from multiprocessing import freeze_support

if __name__=='__main__':
    freeze_support()

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reproducibility
    torch.manual_seed(123)
    if device == 'cuda':
        torch.cuda.manual_seed_all(123)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # setup image set
    train_X = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_X = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    # setup data loader
    batch_size = 64
    train_loader = DataLoader(train_X, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_X, batch_size=128, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    # Model
    layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=784, out_features=256, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=10, bias=True)
    ).to(device)

    print(layer)

    # Optimizer
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

    # Training
    for epoch in range(10):
        for idx, (images, labels) in enumerate(train_loader):
            # data to cuda
            images, labels = torch.FloatTensor(images).to(device), torch.LongTensor(labels).to(device)

            # output
            output = layer(images)

            # loss
            loss = F.cross_entropy(input=output, target=labels)

            # gradient initialize
            optimizer.zero_grad()

            # calculate gradient
            loss.backward()

            # update parameters
            optimizer.step()

            # calculate accuracy
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1)
            accuracy = torch.eq(pred, labels).float().mean()
            if idx % batch_size == 0:
                print(f'train iteration: {idx}, loss: {loss.item()}')

        print(f'epoch: {epoch+1} completed\n')

    # Evaluation
    with torch.no_grad():
        accuracy = 0
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = torch.FloatTensor(images).to(device), torch.LongTensor(labels).to(device)

            # get output of model
            output = layer(images)

            # calculate loss
            loss = F.cross_entropy(input=output, target=labels)

            # calculate accuracy
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1)
            accuracy += torch.eq(pred, labels).float().mean()
        accuracy = accuracy / len(test_loader)

        print(f'test accuracy: {accuracy}')