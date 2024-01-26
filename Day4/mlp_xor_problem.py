import torch

#setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)  # cuda 관련 모든 난수 생성기에 시드 설정

# input
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)  # set input
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)  # set ground-truth.

# setup model
layer = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=100, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=100, out_features=1, bias=True)
).to(device)

# setup criterion
criterion = torch.nn.MSELoss()

# setup optimizer
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

# training
for epoch in range(1001):
    # extract output of layer
    output = layer(X)

    # calculate loss
    loss = criterion(output, Y)

    # gradient initialize
    optimizer.zero_grad()

    # calculate gradient
    loss.backward()

    # update parameters
    optimizer.step()

    # display
    if epoch % 100 == 0:
        print(f'[{epoch}] Loss: {loss.item()}')

result = layer(torch.FloatTensor([[0, 0]]).to(device))[0].item()
print(f'result of (0, 0): {result}')
result = layer(torch.FloatTensor([[0, 1]]).to(device))[0].item()
print(f'result of (0, 1): {result}')
result = layer(torch.FloatTensor([[1, 0]]).to(device))[0].item()
print(f'result of (1, 0): {result}')
result = layer(torch.FloatTensor([[1, 1]]).to(device))[0].item()
print(f'result of (1, 1): {result}')