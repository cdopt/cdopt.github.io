from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import cdopt
from cdopt.nn import Conv2d_cdopt
from cdopt.manifold_torch import stiefel_torch
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel, the `manifold_class` can be chosen as any 
        # manifold class provided in `cdopt.manifold_torch`
        self.conv1 = Conv2d_cdopt(1, 6, 5, manifold_class=stiefel_torch)
        self.conv2 = Conv2d_cdopt(6, 16, 5, manifold_class=stiefel_torch)
        self.fc1 = nn.Linear(256, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Add layer.quad_penalty() to the loss function.
        loss = F.nll_loss(output, target) + 0.05*(model.conv1.quad_penalty()
                                         + model.conv2.quad_penalty())

        loss.backward()
        optimizer.step()


device = torch.device("cuda")
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr= 0.5)
transform=transforms.Compose([ transforms.ToTensor(), ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64,
                    shuffle=True, pin_memory=True)
for epoch in range(10):
    train(model, device, train_loader, optimizer, epoch)
