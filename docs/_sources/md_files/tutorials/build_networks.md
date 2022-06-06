# Training neural networks with constrained weights

Training deep neural networks is usually thought to be challenging both theoretically and practically, for which the vanishing/exploding gradients is one of the most important reasons.  To address such issue, several recent works focus on imposing Riemannian constraints to the weights of the layers in these deep neural networks. For example, some existing works demonstrate that the orthogonal constraints can stabilize the distribution of activations over layers within convolutional neural networks and make their optimization more efficient. And they observe encouraging improvements in the accuracy and robustness of the networks with orthogonal constraints. 





CDOpt supports PyTorch functions in addition to Manifold optimization. Researchers and developers can easily train neurnal networks with constrained weights based on the combination of CDOpt and PyTorch. Compared with existing PyTorch-based Riemannian optimization packages, CDOpt has the following features,

* CDOpt utilizes tensor computation and GPU/TPU acceleration based on PyTorch.
* CDOpt is compatible to all the optimizers provided in `torch.optim` and `torch_optimizers`.
* CDOpt allows building networks by the mixture of layers from `torch.nn` and `cdopt.utils_torch.nn`. The layers provided by CDOpt are plug-in components to various neural networks. 



Let us start with a simple example on training neural networks with orthogonal weights. We first import essential packages. 

```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import cdopt 
from cdopt.utils_torch.nn.modules import Linear_cdopt, Conv2d_cdopt
from cdopt.manifold_torch import stiefel_torch
```

Then we build the neural network

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = Linear_cdopt(9216, 128, manifold_class= stiefel_torch)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```



Next, we define the training and testing functions.

```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) + 0.02* model.fc1.quad_penalty()
        # loss = F.nll_loss(output, target) +  0.01 * model.conv1.quad_penalty()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```



We then set the arguments 

```python
class ARGS():
    pass
args = ARGS()
args.batch_size = 64
args.test_batch_size = 1000
args.epochs = 14
args.lr = 0.1
args.gamma = 0.7 
args.no_cuda = False
args.seed = 1
args.log_interval = 10
args.save_model = False 
args.dry_run = False
```

and define the dataset

```python
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    device = torch.device("cuda")
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```



Finally, we start training the neural network.

```python
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
```

 

## Supported components

This would be an ever increasing list of features. CDOpt currently supports:

### Manifolds

- All the manifolds in `cdopt.manifold_torch`.

### Optimizers

- All the optimizers in `torch.optim`.
- All the optimizers in `torch_optimizer`.

### Layers

- Linear layers and Bilinear layers.
- Convolutional layers: Conv1d, Conv2d, Conv3d. 