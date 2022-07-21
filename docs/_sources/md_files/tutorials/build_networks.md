# Training neural networks with constraints

Training deep neural networks is usually thought to be challenging both theoretically and practically, for which the vanishing/exploding gradients is one of the most important reasons.  To address such issue, several recent works focus on imposing Riemannian constraints to the weights of the layers in these deep neural networks. For example, some existing works demonstrate that the orthogonal constraints can stabilize the distribution of activations over layers within convolutional neural networks and make their optimization more efficient. And they observe encouraging improvements in the accuracy and robustness of the networks with orthogonal constraints. 





CDOpt supports PyTorch functions in addition to Manifold optimization. Researchers and developers can easily train neural networks with constrained weights based on the combination of CDOpt and PyTorch. Compared with existing PyTorch-based Riemannian optimization packages, CDOpt has the following features,

* CDOpt utilizes tensor computation and GPU/TPU acceleration based on PyTorch and JAX.
* CDOpt is compatible to all the optimizers provided in `torch.optim`,`torch_optimizers` and Optax.
* CDOpt provides plug-in neural layers  in `cdopt.nn` and `cdopt.linen`.  These layers can be directly plugged in any network built by PyTorch and JAX. 



## Training by PyTorch

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
from cdopt.nn.modules import Linear_cdopt, Conv2d_cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.nn import get_quad_penalty
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
        self.fc1 = Linear_cdopt(9216, 128, manifold_class= stiefel_torch, penalty_param = 0.02)
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



Next, we define the training and testing functions. DO NOT forget to add the quadratic penalty term to the loss function by the `get_quad_penalty()` function from `cdopt.nn`. 

```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) + get_quad_penalty(model)
        # loss = F.nll_loss(output, target) +  0.02 * model.conv1.quad_penalty()

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

 



## Training by JAX and FLAX

Let us start with a simple example on training neural networks with orthogonal weights by FLAX, a neural network library developed from JAX . We first import essential packages. 

```python
import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import tensorflow_datasets as tfds     # TFDS for MNIST

import cdopt
from cdopt.linen import Conv_cdopt, Dense_cdopt
from cdopt.manifold_jax import sphere_jax, stiefel_jax, euclidean_jax
```



Then we build the network by the neural layers from `cdopt.linen`. 

```python
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x, quad_penalty = Conv_cdopt(features=32, kernel_size=(3, 3), manifold_class = sphere_jax)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x, quad_penalty
```



Then we define the cross entropy loss and metrics

```python
def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
  
def compute_metrics(*, logits, labels, feas = 0):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'feas': feas
  }
  return metrics
```



Then we define how to train the network by utilizing the `train_state` class provided in FLAX,

```python
def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)
      
      
@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits, quad_penalty = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label']) + 0.05*quad_penalty
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics
  
def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state
```



Then we define the test steps,

```python
@jax.jit
def eval_step(params, batch):
  logits, quad_penalty = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'], feas = quad_penalty)
  
  
def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy'], summary['feas']
```



Next, we load the dataset by Tensorflow,

```python
def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds
  
train_ds, test_ds = get_datasets()
```



Finally, we set the arguments and start the training,

```python
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

learning_rate = 0.05
momentum = 0.9

state = create_train_state(init_rng, learning_rate, momentum)
num_epochs = 10
batch_size = 64
for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  # Evaluate on the test set after each training epoch 
  test_loss, test_accuracy, feas = eval_model(state.params, test_ds)
  print(' test epoch: %d, loss: %.2f, accuracy: %.2f, feas: %.2e' % (
      epoch, test_loss, test_accuracy * 100, feas))
```





## Supported components

This would be an ever increasing list of features. CDOpt currently supports:

### Manifolds

- All the manifolds in `cdopt.manifold_torch` and `cdopt.manifold_jax`. 

### Optimizers

- All the optimizers from PyTorch.
- All the optimizers from Torch-optimizer.
- All the optimizers from Optax.

### Layers

For PyTorch

- Linear layers and Bilinear layers.
- Convolutional layers: Conv1d, Conv2d, Conv3d. 
- Recurrent Layers: RNN, LSTM, GRU, and their [cells](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell). 
- 

For JAX/FLAX:

* Linear layers
* Convolutional layers