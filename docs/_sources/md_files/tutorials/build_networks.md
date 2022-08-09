# Training neural networks with manifold constraints

Training deep neural networks is usually thought to be challenging both theoretically and practically, for which the vanishing/exploding gradients is one of the most important reasons.  To address such issue, several recent works focus on imposing Riemannian constraints to the weights of the layers in these deep neural networks. For example, some existing works demonstrate that the orthogonal constraints can stabilize the distribution of activations over layers within convolutional neural networks and make their optimization more efficient. And they observe encouraging improvements in the accuracy and robustness of the networks with orthogonal constraints. 





CDOpt supports PyTorch functions in addition to Manifold optimization. Researchers and developers can easily train neural networks with constrained weights based on the combination of CDOpt and PyTorch. Compared with existing PyTorch-based Riemannian optimization packages, CDOpt has the following features,

* CDOpt utilizes tensor computation and GPU/TPU acceleration based on PyTorch and JAX.
* CDOpt is compatible to all the optimizers provided in `torch.optim`,`torch_optimizers` and Optax.
* CDOpt provides plug-in neural layers  in `cdopt.nn` and `cdopt.linen`.  These layers can be directly plugged in any network built by PyTorch and JAX. 



## Supported components

This would be an ever increasing list of features. CDOpt currently supports:

**Manifolds**

- All the manifolds in `cdopt.manifold_torch` and `cdopt.manifold_jax`. 



**Optimizers**

- All the optimizers from PyTorch.
- All the optimizers from Torch-optimizer.
- All the optimizers from Optax.



**Neural layers**

For PyTorch:

- Linear layers and Bilinear layers.
- Convolutional layers: Conv1d, Conv2d, Conv3d. 
- Recurrent Layers: RNN, LSTM, GRU, and their [cells](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell). 



For JAX/FLAX:

* Linear layers
* Convolutional layers



## Impose manifold constraints by predefined layers

For those users that aims to train neural networks with manifold constraints, CDOpt provides various predefined neural layers in `cdopt.nn` and `cdopt.linen` modules for PyTorch and Flax, respectively. These predefined layers in CDOpt preserve the same APIs as the layers from PyTorch and Flax, hence users can plug these layers into the neural networks with minimal modification to the standard PyTorch or Flax codes.

### Training by PyTorch

`cdopt.nn` provides various of predefined layers for PyTorch, which inherit the same APIs as standard neural layers from `torch.nn`.  In the instantiation of these neural layers, we need to provide the `manifold_class` argument to set the type of manifold constraints, use `penalty_param` to set the penalty parameters, and choose the `weight_var_transfer` argument to determine how the weights of the layers are transferred into the variables of the manifolds. 



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

Then we build the neural network, where we restrict the weights of the first FC layer on the Stiefel manifold, and set the penalty parameter as 0.02. 

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
        # equivalent to 
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



We then set the arguments and load the dataset

```python
use_cuda = torch.cuda.is_available()

torch.manual_seed(1)

train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 1000}
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
optimizer = optim.Adadelta(model.parameters(), lr=0.1)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
```



Finally, we start training the neural network.

```python
for epoch in range(1, 11):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
```

 



### Training by JAX and FLAX

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





## Impose manifold constraints by `set_constraint_dissolving()`

Furthermore, for those neural layers that are not predefined in `cdopt.nn`, CDOpt provides a simple way to add manifold constraints to the parameters of these neural layers. Through the `set_constraint_dissolving` function from `cdopt.nn.utils.set\_constraints`, users can set the manifold constraints to the layers by just providing the neural layers, the name of target parameters and the manifold class.  The following example illustrates how to set the manifold constraints to the first full connect layer for LeNet. 

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)  # 5*5 from image dimension 
        set_constraint_dissolving(self.fc1, 'weight', manifold_class = stiefel_torch, penalty_param= 0.02)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
```







## Functional API for modules in PyTorch

PyTorch introduces a new feature to functionally apply Module computation with a given set of parameters. Sometimes, the traditional PyTorch Module usage pattern that maintains a static set of parameters internally is too restrictive. This is often the case when implementing algorithms for meta-learning, where multiple sets of parameters may need to be maintained across optimizer steps. Based on the functions from`torch.nn.utils.stateless`, we develop functions from `cdopt.nn.utils.stateless`, which allows the 

- Module/feasibility computation with full flexibility over the set of parameters used
- No need to reimplement your module in a functional way
- Any parameter or buffer present in the module can be swapped with an externally-defined value for use in the call. Naming for referencing parameters / buffers follows the fully-qualified form in the module’s `state_dict()`



Here is an simple example:

```python
import torch
import cdopt
from torch import nn
from cdopt.manifold_torch import stiefel_torch
from cdopt.nn.utils.stateless import functional_call, get_quad_penalty_call, functional_quad_penalty_call

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = cdopt.nn.Linear_cdopt(3, 3, manifold_class= stiefel_torch, penalty_param=0.1)
        self.bn = nn.BatchNorm1d(3)
        self.fc2 = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc2(self.bn(self.fc1(x)))

m = MyModule()

# Define parameter / buffer values to use during module computation.
my_weight = torch.randn(3, 3, requires_grad=True)
my_bias = torch.tensor([1., 2., 3.], requires_grad=True)
params_and_buffers = {
    'fc1.weight': my_weight,
    'fc1.bias': my_bias,
    # Custom buffer values can be used too.
    'bn.running_mean': torch.randn(3),
}

# Apply module computation to the input with the specified parameters / buffers.
inp = torch.randn(5, 3)
output1 = functional_call(m, params_and_buffers, inp)
quad_penalty1 = get_quad_penalty_call(m,params_and_buffers)
output2, quad_penalty2 = functional_quad_penalty_call(m, params_and_buffers, inp)
```

