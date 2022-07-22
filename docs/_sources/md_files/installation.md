# Installation

CDOpt is compatible with Python 3.6+, and depends on NumPy and SciPy. CDOpt is available to install via the [Python Package Index](https://pypi.org/project/cdopt/), and can be installed via `pip`,



```shell
pip install cdopt --upgrade
```



Moreover, to use the advanced features of CDOpt, including the automatic differentiation, customized manifolds, CUDA supports and training neural networks, the users need to install some numerical packages, such as [autograd](https://github.com/HIPS/autograd), [PyTorch](https://pytorch.org/), or [JAX](https://jax.readthedocs.io/en/latest/index.html). We strongly recommend the users to install the following packages

* Autograd >=1.0
* PyTorch >= 1.7.0
* JAX >= 0.3.12 (optional since JAX is not available on windows platform) 

 Please follow the installation guide for [autograd](https://github.com/HIPS/autograd), [PyTorch](https://pytorch.org/get-started/locally/), and [JAX](https://jax.readthedocs.io/en/latest/installation.html) to install these packages. 