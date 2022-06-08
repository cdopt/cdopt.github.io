# Welcome to CDOpt

**A Python toolbox for optimization on closed Riemannian manifolds with support for deep learning**



Riemannian optimization is a powerful framework to tackle nonlinear optimization problems with structural equality constraints. By transforming these Riemannian optimization problems into the minimization of constraint dissolving functions, CDOpt allows for elegant and direct implementations of various unconstrained optimization approaches. 



The constraint dissolving approaches have the following advantages:

* **Direct optimization & high efficiency:** `cdopt` is developed from the [constraint dissolving approaches](https://arxiv.org/abs/2203.10319), which transforms Riemannian optimization problems to unconstrained optimization problem. Therefore, we can utilize various highly efficient solvers for unconstrained optimization, and directly apply them to solve Riemannian optimization problems. Benefited from the rich expertise gained over the decades for unconstrained optimization, CDOpt is very efficient and avoids the difficulties in developing Riemannian optimization solvers. 
* **Easy to use:** CDOpt provides various plug-in layers for PyTorch packages. With only minor changes in the original codes, users can easily train the neural network while constrain the weights over various of closed manifolds. 
* **High compatibility:** CDOpt has high compatibility with various numerical packages, including `numpy`, `PyTorch`, `JAX`, etc. Users can directly apply the advanced features of these numerical packages to accelerate the optimization, including the automatic differentiation, CUDA supports, distributed optimization supports, just-in-time compilations, etc. 
* **Various constraints:** The optimization problem in CDOpt  can be constructed only from the expressions of the objective and constraints. Different from existing Riemannian optimization packages, we can easily and directly describe Riemannian optimization problems in CDOpt without any geometrical materials of the Riemannian manifold (e.g., retractions, tangent spaces, vector-transports, etc.).



```{tableofcontents}
```
