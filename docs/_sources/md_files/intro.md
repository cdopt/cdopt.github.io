# Welcome to CDOpt

**A Python toolbox for optimization on closed Riemannian manifolds with supports for deep learning**



Riemannian optimization is a powerful framework to tackle nonlinear optimization problems with structural equality constraints. By transforming these Riemannian optimization problems into the minimization of constraint dissolving functions, CDOpt allows for elegant and direct implementation various unconstrained optimization approaches for Riemannian optimization problems. CDOpt also provides user-friendly frameworks for training manifold constrained neural networks by PyTorch and Flax.



The constraint dissolving approaches have the following advantages:

* **Direct optimization & high efficiency:** CDOpt is developed from the *[constraint dissolving approaches](https://arxiv.org/abs/2203.10319)*, which transforms Riemannian optimization problems to unconstrained ones. Therefore, we can utilize various highly efficient solvers for unconstrained optimization, and directly apply them to solve Riemannian optimization problems. Benefited from the rich expertise gained over the decades for unconstrained optimization, CDOpt is very efficient and naturally avoids the difficulties in developing specialized solvers for Riemannian optimization.
* **Plug-in neural layers:** CDOpt provides various plug-in neural layers for PyTorch and \Pkg{Flax} packages. With only minor changes in the original codes, users can easily build and train the neural network while constrain the weights over various manifolds.
* **High efficiency:** CDOpt has high compatibility with various numerical backends, including NumPy, SciPy, PyTorch, JAX, Flax, etc . Users can directly apply the advanced features of these packages to accelerate optimization, including the automatic differentiation, GPU/TPU supports, distributed optimization frameworks, just-in-time (JIT) compilation, etc.
* **Customized constraints:** The manifold classes in CDOpt can be constructed only from the expressions of constraints. Users can easily and directly describe Riemannian optimization problems in CDOpt without any geometrical materials of the Riemannian manifold (e.g., retractions and their inverse, vector-transports, etc.).



```{tableofcontents}
```
