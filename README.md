# Welcome to CDOpt

**A Python toolbox for optimization on closed Riemannian manifolds with support for automatic differentiation**



Riemannian optimization is a powerful framework to tackle nonlinear optimization problems with structural equality constraints. By transforming these Riemannian optimization problems into the minimization of so-called constraint dissolving functions, `cdopt` allows for elegant and direct implementations of various unconstrained optimization approaches. 



The constraint dissolving approaches have the following advantages:

* **Direct optimization:** `cdopt` is developed from the [constraint dissolving approaches](https://arxiv.org/abs/2203.10319), which transforms Riemannian optimization problems to unconstrained optimization problem. Therefore, we can utilize various highly efficient solvers for unconstrained optimization, and directly apply them to solve Riemannian optimization problems. Benefited from the rich expertise gained over the decades for unconstrained optimization, `cdopt` is very efficient and avoids the difficulties in developing Riemannian optimization solvers. 
* **Easy construction:** The optimization problem in`cdopt` can be constructed only from the expressions of the objective and constraints. Different from existing Riemannian optimization packages, we can easily and directly describe Riemannian optimization problems in `cdopt` without any geometrical materials of the Riemannian manifold (e.g., retractions, tangent spaces, vector-transports, etc.).
* **High compatibility :** `cdopt` has high compatibility with various numerical packages, including `numpy`, `PyTorch`, `JAX`, `tensorflow`, etc. Users can directly apply the advanced features of these numerical packages to accelerate the optimization, including the automatic differentiation, CUDA supports, distributed optimization supports, just-in-time compilations, etc. 



```{tableofcontents}
```
