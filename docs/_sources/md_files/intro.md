# Welcome to CDOpt

**A Python toolbox for optimization on Riemannian manifolds with supports for deep learning**



Riemannian optimization is a powerful framework to tackle nonlinear optimization problems with structural equality constraints. By transforming these Riemannian optimization problems into the minimization of the *[constraint dissolving functions](https://arxiv.org/abs/2203.10319)*, CDOpt allows for elegant and direct implementation various unconstrained optimization approaches for Riemannian optimization problems. CDOpt also provides user-friendly frameworks for training manifold constrained neural networks by PyTorch and Flax.



CDOpt have the following key features: 

* **Dissolved constraints:** CDOpt transforms Riemannian optimization problems into equivalent unconstrained optimization problems. Therefore, we can utilize various highly efficient solvers for unconstrained optimization, and directly apply them to solve Riemannian optimization problems. Benefiting from the rich expertise gained over decades for unconstrained optimization, CDOpt is very efficient and naturally avoids the difficulties in extending the unconstrained optimization solvers to their Riemannian versions.
* **High compatibility:** CDOpt has high compatibility with various numerical backends, including NumPy, SciPy, PyTorch, JAX, Flax, etc . Users can directly apply the advanced features of these packages to accelerate optimization, including the automatic differentiation, GPU/TPU supports, distributed optimization frameworks, just-in-time (JIT) compilation, etc.
* **Customized constraints:** CDOpt dissolves manifold constraints without involving any geometrical material of the manifold in question. Therefore, users can directly define various Riemannian manifolds in CDOpt through their constraint expressions $c(x)$.
* **Plug-in neural layers:** CDOpt provides various plug-in neural layers for [PyTorch](https://pytorch.org/) and [Flax](https://flax.readthedocs.io/) packages. With minor changes in the standard PyTorch/Flax codes, users can easily build and train neural networks with various manifold constraints. 



```{tableofcontents}
```
