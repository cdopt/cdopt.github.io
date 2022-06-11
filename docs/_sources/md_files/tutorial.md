# Tutorials

CDOpt is an easy-to-use modular package that separates the manifold modules, the problems descriptions, the automatic differentiation packages and solvers apart. All of the automatic differentiations are done behind the scenes so the amount of setup that user needs to do is minimal. Usually only the following steps are required:

1. Instantiate a manifold $\mathcal{M}$ from the `cdopt.manifold` or define $\mathcal{M} = \{x \in \mathbb{R}^n: c(x) = 0 \}$ by the `cdopt.core.basic_manifold` module.
2. Define a cost function $f: \mathbb{R}^{m\times s} \to \mathbb{R}$ to minimize over the manifold $\mathcal{M}$.  
3. Using the `cdopt.core.problem` as a high-level interface to describe the optimization problem. 
4. Retrieve the corresponding constraint dissolving function and its differentials. Instantiate a solver from various of existing packages, including `scipy.optimize` and `torch.optim`, to minimize the constraint dissolving function without any constraints. 

It is worth mentioning that the `problem` class in CDOpt integrates various pre-processing and concurrency checking steps for the optimization problems. Moreover, it provides integrated APIs for calling the related solvers. Therefore, although we can run the solvers without the `cdopt.core.problem` interface, using  `cdopt.core.problem` to define the problem is always recommended. 







```{tableofcontents}
```



