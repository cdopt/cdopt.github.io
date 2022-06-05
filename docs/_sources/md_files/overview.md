# Overview

## Background

This project illustrates that we could directly apply the build-in functions in `scipy` package to solve a broad class of optimization problems on Riemannian manifolds by the [constraint dissolving approaches](https://arxiv.org/abs/2203.10319). In this project, we consider a class of optimization problems that can be expressed as 


$$
\begin{aligned}
	\min_{x \in \mathbb{R}^n}\quad &f(x) \\
	\text{s. t.} \quad &c(x) = 0.
\end{aligned}
\qquad\qquad \text{(OCP)}
$$


Here $\mathcal{M}$ refers to the feasible region of (OCP), namely, $\mathcal{M} := \{x \in \mathbb{R}^n: c(x) = 0\}$. Besides, we assume (OCP) satisfies the following assumptions:

* $f: \mathbb{R}^n \mapsto \mathbb{R}$ is locally Lipschitz continuous in $\mathbb{R}^n$.

* $c: \mathbb{R}^n \mapsto \mathbb{R}^p$ is twice continuous differentiable in $\mathbb{R}^n$.

* The linear independence constraint qualification (LICQ) holds at every $x \in \mathcal{M}$.  That is, the transposed Jacobian of $c$, denoted as $J_c:= [\nabla c_1(x), \nabla c_2(x), ..., \nabla c_p(x)] \in \mathbb{R}^{n\times p}$ , has full column rank at every $x \in \mathcal{M}$. 

  

As $c$ is smooth,  it is easy to verify that $\mathcal{M}$ is a Riemannian manifold embedded in $\mathbb{R}^n$. Therefore, (OCP) covers optimization problems over a wide range of Riemannian manifolds. Those Riemannian manifolds include sphere, oblique manifold, (generalized) Stiefel manifold, (generalized) Grassmann manifold, hyperbolic manifold, symplectic Stiefel manifold, etc. Interested readers could refer to the [webpage](https://www.manopt.org/tutorial.html#manifolds) for more examples on closed Riemannian manifolds. 



Existing Riemannian optimization packages, including PyManopt, and Geoopt, are developed based on the geometrical materials of the Riemannian manifold. Therefore, these packages only utilize the Riemannian solvers, which are quite limited when compared with unconstrained optimization solvers. Moreover, as determining the geometrical materials for unknown Riemannian manifold are typically challenging, these Riemannian optimization packages only support pre-defined Riemannian manifolds, which are restricted to several well-known cases. Furthermore, as the modules are highly specific, these Riemannian packages are generally hard for users to integrate it with existing popular machine learning frameworks.



To overcome these issues and facilitate both high performance and ease of use in machine and deep learning, we introduce a new software package called cdopt, which is developed based on the *constraint dissolving approaches* for Riemannian optimization with some key features,

* **Direct optimization:** `cdopt` is developed from the [constraint dissolving approaches](https://arxiv.org/abs/2203.10319), which transforms Riemannian optimization problems to unconstrained optimization problem. Therefore, we can utilize various highly efficient solvers for unconstrained optimization, and directly apply them to solve Riemannian optimization problems. Benefited from the rich expertise gained over the decades for unconstrained optimization, `cdopt` is very efficient and avoids the difficulties in developing Riemannian optimization solvers. 
* **Easy construction:** The optimization problem in`cdopt` can be constructed only from the expressions of the objective and constraints. Different from existing Riemannian optimization packages, we can easily and directly describe Riemannian optimization problems in `cdopt` without any geometrical materials of the Riemannian manifold (e.g., retractions, tangent spaces, vector-transports, etc.).
* **High compatibility :** `cdopt` has high compatibility with various numerical packages, including `numpy`, `PyTorch`, `JAX`, `tensorflow`, etc. Users can directly apply the advanced features of these numerical packages to accelerate the optimization, including the automatic differentiation, CUDA supports, distributed optimization supports, just-in-time compilations, etc. 





## A deeper introduction to constraint dissolving approaches

The *constraint dissolving approaches* aim to reshape (OCP) as an **unconstrained optimization problem** that adapts to unconstrained optimization approaches while keeping its stationary points unchanged. Therefore, constraint dissolving approaches enable **direct implementation** of unconstrained optimization approaches to solve (OCP). 

Given $c(x)$, we first consider a mapping $\mathcal{A}: \mathbb{R}^n \mapsto \mathbb{R}^n$ that satisfies the following assumptions,

* $\mathcal{A}$ is locally Lipschitz continuous in $\mathrm{R}^n$.
* $\mathcal{A}(x) = x$ holds for any $x \in \mathcal{M}$.
* For any $x \in \mathcal{M}$, the Jacobian of $c(\mathcal{A}(x))$ equals $0$, i.e., the Jacobian of $\mathcal{A}$, denoted as $J_A(x):=[\nabla \mathcal{A}_1(x), \nabla \mathcal{A}_2(x), ..., \nabla \mathcal{A}_n(x)]$, satisfies $J_A(x) J_c(x) = 0$ for any $x \in \mathcal{M}$. 

Based on the mapping $\mathcal{A}$,  we propose the constraint dissolving function for (OCP),


$$
h(x) := f(\mathcal{A}(x)) + \frac{\beta}{2} ||c(x)||_2^2. \qquad\qquad \text{(CDF)}
$$


Here $\beta > 0$ is a penalty parameter for the quadratic penalty term. As described in [1], CDF and OCP have the same stationary points, local minimizers and Lojasiewicz exponents in a neighborhood of $\mathcal{M}$. Interested readers could found the detailed proof in the [paper](https://arxiv.org/abs/2203.10319).

Noting that CDF has the same order of smoothness as the objective function $f(x)$. Therefore, the gradient of CDF is easy to achieve from $f(x)$:


$$
\nabla h(x) = J_A(x) \nabla f(\mathcal{A}(x)) + \beta J_c(x) c(x).
$$


Therefore, various approaches that is designed for unconstrained optimization can be **directly** applied to handle Riemannian optimization problems through CDF. Their convergence properties nearly automatically hold from the prior works on unconstrained cases. 

For generalized constraints $c(x) = 0$, we can design $\mathcal{A}$ by 


$$
\mathcal{A}(x) := x - J_c(x) \left(J_c(x)^T J_c(x) + \alpha||c(x)||^2  \right)^{-1}c(x).
$$


For several well-known manifolds, we also provide compact expressions for $\mathcal{A}$ in the following table. 

| Name                           | Expression of $c$                                            | Expression of $\mathcal{A}$                                  |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Sphere                         | $\left\{ x \in \mathbb{R}^{n}: x^\top x = 1 \right\}$        | $x \mapsto 2x\left(1 + x^\top x \right)^{-1} $               |
| Oblique manifold               | $\left\{ X \in \mathbb{R}^{m\times s}: \mathrm{Diag} (X ^\top X) = I_s \right\}$ | $X \mapsto 2X\left( I_s + \mathrm{Diag}(X^\top X) \right)^{-1} $ |
| Stiefel manifold               | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top X = I_s \right\}$ | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^\top X \right)$ |
| Grassmann manifold             | $\left\{ \mathrm{range}(X): X \in \mathbb{R}^{m\times s}, X ^\top X = I_s \right\}$ | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^\top X \right)$ |
| Complex Stiefel manifold       | $\left\{ X \in \mathbb{C}^{m\times s}: X^H X = I_s \right\}$ | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^H X \right)$ |
| Complex Grassmann manifold     | $\left\{ \mathrm{range}(X): X \in \mathbb{C}^{m\times s},  X^H X = I_s \right\}$ | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^H X \right)$ |
| Generalized Stiefel manifold   | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top B X = I_s \right\}$, $B$ is positive definite | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^\top B X \right)$ |
| Generalized Grassmann manifold | $\left\{ \mathrm{range}(X): X \in \mathbb{R}^{m\times s}, X ^\top B X = I_s \right\}$, $B$ is positive definite | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^\top B X \right)$ |
| Hyperbolic manifold            | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top B X = I_s \right\}$, $\lambda_{\min}(B)< 0 < \lambda_{\max}(B)$ | $X \mapsto X\left( \frac{3}{2}I_s - \frac{1}{2} X^\top B X \right)$ |
| Symplectic Stiefel manifold    | $\left\{ X \in \mathbb{R}^{2m\times 2s}: X ^\top Q_m X = Q_s \right\}$, $Q_m := \left[ \begin{smallmatrix}	{\bf 0}_{m\times m} & I_m\\			 -I_m & {\bf 0}_{m\times m}			\end{smallmatrix}\right]$ | $X \mapsto X \left(\frac{3}{2}I_{2s} + \frac{1}{2}Q_sX^\top Q_mX \right)$ |
| ...                            | ...                                                          | ...                                                          |



