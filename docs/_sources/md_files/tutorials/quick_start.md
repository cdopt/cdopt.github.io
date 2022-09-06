# Quickstart



## A simple example

We begin with a simple optimization problem with orthogonality constraints,



$$
\begin{aligned}
		\min_{X \in \mathbb{R}^{m\times s}}\quad &f(X) = -\frac{1}{2}\mathrm{tr}(X^\top HX)\\
		\text{s.t.} \quad & X^\top X = I_s,
	\end{aligned}
$$



where $H \in \mathbb{R}^{m\times m}$ is a symmetric matrix, and the gradient of $f$ can be expressed as 



$$
\nabla f(X) = -HX.
$$



The constraints on the matrix $X$ require that $X$ should be an orthogonal matrix, i.e., $X$ lies on the Stiefel manifold, 



$$
\mathcal{S}_{m,s} = \{X \in \mathbb{R}^{m\times s}: X^\top X = I_s  \}.
$$



The following is a minimal working example of how to solve the above problem using CDOpt for a random symmetric matrix. As indicated in the introduction above, we follow four simple steps: we instantiate the manifold, create the cost function (using PyTorch in this case), define a problem instance which we pass the manifold and the cost function, and run the minimization problem using one of the existing unconstrained optimization solvers. 

```python
# Import basic functions
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
import torch 
import cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.core.problem import problem
import time

# Set parameters
m = 200  # column size
s = 8    # row size
beta = 100  # penalty parameter
local_device = torch.device('cpu')  # the device to perform the computation
local_dtype = torch.float64  # the data type of the pytorch tensor

# Define object function
H = torch.randn(m,m).to(device =local_device, dtype = local_dtype)
H = H+H.T 

def obj_fun(X):
    return -0.5 * torch.sum( X * (H@X)) 


# Set optimization problems and retrieve constraint dissolving functions.
M = stiefel_torch((m,s), device =local_device, dtype = local_dtype )
problem_test = problem(M, obj_fun, beta = beta)

cdf_fun_np = problem_test.cdf_fun_vec_np
cdf_grad_np = problem_test.cdf_grad_vec_np
cdf_hvp_np = problem_test.cdf_hvp_vec_np

# Implement L-BFGS solver from scipy.optimize
Xinit = problem_test.Xinit_vec_np
t_start = time.time()
out_msg = sp.optimize.minimize(cdf_fun_np, Xinit,method='L-BFGS-B',jac = cdf_grad_np)
t_run = time.time() - t_start

# Statistics
feas = M.Feas_eval(M.v2m(M.array2tensor(out_msg.x)))   # Feasibility
stationarity = np.linalg.norm(out_msg['jac'],2)   # stationarity
result_lbfgs = [out_msg['fun'], out_msg['nit'], out_msg['nfev'],stationarity,feas, t_run]
print('& L-BFGS & {:.2e} & {:} & {:} & {:.2e} & {:.2e} & {:.2f} \\\\'.format(*result_lbfgs))
```



Now let us take a deeper look at the code step by step. First, we imports necessary packages and set the parameters for the optimization problem. 

```python
# Import basic functions
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
import torch 
import cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.core.problem import problem
import time

# Set parameters
m = 200  # column size
s = 8    # row size
beta = 100  # penalty parameter
local_device = torch.device('cpu')  # the device to perform the computation
local_dtype = torch.float64  # the data type of the pytorch tensor
```



Then we describe the objective function, where the variables are PyTorch tensors. The cost function can be compatible to the automatic differentiation (AD) packages in PyTorch. Otherwise, we need to manually provide the gradient and Hessian-vector produce of the objective function. 

```python
# Define object function
H = torch.randn(m,m).to(device =local_device, dtype = local_dtype)
H = H + H.T 

def obj_fun(X):
    return -0.5 * torch.sum( X * (H@X))  
```



Then we call `stiefel_torch` to generate a structure that describes the Stiefel manifold $\mathcal{S}_{n,p}$. This manifold corresponds to the constraint appearing in our optimization problem. For other constraints, take a look at the [various supported manifolds](#manifolds) for details. The second instruction creates a structure named `problem_test`. Here the gradients and hessians of the objective function are not necessary, as they can be computed by the AD packages provided by PyTorch.  

```python
# Set optimization problems and retrieve constraint dissolving functions.
M = stiefel_torch((m,s), device =local_device, dtype = local_dtype )
problem_test = problem(M, obj_fun, beta = beta)
```



After describe the optimization problem, we can directly retrieve the function value, gradients, Hessian-vector product of the corresponding constraint dissolving function. 

```python
# Get the objective function, gradient, hessian-vector product 
# of the corresponding constraint dissolving function
cdf_fun_np = problem_test.cdf_fun_vec_np   
cdf_grad_np = problem_test.cdf_grad_vec_np
cdf_hvp_np = problem_test.cdf_hvp_vec_np
```



Finally, we call the L-BFGS solver from `scipy.optimize` package to minimize the constraint dissolving function over $\mathbb{R}^{n\times p}$. 

```python
# Implement L-BFGS solver from scipy.optimize
Xinit = problem_test.Xinit_vec_np
t_start = time.time()
out_msg = sp.optimize.minimize(cdf_fun_np, Xinit, method='L-BFGS-B',jac = cdf_grad_np)
t_run = time.time() - t_start


# Statistics
X_tensor = M.array2tensor(out_msg.x)   # Transfer the numpy 1D array to tensor
X_var = M.v2m(X_tensor)   # Transfer the 1D tensor to the tensor with shape M.var_shape.
feas = M.Feas_eval(X_var)   # Evaluate the feasibility
stationarity = np.linalg.norm(out_msg['jac'],2)   # stationarity
result_lbfgs = [out_msg['fun'], out_msg['nit'], out_msg['nfev'],stationarity,feas, t_run]
print('& L-BFGS & {:.2e} & {:} & {:} & {:.2e} & {:.2e} & {:.2f} \\\\'.format(*result_lbfgs))
```

 





## Manifolds

For several well-known manifolds, we provide build-in expressions for $\mathcal{A}$ in the following table. We strongly suggest you to use the pre-defined manifold class if it is included in the following table. 

| Name                           | Expression of $c$                                            | Pre-defined structure by Numpy | Pre-defined structure by PyTorch | Pre-defined structure by JAX |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------ | -------------------------------- | ---------------------------- |
| Euclidean space                | No constraint                                                | `euclidean_np`                 | `euclidean_torch`                | `euclidean_jax`              |
| Sphere                         | $\left\{ x \in \mathbb{R}^{n}: x^\top x = 1 \right\}$        | `sphere_np`                    | `sphere_torch`                   | `sphere_jax`                 |
| Oblique manifold               | $\left\{ X \in \mathbb{R}^{m\times s}: \mathrm{Diag} (X  X^\top) = I_m \right\}$ | `obluqie_np`                   | `obluqie_torch`                  | `obluqie_jax`                |
| Stiefel manifold               | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top X = I_s \right\}$ | `stiefel_np`                   | `stiefel_torch`                  | `stiefel_jax`                |
| Grassmann manifold             | $\left\{ \mathrm{range}(X): X \in \mathbb{R}^{m\times s}, X ^\top X = I_s \right\}$ | `stiefel_np`                   | `stiefel_torch`                  | `stiefel_jax`                |
| Generalized Stiefel manifold   | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top B X = I_s \right\}$, $B$ is positive definite | `generalized_stiefel_np`       | `generalized_stiefel_torch`      | `generalized_stiefel_jax`    |
| Generalized Grassmann manifold | $\left\{ \mathrm{range}(X): X \in \mathbb{R}^{m\times s}, X ^\top B X = I_s \right\}$, $B$ is positive definite | `generalized_stiefel_np`       | `generalized_stiefel_torch`      | `generalized_stiefel_jax`    |
| Hyperbolic manifold            | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top B X = I_s \right\}$, $\lambda_{\min}(B)< 0 < \lambda_{\max}(B)$ | `hyperbolic_np`                | `hyperbolic_torch`               | `hyperbolic_jax`             |
| Symplectic Stiefel manifold    | $\left\{ X \in \mathbb{R}^{2m\times 2s}: X ^\top Q_m X = Q_s \right\}$, $Q_m := \left[ \begin{smallmatrix}	{\bf 0}_{m\times m} & I_m\\			 -I_m & {\bf 0}_{m\times m}			\end{smallmatrix}\right]$ | `symp_stiefel_np`              | `symp_stiefel_torch`             | `symp_stiefel_jax`           |
| Complex sphere                 | $\{x \in \mathbb{C}^n : x^H x = 1  \}$                       | `complex_shpere_np`            | `complex_shpere_torch`           | `complex_shpere_jax`         |
| Complex oblique manifold       | $\left\{ X \in \mathbb{C}^{m\times s}: \mathrm{Diag} (X  X^H) = I_m \right\}$ | `complex_oblique_np`           | `complex_oblique_torch`          | `complex_oblique_jax`        |
| Complex Stiefel manifold       | $\left\{ X \in \mathbb{C}^{m\times s}: X ^H X = I_s \right\}$ | `complex_stiefel_np`           | `complex_stiefel_torch`          | `complex_stiefel_jax`        |
| ...                            | ...                                                          | ...                            |                                  |                              |





 

## Solvers

In CDOpt, the Riemannian optimization problems are transferred into the unconstrained minimization of the constraint dissolving functions, which can be solved by various of existing solvers. As far as we tested, the solvers from [PDFO](https://www.pdfo.net/index.html), [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize), [PyTorch](https://pytorch.org/docs/stable/optim.html), [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer), and [jaxopt](https://github.com/google/jaxopt) packages can be directly applied to minimize the constraint dissolving functions yielded by CDOpt. 









## Automatic differentiation backbones

CDOpt relies on the automatic differentiation (AD) packages to compute the derivatives of the objective function and build the constraint dissolving mapping. In CDOpt, we provide various of plug-in AD backbones in `cdopt.core` based on autograd and PyTorch packages. Moreover, one can easily build his own AD backbones based on other packages, including the `jax` and `tensorflow`.  













## CUDA support

CDOpt utilizes the CUDA supports from the employed backbones. Both the computation of constraint dissolving mappings and the unconstrained minimization of the constraint dissolving functions can be accelerated by the CUDA support of the selected backbones. 



For example, by setting the `local_device = torch.device('cuda')` in the following code blocks, all the computations for CDF are accelerated by CUDA.    

```python
# Import basic functions
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
import torch 
import cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.core.problem import problem
import time

# Set parameters
m = 200  # column size
s = 8    # row size
beta = 100  # penalty parameter
local_device = torch.device('cuda')  # the device to perform the computation
local_dtype = torch.float64  # the data type of the pytorch tensor

# Define object function
H = torch.randn(m,m).to(device =local_device, dtype = local_dtype)
H = H+H.T 

def obj_fun(X):
    return -0.5 * torch.sum( X * (H@X)) 


# Set optimization problems and retrieve constraint dissolving functions.
M = stiefel_torch((m,s), device =local_device, dtype = local_dtype )
problem_test = problem(M, obj_fun, beta = beta)
```



On the one hand, we can use the interface provided by `cdopt.core.problem` class to retrieve the constraint dissolving function that adopts `scipy.optimize` package. 

```python
cdf_fun_np = problem_test.cdf_fun_vec_np   
cdf_grad_np = problem_test.cdf_grad_vec_np
cdf_hvp_np = problem_test.cdf_hvp_vec_np

# Implement L-BFGS solver from scipy.optimize
Xinit = M.tensor2array(M.Init_point())
t_start = time.time()
out_msg = sp.optimize.minimize(cdf_fun_np, Xinit.flatten(),method='L-BFGS-B',jac = cdf_grad_np)
t_end = time.time() - t_start

# Statistics
feas = M.Feas_eval(M.v2m(M.array2tensor(out_msg.x)))   # Feasibility
stationarity = np.linalg.norm(out_msg['jac'],2)   # stationarity
result_lbfgs = [out_msg['fun'], out_msg['nit'], out_msg['nfev'],stationarity,feas, t_end]
print('& L-BFGS & {:.2e} & {:} & {:} & {:.2e} & {:.2e} & {:.2f} \\\\'.format(*result_lbfgs))
```



On the other hand, we can also employ the solvers from `torch.optim` and `torch_optimizer` packages to minimize the constraint dissolving function. 

```python
cdf_fun = problem_test.cdf_fun
class Model(torch.nn.Module):
    def __init__(self, Xinit = None):
        super().__init__()
        self.X = torch.nn.Parameter(M.Init_point(Xinit))
    
    def forward(self):
        return cdf_fun(self.X)

model = Model(M.Init_point())
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

def train_epoch(epoch):
    def closure():
        optim.zero_grad()
        loss = model()
        loss.backward()
        return loss
    optim.step(closure)

from tqdm import tqdm

for jj in tqdm(range(2000)):
    train_epoch(jj)
    
X_fin = model.X
M.Feas_eval(X_fin)
```

















