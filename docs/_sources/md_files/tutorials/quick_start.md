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



The constraints on the matrix $X$ require that $X$ is an orthogonal matrix, i.e., $X$ lies on the Stiefel manifold, 



$$
	\mathcal{S}_{m,s} = \{X \in \mathbb{R}^{m\times s}: X^\top X = I_s  \}. 
$$



The following is a minimal working example of how to solve the above problem using `cdopt` for a random symmetric matrix. As indicated in the introduction above, we follow four simple steps: we instantiate the manifold, create the cost function (using Autograd in this case), define a problem instance which we pass the manifold and the cost function, and run the minimization problem using one of the existing unconstrained optimization solvers. 

```python
# Import basic functions
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
import torch 
import cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.core.problem import Problem
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
problem_test = Problem(M, obj_fun, beta = beta)

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



Now let us take a deeper look at the code step by step. First, `cdopt` imports necessary packages and set the parameters.

```python
# Import basic functions
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
import torch 
import cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.core.problem import Problem
import time

# Set parameters
m = 200  # column size
s = 8    # row size
beta = 100  # penalty parameter
local_device = torch.device('cpu')  # the device to perform the computation
local_dtype = torch.float64  # the data type of the pytorch tensor
```



Then we describe the objective function, where the variables are PyTorch tensors. The cost function should be a 

```python
# Define object function
H = torch.randn(m,m).to(device =local_device, dtype = local_dtype)
H = H+H.T 

def obj_fun(X):
    return -0.5 * torch.sum( X * (H@X))  
```



Then we call `stiefel_torch` to generate a structure that describes the Stiefel manifold $\mathcal{S}_{n,p}$. This manifold corresponds to the constraint appearing in our optimization problem. For other constraints, take a look at the [various supported manifolds](#manifolds) for details. The second instruction creates a structure named `problem_test`. Here the gradients and hessians of the objective function are not necessary, as they can be computed by the automatic differentiation (AD) packages. 

```python
# Set optimization problems and retrieve constraint dissolving functions.
M = stiefel_torch((m,s), device =local_device, dtype = local_dtype )
problem_test = Problem(M, obj_fun, beta = beta)
```



After describe the optimization problem, we can directly retrieve the function value, gradients, Hessian of the corresponding constraint dissolving function. 

```python
# Get the objective function, gradient, hessian-vector product 
# of the corresponding constraint dissolving function
cdf_fun_np = problem_test.cdf_fun_vec_np   
cdf_grad_np = problem_test.cdf_grad_vec_np
cdf_hvp_np = problem_test.cdf_hvp_vec_np
```



Finally, we call the unconstraint solver from `scipy.optimize` package to minimize the constraint dissolving function over $\mathbb{R}^{n\times p}$. 

```python
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

 





## Manifolds

For several well-known manifolds, we provide build-in expressions for $\mathcal{A}$ in the following table. We strongly suggest you to use the provided structures to define the manifold if it is included in the following table. 

| Name                           | Expression of $c$                                            | Pre-defined structure from `autograd` | Pre-defined structure from`PyTorch`  |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------- | ------------------------------------ |
| Euclidean space                | No constraint                                                | `manifold.euclidean_np`               | `manifold.euclidean_torch`           |
| Sphere                         | $\left\{ x \in \mathbb{R}^{n}: x^\top x = 1 \right\}$        | `manifold.sphere_np`                  | `manifold.sphere_torch`              |
| Oblique manifold               | $\left\{ X \in \mathbb{R}^{m\times s}: \mathrm{Diag} (X ^\top X) = I_s \right\}$ | `manifold.obluqie_np`                 | `manifold.obluqie_torch`             |
| Stiefel manifold               | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top X = I_s \right\}$ | `manifold.stiefel_np`                 | `manifold.stiefel_torch`             |
| Grassmann manifold             | $\left\{ \mathrm{range}(X): X \in \mathbb{R}^{m\times s}, X ^\top X = I_s \right\}$ | `manifold.stiefel_np`                 | `manifold.stiefel_torch`             |
| Generalized Stiefel manifold   | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top B X = I_s \right\}$, $B$ is positive definite | `manifold.generalized_stiefel_np`     | `manifold.generalized_stiefel_torch` |
| Generalized Grassmann manifold | $\left\{ \mathrm{range}(X): X \in \mathbb{R}^{m\times s}, X ^\top B X = I_s \right\}$, $B$ is positive definite | `manifold.generalized_stiefel_np`     | `manifold.generalized_stiefel_torch` |
| Hyperbolic manifold            | $\left\{ X \in \mathbb{R}^{m\times s}: X ^\top B X = I_s \right\}$, $\lambda_{\min}(B)< 0 < \lambda_{\max}(B)$ | `manifold.hyperbolic_np`              | `manifold.hyperbolic_torch`          |
| Symplectic Stiefel manifold    | $\left\{ X \in \mathbb{R}^{2m\times 2s}: X ^\top Q_m X = Q_s \right\}$, $Q_m := \left[ \begin{smallmatrix}	{\bf 0}_{m\times m} & I_m\\			 -I_m & {\bf 0}_{m\times m}			\end{smallmatrix}\right]$ | `manifold.symp_stiefel_np`            | `manifold.symp_stiefel_torch`        |
| ...                            | ...                                                          | ...                                   |                                      |







 

## Solvers

`cdopt` package does not contain any solvers. However, the unconstrained minimization of the constraint dissolving functions can be solved by various of existing solvers. The solvers from `scipy.optimize`, `torch.optim` and `torch_optimizer` can be directly applied to minimize CDF over $\mathbb{R}^n$. 









## Automatic differentiation backbones

`cdopt` relies on the automatic differentiation (AD) packages to compute the derivatives of the objective function and build the constraint dissolving mapping. In `cdopt`, we provide various of plug-in AD backbones in `cdopt.core` based on `autograd` and `torch.autograd` packages. Moreover, one can easily build his own AD backbones by other packages, including the `jax` and `tensorflow`.  













## CUDA support

The CUDA support for`cdopt` relies on the employed backbones. Both the computation of constraint dissolving mappings and the unconstrained minimization of CDF can be accelerated by the CUDA support of the selected backbones. 



For example, by setting the `local_device = torch.device('cuda')` in the following code blocks, all the computations for CDF are accelerated by CUDA.    

```python
# Import basic functions
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
import torch 
import cdopt
from cdopt.manifold_torch import stiefel_torch
from cdopt.core.problem import Problem
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
problem_test = Problem(M, obj_fun, beta = beta)
```



On the one hand, we can use the interface provided by `cdopt.core.Problem` class to retrieve the constraint dissolving function that adopts `scipy.optimize` package. 

```python
cdf_fun_np = problem_test.cdf_fun_vec_np   
cdf_grad_np = problem_test.cdf_grad_vec_np
cdf_hvp_np = problem_test.cdf_hvp_vec_np

# Implement L-BFGS solver from scipy.optimize
Xinit = M.tensor2array(M.Init_point())
out_msg = sp.optimize.minimize(cdf_fun_np, Xinit.flatten(),method='L-BFGS-B',jac = cdf_grad_np)

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
        (loss).backward()
        return loss
    optim.step(closure)

from tqdm import tqdm

for jj in tqdm(range(2000)):
    train_epoch(jj)
    
X_fin = model.X
M.Feas_eval(X_fin)
```

















