 

# stiefel_torch

`CLASS stiefel_torch(var_shape, device = torch.device('cpu'), dtype = torch.float64)`

This manifold class defines the Stiefel manifold, i.e. 


$$
\{X \in \mathbb{R}^{n\times p}: X^T X = I_p \}.
$$





##  **Parameters:**

* **var_shape** (tuple of ints) -- The shape of the variables of the manifold. `var_shape = (*self.dim, self._n, self_p)`
* **device** (PyTorch device) -- The object representing the device on which a [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) is or will be allocated.
* **dtype** (PyTorch dtype) -- The object that represents the data type of a [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor).





The variables in `stiefel_torch` is not restricted in 2D-tensors, they can be Nd-tensors whose `.size()` is `var_shape`. In that cases, for each index `(i,j,k,...)`, we have `torch.matmul( X(i,j,k,...,:,:).T ,X(i,j,k,...,:,:)) ` is a $p\times p$-dimensional identity matrix. 



## **Attributes:**



`A(x)` (callable) 

The constraint dissolving mapping $\mathcal{A}(x)$. `A(X)`  is set as `1.5 * X - torch.matmul(X , (torch.matmul(X.transpose(-2,-1), X)/2))`. 



`C(X)` (callable)

Describe the constraints $c$. `C(X)` returns `torch.matmul(X.transpose(-2,-1), X) - self.Ip`.



`_parameter()` (OrdDict)

The ordered dictionary that contains all the variables that changes when `device` and `dtype` changes. It contains

`self._parameters['Ip'] = torch.diag_embed(torch.ones((*self.dim, self._p), device=self.device, dtype=self.dtype))`



`m2v(x)` (callable)

Flatten the variable of the manifold.



`v2m(x)` (callable) 

Recover flattened variables to its original shape as `variable_shape`. 



`Init_point(Xinit = None)` (callable)

Generate the initial point. 



`tensor2array(x)` (callable)

Transfer the variable of the manifold to the numpy Nd-array while keep its shape. Default settings are provided in the `core.backbone_torch`. 



`array2tensor(x)` (callable)

Transfer the numpy Nd-array to the variable of the manifold while keep its shape. Default settings are provided in the `core.backbone_torch`. 





`JC(x, lambda)` (callable)

The Jacobian of `C(x)`. 



`JC_transpose(x, lambda)` (callable)

The transpose of $J_c(x)$, expressed by matrix-vector production. 




`JA(x, d)` (callable)

The transposed Jacobian of $\mathcal{A}(x)$. 



`JA_transpose(x, d)` (callable) 

The transpose (or adjoint) of `JA(x)`, i.e. $\lim_{t \to 0} \frac{1}{t}(J_A(x+td) -J_A(x)) $. 



`C_quad_penalty(x)` (callable)

Returns the quadratical penalty term $||c(x)||^2$. 



`hessA(X, U, D)` (callable)

Returns the Hessian of $\mathcal{A}(x)$ in a tensor-vector product form. 



`hess_feas(X, D)` (callable)

Returns the hessian-vector product of $\frac{1}{2} ||c(x)||^2$. 



`Feas_eval(X)` (callable)

Returns the feasibility of $x$, measured by value of $||c(x)||$. 



`Post_process(X)` (callable)

Return the post-processing for `X` to achieve a point with better feasibility. 



`generate_cdf_fun(obj_fun, beta)` (callable)

Return the function value of the constraint dissolving function. `obj_fun` is a callable function that returns the value of $f$ at $x$. `beta` is a float object that refers to the penalty parameter in the constraint dissolving function. 



`generate_cdf_grad(obj_grad, beta)` (callable)

Return the gradient of the constraint dissolving function. `obj_grad` is a callable function that returns the gradient of $f$ at $x$. `beta` is a float object that refers to the penalty parameter in the constraint dissolving function. 



`generate_cdf_hess(obj_grad, obj_hvp, beta)` (callable)

Return the hessian of the constraint dissolving function. `obj_grad` is a callable function that returns the gradient of $f$ at $x$. `obj_hvp` is the hessian-vector product of $f$ at $x$, i.e., $\nabla^2 h(x)[d]$.  `beta` is a float object that refers to the penalty parameter in the constraint dissolving function. 







