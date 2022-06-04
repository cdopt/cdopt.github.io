# cdopt.core

## cdopt.core.Problem

`CLASS cdopt.core.Problem(manifold, obj_fun, obj_grad=None, obj_hvp=None, beta = 0, enable_autodiff = True, backbone = 'torch',  **kwargs)`

Problem class to define a Riemannian optimization problem. 



**Parameters:**

* **manifold** (cdopt.manifold) 
  -- The manifold to optimize over.
* **obj_fun** (callable)
  --  A callable function that maps a point in the space to a scalar. 
* **obj_grad=None** (callable, optional) 
  -- The Euclidean gradient of the objective function.
* **obj_hvp=None** (callable, optional)
  -- The Euclidean hessian-vector product of the objective function.
* **beta = 0** (float, optional)
  -- The penalty parameter.
* **enable_autodiff  = True** (bool, optional) 
  -- Controls whether to perform the automatic differentiation packages to automatically compute essential materials. If `False`, the `Problem` class does not use any AD package. In that cases, users must provide the expression of `obj_grad` and `obj_hvp`. 
* **backbone = 'torch'** (str or core.backbone class, optional)
  -- Determines the automatic differentiation packages.  If ``'torch'``, the ``Problem`` class uses the `torch.autograd` to automatically compute essential materials. If ``'autograd'``, the `Problem` class uses the `autograd` package. Otherwise, it can be set as the user-defined backbone class. 





**Attributes:**

`obj_fun(x)` (callable)

​				Returns the function value of $f$. 



`obj_grad(x)` (callable)

​				Returns the gradient of $f$, which has the same shape as $x$. 



`obj_hvp(x, v)` (callable)

​				Returns the hessian-vector product of $\nabla^2 f(x)[v] := \lim_{t \to 0} \frac{1}{t}(\nabla f(x+t) - \nabla f(x))$, which has the same shape as $x$. 



`cdf_fun(x)` (callable)

Returns the function value of the corresponding constraint dissolving function $h(x)$.



`cdf_grad(x)` (callable) 

Returns the gradient of   the corresponding constraint dissolving function. 



`cdf_hvp(x,v)` (callable) 

Returns the hessian-vector product of $\nabla^2 h(x)[v]$. 



`cdf_fun_vectorized(x_vec)` (callable)

Returns the function value of the corresponding constraint dissolving function $h(x)$. Here `x_vec` is a 1D-array or tensor. 



`cdf_grad_vectorized(x_vec)` (callable) 

Returns the gradient of  the corresponding constraint dissolving function. Here `x_vec` is a 1D-array  or tensor. 



`cdf_hvp_vectorized(x_vec,v_vec)` (callable) 

Returns the hessian-vector product of $\nabla^2 h(x)[v]$. Here `x_vec` and `v_vec` are 1D-arrays or tensors. 





## cdopt.core.backbone_autograd

`class backbone_autograd(*args, **kwargs)`

The AD backbone that utilizes the AD algorithms from `autograd` package to compute the differentials of objective function, constraints and constraint dissolving function. 



**Parameters**

`backbone_autograd` does not require any input parameters.





**Attributes:**

`auto_diff_vjp(fun, X, D)` (callable)

Returns the vector-Jacobian product for `fun` at `X`  with direction `D`.



`auto_diff_jvp(fun, X, D)` (callable)

Returns the Jacobian-vector product for `fun` at `X`  with direction `D`. 



`auto_diff_jacobian(fun, X)` (callable)

Returns the Jacobian matrix for `fun` at `X`. 



`autodiff(self, obj_fun, obj_grad = None, manifold_type = 'S')` (callable)

Returns the `tuple(obj_grad, obj_hvp)` , where `obj_grad` refers to the gradient of the `obj_fun`, and `obj_hvp` refers to the hessian-vector product of `obj_fun`. 



`array2tensor(X_array)` (callable)

Returns the variable the adapts the `autograd` package from the numpy array `X_array`. 



`tensor2array(X_tensor)` (callable)

Returns the numpy array from the variable that adapts the `autograd` package. 



`solve` (callable)

Alias of the `autograd.numpy.linalg.solve`. In general, `solve` should act as the `numpy.linalg.solve` and compatible with the AD backbone.   

 

`identity_matrix` (callable)

Alias of the `autograd.numpy.eye`. In general, `identity_matrix` should return an identity matrix, whose type should be compatible with the AD backbone. 



`zero_matrix` (callable)

Alias of the `autograd.numpy.zeros`. In general, `zero_matrix` should return an zero matrix, whose type should be compatible with the AD backbone. 



`func_sum` (callable)

Alias of the `autograd.numpy.sum`. In general, `func_sum` should act as the `numpy.sum` and compatible with the AD backbone.   



`func_sqrt` (callable)

Alias of the `autograd.numpy.sqrt`. In general, `func_sqrt` should act as the `numpy.sqrt` and compatible with the AD backbone.   



`var_type` (str)

Record the type of the variables in the backbone. 





## cdopt.core.backbone_torch

`class backbone_autograd(*args, **kwargs)`

The AD backbone that utilizes the AD algorithms from `autograd` package to compute the differentials of objective function, constraints and constraint dissolving function. 



**Parameters**

* `device = torch.device('cup')` (torch device) -- Specify the location of the variables. 
* `dtype = torch.float64 ` (torch types) -- Specify the type of the variables. 





**Attributes:**

`auto_diff_vjp(fun, X, D)` (callable)

Returns the vector-Jacobian product for `fun` at `X`  with direction `D`.



`auto_diff_jvp(fun, X, D)` (callable)

Returns the Jacobian-vector product for `fun` at `X`  with direction `D`. 



`auto_diff_jacobian(fun, X)` (callable)

Returns the Jacobian matrix for `fun` at `X`. 



`autodiff(self, obj_fun, obj_grad = None, manifold_type = 'S')` (callable)

Returns the `tuple(obj_grad, obj_hvp)` , where `obj_grad` refers to the gradient of the `obj_fun`, and `obj_hvp` refers to the hessian-vector product of `obj_fun`. 



`array2tensor(X_array)` (callable)

Returns the pytorch tensor from the numpy array `X_array`. 



`tensor2array(X_tensor)` (callable)

Returns the numpy array from the pytorch tensor. 



`solve` (callable)

Alias of the `torch.linalg.solve`. 

 

`identity_matrix` (callable)

Alias of the `torch.eye`. 



`zero_matrix` (callable)

Alias of the `torch.zeros`. 



`func_sum` (callable)

Alias of the `toch.sum`. 



`func_sqrt` (callable)

Alias of the `torch.sqrt`. 



`var_type` (str)

Record the type of the variables in the backbone. 