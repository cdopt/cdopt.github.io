 

# generalized_stiefel_np

`CLASS generalized_stiefel_np(var_shape, B = lambda X:X )`

This manifold class defines the generalized Stiefel manifold, i.e. 


$$
\{X \in \mathbb{R}^{n\times p}: X^T B X = I_p \},
$$

where $B \in \mathbb{R}^{n\times n}$ is a positive-definite matrix. 



##  **Parameters:**

* **var_shape** ( (int, int) ) -- The shape of the variables of the manifold. `len(var_shape)` must equal to $2$. 
* **`B = lambda X:X `**: (callable )-- The function that maps $X$ to $BX$. 







## **Attributes:**



`A(x)` (callable) 

The constraint dissolving mapping $\mathcal{A}(x)$. `A(X)`  is set as `1.5 * X - X @ (X.T @ self.B(X) /2)`. 



`C(X)` (callable)

Describe the constraints $c$. `C(X)` returns `X.T @ self.B(X) - self.Ip`.



`m2v(x)` (callable)

Flatten the variable of the manifold.



`v2m(x)` (callable) 

Recover flattened variables to its original shape as `variable_shape`. 



`Init_point(Xinit = None)` (callable)

Generate the initial point. 



`tensor2array(x)` (callable)

Transfer the variable of the manifold to the numpy Nd-array while keep its shape. Default settings are provided in the `core.backbone_autograd`. 



`array2tensor(x)` (callable)

Transfer the numpy Nd-array to the variable of the manifold while keep its shape. Default settings are provided in the `core.backbone_autograd`. 





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







