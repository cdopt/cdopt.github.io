# cdopt.nn.module.utils

`cdopt.nn.module.utils` contains some useful functions for `cdopt.nn` module, and all its contained functions can be accessed  in `cdopt.nn`.





## get_quad_penalty

`sum_quad_pen = cdopt.nn.get_quad_penalty(module)`

The function that returns the sum of all the quadratic penalty terms in the `module`. 

Input:

* **module**: (`torch.nn.Module`) The `torch.nn.Module` class that composed of the layers from `torch.nn` and `cdopt.nn`. 



Output:

* The sum of all the quadratic penalty terms in the children of the `module`, where the penalty parameter equals to `penalty_param` attributes of each children of the module. 



## wvt_flatten2d

`weight_to_var, var_to_weight, var_shape = cdopt.nn.wvt_flatten2d(tensor_shape)`

The function that determines the tensor of the shape $(n_1, n_2,...n_m)$ be converted to an 2D tensor that of the shape $(n_1 * n_3 * n_4 *...* n_m, n_2)$.

 

Inputs:

* **tensor_shape**: (Tuple of ints) The shape of the weight matrix of the layer. 



Outputs:

* **weight_to_var**: (Callable)  The function that transform the weight matrix of the layer to the 2D tensor. 
* **var_to_weight**: (Callable)  The function that transform the 2D tensor to the weight matrix of the layer.
* **var_shape **: (Tuple of ints) The shape of the 2D tensor. 





## wvt_flatten2d_transp

`weight_to_var, var_to_weight, var_shape = cdopt.nn.wvt_flatten2d_transp(tensor_shape)`

The function that determines the tensor of the shape $(n_1, n_2,...n_m)$ be converted to an 2D tensor that of the shape $(n_2 * n_3 * n_4 *...* n_m, n_1)$.

 

Inputs:

* **tensor_shape**: (Tuple of ints) The shape of the weight matrix of the layer. 



Outputs:

* **weight_to_var**: (Callable)  The function that transform the weight matrix of the layer to the 2D tensor. 
* **var_to_weight**: (Callable)  The function that transform the 2D tensor to the weight matrix of the layer.
* **var_shape **: (Tuple of ints) The shape of the 2D tensor. 





## wvt_identical

`weight_to_var, var_to_weight, var_shape = cdopt.nn.wvt_identical(tensor_shape)`

The function that determines the tensor of the shape $(n_1, n_2,...n_m)$ be converted to an tensor that of the same shape.

 

Inputs:

* **tensor_shape**: (Tuple of ints) The shape of the weight matrix of the layer. 



Outputs:

* **weight_to_var**: (Callable)  The function that transform the weight matrix of the layer to the 2D tensor. 
* **var_to_weight**: (Callable)  The function that transform the 2D tensor to the weight matrix of the layer.
* **var_shape **: (Tuple of ints) The shape of the tensor. 

