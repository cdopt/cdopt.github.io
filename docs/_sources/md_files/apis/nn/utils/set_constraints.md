# cdopt.nn.utils.set_constraints



## set_constraints.set_constraint_dissolving

`set_constraint_dissolving(module, attr_name, manifold_class = euclidean_torch, weight_var_transfer = None, manifold_args = {}, penalty_param = 0)`

Set the manifold constraints to the attribute `attr_name` to the Module `module`.  



Parameters:

- **module** ([*torch.nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – the module to be set the manifold constraints
- **attr_name** (str) – the name of the attribute to be set the manifold constraints
- **manifold_class** (*cdopt.manifold*) – the manifold classes from `cdopt.manifold_torch`.
- **weight_var_transfer** (Callable) – the function that determines how the variables from `manifold_class` are transformed to the attribute `attr_name` of the `module`. 
- **manifold_args** (dict) – arguments to be passed when  instantiating the manifold class from `manifold_class`
- **penalty_param** (float) – the value of the penalty parameter. 



Returns:

* The module with manifold constraints on its attribute `attr_name`. 





```{note}
The `set_constraint_dissolving()` function is developed based on the `torch.nn.utils.parametrize` functions. Therefore, the module returned belongs to the `ParametrizationList` module in PyTorch, which is different from the predefined neural layers from `cdopt.nn`. 

The list of parametrizations on the tensor weight will be accessible under `module.parametrizations`. And the variables of the manifold will be accessible under `module.parametrizations[attr_name].original`. Moreover, the constraint dissolving mapping $\mathcal{A}$ can be accessed at `module.parametrizations[attr_name].A`, the constriants can be accessed at `module.parametrizations[attr_name].C`, the quadratic penalty term can be accessed at `module.parametrizations[attr_name].quad_penalty()`, and the penalty parameter can be accessed at `module.parametrizations[attr_name].penalty_param`. 
```



