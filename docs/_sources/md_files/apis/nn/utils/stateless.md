# cdopt.nn.utils.stateless



## stateless.functional_call

`cdopt.nn.utils.stateless.functional_call(module, parameters_and_buffers, args, kwargs=None)`

Performs a functional call on the module by replacing the module parameters and buffers with the provided ones. This is the copy of the [`functional_call` from PyTorch >=1.12](https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html#torch.nn.utils.stateless.functional_call). 



Parameters:

- **module** ([*torch.nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – the module to call
- **parameters_and_buffers** (*dict of str and Tensor*) – the parameters that will be used in the module call.
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – arguments to be passed to the module call
- **kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – keyword arguments to be passed to the module call



Returns:

* the result of calling `module`.





## stateless.get_quad_penalty_call

`cdopt.nn.utils.stateless.get_quad_penalty_call(module, parameters_and_buffers)`

Call `cdopt.nn.get_quad_penalty()` to the module by replacing the module parameters and buffers with the provided ones. 



Parameters:

- **module** ([*torch.nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – the module to call
- **parameters_and_buffers** (*dict of str and Tensor*) – the parameters that will be used in the module call.



Returns:

* the result of calling `cdopt.nn.get_quad_penalty()` to the `module`.







## stateless.functional_quad_penalty_call

`cdopt.nn.utils.stateless.functional_quad_penalty_call(module, parameters_and_buffers, args, kwargs=None)`

Calling the function and `cdopt.nn.get_quad_penalty()` simultaneously on the module by replacing the module parameters and buffers with the provided ones.



Parameters:

- **module** ([*torch.nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – the module to call
- **parameters_and_buffers** (*dict of str and Tensor*) – the parameters that will be used in the module call.
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – arguments to be passed to the module call
- **kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – keyword arguments to be passed to the module call



Returns:

* the result of calling `module` and `cdopt.nn.get_quad_penalty()`. 