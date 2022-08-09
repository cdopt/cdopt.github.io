# cdopt.nn.utils.modified_apply



## modified_apply

`cdopt.nn.utils.modified_apply.modified_apply(module, fn)`

Applies `fn` recursively to every submodule (as returned by `.children()`) as well as self. Typical use includes initializing the parameters of a model (see also [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html#nn-init-doc)) or move the parameters and buffers to other devices (see also [torch.nn.Module.to](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=appl#torch.nn.Module.to)). 



```{note}
The `modified_apply()` is developed based the `apply()` method from [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). In `torch.nn.Module`, when its parameters and buffers are moved to an different device by the `to()` method based on its `apply()` method, the parameters are moved to the target device in-place, but the buffers are **copied** to the target device. That leads to great inconvenience to the CDOpt package since some of the variables of the manifold classes, such as the identity matrix in the Stiefel manifold, are registered as the buffers of the neural layers. 

Therefore, CDOpt replaces the `apply()` function of all its neural layers by the `modified_apply()` function. In `modified_apply()` function, the parameters and buffers have the same behavior, and the variables of the manifold classes that are registered as the buffers can be properly moved to other devices by the `to()` method. 
```





Parameters:

- **fn** ([`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module) -> None) – function to be applied to each submodule

Returns:

* The `module`.



