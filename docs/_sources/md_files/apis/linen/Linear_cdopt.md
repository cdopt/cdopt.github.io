# linen.linear



## Dense_cdopt

`class cdopt.linen.Dense_cdopt(features, use_bias=True, dtype=None, param_dtype=<class 'jax.numpy.float32'>, precision=None, kernel_init=<function variance_scaling.<locals>.init>, bias_init=<function zeros>, parent=<flax.linen.module._Sentinel object>, name=None, manifold_class = euclidean_jax, weight_var_transfer = <function>, manifold_args )`

A linear transformation applied over the last dimension of the input.

Attributes:

* **features**: the number of output features.

* **use_bias**: whether to add a bias to the output (default: True).

* **dtype**: the dtype of the computation (default: infer from input and params).

* **param_dtype**: the dtype passed to parameter initializers (default: float32).

* **precision**: numerical precision of the computation see `jax.lax.Precision` for details.

* **kernel_init**: initializer function for the weight matrix.

* **bias_init**: initializer function for the bias.
* **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
* **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 
* **weight_var_transfer** (callable) -- The function that transfer the weights (3D-tensor) to the shape of the variables of the manifold.   



## Conv_cdopt

`class cdopt.linen.Conv_cdopt(features, kernel_size, strides=1, padding='SAME', input_dilation=1, kernel_dilation=1, feature_group_count=1, use_bias=True, mask=None, dtype=None, param_dtype=<class 'jax.numpy.float32'>, precision=None, kernel_init=<function variance_scaling.<locals>.init>, bias_init=<function zeros>, parent=<flax.linen.module._Sentinel object>, name=None, manifold_class = euclidean_jax, weight_var_transfer = <function>, manifold_args )`

Convolution Module wrapping `lax.conv_general_dilated`. This is the channels-last convention, i.e. NHWC for a 2d convolution and NDHWC for a 3D convolution. Note: this is different from the input convention used by `lax.conv_general_dilated`, which puts the spatial dimensions last.

Attributes:

* **features**: the number of output features.

* **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
* **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 
* **weight_var_transfer** (callable) -- The function that transfer the weights (3D-tensor) to the shape of the variables of the manifold.   



## ConvTranspose_cdopt

`class cdopt.linen.ConvTranspose_cdopt(features, kernel_size, strides=1, padding='SAME', input_dilation=1, kernel_dilation=1, feature_group_count=1, use_bias=True, mask=None, dtype=None, param_dtype=<class 'jax.numpy.float32'>, precision=None, kernel_init=<function variance_scaling.<locals>.init>, bias_init=<function zeros>, parent=<flax.linen.module._Sentinel object>, name=None, manifold_class = euclidean_jax, weight_var_transfer = <function>, manifold_args )`

Convolution Module wrapping `lax.conv_transpose`.



Attributes:

* **features**: the number of output features.

* **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
* **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 
* **weight_var_transfer** (callable) -- The function that transfer the weights (3D-tensor) to the shape of the variables of the manifold.   