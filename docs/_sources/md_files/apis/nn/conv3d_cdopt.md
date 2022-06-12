# Conv3d_cdopt

`CLASS cdopt.nn.Conv3d_cdopt(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, manifold_class = euclidean_torch,  weight_var_transfer = None, **kwargs)`

Applies a 3D convolution over an input signal composed of several input planes. The basic introduction to convolution can be found at [`torch.nn.Conv3d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d). 

This module supports [TensorFloat32](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).

- `stride` controls the stride for the cross-correlation, a single number or a one-element tuple.

- `padding` controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or a tuple of ints giving the amount of implicit padding applied on both sides.

- `dilation` controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation` does.

- `groups` controls the connections between inputs and outputs. `in_channels` and `out_channels` must both be divisible by `groups`. For example,

  > - At `groups=1`, all inputs are convolved to all outputs.
  > - At `groups=2`, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
  > - At `groups= in_channels`, each input channel is convolved with its own set of filters (of size $\frac{\text{out\_channels}}{\text{in\_channels}}$).



The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:

> - a single `int` – in which case the same value is used for the height and width dimension
> - a `tuple` of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension



## Parameters:

- **in_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels in the input image
- **out_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels produced by the convolution
- **kernel_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – Size of the convolving kernel
- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Stride of the convolution. Default: 1
- **padding** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – Padding added to all four sides of the input. Default: 0
- **padding_mode** (*string**,* *optional*) – `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'zeros'`
- **dilation** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Spacing between kernel elements. Default: 1
- **groups** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Number of blocked connections from input channels to output channels. Default: 1
- **bias** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True`, adds a learnable bias to the output. Default: `True`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **kwargs** - The additional key-word arguments that helps to define the manifold constraints. 
- **weight_var_transfer** (callable) -- The function that transfer the weights (3D-tensor) to the shape of the variables of the manifold.   
  - The `weight_var_transfer` is called by  
    `weight_to_var, var_to_weight, var_shape = weight_var_transfer( tensor_shape )`
  - The inputs of `weight_var_transfer` should be the `size` of the weights. As for the outputs, `weight_to_var` is the callable function that transfer the weights to the variables of the manifold. `var_to_weight` is the callable function that transfers the variables of the manifold to the weights. `var_shape ` is a tuple of ints that refers to the shape of the variables of the manifolds. 
  - Default: 
    - `var_shape = (torch.prod(torch.tensor(tensor_shape[:-1])), torch.tensor(tensor_shape[-1]))`
    - `weight_to_var = lambda X_tensor: torch.reshape(X_tensor, var_shape)`
    - `var_to_weight = lambda X_var: torch.reshape(X_var, tensor_shape)`



## Shapes:

- Input: $(N, C_{in}, D_{in}, H_{in}, W_{in})$ or $(C_{in}, D_{in},H_{in}, W_{in})$.
- Output: $(N, C_{out}, D_{out}, H_{out}, W_{out})$ or $(C_{out}, D_{out}, H_{out}, W_{out})$, 

- where

  $D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor$,

  $H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor$,

  $W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor$.





## Attributes:

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable weights of the module of shape $(\mathrm{out\_channels}, \frac{\mathrm{in\_channels}}{\mathrm{groups}},\mathrm{kernel\_size[0]}, \mathrm{kernel\_size[1]})$. .  The values are initialized from `var_to_weight(manifold.Init_point(weight_to_var(Xinit)))`, where $\mathrm{Xinit}\sim \mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = \frac{\mathrm{groups}}{C_\mathrm{in} * \prod_{i=0}^{2}\mathrm{kernel\_size}[i]}$.
- **bias** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable bias of the module of shape (out_channels). If `bias` is `True`, then the values of these weights are sampled from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = \frac{\mathrm{groups}}{C_\mathrm{in} * \prod_{i=0}^{2}\mathrm{kernel\_size}[i]}$.
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 



## Example:

```python
# With square kernels and equal stride
m_layer = cdopt.nn.Conv3d_cdopt(16, 33, 3, stride=2, manifold_class=cdopt.manifold_torch.sphere_torch)
# non-square kernels and unequal stride and with padding
m_layer = cdopt.nn.Conv3d_cdopt(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0), manifold_class=cdopt.manifold_torch.sphere_torch)
input = torch.randn(20, 16, 10, 50, 100)
output = m_layer(input)
print(output.size())
```

