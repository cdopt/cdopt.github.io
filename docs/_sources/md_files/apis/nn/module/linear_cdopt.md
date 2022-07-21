# Linear_cdopt

`CLASS cdopt.nn.Linear_cdopt(in_features, out_features, bias=True, device=None, dtype=None, manifold_class = euclidean_torch, penalty_param = 0, weight_var_transfer = None, manifold_args = {})`

Applies a linear transformation to the incoming data: $y = x A^T + b$, where the weight matrix $A$ is restricted over the manifold specified by `manifold_class`. 

This module supports [TensorFloat32](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere), and is developed based on [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear). 



## Parameters:

- **in_features** – size of each input sample
- **out_features** – size of each output sample
- **bias** – If set to `False`, the layer will not learn an additive bias. Default: `True`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **penalty_param** – The penalty parameter for the quadratic penalty terms in constraint dissolving function
- **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 



## Shape:

- Input: $(*, H_{in})$ where $∗$ means any number of dimensions including none and $H_{in} = \mathrm{in\_features}$.
- Output: $(*, H_{out})$ where all but the last dimension are the same shape as the input and $H_{out} = \mathrm{out\_features}$.



## Attributes:

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as $(\mathrm{out\_features}, \mathrm{in\_features})$ if $\mathrm{out\_features} \geq \mathrm{in\_features}$. Otherwise, it is set as $(\mathrm{in\_features}, \mathrm{out\_features})$.
- **weight** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable weights of the module of shape $(\mathrm{out\_features}, \mathrm{in\_features})$. The values are initialized from `manifold.Init_point(Xinit)`, where $\mathrm{Xinit}\sim \mathcal{U}(-\sqrt{k}, \sqrt{k})$ with $k = \frac{1}{\mathrm{in\_features}}$. 
- **bias** – the learnable bias of the module of shape $(\mathrm{out\_features})$. If `bias` is `True`, the values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = \frac{1}{\mathrm{in\_features}}$.
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example:

```python
my_layer = cdopt.nn.Linear_cdopt(20, 30, manifold_class = cdopt.manifold_torch.symp_stiefel_torch)
input = torch.randn(128, 20)
output = my_layer(input)
print(output.size())
# expected to print torch.Size([128, 30])
```

