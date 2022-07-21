# Bilinear

`CLASS cdopt.nn.Bilinear_cdopt(in1_features: int, in2_features: int, out_features: int, bias: bool = True, device=None, dtype=None, manifold_class = euclidean_torch, penalty_param = 0, weight_var_transfer = None, manifold_args = {})`

Applies a bilinear transformation to the incoming data: $y = x_1^T A x_2 + b$, where the weight matrix $A$ is restricted over the manifold specified by `manifold_class`. 



## Parameters:

- **in1_features** – size of each first input sample
- **in2_features** – size of each second input sample
- **out_features** – size of each output sample
- **bias** – If set to False, the layer will not learn an additive bias. Default: `True`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **penalty_param** – The penalty parameter for the quadratic penalty terms in constraint dissolving function
- **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 
- **weight_var_transfer** (callable) -- The function that transfer the weights (3D-tensor) to the shape of the variables of the manifold.   
  - The `weight_var_transfer` is called by  
    `weight_to_var, var_to_weight, var_shape = weight_var_transfer( (out_features, in1_features, in2_features) )`
  - The inputs of `weight_var_transfer` should be the `size` of the weights. As for the outputs, `weight_to_var` is the callable function that transfer the weights to the variables of the manifold. `var_to_weight` is the callable function that transfers the variables of the manifold to the weights. `var_shape ` is a tuple of ints that refers to the shape of the variables of the manifolds. 
  - Default: 
    - `var_shape = (in1_features * in2_features, out_feature_total)`
    - `weight_to_var = lambda X_tensor: torch.reshape(X_tensor, var_shape)`
    - `var_to_weight = lambda X_var: torch.reshape(X_var, tensor_shape)`





## Shapes:

- Input1: $(*, H_{in1})$ where $∗$ means any number of dimensions including none and $H_{in1} = \mathrm{in1\_features}$.
- Input2: $(*, H_{in2})$ where $∗$ means any number of dimensions including none and $H_{in2} = \mathrm{in2\_features}$.
- Output: $(*, H_{out})$ where all but the last dimension are the same shape as the input and $H_{out} = \mathrm{out\_features}$.



## Attributes:

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable weights of the module of shape $(\mathrm{out\_features}, \mathrm{in1\_features}, \mathrm{in2\_features})$. The values are initialized from `var_to_weight(manifold.Init_point(weight_to_var(Xinit)))`, where $\mathrm{Xinit}\sim \mathcal{U}(-\sqrt{k}, \sqrt{k})$ with $k = \frac{1}{\mathrm{in1\_features}}$. 
- **bias** – the learnable bias of the module of shape $(\mathrm{out\_features})$. If `bias` is `True`, the values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = \frac{1}{\mathrm{in1\_features}}$.
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example:

```python
my_layer = cdopt.nn.Bilinear_cdopt(20, 30, 40, manifold_class=cdopt.manifold_torch.sphere_torch)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = my_layer(input1, input2)
print(output.size())
```

