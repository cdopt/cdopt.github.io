# RNNCell_cdopt

`CLASS cdopt.nn.RNNCell_cdopt(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None, manifold_class = euclidean_torch, **kwargs)`

An Elman RNN cell with tanh or ReLU non-linearity,


$$
h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h + b_{hh}),
$$
where the weight for hidden states $W_{hh}$ is constrained over the manifold defined by `manifold_class`. 

If `nonlinearity` is `relu`, then ReLU is used in place of tanh.



## Parameters

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **nonlinearity** – The non-linearity to use. Can be either `'tanh'` or `'relu'`. Default: `'tanh'`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **kwargs** - The additional key-word arguments that helps to define the manifold constraints. 
- **weight_var_transfer** (callable) -- The function that transfer the weights (3D-tensor) to the shape of the variables of the manifold.   
  - The `weight_var_transfer` is called by  
    `weight_to_var, var_to_weight, var_shape = weight_var_transfer( tensor_shape )`
  - The inputs of `weight_var_transfer` should be the `size` of the weights. As for the outputs, `weight_to_var` is the callable function that transfer the weights to the variables of the manifold. `var_to_weight` is the callable function that transfers the variables of the manifold to the weights. `var_shape ` is a tuple of ints that refers to the shape of the variables of the manifolds. 
  - Default: 
    - `weight_to_var = lambda X_tensor: X_tensor`
    - `var_to_weight = lambda X_var: X_var `
    - `var_shape = tensor_shape `



## Shapes

- input: $(N, H_{in})$ or $(H_{in})$ tensor containing input features where $H_{in}$ = input_size.
- hidden: $(N, H_{out})$ or $(H_{out})$ tensor containing the initial hidden state where $H_{out} = \mathrm{hidden\_size}$. Defaults to zero if not provided.
- output: $(N, H_{out})$ or $(H_{out})$ tensor containing the next hidden state.





## Attributes

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight_ih** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable input-hidden weights, of shape (hidden_size, input_size)
- **weight_hh** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
- **bias_ih** – the learnable input-hidden bias, of shape (hidden_size)
- **bias_hh** – the learnable hidden-hidden bias, of shape (hidden_size)
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example

```python
rnn = cdopt.nn.RNNCell_cdopt(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
    
print(output)
```

