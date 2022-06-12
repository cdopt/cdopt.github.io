# GRU_cdopt

`CLASS cdopt.nn.GRU_cdopt(*args, **kwargs)`

Applies a multi-layer gated recurrent unit (GRU) RNN where the weights for hidden states are restricted on the manifold defined by `manifold_class`. The basic introduction to convolution can be found at [`torch.nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html). 



## Parameters

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: `False`
- **dropout** – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to `dropout`. Default: 0
- **bidirectional** – If `True`, becomes a bidirectional GRU. Default: `False`
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

### Inputs

`input, h_0`

- **input**: tensor of shape $(L, H_{in})$ for unbatched input, $(L, N, H_{in})$ when `batch_first=False` or $(N, L, H_{in})$ when `batch_first=True` containing the features of the input sequence. The input can also be a packed variable length sequence. See [`torch.nn.utils.rnn.pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) or [`torch.nn.utils.rnn.pack_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence) for details.
- **h_0**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.

where:


$$
\begin{aligned} 
N ={} & \text{batch size} \\ 
L ={} & \text{sequence length} \\ D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\ 
H_{in} ={} & \mathrm{input\_size} \\ 
H_{out} ={} & \mathrm{hidden\_size} 
\end{aligned}
$$

### Outputs

`output, h_n`

- **output**: tensor of shape $(L, H_{in})$ for unbatched input, $(L, N, D * H_{out})$ when `batch_first=False` or $(N, L, D * H_{out})$ when `batch_first=True` containing the output features (h_t) from the last layer of the GRU, for each t. If a [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) has been given as the input, the output will also be a packed sequence.
- **h_n**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the final hidden state for each element in the batch.





## Attributes

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight_ih_l[k]** – the learnable input-hidden weights of the $\text{k}^{th}$ layer (W_ir|W_iz|W_in), of shape (3*hidden_size, input_size) for k = 0. Otherwise, the shape is (3*hidden_size, num_directions * hidden_size)
- **weight_hh_l[k]** – the learnable hidden-hidden weights of the $\text{k}^{th}$ layer (W_hr|W_hz|W_hn), of shape (3*hidden_size, hidden_size)
- **bias_ih_l[k]** – the learnable input-hidden bias of the $\text{k}^{th}$ layer (b_ir|b_iz|b_in), of shape (3*hidden_size)
- **bias_hh_l[k]** – the learnable hidden-hidden bias of the $\text{k}^{th}$ layer (b_hr|b_hz|b_hn), of shape (3*hidden_size)
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example

```python
rnn = cdopt.nn.GRU_cdopt(10, 20, 2, manifold_class = cdopt.manifold_torch.stiefel_torch)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(output.size())
```

