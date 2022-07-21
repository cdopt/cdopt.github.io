# RNN_cdopt

`CLASS cdopt.nn.RNN_cdopt(*args, **kwargs)`

Applies a multi-layer Elman RNN where the weights for hidden states are restricted on the manifold defined by `manifold_class`. The basic introduction to convolution can be found at [`torch.nn.RNN`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN). 



## Parameters

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
- **nonlinearity** – The non-linearity to use. Can be either `'tanh'` or `'relu'`. Default: `'tanh'`
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: `False`
- **dropout** – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to `dropout`. Default: 0
- **bidirectional** – If `True`, becomes a bidirectional RNN. Default: `False`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **penalty_param** – The penalty parameter for the quadratic penalty terms in constraint dissolving function
- **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 
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

- **output**: tensor of shape $(L, D * H_{out})$ for unbatched input, $(L, N, D * H_{out})$ when `batch_first=False` or $(N, L, D * H_{out})$ when `batch_first=True` containing the output features (h_t) from the last layer of the RNN, for each $t$. If a [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) has been given as the input, the output will also be a packed sequence.
- **h_n**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the final hidden state for each element in the batch.





## Attributes

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight_ih_l[k]** – the learnable input-hidden weights of the k-th layer, of shape $(\mathrm{hidden\_size}, \mathrm{input\_size})$ for k = 0. Otherwise, the shape is $(\mathrm{hidden\_size}, \mathrm{num\_directions }* \mathrm{hidden\_size})$
- **weight_hh_l[k]** – the learnable hidden-hidden weights of the k-th layer, of shape $(\mathrm{hidden\_size}, \mathrm{hidden\_size})$
- **bias_ih_l[k]** – the learnable input-hidden bias of the k-th layer, of shape $(\mathrm{hidden\_size},)$
- **bias_hh_l[k]** – the learnable hidden-hidden bias of the k-th layer, of shape $(\mathrm{hidden\_size},)$
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example

```python
rnn = cdopt.nn.RNN_cdopt(10, 20, 2, manifold_class = cdopt.manifold_torch.stiefel_torch)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(output.size())
```

