# LSTM_cdopt

`CLASS cdopt.nn.LSTM_cdopt(*args, **kwargs)`

Applies a multi-layer Elman RNN where the weights for hidden states are restricted on the manifold defined by `manifold_class`. The basic introduction to convolution can be found at [`torch.nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html). 



## Parameters

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: `False`
- **dropout** – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`. Default: 0
- **bidirectional** – If `True`, becomes a bidirectional LSTM. Default: `False`
- **proj_size** – If `> 0`, will use LSTM with projections of corresponding size. Default: 0
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

`input, (h_0, c_0)`

- **input**: tensor of shape $(L, H_{in})$ for unbatched input, $(L, N, H_{in})$ when `batch_first=False` or $(N, L, H_{in})$ when `batch_first=True` containing the features of the input sequence. The input can also be a packed variable length sequence. See [`torch.nn.utils.rnn.pack_padded_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) or [`torch.nn.utils.rnn.pack_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence) for details.
- **h_0**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.
- **c_0**: tensor of shape $(D * \text{num\_layers}, H_{cell})$ for unbatched input or $(D * \text{num\_layers}, N, H_{cell})$ containing the initial cell state for each element in the input sequence. Defaults to zeros if (h_0, c_0) is not provided.

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

`output, (h_n, c_n)`

- **output**: tensor of shape $(L, D * H_{out})$ for unbatched input, $(L, N, D * H_{out})$ when `batch_first=False` or $(N, L, D * H_{out})$ when `batch_first=True` containing the output features (h_t) from the last layer of the RNN, for each $t$. If a [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) has been given as the input, the output will also be a packed sequence.
- **h_n**: tensor of shape $(D * \text{num\_layers}, H_{out})$ for unbatched input or $(D * \text{num\_layers}, N, H_{out})$ containing the final hidden state for each element in the batch.
- **c_n**: tensor of shape $(D * \text{num\_layers}, H_{cell})$ for unbatched input or $(D * \text{num\_layers}, N, H_{cell})$ containing the final cell state for each element in the sequence.





## Attributes

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 
- **weight_ih_l[k]** – the learnable input-hidden weights of the $\text{k}^{th}$ layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) for k = 0. Otherwise, the shape is (4*hidden_size, num_directions * hidden_size). If `proj_size > 0` was specified, the shape will be (4*hidden_size, num_directions * proj_size) for k > 0
- **weight_hh_l[k]** – the learnable hidden-hidden weights of the $\text{k}^{th}$ layer (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size). If `proj_size > 0` was specified, the shape will be (4*hidden_size, proj_size).
- **bias_ih_l[k]** – the learnable input-hidden bias of the $\text{k}^{th}$ layer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)
- **bias_hh_l[k]** – the learnable hidden-hidden bias of the $\text{k}^{th}$ layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size)
- **weight_hr_l[k]** – the learnable projection weights of the $\text{k}^{th}$ layer of shape (proj_size, hidden_size). Only present when `proj_size > 0` was specified.
- **weight_ih_l[k]_reverse** – Analogous to weight_ih_l[k] for the reverse direction. Only present when `bidirectional=True`.
- **weight_hh_l[k]_reverse** – Analogous to weight_hh_l[k] for the reverse direction. Only present when `bidirectional=True`.
- **bias_ih_l[k]_reverse** – Analogous to bias_ih_l[k] for the reverse direction. Only present when `bidirectional=True`.
- **bias_hh_l[k]_reverse** – Analogous to bias_hh_l[k] for the reverse direction. Only present when `bidirectional=True`.
- **weight_hr_l[k]_reverse** – Analogous to weight_hr_l[k] for the reverse direction. Only present when `bidirectional=True` and `proj_size > 0` was specified.





## Example

```python
rnn = cdopt.nn.LSTM_cdopt(10, 20, 2, manifold_class = cdopt.manifold_torch.stiefel_torch)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.size())
```

