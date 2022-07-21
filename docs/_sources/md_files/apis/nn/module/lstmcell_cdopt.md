# LSTMCell_cdopt

`CLASS cdopt.nn.LSTMCell_cdopt(input_size, hidden_size, bias=True, device=None, dtype=None, manifold_class = euclidean_torch, penalty_param = 0, manifold_args = {})`

A long short-term memory (LSTM) cell,


$$
\begin{array}{ll} i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\ f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\ g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\ o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\ c' = f * c + i * g \\ h' = o * \tanh(c') \\ \end{array}
$$
where $\sigma$ is the sigmoid function, and $*$ is the Hadamard product, and the weight for hidden states $W_{hh}$ is constrained over the manifold defined by `manifold_class`. 



## Parameters

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **penalty_param** – The penalty parameter for the quadratic penalty terms in constraint dissolving function
- **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 



## Shapes

### Inputs

`input, (h_0, c_0)`

- **input** of shape (batch, input_size) or (input_size): tensor containing input features

- **h_0** of shape (batch, hidden_size) or (hidden_size): tensor containing the initial hidden state

- **c_0** of shape (batch, hidden_size) or (hidden_size): tensor containing the initial cell state

  If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.



### Outputs

`h_1, c_1`

- **h_1** of shape (batch, hidden_size) or (hidden_size): tensor containing the next hidden state
- **c_1** of shape (batch, hidden_size) or (hidden_size): tensor containing the next cell state



## Attributes

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight_ih** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable input-hidden weights, of shape (4*hidden_size, input_size)
- **weight_hh** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable hidden-hidden weights, of shape (4*hidden_size, hidden_size)
- **bias_ih** – the learnable input-hidden bias, of shape (4*hidden_size)
- **bias_hh** – the learnable hidden-hidden bias, of shape (4*hidden_size)
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example

```python
rnn = cdopt.nn.LSTMCell_cdopt(10, 20, manifold_class = cdopt.manifold_torch.stiefel_torch) # (input_size, hidden_size)
input = torch.randn(2, 3, 10) # (time_steps, batch, input_size)
hx = torch.randn(3, 20) # (batch, hidden_size)
cx = torch.randn(3, 20)
output = []
for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)

output = torch.stack(output, dim=0)
print(output)
```

