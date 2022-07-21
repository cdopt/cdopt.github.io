# GRUCell_cdopt

`CLASS cdopt.nn.GRUCell_cdopt(input_size, hidden_size, bias=True, device=None, dtype=None, manifold_class = euclidean_torch, penalty_param = 0, manifold_args = {})`

A gated recurrent unit (GRU) cell


$$
\begin{array}{ll} r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\ z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\ n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\ h' = (1 - z) * n + z * h \end{array}
$$



where $\sigma$ is the sigmoid function, and $*$ refers to the Hadamard product, and the weight for hidden states $W_{hh}$ is constrained over the manifold defined by `manifold_class`. 



## Parameters

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **manifold_class** – The manifold class for the weight matrix. Default: `cdopt.manifold_torch.euclidean_torch`
- **penalty_param** – The penalty parameter for the quadratic penalty terms in constraint dissolving function
- **manifold_args** - The additional key-word arguments that helps to define the manifold constraints. 



## Shapes

### Inputs

`input, hidden`

- **input** : tensor containing input features
- **hidden** : tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.



### Outputs

`h'`

- **h’** : tensor containing the next hidden state for each element in the batch



### Shapes

- input: $(N, H_{in})$ or $(H_{in})$ tensor containing input features where $H_{in}$ = input_size.
- hidden: $(N, H_{out})$ or $(H_{out})$ tensor containing the initial hidden state where $H_{out}$ = hidden_size. Defaults to zero if not provided.
- output: $(N, H_{out})$ or $(H_{out})$ tensor containing the next hidden state.





## Attributes

- **manifold** (cdopt manifold class) -- the manifold that defines the constraints.  The shape of the variables in `manifold` is set as `var_shape`. 
- **weight_ih** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable input-hidden weights, of shape (3*hidden_size, input_size)
- **weight_hh** ([*torch.Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the learnable hidden-hidden weights, of shape (3*hidden_size, hidden_size)
- **bias_ih** – the learnable input-hidden bias, of shape (3*hidden_size)
- **bias_hh** – the learnable hidden-hidden bias, of shape (3*hidden_size)
- **quad_penalty** (callable) -- the function that returns the quadratic penalty terms of the weights. Its return value equals to $||\mathrm{manifold.C}(\mathrm{weight})||^2$. 





## Example

```python
rnn = cdopt.nn.GRUCell_cdopt(10, 20, manifold_class = cdopt.manifold_torch.stiefel_torch)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)

print(output)
```

