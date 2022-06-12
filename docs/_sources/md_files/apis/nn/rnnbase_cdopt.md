# RNNBase_cdopt

`CLASS cdopt.nn.RNNBase_cdopt(mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None, manifold_class = euclidean_torch, weight_var_transfer = None, **kwargs)`

The base class of all RNN classes in `cdopt.nn`, which is developed based on [`torch.nn.RNNBase`](https://pytorch.org/docs/stable/generated/torch.nn.RNNBase.html#torch.nn.RNNBase).