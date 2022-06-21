import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

from torch import Tensor
import torch
from torch.nn import Parameter, init


from ...manifold_torch import euclidean_torch




class Linear_cdopt(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, manifold_class = euclidean_torch, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_cdopt, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if self.in_features >= self.out_features:
            self.manifold = manifold_class((self.in_features, self.out_features), device= device, dtype= dtype, **kwargs)
            self.A = lambda weight: self.manifold.A(weight.T).T
            self.C = lambda weight: self.manifold.C(weight.T).T
        else:
            self.manifold = manifold_class((self.out_features, self.in_features), device= device, dtype= dtype, **kwargs)
            self.A = self.manifold.A
            self.C = self.manifold.C

        
        
        for key, param in self.manifold._parameters.items():
            self._parameters[key] = param



        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.quad_penalty = lambda : torch.sum(self.C(self.weight)**2)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.in_features >= self.out_features:
            self.weight = Parameter(self.manifold.Init_point(self.weight.T).T)
        else:
            self.weight = Parameter(self.manifold.Init_point(self.weight))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.A(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )




class Bilinear_cdopt(nn.Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`
    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input1: :math:`(*, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(*, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`
    Examples::
        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None, manifold_class = euclidean_torch, weight_var_transfer = None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bilinear_cdopt, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features


        if weight_var_transfer is None:

            def weight_var_transfer(tensor_shape):
                in_feature_total = self.in1_features * self.in2_features
                out_feature_total = self.out_features

                if in_feature_total >= out_feature_total:
                    var_shape = (in_feature_total, out_feature_total)
                    weight_to_var = lambda X_tensor: torch.reshape(X_tensor, var_shape)
                    var_to_weight = lambda X_var: torch.reshape(X_var, tensor_shape)
                else:
                    var_shape = (out_feature_total, in_feature_total)
                    var_transp_shape = (in_feature_total, out_feature_total)
                    weight_to_var = lambda X_tensor: torch.reshape(X_tensor, var_transp_shape).T 
                    var_to_weight = lambda X_var: torch.reshape( X_var.T, tensor_shape )

                return weight_to_var, var_to_weight, var_shape
        

        self.weight_to_var, self.var_to_weight, self.var_shape = weight_var_transfer( (self.out_features, self.in1_features, self.in2_features) )



        self.manifold = manifold_class(self.var_shape , device= device, dtype= dtype, **kwargs)
        self.A = lambda weight: self.var_to_weight( self.manifold.A( self.weight_to_var(weight) )  )
        self.C = lambda weight: self.var_to_weight( self.manifold.C( self.weight_to_var(weight) )  )

        for key, param in self.manifold._parameters.items():
            self._parameters[key] = param
    

        self.weight = Parameter(torch.empty((out_features, in1_features, in2_features), **factory_kwargs))

        self.quad_penalty = lambda : torch.sum(self.C(self.weight)**2)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)


        self.weight = Parameter(  self.var_to_weight( self.manifold.Init_point( self.weight_to_var(self.weight) ) )  )



        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.A(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )
