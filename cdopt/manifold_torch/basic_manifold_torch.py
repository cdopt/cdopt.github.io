# import numpy as np
import abc
import functools
import torch
import numpy as np
from collections import OrderedDict


from ..manifold import basic_manifold, complex_basic_manifold
# from autograd import grad, jacobian
# from autograd.numpy.linalg import solve, pinv

class basic_manifold_torch(basic_manifold):
    def __init__(self,name,variable_shape, constraint_shape, device = torch.device('cpu'), dtype = torch.float64,  regularize_value = 0.01, safegurad_value = 0) -> None:
        self.name = name
        self.shape = (variable_shape,) 
        self.var_shape = variable_shape
        self.con_shape = constraint_shape
        self.device = device
        self.dtype = dtype
        self.regularize_value = regularize_value
        self.safegurad_value = safegurad_value

        self._parameters = OrderedDict()

        
        # self.manifold_type = 'S'
        # self.backbone = 'torch'
        # self.var_type = 'torch'

        # self.Ip = torch.eye()

        
        super().__init__(self.name,self.var_shape, self.con_shape,  backbone = 'torch',regularize_value = self.regularize_value, safegurad_value = self.safegurad_value, device= self.device ,dtype= self.dtype)

        
    

    # In class basic_manifold, only the expression for C(X) is required.
    # Only accepts single blocks of variables. 
    # For multiple blocks of variables, please uses product_manifold class. 
    # The generating graph can be expressed as 
    #  C -> JC -> JC_transpose -> hess_feas ----
    #  |                                        |-> generate_cdf_fun, generate_cdf_grad, generate_cdf_hess
    #  --> A -> JA -> JA_transpose -> hessA ----


    def _raise_not_implemented_error(method):
        """Method decorator which raises a NotImplementedError with some meta
        information about the manifold and method if a decorated method is
        called.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            raise NotImplementedError(
                "Manifold class '{:s}' provides no implementation for "
                "'{:s}'".format(self._get_class_name(), method.__name__))
        return wrapper



    @_raise_not_implemented_error
    def C(self,X):
        """Returns the expression of the constraints
        """



    def v2m(self,x):
        return torch.reshape(x, self.var_shape)



    def m2v(self,X):
        return X.flatten()



    def Init_point(self, Xinit = None):
        Xinit = torch.randn(*self.var_shape).to(device = self.device, dtype = self.dtype)
        Xinit.requires_grad = True
        return Xinit


    


    

class complex_basic_manifold_torch(complex_basic_manifold):
    def __init__(self,name,variable_shape, constraint_shape, device = torch.device('cpu'), dtype = torch.float64,  regularize_value = 0.01, safegurad_value = 0) -> None:
        self.name = name
        self.shape = (variable_shape,) 
        self.var_shape = variable_shape
        self.con_shape = constraint_shape
        self.device = device
        self.dtype = dtype
        self.regularize_value = regularize_value
        self.safegurad_value = safegurad_value

        self._parameters = OrderedDict()

        
        # self.manifold_type = 'S'
        # self.backbone = 'torch'
        # self.var_type = 'torch'

        # self.Ip = torch.eye()

        
        super().__init__(self.name,self.var_shape, self.con_shape,  backbone = 'torch',regularize_value = self.regularize_value, safegurad_value = self.safegurad_value, device= self.device ,dtype= self.dtype)

        
    

    # In class basic_manifold, only the expression for C(X) is required.
    # Only accepts single blocks of variables. 
    # For multiple blocks of variables, please uses product_manifold class. 
    # The generating graph can be expressed as 
    #  C -> JC -> JC_transpose -> hess_feas ----
    #  |                                        |-> generate_cdf_fun, generate_cdf_grad, generate_cdf_hess
    #  --> A -> JA -> JA_transpose -> hessA ----


    def _raise_not_implemented_error(method):
        """Method decorator which raises a NotImplementedError with some meta
        information about the manifold and method if a decorated method is
        called.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            raise NotImplementedError(
                "Manifold class '{:s}' provides no implementation for "
                "'{:s}'".format(self._get_class_name(), method.__name__))
        return wrapper



    @_raise_not_implemented_error
    def C(self,X):
        """Returns the expression of the constraints
        """

    def array2tensor(self, X_array):
        dim = len(X_array)
        X_real = torch.as_tensor(X_array[:int(dim/2)])
        X_imag = torch.as_tensor(X_array[int(dim/2):])
        X =  torch.complex(X_real, X_imag).to(device=self.device, dtype=self.dtype)
        X.requires_grad = True 
        return X 


    def tensor2array(self, X_tensor, np_dtype = np.float64):
        X_real = X_tensor.real.detach().cpu().numpy()
        X_imag = X_tensor.imag.detach().cpu().numpy()

        return np.concatenate((np_dtype(X_real), np_dtype(X_imag)) )




    def v2m(self,x):
        return torch.reshape(x, self.var_shape)



    def m2v(self,X):
        return X.flatten()



    def C_quad_penalty(self, X):
        CX = self.C(X)
        return self.sum( CX * CX.conj())


    def Feas_eval(self, X):
        CX = self.C(X)
        return self.sqrt(self.sum( CX * CX.conj() )).real


    def generate_cdf_fun(self, obj_fun, beta):
        return lambda X: (obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)).real



    def Init_point(self, Xinit = None):
        Xinit = torch.randn(*self.var_shape,device = self.device, dtype = self.dtype)
        Xinit.requires_grad = True
        return Xinit
