# import numpy as np
import abc
import functools
import numpy as np
# from autograd import grad, jacobian
# from autograd.numpy.linalg import solve, pinv

from ..manifold import basic_manifold
class basic_manifold_np(basic_manifold):
    def __init__(self,name,variable_shape, constraint_shape,  regularize_value = 0.01, safegurad_value = 0) -> None:
        self.name = name
        self.shape = (variable_shape,) 
        self.var_shape = variable_shape
        self.con_shape = constraint_shape
        self.regularize_value = regularize_value
        self.safegurad_value = safegurad_value

        
        # self.manifold_type = 'S'
        # self.backbone = 'torch'
        # self.var_type = 'torch'

        # self.Ip = torch.eye()

        
        super().__init__(self.name,self.var_shape, self.con_shape,  regularize_value = self.regularize_value, safegurad_value = self.safegurad_value, backbone = 'autograd',)

        
    

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

    # self.constraint_shape = np.shape(self.C(np.random.randn(*self.shape )))


    def v2m(self,x):
        return np.reshape(x, self.var_shape)



    def m2v(self,X):
        return X.flatten()


    def array2tensor(self, X_array):
        return X_array

    def tensor2array(self, X_tensor):
        return X_tensor


    def Init_point(self, Xinit = None):
        Xinit = np.random.randn(*self.var_shape)
        return Xinit


    
class complex_basic_manifold_np(basic_manifold):
    def __init__(self,name,variable_shape, constraint_shape,  regularize_value = 0.01, safegurad_value = 0) -> None:
        self.name = name
        self.shape = (variable_shape,) 
        self.var_shape = variable_shape
        self.con_shape = constraint_shape
        self.regularize_value = regularize_value
        self.safegurad_value = safegurad_value

        
        # self.manifold_type = 'S'
        # self.backbone = 'torch'
        # self.var_type = 'torch'

        # self.Ip = torch.eye()

        
        super().__init__(self.name,self.var_shape, self.con_shape,  regularize_value = self.regularize_value, safegurad_value = self.safegurad_value, backbone = 'autograd',)

        
    

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

    # self.constraint_shape = np.shape(self.C(np.random.randn(*self.shape )))


    
    def array2tensor(self, X_array):
        dim = len(X_array)
        X_real = X_array[:int(dim/2)]
        X_imag = X_array[int(dim/2):]

        return X_real + 1j * X_imag 


    def tensor2array(self, X_tensor, np_dtype = np.float64):
        X_real = X_tensor.real
        X_imag = X_tensor.imag

        return np.concatenate((X_real, X_imag) )


    def v2m(self,x):
        return np.reshape(x, self.var_shape)



    def m2v(self,X):
        return X.flatten()




    def Init_point(self, Xinit = None):
        Xinit_real = np.random.randn(*self.var_shape)
        Xinit_imag = np.random.randn(*self.var_shape)
        return Xinit_real + 1j * Xinit_imag



    def C_quad_penalty(self, X):
        CX = self.C(X)
        return self.sum( CX * CX.conjugate())


    def Feas_eval(self, X):
        CX = self.C(X)
        return self.sqrt(self.sum( CX * CX.conjugate() )).real


    def generate_cdf_fun(self, obj_fun, beta):
        return lambda X: (obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)).real