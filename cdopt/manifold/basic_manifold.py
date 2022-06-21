# import numpy as np
import abc
import functools
import numpy as np
# import torch
# from core.autodiff_torch import   linear_map_adjoint, auto_diff_vjp, auto_diff_jvp
# from autograd import grad, jacobian
# from autograd.numpy.linalg import solve, pinv

class basic_manifold(metaclass=abc.ABCMeta):
    def __init__(self,name,variable_shape, constraint_shape, backbone = 'torch',  regularize_value = 0.01, safeguard_value = 0, **kwargs) -> None:
        self.name = name
        self.shape = (variable_shape,) 
        self.var_shape = variable_shape
        self.con_shape = constraint_shape
        self.regularize_value = regularize_value
        self.safeguard_value = safeguard_value

        self.var_num = np.prod(self.var_shape)
        self.con_num = np.prod(self.con_shape)

        
        self.manifold_type = 'S'
        

        if backbone == 'torch':
            from ..core.backbone_torch import backbone_torch
            self.backbone = backbone_torch(**kwargs)


        elif backbone == 'autograd':
            from ..core.backbone_autograd import backbone_autograd
            self.backbone = backbone_autograd(**kwargs)

        else:
            # Custom AD packages
            self.backbone = backbone(*kwargs)


        self.var_type = self.backbone.var_type

        self.vjp = self.backbone.auto_diff_vjp
        self.jvp = self.backbone.auto_diff_jvp
        self.jacobian = self.backbone.auto_diff_jacobian


        self.solve = self.backbone.solve
        self.identity_matrix =  self.backbone.identity_matrix
        self.zero_matrix =   self.backbone.zero_matrix
        self.sum = self.backbone.func_sum 
        self.sqrt = self.backbone.func_sqrt 

        # self.array2tensor = self.backbone.array2tensor
        # self.tensor2array = self.backbone.tensor2array
            # autograd.vjp has low compatibality to the autograd.linalg.solve or autograd.linalg.pinv
            # hence we choose the linear_map_adjoint to generate jvp from vjp.
            # On the other hand, mathematically, jvp and vjp are equivalent. Therefore, it would be better 
            # directly generate jvp from vjp, rather than calling jvp from the autodiff backbone. 
            # In the current version, we still keep the self.jvp, but it may be modified or removed in the following versions. 

        
    

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

    @_raise_not_implemented_error
    def v2m(self,x):
        """
        transfer vector to variables of the backbone
        """


    @_raise_not_implemented_error
    def m2v(self,X):
        """
        Flatten variables of the backbone
        """

    # @_raise_not_implemented_error
    def array2tensor(self, X_array):
        """
        Transfer the numpy array to variables
        """
        return self.backbone.array2tensor(X_array)

    # @_raise_not_implemented_error
    def tensor2array(self, X_tensor):
        """
        Transfer the variables to array
        """
        return self.backbone.tensor2array(X_tensor)


    @ _raise_not_implemented_error
    def Init_point(self, Xinit = None):
        """
        Generate initial point
        """


    
    # def A(self, X):
    #     local_JC =  lambda y: self.C(self.v2m(y)).flatten()
    #     JC_mat = self.jacobian(local_JC,self.m2v(X))
    #     # print(JC_mat.size())
    #     C_vec = (self.C(X)).flatten()
    #     # print((self.solve(  0.01 * torch.eye(C_vec.size()[0]).to(device = self.device, dtype = self.dtype) + (JC_mat @ JC_mat.T), C_vec   ) ).size())
    #     AX_vec = self.m2v(X) - self.pinv(JC_mat) @ C_vec
    #     return self.v2m(AX_vec)

    def A(self, X):
        local_JC =  lambda y: self.C(self.v2m(y)).flatten()
        JC_mat = self.jacobian(local_JC,self.m2v(X))
        C_vec = (self.C(X)).flatten()
        # AX_vec = self.m2v(X) - (JC_mat.T).detach() @ self.solve(  ((self.regularize_value* self.C_quad_penalty(X) + self.safeguard_value) * torch.eye(C_vec.size()[0]).to(device = self.device, dtype = self.dtype) + (JC_mat @ JC_mat.T)).detach(), C_vec   ) 

        # AX_vec = self.m2v(X) - self.pinv(JC_mat) @ C_vec
        AX_vec = self.m2v(X) - JC_mat.T @ self.solve(  (self.regularize_value* self.C_quad_penalty(X) + self.safeguard_value) * self.identity_matrix(self.con_num) + (JC_mat @ JC_mat.T), C_vec   ) 
        return self.v2m(AX_vec)


    # def A(self, X):
    #     local_JC =  lambda y: self.C(self.v2m(y)).flatten()
    #     JC_mat = self.jacobian(local_JC)(self.m2v(X))
    #     C_vec = (self.C(X)).flatten()
        
    #     AX_vec = self.m2v(X) - JC_mat.T @ self.solve( self.offset * np.eye(np.size(C_vec)) + (JC_mat @ JC_mat.T), C_vec ) 
    #     return self.v2m(AX_vec)

    def JA(self, X, D):
        return self.vjp(self.A, X, D)


    
    def JC(self, X, Lambda):
        return self.vjp(self.C, X, Lambda)

    def JA_transpose(self, X, D):
        return self.jvp(self.A, X, D)
        # return self.linear_map_adjoint( lambda T: self.JA(X,T), D )(X)

    def JC_transpose(self, X, D):
        return self.jvp(self.C, X, D)
        # return self.linear_map_adjoint( lambda T: self.JC(X,T), D )(self.zero_matrix(self.con_shape))


    def C_quad_penalty(self, X):
        return self.sum(self.C(X) **2)



    def hessA(self, X, gradf, D):
        return self.vjp(lambda X: self.JA(X, gradf), X, D)

    def hess_feas(self, X, D):
        return self.vjp(lambda X: self.JC(X, self.C(X)), X, D)


    def Feas_eval(self, X):
        return self.sqrt(self.sum( self.C(X) **2 ))





    def Post_process(self,X):
        
        return X



    def generate_cdf_fun(self, obj_fun, beta):
        return lambda X: obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)




    def to_cdf_fun(self, beta = 0):
        def decorator_cdf_obj(obj_fun):
            return self.generate_cdf_fun(obj_fun, beta )
            # return lambda X: obj_fun(self.A(X)) + (beta) * self.C_quad_penalty(X)
            
        return decorator_cdf_obj



    def generate_cdf_grad(self, obj_grad, beta):
        return lambda X:  self.JA(X,obj_grad(self.A(X))) + beta * self.JC(X, self.C(X)) 



    def to_cdf_grad(self, beta = 0):
        def decorator_cdf_grad(obj_grad):
            return self.generate_cdf_grad(obj_grad, beta )
            
        return decorator_cdf_grad




    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        return lambda X, D: self.JA( X, obj_hess( self.A(X), self.JA_transpose(X,D) ) ) + self.hessA(X, obj_grad(X), D) + beta * self.hess_feas(X, D)





    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        return self.generate_cdf_hess( obj_grad, obj_hess, beta)






class complex_basic_manifold(metaclass=abc.ABCMeta):
    def __init__(self,name,variable_shape, constraint_shape, backbone = 'torch',  regularize_value = 0.01, safeguard_value = 0, **kwargs) -> None:
        self.name = name
        self.shape = (variable_shape,) 
        self.var_shape = variable_shape
        self.con_shape = constraint_shape
        self.regularize_value = regularize_value
        self.safeguard_value = safeguard_value

        self.var_num = np.prod(self.var_shape)
        self.con_num = np.prod(self.con_shape)

        
        self.manifold_type = 'S'
        

        if backbone == 'torch':
            from ..core.backbone_torch import backbone_torch
            self.backbone = backbone_torch(**kwargs)


        elif backbone == 'autograd':
            from ..core.backbone_autograd import backbone_autograd
            self.backbone = backbone_autograd(**kwargs)

        else:
            # Custom AD packages
            self.backbone = backbone(*kwargs)


        self.var_type = self.backbone.var_type

        self.vjp = self.backbone.auto_diff_vjp
        self.jvp = self.backbone.auto_diff_jvp
        self.jacobian = self.backbone.auto_diff_jacobian


        self.solve = self.backbone.solve
        self.identity_matrix =  self.backbone.identity_matrix
        self.zero_matrix =   self.backbone.zero_matrix
        self.sum = self.backbone.func_sum 
        self.sqrt = self.backbone.func_sqrt 

        # self.array2tensor = self.backbone.array2tensor
        # self.tensor2array = self.backbone.tensor2array
            # autograd.vjp has low compatibality to the autograd.linalg.solve or autograd.linalg.pinv
            # hence we choose the linear_map_adjoint to generate jvp from vjp.
            # On the other hand, mathematically, jvp and vjp are equivalent. Therefore, it would be better 
            # directly generate jvp from vjp, rather than calling jvp from the autodiff backbone. 
            # In the current version, we still keep the self.jvp, but it may be modified or removed in the following versions. 

        
    

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

    @_raise_not_implemented_error
    def v2m(self,x):
        """
        transfer vector to variables of the backbone
        """


    @_raise_not_implemented_error
    def m2v(self,X):
        """
        Flatten variables of the backbone
        """

    # @_raise_not_implemented_error
    def array2tensor(self, X_array):
        """
        Transfer the numpy array to variables
        """
        return self.backbone.array2tensor(X_array)

    # @_raise_not_implemented_error
    def tensor2array(self, X_tensor):
        """
        Transfer the variables to array
        """
        return self.backbone.tensor2array(X_tensor)


    @ _raise_not_implemented_error
    def Init_point(self, Xinit = None):
        """
        Generate initial point
        """


    
    # def A(self, X):
    #     local_JC =  lambda y: self.C(self.v2m(y)).flatten()
    #     JC_mat = self.jacobian(local_JC,self.m2v(X))
    #     # print(JC_mat.size())
    #     C_vec = (self.C(X)).flatten()
    #     # print((self.solve(  0.01 * torch.eye(C_vec.size()[0]).to(device = self.device, dtype = self.dtype) + (JC_mat @ JC_mat.T), C_vec   ) ).size())
    #     AX_vec = self.m2v(X) - self.pinv(JC_mat) @ C_vec
    #     return self.v2m(AX_vec)

    def A(self, X):
        local_JC =  lambda y: self.C(self.v2m(y)).flatten()
        JC_mat = self.jacobian(local_JC,self.m2v(X))
        C_vec = (self.C(X)).flatten()
        # AX_vec = self.m2v(X) - (JC_mat.T).detach() @ self.solve(  ((self.regularize_value* self.C_quad_penalty(X) + self.safeguard_value) * torch.eye(C_vec.size()[0]).to(device = self.device, dtype = self.dtype) + (JC_mat @ JC_mat.T)).detach(), C_vec   ) 

        # AX_vec = self.m2v(X) - self.pinv(JC_mat) @ C_vec
        AX_vec = self.m2v(X) - JC_mat.T.conj() @ self.solve(  (self.regularize_value* self.C_quad_penalty(X) + self.safeguard_value) * self.identity_matrix(self.con_num) + (JC_mat @ JC_mat.T.conj()), C_vec   ) 
        return self.v2m(AX_vec)


    # def A(self, X):
    #     local_JC =  lambda y: self.C(self.v2m(y)).flatten()
    #     JC_mat = self.jacobian(local_JC)(self.m2v(X))
    #     C_vec = (self.C(X)).flatten()
        
    #     AX_vec = self.m2v(X) - JC_mat.T @ self.solve( self.offset * np.eye(np.size(C_vec)) + (JC_mat @ JC_mat.T), C_vec ) 
    #     return self.v2m(AX_vec)

    def JA(self, X, D):
        return self.vjp(self.A, X, D)


    
    def JC(self, X, Lambda):
        return self.vjp(self.C, X, Lambda)

    def JA_transpose(self, X, D):
        return self.jvp(self.A, X, D)
        # return self.linear_map_adjoint( lambda T: self.JA(X,T), D )(X)

    def JC_transpose(self, X, D):
        return self.jvp(self.C, X, D)
        # return self.linear_map_adjoint( lambda T: self.JC(X,T), D )(self.zero_matrix(self.con_shape))


    def C_quad_penalty(self, X):
        return self.sum(self.C(X) **2)



    def hessA(self, X, gradf, D):
        return self.vjp(lambda X: self.JA(X, gradf), X, D)

    def hess_feas(self, X, D):
        return self.vjp(lambda X: self.JC(X, self.C(X)), X, D)


    def Feas_eval(self, X):
        return self.sqrt(self.sum( self.C(X) **2 ))





    def Post_process(self,X):
        
        return X



    def generate_cdf_fun(self, obj_fun, beta):
        return lambda X: obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)




    def to_cdf_fun(self, beta = 0):
        def decorator_cdf_obj(obj_fun):
            return self.generate_cdf_fun(obj_fun, beta )
            # return lambda X: obj_fun(self.A(X)) + (beta) * self.C_quad_penalty(X)
            
        return decorator_cdf_obj



    def generate_cdf_grad(self, obj_grad, beta):
        return lambda X:  self.JA(X,obj_grad(self.A(X))) + beta * self.JC(X, self.C(X)) 



    def to_cdf_grad(self, beta = 0):
        def decorator_cdf_grad(obj_grad):
            return self.generate_cdf_grad(obj_grad, beta )
            
        return decorator_cdf_grad




    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        return lambda X, D: self.JA( X, obj_hess( self.A(X), self.JA_transpose(X,D) ) ) + self.hessA(X, obj_grad(X), D) + beta * self.hess_feas(X, D)





    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        return self.generate_cdf_hess( obj_grad, obj_hess, beta)


    

    