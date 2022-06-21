import numpy as np

import torch 




class backbone_torch:
    def __init__(self,*args, **kwargs) -> None:
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cpu')

        if 'dtype' in kwargs.keys():
            self.dtype = kwargs['dtype']
        else:
            self.device = torch.float64

        self.solve = torch.linalg.solve
        self.identity_matrix = lambda *args, **kwargs: torch.eye(*args, **kwargs).to(device = self.device, dtype = self.dtype)
        self.zero_matrix =  lambda *args, **kwargs: torch.zeros(*args, **kwargs).to(device = self.device, dtype = self.dtype)
        self.func_sum = torch.sum 
        self.func_sqrt = torch.sqrt
        self.var_type = 'torch'

    def auto_diff_vjp(self, fun, X, D):
        fun_vjp, val = torch.autograd.functional.vjp(fun, X, D, create_graph=True)
        # val.requires_grad = True
        return val

    def auto_diff_jvp(self, fun, X, D):
        val, jvp_result = torch.autograd.functional.jvp(fun, X, v = D, create_graph=True)
        # jvp_result.requires_grad = True
        return jvp_result


    def auto_diff_jacobian(self, fun, X):
        return torch.autograd.functional.jacobian(fun, X, create_graph=True, strict=False, vectorize=True)

    # def linear_map_adjoint(fun,D):
    #     # D.requires_grad_(requires_grad=False)
    #     def test_fun(U):
    #         # print(U)
    #         U.requires_grad = True
    #         # D.requires_grad = True

    #         return torch.autograd.grad(outputs=( torch.sum(D *fun(U)) ), inputs=U, retain_graph = True, create_graph= True)[0]
    #         # ( torch.sum(D *fun(U)) ).backward(gradient= U)
    #         # return U.grad
            
    #     return test_fun


    def linear_map_adjoint(self, fun,D):
        def test_fun(U):
            U.detach()
            U.requires_grad = True

            # return torch.autograd.functional.vjp(torch.sum(D *fun(U)), U, v = D, create_graph=True)[1]
            return torch.autograd.grad(outputs=( torch.sum(D *fun(U)) ), inputs=U, retain_graph = True, create_graph= True)[0]
            
        return test_fun


    def autodiff(self, obj_fun, obj_grad = None, manifold_type = 'S'):
        if obj_grad is not None:
            local_obj_grad = obj_grad
        else:
            if manifold_type == 'S':
                def local_grad_tmp(X):
                    return torch.autograd.grad(outputs=obj_fun(X), inputs=X, retain_graph = True, create_graph= True)[0]
            else:
                def local_grad_tmp(*args):
                    return torch.autograd.grad(outputs=obj_fun(*args), inputs=args, retain_graph = True, create_graph= True)
            local_obj_grad = local_grad_tmp




        def local_obj_hess(X, D):
            fprime, f_vjp = torch.autograd.functional.jvp(local_obj_grad, X, D, create_graph=True)
            return f_vjp

        return local_obj_grad, local_obj_hess


    def array2tensor(self, X_array):
        X = torch.as_tensor(X_array).to(device = self.device, dtype = self.dtype)
        X.requires_grad = True 
        return X

    def tensor2array(self, X_tensor, np_dtype = np.float64):
        return np_dtype(X_tensor.detach().cpu().numpy())


    def jit(self, fun):
        return fun
