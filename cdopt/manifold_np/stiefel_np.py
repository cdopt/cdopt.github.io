import numpy as np


from .basic_manifold_np import basic_manifold_np


class stiefel_np(basic_manifold_np):
    def __init__(self, var_shape) -> None:
        n = var_shape[0]
        p = var_shape[1]
        self._n = n
        self._p = p
        

        
        self.Ip = np.eye(self._p)


        super().__init__('Stiefel',(n,p), (p,p))



    def Phi(self, M):
        return (M + M.T )/2



    def A(self, X):
        XX = X.T @ X
        return 1.5 * X - X @ (XX /2)


    def JA(self, X, G):
        return G @ ( self.Ip - 0.5 * self.C(X) )  - X @ self.Phi(X.T @ G)

    def JA_transpose(self,X,G):
        # JA is self-adjoint
        return self.JA(X,G)

    def hessA(self, X, gradf, D):
        return - D @ self.Phi( X.T @ gradf  ) - X @ self.Phi(D.T @ gradf) - gradf @ self.Phi( D.T @ X )


    def JC(self, X, Lambda):
        return 2*X @ self.Phi(Lambda)

    
    def C(self, X):
        return X.T @ X - self.Ip 

    def C_quad_penalty(self, X):
        return np.sum(self.C(X) ** 2)


    def hess_feas(self, X, D):
        return 4*X @ self.Phi( X.T @ D ) + 2*D @ self.C(X)

    



    def Feas_eval(self, X):
        return np.linalg.norm( self.C(X) , 'fro')

    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit = np.random.randn(self._n, self._p)
        

        if np.linalg.norm(Xinit.T @ Xinit - self.Ip, 'fro') > 1e-6:
            Xinit, Rinit = np.linalg.qr(Xinit)
        return Xinit

    def Post_process(self,X):
        UX, SX, VX = np.linalg.svd(X, full_matrices = False)
        return UX @ VX



    def generate_cdf_fun(self, obj_fun, beta):
        def local_obj_fun(X):
            CX = self.C(X)
            AX = X - 0.5 * X@ CX
            return obj_fun(AX) + (beta/2) * np.sum(CX **2)

        



        return local_obj_fun  


    # def to_cdf_fun(self, beta = 0):
    #     def decorator_cdf_obj(obj_fun):
    #         return self.generate_cdf_fun(obj_fun, beta )
    #         # return lambda X: obj_fun(self.A(X)) + (beta) * self.C_quad_penalty(X)
            
    #     return decorator_cdf_obj



    def generate_cdf_grad(self, obj_grad, beta):
        def local_grad(X):
            CX = self.C(X)
            AX = X - 0.5 * X@CX
            gradf = obj_grad(AX)
            XG = self.Phi(X.T @ gradf)

            # local_JA_gradf = gradf @ (np.eye(self._p) - 0.5 * CX) - X @ XG 
            
            # local_JC_CX = 2 * X @(CX)

            return gradf @ (self.Ip - 0.5 * CX) +  X @  ( 2* beta * CX - XG)

        return local_grad  

    # def to_cdf_grad(self, beta = 0):
    #     def decorator_cdf_grad(obj_grad):
    #         return self.generate_cdf_grad(obj_grad, beta )
    #         # return lambda X: obj_fun(self.A(X)) + (beta) * self.C_quad_penalty(X)
            
    #     return decorator_cdf_grad



    #TODO 
    # 2022/01/05
    # The simplest expression for self.generate_cdf_hess()  is
    # self.JA( X, hessf( self.JA_transpose(X,D) ) ) + self.hessA(X, gradf, D) + beta * self.hess_feas(X, D)
    # However, it repeatively computes A(X), X.T @ D and C(X), leading to inferior efficiency in practice.
    # A better implementation is presented below.
    # However, the function self.generate_cdf_hess()  is still not well-optimized. 
    # In the nest version I will rewrite this function for a better performance.

    
    # 2022/01/07
    # Rewite  self.generate_cdf_grad() and self.generate_cdf_hess()
    


    # def generate_cdf_hess(self, obj_grad, obj_hess, beta):
    #     def local_hess(X, D):
    #         CX = self.C(X)
    #         AX = X - 0.5 * X@CX
    #         gradf = obj_grad(AX)
    #         XG = self.Phi(X.T @ gradf)
    #         XD = self.Phi(X.T @ D)

    #         local_JAT_D = D @ (np.eye(self._p) - 0.5 * CX) - X @ XD 
    #         local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
    #         local_JA_objhess_JAT_D = local_objhess_JAT_D @ (np.eye(self._p) - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D )


    #         local_hessA_objgrad_D = - D @ XG - X @ self.Phi(D.T @ gradf) - gradf @ XD

    #         local_hess_feas = 4*X @ XD + 2*D @ CX


    #         return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

    #     return local_hess




    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            CX = self.C(X)
            AX = X - 0.5 * X@CX
            gradf = obj_grad(AX)
            XG = self.Phi(X.T @ gradf)
            XD = self.Phi(X.T @ D)

            local_JAT_D = D @ (self.Ip - 0.5 * CX) - X @ XD 
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ (np.eye(self._p) - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D )

            # local_hessA_objgrad_D = - D @ XG - X @ self.Phi(D.T @ gradf) - gradf @ XD

            # local_hess_feas = 4*X @ XD + 2*D @ CX
            # return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

            return local_objhess_JAT_D @ (self.Ip - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D + self.Phi(D.T @ gradf) - 4* beta * XD) + D @ (2*beta*CX - XG) - gradf @ XD



        return local_hess



    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            CX = self.C(X)
            AX = X
            gradf = obj_grad(AX)
            XG = self.Phi(X.T @ gradf)
            XD = self.Phi(X.T @ D)

            local_JAT_D = D  - X @ XD 
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ (np.eye(self._p) - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D )

            # local_hessA_objgrad_D = - D @ XG - X @ self.Phi(D.T @ gradf) - gradf @ XD

            # local_hess_feas = 4*X @ XD + 2*D @ CX
            # return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

            return local_objhess_JAT_D  -  X @ self.Phi( X.T @ local_objhess_JAT_D + self.Phi(D.T @ gradf) - 4* beta * XD) + D @ (2*beta*CX - XG) - gradf @ XD



        return local_hess

        # return local_hess



# import numpy as np
# import torch
# from torch import nn

# from numpy.linalg import svd
# from torch._C import device

# class stiefel_torch:
#     def __init__(self, n, p, device = torch.device('cpu'), dtype = torch.float64) -> None:
#         self._n = n
#         self._p = p
#         self.dim = n*p 
#         self.device = device
#         self.dtype = dtype

        
#         self.Ip = torch.eye(self._p).to(device = self.device, dtype = self.dtype)



#     def Phi(self, M):
#         return (M + M.T)/2


#     def A(self, X):
#         XX = X.T @ X
#         return 1.5 * X - X @ (XX /2)


    
#     def C(self, X):
#         return X.T @ X - self.Ip

#     def Feas_eval(self, X):
#         return torch.linalg.norm( self.C(X) , 'fro')

#     def Init_point(self, Xinit = None):
#         if Xinit is None:
#             # Xinit = np.random.randn(self._n, self._p)
#             Xinit = torch.randn(self._n, self._p).to(device = self.device, dtype = self.dtype)
            
#         if self.Feas_eval(Xinit) > 1e-6:
#             Xinit, Rinit = torch.linalg.qr(Xinit)
#         return Xinit

#     def Post_process(self,X):
#         UX, SX, VX = torch.linalg.svd(X, full_matrices = False)
#         return UX @ VX