import numpy as np
import torch

# from numpy.linalg import svd, eig
from .basic_manifold_torch import basic_manifold_torch
from torch.nn.parameter import Parameter


class generalized_stiefel_torch(basic_manifold_torch):
    def __init__(self, var_shape, B = lambda X:X , device = torch.device('cpu'), dtype = torch.float64) -> None:
        n = var_shape[0]
        p = var_shape[1]
        self._n = n
        self._p = p
        self.dim = n*p 
        self.var_shape = (n,p)
        self.manifold_type = 'S'

        self.B = B 
        # Here we require B be a function since it could fully utilize the
        # low-rank structure of B
        #  when one wishes to input a matrix B_mat
        #  we recommand to use lambda X: B@X instead

        self.device = device
        self.dtype = dtype

        

        
        super().__init__('Generalized_Stiefel',(n,p), (p,p),   device= self.device ,dtype= self.dtype)

        self._parameters['Ip'] = Parameter(torch.eye(self._p).to(device = self.device, dtype = self.dtype),False)
        self.Ip = self._parameters['Ip']


    def Phi(self, M):
        return (M + M.T)/2


    def A(self, X):
        XX = X.T @ self.B(X)
        return 1.5 * X - X @ (XX /2)


    def JA(self, X, D):
        BX = self.B(X)
        return D @ ( 1.5 * self.Ip - 0.5 * X.T @ BX  )  - BX @ self.Phi(X.T @ D)


    def JA_transpose(self, X, D):
        BX = self.B(X)
        return D @ ( 1.5 * self.Ip - 0.5 * X.T @ BX  ) - X @ self.Phi( D.T @ BX )


    def hessA(self, X, gradf, D):
        BX = self.B(X)

        return - self.B(D) @ self.Phi( X.T @ gradf  ) - BX @ self.Phi(D.T @ gradf) - gradf @ self.Phi( D.T @ BX )



    
    def C(self, X):
        return X.T @ self.B(X) - self.Ip

    def C_quad_penalty(self, X):
        return torch.sum(self.C(X) ** 2)



    def JC(self, X, Lambda):
        return 2 * self.B(X) @ self.Phi( Lambda  )


    def hess_feas(self, X, D):
        BX = self.B(X)
        return 4*BX @ self.Phi( BX.T @ D ) + 2*self.B(D) @ (X.T @ BX- self.Ip)



    def Feas_eval(self, X):
        return torch.linalg.norm( self.C(X) , 'fro')

    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit = torch.randn(self._n, self._p).to(device = self.device, dtype = self.dtype)
        else:
            Xinit = Xinit.detach()
            
        if torch.linalg.norm(self.C(Xinit), 'fro') > 1e-6:
            Linit = torch.linalg.cholesky(Xinit.T @ self.B(Xinit))
            Xinit = torch.linalg.solve(Linit, Xinit.T).T

        Xinit.requires_grad = True

        
        return Xinit

    def Post_process(self,X):
        L_process = torch.linalg.cholesky(X.T @ self.B(X))
        X = torch.linalg.solve(L_process, X.T).T
        return X





    def generate_cdf_fun(self, obj_fun, beta):
        return  lambda X: obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)
        



    def generate_cdf_grad(self, obj_grad, beta):
        def local_grad(X):
            BX= self.B(X)
            CX = X.T @ BX - self.Ip
            AX = X - 0.5 * X@CX
            gradf = obj_grad(AX)
            XG = self.Phi(X.T @ gradf)

            # local_JA_gradf = gradf @ (self.Ip - 0.5 * CX) - X @ XG 
            
            # local_JC_CX = 2 * X @(CX)

            return gradf @ (self.Ip - 0.5 * CX) +  BX @  ( 2* beta * CX - XG) 

        return local_grad  




    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            BX= self.B(X)
            CX = X.T @ BX - self.Ip
            AX = X - 0.5 * X@CX
            gradf = obj_grad(AX)
            # XG = self.Phi(X.T @ gradf)



            local_JAT_D =  D @ ( self.Ip - 0.5 * CX  ) - X @ self.Phi( D.T @ BX )
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ ( self.Ip - 0.5 * CX  )  - BX @ self.Phi(X.T @ local_objhess_JAT_D)

            # local_hessA_objgrad_D = - self.B(D) @ self.Phi( X.T @ gradf  ) - BX @ self.Phi(D.T @ gradf) - gradf @ self.Phi( D.T @ BX )

            # local_hess_feas = 4*BX @ self.Phi( BX.T @ D ) + 2*self.B(D) @ CX



            return local_objhess_JAT_D @ ( self.Ip - 0.5 * CX  )  - BX @ (self.Phi(X.T @ local_objhess_JAT_D) + self.Phi(D.T @ gradf)  - 4* beta * self.Phi( BX.T @ D )   ) - self.B(D) @ self.Phi( X.T @ gradf -2*beta * CX ) - gradf @ self.Phi( D.T @ BX )


        return local_hess



    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            AX = self.A(X)
            gradf = obj_grad(AX)

            local_JAT_D = self.JA_transpose(X, D)
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            local_JA_objhess_JAT_D = self.JA(X, local_objhess_JAT_D)

            local_hessA_objgrad_D = self.hessA(X, gradf, D)

            local_hess_feas = self.hess_feas(X,D)


            return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

        return local_hess
