import abc
import numpy as np 
import scipy as sp 
import jax 
import jax.numpy as jnp 
from cdopt.manifold import basic_manifold
from jax import random

from numpy.linalg import svd

from .basic_manifold_jax import basic_manifold_jax


class generalized_stiefel_jax(basic_manifold_jax):
    def __init__(self, var_shape, B = lambda X:X ) -> None:
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


        

        
        super().__init__('Generalized_Stiefel',(n,p), (p,p))

        self.Ip = jnp.eye(self._p) 


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
        return jnp.sum(self.C(X) ** 2)



    def JC(self, X, Lambda):
        return 2 * self.B(X) @ self.Phi( Lambda  )


    def hess_feas(self, X, D):
        BX = self.B(X)
        return 4*BX @ self.Phi( BX.T @ D ) + 2*self.B(D) @ (X.T @ BX- self.Ip)



    def Feas_eval(self, X):
        return jnp.linalg.norm( self.C(X) , 'fro')

    def Init_point(self, Xinit = None, seed = 0):
        if Xinit is None:
            key = random.PRNGKey(seed)
            Xinit = random.uniform(key, self.var_shape)
            
        if np.linalg.norm(self.C(Xinit), 'fro') > 1e-6:
            Linit = np.linalg.cholesky(Xinit.T @ self.B(Xinit))
            Xinit = np.linalg.solve(Linit, Xinit.T).T


        
        return Xinit

    def Post_process(self,X):
        L_process = np.linalg.cholesky(X.T @ self.B(X))
        X = np.linalg.solve(L_process, X.T).T
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
