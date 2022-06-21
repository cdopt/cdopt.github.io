import abc
import numpy as np 
import scipy as sp 
import jax 
import jax.numpy as jnp 
from cdopt.manifold import basic_manifold
from jax import random


from .basic_manifold_jax import basic_manifold_jax


class stiefel_jax(basic_manifold_jax):
    def __init__(self, var_shape) -> None:
        if len(var_shape) >= 2:
            self._n = var_shape[-2]
            self._p = var_shape[-1]
            self.dim = var_shape[:-2]
        else:
            print("The length of var_shape should be no less than 2.")
            raise TypeError

        super().__init__('stiefel',var_shape, (*self.dim, self._p,self._p))
        
        self.Ip = jnp.reshape(jnp.outer(jnp.ones(self.dim), jnp.eye(self._p)),(*self.dim, self._p,self._p))


        



    def Phi(self, M):
        return (M + M.swapaxes(-2,-1) )/2



    def A(self, X):
        XX = jnp.matmul(X.swapaxes(-2,-1), X)
        return 1.5 * X - jnp.matmul(X , (XX /2))


    def JA(self, X, G):
        return jnp.matmul(G , ( self.Ip - 0.5 * self.C(X) ))  - jnp.matmul(X , self.Phi(jnp.matmul(X.swapaxes(-2,-1), G)) )

    def JA_swapaxes(self,X,G):
        # JA is self-adjoint
        return self.JA(X,G)

    def hessA(self, X, gradf, D):
        return - jnp.matmul(D , self.Phi( jnp.matmul(X.swapaxes(-2,-1), gradf)  )) - jnp.matmul(X , self.Phi( jnp.matmul(D.swapaxes(-2,-1) , gradf)  )) - jnp.matmul(gradf , self.Phi( jnp.matmul(D.swapaxes(-2,-1) , X) ))


    def JC(self, X, Lambda):
        return jnp.matmul(2*X , self.Phi(Lambda))

    
    def C(self, X):
        # return X.T @ X - self.Ip.to(device=X.device, dtype = X.dtype)
        return jnp.matmul(X.swapaxes(-2,-1), X) - self.Ip

    def C_quad_penalty(self, X):
        return jnp.sum(self.C(X) ** 2)


    def hess_feas(self, X, D):
        return jnp.matmul(4*X , self.Phi( jnp.matmul(X.swapaxes(-2,-1), D) )  ) + 2* jnp.matmul(D , self.C(X))

    



    def Feas_eval(self, X):
        return jnp.linalg.norm( self.C(X).flatten() , 2)

    def Init_point(self, Xinit = None, seed = 0):
        if Xinit is None:
            key = random.PRNGKey(seed)
            Xinit = random.uniform(key, self.var_shape)

        if self.Feas_eval(Xinit) > 1e-6:
            UX, SX, VX = jnp.linalg.svd(Xinit, full_matrices= False)
            Xinit = jnp.matmul(UX, VX)
        
        return Xinit

    def Post_process(self,X):
        UX, SX, VX = jnp.linalg.svd(X, full_matrices= False)
        return jnp.matmul(UX, VX)



    def generate_cdf_fun(self, obj_fun, beta):
        def local_obj_fun(X):
            CX = self.C(X)
            AX = X - 0.5 * X@ CX
            return obj_fun(AX) + (beta/2) * jnp.sum(CX **2)

        



        return local_obj_fun  




    def generate_cdf_grad(self, obj_grad, beta):
        def local_grad(X):
            CX = self.C(X)
            AX = X - 0.5 * jnp.matmul(X,CX)
            gradf = obj_grad(AX)
            XG = self.Phi( jnp.matmul(X.swapaxes(-2,-1) , gradf) )

            # local_JA_gradf = gradf @ (np.eye(self._p) - 0.5 * CX) - X @ XG 
            
            # local_JC_CX = 2 * X @(CX)

            return jnp.matmul(gradf , (self.Ip - 0.5 * CX)) +  jnp.matmul(X ,  ( 2* beta * CX - XG))

        return local_grad  



    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            CX = self.C(X)
            AX = X - 0.5 *  jnp.matmul(X,CX)
            gradf = obj_grad(AX)
            XG = self.Phi( jnp.matmul(X.swapaxes(-2,-1) , gradf) )
            XD = self.Phi( jnp.matmul(X.swapaxes(-2,-1) , D) )

            local_JAT_D = jnp.matmul(D , (self.Ip - 0.5 * CX)) - jnp.matmul(X , XD)
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ (np.eye(self._p) - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D )

            # local_hessA_objgrad_D = - D @ XG - X @ self.Phi(D.T @ gradf) - gradf @ XD

            # local_hess_feas = 4*X @ XD + 2*D @ CX
            # return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

            return (   jnp.matmul(local_objhess_JAT_D , (self.Ip - 0.5 * CX)) 
                    -  jnp.matmul(X , self.Phi( jnp.matmul(X.swapaxes(-2,-1) , local_objhess_JAT_D) + self.Phi(jnp.matmul(D.swapaxes(-2,-1) , gradf)) - 4* beta * XD) )
                    + jnp.matmul(D , (2*beta*CX - XG) - jnp.matmul(gradf , XD)  )   )



        return local_hess



    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            CX = self.C(X)
            AX = X - 0.5 *  jnp.matmul(X,CX)
            gradf = obj_grad(AX)
            XG = self.Phi( jnp.matmul(X.swapaxes(-2,-1) , gradf) )
            XD = self.Phi( jnp.matmul(X.swapaxes(-2,-1) , D) )

            local_JAT_D = D  - jnp.matmul(X , XD) 
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ (np.eye(self._p) - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D )

            # local_hessA_objgrad_D = - D @ XG - X @ self.Phi(D.T @ gradf) - gradf @ XD

            # local_hess_feas = 4*X @ XD + 2*D @ CX
            # return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

            return local_objhess_JAT_D  -  jnp.matmul(X , self.Phi( jnp.matmul(X.swapaxes(-2,-1) , local_objhess_JAT_D) + self.Phi(jnp.matmul(D.swapaxes(-2,-1) , gradf)) - 4* beta * XD) ) + jnp.matmul(D , (2*beta*CX - XG) - jnp.matmul(gradf , XD)  ) 


        return local_hess

        # return local_hess



# import numpy as np
# import jnp
# from jnp import nn

# from numpy.linalg import svd
# from jnp._C import device

# class stiefel_jnp:
#     def __init__(self, n, p, device = jnp.device('cpu'), dtype = jnp.float64) -> None:
#         self._n = n
#         self._p = p
#         self.dim = n*p 
#         self.device = device
#         self.dtype = dtype

        
#         self.Ip = jnp.eye(self._p).to(device = self.device, dtype = self.dtype)



#     def Phi(self, M):
#         return (M + M.T)/2


#     def A(self, X):
#         XX = X.T @ X
#         return 1.5 * X - X @ (XX /2)


    
#     def C(self, X):
#         return X.T @ X - self.Ip

#     def Feas_eval(self, X):
#         return jnp.linalg.norm( self.C(X) , 'fro')

#     def Init_point(self, Xinit = None):
#         if Xinit is None:
#             # Xinit = np.random.randn(self._n, self._p)
#             Xinit = jnp.randn(self._n, self._p).to(device = self.device, dtype = self.dtype)
            
#         if self.Feas_eval(Xinit) > 1e-6:
#             Xinit, Rinit = jnp.linalg.qr(Xinit)
#         return Xinit

#     def Post_process(self,X):
#         UX, SX, VX = jnp.linalg.svd(X, full_matrices = False)
#         return UX @ VX