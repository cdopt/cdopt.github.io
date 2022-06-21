import abc
import numpy as np 
import scipy as sp 
import jax 
import jax.numpy as jnp 
from cdopt.manifold import basic_manifold
from jax import random


from .basic_manifold_jax import basic_manifold_jax

class oblique_jax(basic_manifold_jax):
    def __init__(self, var_shape) -> None:
        self.dim = var_shape[:-1]
        self._p = var_shape[-1]


        

        super().__init__('oblique',var_shape, (*self.dim,1 ))

    


    def A(self, X):
        X_rvec = jnp.sum( X * X, -1, keepdims=True )
        return (2*X)/( 1 + X_rvec )


    def JA(self, X, G):
        XG = jnp.sum(X*G, -1, keepdims=True)
        X_rvec = jnp.sum( X * X, -1, keepdims=True ) + 1
        return (2*G - (4*X *XG)/X_rvec )/X_rvec

    def JA_transpose(self, X, D):
        return self.JA(X,D) 


    def hessA(self, X, gradf, D):
        XG = jnp.sum(X*gradf,-1, keepdims=True)
        XD = jnp.sum(X * D, -1, keepdims=True)
        GD = jnp.sum(gradf * D, -1, keepdims=True)
        X_rvec = jnp.sum( X **2, -1, keepdims=True  ) +1
        return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * XG * XD)


    def JC(self, X, Lambda):
        return 2*X * Lambda

    
    def C(self, X):
        return jnp.sum( X * X, -1, keepdims=True ) - 1

    


    def C_quad_penalty(self, X):
        return jnp.sum( self.C(X) **2  )

    def Feas_eval(self, X):
        return jnp.sqrt(self.C_quad_penalty(X))



    def hess_feas(self, X, D):
        return 2 * D * self.C(X) + 4 * X * jnp.sum(X*D, -1, keepdims=True)




    def Init_point(self, Xinit = None, seed = 0):
        if Xinit is None:
            key = random.PRNGKey(seed)
            Xinit = random.normal(key, self.var_shape)
        
        X_rvec = jnp.sqrt(jnp.sum( Xinit * Xinit, -1, keepdims= True ))
        Xinit = Xinit / X_rvec
        
        return Xinit

    def Post_process(self,X):
        X_rvec = jnp.sqrt(jnp.sum( X * X, -1, keepdims= True ))
        X = X / X_rvec
        return X
