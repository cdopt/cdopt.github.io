import abc
import numpy as np 
import scipy as sp 
import jax 
import jax.numpy as jnp 
from cdopt.manifold import basic_manifold
from jax import random


from .basic_manifold_jax import basic_manifold_jax

class sphere_jax(basic_manifold_jax):
    def __init__(self, var_shape) -> None:


        super().__init__('sphere',var_shape, (1,))
    


    def A(self, X):
        # X_rvec = torch.sum( X * X, 1 )[:, None]
        return (2*X)/( 1 + jnp.sum( X**2 ) )


    def JA(self, X, G):
        XG = jnp.sum(X*G)
        X_rvec = jnp.sum( X **2  ) +1
        return (2*G - ( (4*XG)*X )/X_rvec )/X_rvec

    def JA_transpose(self, X, D):
        return self.JA(X,D) 


    def hessA(self, X, gradf, D):
        XG = jnp.sum(X*gradf)
        XD = jnp.sum(X * D)
        GD = jnp.sum(gradf * D)
        X_rvec = jnp.sum( X **2  ) +1
        return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * XG * XD)


    


    def JC(self, X, Lambda):
        return 2*X * Lambda

    
    def C(self, X):
        return jnp.sum( X **2 ) - 1

    def hess_feas(self, X, D):
        return 2 * D * self.C(X) + 4 * X * jnp.sum(X*D)


    def C_quad_penalty(self, X):
        return jnp.sum( self.C(X) **2  )

    def Feas_eval(self, X):
        return jnp.sqrt(self.C_quad_penalty(X))






    def Init_point(self, Xinit = None, seed = 0):
        if Xinit is None:
            key = random.PRNGKey(seed)
            Xinit = random.normal(key, self.var_shape)
        
        
        X_rvec = jnp.sqrt(jnp.sum( Xinit * Xinit ))
        Xinit = Xinit / X_rvec
        
        return Xinit

    def Post_process(self,X):
        X_rvec = jnp.sqrt(jnp.sum( X **2  ))
        X = X / X_rvec
        return X
