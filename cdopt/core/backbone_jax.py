from cmath import sqrt
import numpy as np
import jax
import jax.numpy as jnp

from jax import grad, jvp, vjp,  jit



class backbone_jax:
    def __init__(self, *args, **kwargs) -> None:
        self.solve = jnp.linalg.solve 
        self.identity_matrix = jnp.eye
        self.zero_matrix = jnp.zeros
        self.func_sum = jnp.sum 
        self.func_sqrt = jnp.sqrt 
        self.var_type = 'jax'
        
    def dir_grad(self,mappings,X, D):
        def local_fun(Y):
            return jax.sum( D * mappings(Y) )
        return grad(local_fun)(X)




    def auto_diff_vjp(self,fun, X, D):
        val, fun_vjp = vjp(fun, X)
        return fun_vjp(D)[0]

    def auto_diff_jvp(self, fun, X, D):
        val, jvp_result = jvp(fun, (X,), (D,))
        return jvp_result

    def auto_diff_jacobian(self, fun, X):
        return jax.jacrev(fun)(X)



    def linear_map_adjoint(fun,D):
        test_fun = lambda U: jnp.sum(D *fun(U))
        return grad(test_fun)
        # return lambda X: auto_diff_vjp(fun, X, D)




    def autodiff(self, obj_fun, obj_grad = None, manifold_type = 'S'):
        if obj_grad is not None:
            local_obj_grad = obj_grad
        else:
            local_obj_grad_tmp = jit(grad(obj_fun))
            local_obj_grad = lambda X: local_obj_grad_tmp(X)


        # def local_obj_hess(X, D):
        #     def directional_grad(X):
        #         return anp.sum( D*  local_obj_grad(X))
        #     return grad(directional_grad)(X)

        local_obj_hess = lambda X, D: self.auto_diff_vjp(local_obj_grad, X, D)
        # local_obj_hess = lambda X, D:  make_hvp(obj_fun, X)(D)

        return local_obj_grad, local_obj_hess


    def array2tensor(self, X_array):
        return jnp.asarray(X_array)

    def tensor2array(self, X_tensor):
        return np.array(X_tensor,dtype=np.float64 ,order = 'F')


    def jit(self, fun):
        return jax.jit(fun)

    
