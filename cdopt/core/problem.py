import numpy as np



class Problem:
    def __init__(self, manifold, obj_fun, obj_grad=None, obj_hvp=None, beta = 0, enable_autodiff = True, backbone = None, enable_jit = True,  **kwargs):
        
        self.manifold = manifold
        self.obj_fun = obj_fun 



        


        if enable_autodiff:
            if obj_grad is None   or   obj_hvp is None:
                if backbone == None:
                    self.backbone = manifold.backbone
                elif backbone == 'jax':
                    # from core.autodiff_jax import autodiff
                    # raise NotImplementedError
                    from ..core.backbone_jax import backbone_jax
                    self.backbone = backbone_jax()
                elif backbone == 'autograd':
                    # from core.autodiff_ag import autodiff
                    from ..core.backbone_autograd import backbone_autograd
                    self.backbone = backbone_autograd()
                elif backbone == 'torch':
                    from ..core.backbone_torch import backbone_torch
                    self.backbone = backbone_torch(**kwargs)
                else: 
                    self.backbone = backbone(**kwargs)
                
                autodiff = self.backbone.autodiff
                    
                if  obj_hvp is None:
                    self.obj_grad, self.obj_hvp = autodiff(obj_fun, obj_grad, manifold.manifold_type)
                else:
                    self.obj_grad,  = autodiff(obj_fun,obj_grad, manifold.manifold_type)
                
            else:
                self.obj_grad = obj_grad 
                self.obj_hvp = obj_hvp 


            

            self.cdf_fun = self.manifold.generate_cdf_fun(self.obj_fun, beta)
            self.cdf_grad = self.manifold.generate_cdf_grad(self.obj_grad, beta)
            self.cdf_hvp = self.manifold.generate_cdf_hess(self.obj_grad, self.obj_hvp, beta)

            if enable_jit:
                self.cdf_fun = self.backbone.jit(self.cdf_fun)
                self.cdf_grad = self.backbone.jit(self.cdf_grad)
                self.cdf_hvp = self.backbone.jit(self.cdf_hvp)




            self.cdf_fun_vec = lambda y: self.cdf_fun(self.manifold.v2m(y))
            self.cdf_grad_vec= lambda y: self.manifold.m2v( self.cdf_grad(self.manifold.v2m(y)) )
            self.cdf_hvp_vec = lambda y,p: self.manifold.m2v( self.cdf_hvp(self.manifold.v2m(y), self.manifold.v2m(p)) )


            # self.cdf_fun_vec_np = lambda y: float(self.manifold.tensor2array(self.cdf_fun_vec( self.manifold.array2tensor(y)) ))
            self.cdf_fun_vec_np = lambda y: float(self.cdf_fun_vec( self.manifold.array2tensor(y)) )
            self.cdf_grad_vec_np = lambda y: self.manifold.tensor2array(self.cdf_grad_vec( self.manifold.array2tensor(y)) )
            self.cdf_hvp_vec_np = lambda y,p: self.manifold.tensor2array(self.cdf_hvp_vec(self.manifold.array2tensor(y), self.manifold.array2tensor(p))   )

        else:

            def _raise_not_implemented_error(*args, **kwargs):
                raise NotImplementedError("Automatical differentiation not enabled")


            self.obj_fun = obj_fun
            self.obj_grad = obj_grad
            self.obj_hvp = obj_hvp

            self.cdf_fun = self.manifold.generate_cdf_fun(self.obj_fun, beta)
            if enable_jit:
                self.cdf_fun = self.backbone.jit(self.cdf_fun)
            
            
            self.cdf_fun_vec = lambda y: self.cdf_fun(self.manifold.v2m(y))

            

            if obj_grad is not None:
                self.cdf_grad = self.manifold.generate_cdf_grad(self.obj_grad, beta)
                if enable_jit:
                    self.cdf_grad = self.backbone.jit(self.cdf_grad)

                self.cdf_grad_vec = lambda y: self.manifold.m2v( self.cdf_grad(self.manifold.v2m(y)) )

            else:
                self.cdf_grad = _raise_not_implemented_error
                self.cdf_grad_vec= _raise_not_implemented_error



            if obj_hvp is not None:
                self.cdf_hvp = self.manifold.generate_cdf_hess(self.obj_grad, self.obj_hvp, beta)
                if enable_jit:
                    self.cdf_hvp = self.backbone.jit(self.cdf_hvp)

                self.cdf_hvp_vec = lambda y,p: self.manifold.m2v( self.cdf_hvp(self.manifold.v2m(y), self.manifold.v2m(p)) )

            else:
                self.cdf_hvp = _raise_not_implemented_error
                self.cdf_hvp_vec = _raise_not_implemented_error




        
            


        