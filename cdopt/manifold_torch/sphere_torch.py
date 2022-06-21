import numpy as np
import torch
from torch import nn


from .basic_manifold_torch import basic_manifold_torch

class sphere_torch(basic_manifold_torch):
    def __init__(self, var_shape, device = torch.device('cpu'), dtype = torch.float64) -> None:
        self.device = device
        self.dtype = dtype


        super().__init__('Sphere',var_shape, (1,),   device= self.device ,dtype= self.dtype)
    


    def A(self, X):
        # X_rvec = torch.sum( X * X, 1 )[:, None]
        return (2*X)/( 1 + torch.sum( X**2 ) )


    def JA(self, X, G):
        XG = torch.sum(X*G)
        X_rvec = torch.sum( X **2  ) +1
        return (2*G - ( (4*XG)*X )/X_rvec )/X_rvec

    def JA_transpose(self, X, D):
        return self.JA(X,D) 


    def hessA(self, X, gradf, D):
        XG = torch.sum(X*gradf)
        XD = torch.sum(X * D)
        GD = torch.sum(gradf * D)
        X_rvec = torch.sum( X **2  ) +1
        return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * XG * XD)


    


    def JC(self, X, Lambda):
        return 2*X * Lambda

    
    def C(self, X):
        return torch.sum( X **2 ) - 1

    def hess_feas(self, X, D):
        return 2 * D * self.C(X) + 4 * X * torch.sum(X*D)


    def C_quad_penalty(self, X):
        return torch.sum( self.C(X) **2  )

    def Feas_eval(self, X):
        return torch.sqrt(self.C_quad_penalty(X))






    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit = torch.randn(*self.var_shape)
        else:
            Xinit = Xinit.detach()
        
        Xinit = torch.as_tensor(Xinit).to(device=self.device, dtype=self.dtype)
        Xinit.requires_grad = True
        X_rvec = torch.sqrt(torch.sum( Xinit * Xinit ))
        Xinit = Xinit / X_rvec
        
        return Xinit

    def Post_process(self,X):
        X_rvec = torch.sqrt(torch.sum( X **2  ))
        X = X / X_rvec
        return X
