import numpy as np
import torch
from torch import nn

from ..manifold_torch import complex_basic_manifold_torch

class complex_oblique_torch(complex_basic_manifold_torch):
    def __init__(self, var_shape, device = torch.device('cpu'), dtype = torch.complex128) -> None:
        self.dim = var_shape[:-1]
        self._p = var_shape[-1]


        self.device = device
        self.dtype = dtype
        

        super().__init__('oblique',var_shape, (*self.dim,1 ),   device= self.device ,dtype= self.dtype)

    


    # def array2tensor(self, X_array):
    #     dim = len(X_array)
    #     X_real = torch.as_tensor(X_array[:int(dim/2)])
    #     X_imag = torch.as_tensor(X_array[int(dim/2):])
    #     X =  torch.complex(X_real, X_imag).to(device=self.device, dtype=self.dtype)
    #     X.requires_grad = True 
    #     return X 


    # def tensor2array(self, X_tensor, np_dtype = np.float64):
    #     X_real = X_tensor.real.detach().cpu().numpy()
    #     X_imag = X_tensor.imag.detach().cpu().numpy()

    #     return np.concatenate((np_dtype(X_real), np_dtype(X_imag)) )



    def A(self, X):
        X_rvec = torch.sum( X * X.conj(), -1, keepdim=True )
        return (2*X)/( 1 + X_rvec )


    def JA(self, X, G):
        XG = torch.sum(X*G.conj(), -1, keepdim=True).real
        X_rvec = torch.sum( X* X.conj() , -1, keepdim=True).real +1
        return (2*G - ( (4*XG)*X )/X_rvec )/X_rvec


    def hessA(self, X, gradf, D):
        XG = torch.sum(X * gradf.conj(), -1, keepdim=True).real 
        XD = torch.sum(X * D.conj(), -1, keepdim=True).real
        GD = torch.sum(gradf * D.conj(), -1, keepdim=True).real
        X_rvec = torch.sum( X * X.conj() , -1, keepdim=True ) +1
        
        return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * XG * XD)



    # def C_quad_penalty(self, X):
    #     CX = self.C(X)
    #     return torch.sum( CX * CX.conj()  )

    # def Feas_eval(self, X):
    #     return torch.sqrt(self.C_quad_penalty(X)).real



    def C(self, X):
        return torch.sum( X * X.conj(), -1, keepdim=True ) - 1

    def JC(self, X, Lambda):
        return 2*X * Lambda.real

    def hess_feas(self, X, D):
        return 2 * D * self.C(X) + 4 * X * torch.sum(X*D.conj(), -1, keepdim= True).real



    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit = torch.randn(*self.var_shape, dtype=self.dtype, device= self.device)
        else:
            Xinit = Xinit.detach().to(device=self.device, dtype=self.dtype)
        
        # Xinit = torch.as_tensor(Xinit)
        Xinit.requires_grad = True
        X_rvec = torch.sqrt(torch.sum( Xinit * Xinit.conj(), -1, keepdim= True ))
        Xinit = Xinit / X_rvec
        
        return Xinit

    def Post_process(self,X):
        X_rvec = torch.sqrt(torch.sum( X * X.conj(), -1, keepdim= True ))
        X = X / X_rvec
        return X



    # def generate_cdf_fun(self, obj_fun, beta):
    #     return lambda X: (obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)).real

