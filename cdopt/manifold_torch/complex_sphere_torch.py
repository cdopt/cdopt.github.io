import numpy as np 
import scipy as sp 
import torch
from ..manifold_torch import complex_basic_manifold_torch
class complex_sphere_torch(complex_basic_manifold_torch):
    def __init__(self, var_shape, device = torch.device('cpu'), dtype = torch.complex128):
        self.device = device
        self.dtype = dtype
        super().__init__('complex_sphere',var_shape, (1,),   device= self.device ,dtype= self.dtype)



    def A(self, X):
        return (2*X)/( 1 + torch.sum( X* X.conj() ) )

    def JA(self, X, G):
        XG = torch.sum(X*G.conj()).real
        X_rvec = torch.sum( X* X.conj() ).real +1
        return (2*G - ( (4*XG)*X )/X_rvec )/X_rvec
        

    def JA_transpose(self, X, D):
        return self.JA(X,D) 

    def hessA(self, X, gradf, D):
        XG = torch.sum(X * gradf.conj()).real 
        XD = torch.sum(X * D.conj()).real
        GD = torch.sum(gradf * D.conj()).real
        X_rvec = torch.sum( X * X.conj()  ) +1
        
        return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * XG * XD)

    # def hessA(self, X, gradf, D):
    #     XG = torch.sum(X*gradf).real
    #     XD = torch.sum(X * D).real
    #     GD = torch.sum(gradf * D).real
    #     X_rvec = torch.sum( X **2  ) +1
    #     return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * XG * XD)

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

    def C(self, X):
        return torch.sum( X * X.conj() ) - 1

    def JC(self, X, Lambda):
        return 2*X * Lambda.real

    def hess_feas(self, X, D):
        return 2 * D * self.C(X) + 4 * X * torch.sum(X*D.conj()).real


    # def C_quad_penalty(self, X):
    #     CX = self.C(X)
    #     return torch.sum( CX * CX.conj()  )

    # def Feas_eval(self, X):
    #     return torch.sqrt(self.C_quad_penalty(X)).real



    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit_real = torch.randn(*self.var_shape)
            Xinit_imag = torch.randn(*self.var_shape)
            Xinit = torch.complex(Xinit_real, Xinit_imag)
        else:
            Xinit = Xinit.detach()
        
        Xinit = torch.as_tensor(Xinit).to(device=self.device, dtype=self.dtype)
        Xinit.requires_grad = True
        X_rvec = torch.sqrt(torch.sum( Xinit * Xinit.conj() ))
        Xinit = Xinit / X_rvec
        
        return Xinit

    def Post_process(self,X):
        X_rvec = torch.sqrt(torch.sum( X * X.conj()  ))
        X = X / X_rvec
        return X


    # def generate_cdf_fun(self, obj_fun, beta):
    #     return lambda X: (obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)).real