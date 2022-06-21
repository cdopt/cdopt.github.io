import numpy as np
import torch
from torch import nn

from ..manifold_torch import complex_basic_manifold_torch
from torch.nn.parameter import Parameter


class complex_stiefel_torch(complex_basic_manifold_torch):
    def __init__(self, var_shape, device = torch.device('cpu'), dtype = torch.complex128) -> None:
        if len(var_shape) >= 2:
            self._n = var_shape[-2]
            self._p = var_shape[-1]
            self.dim = var_shape[:-2]
        else:
            print("The length of var_shape should be no less than 2.")
            raise TypeError

        self.device = device
        self.dtype = dtype

        super().__init__('Stiefel',var_shape, (*self.dim, self._p,self._p),   device= self.device ,dtype= self.dtype)
        # self._parameters = OrderedDict()
        self._parameters['Ip'] =  Parameter(torch.diag_embed(torch.ones((*self.dim, self._p), device=self.device, dtype=self.dtype)), False)
        self.Ip = self._parameters['Ip']


        
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



    def Phi(self, M):
        return (M + M.transpose(-2,-1).conj() )/2

    def C(self, X):
        # return X.T @ X - self.Ip.to(device=X.device, dtype = X.dtype)
        return torch.matmul(X.transpose(-2,-1).conj(), X) - self.Ip

    def A(self, X):
        XX = torch.matmul(X.transpose(-2,-1).conj(), X)
        return 1.5 * X - torch.matmul(X , (XX /2))


    def JA(self, X, G):
        return torch.matmul(G , ( self.Ip - 0.5 * self.C(X) ))  - torch.matmul(X , self.Phi(torch.matmul(X.transpose(-2,-1).conj(), G)) )

    def JA_transpose(self,X,G):
        # JA is self-adjoint
        return self.JA(X,G)

    def hessA(self, X, gradf, D):
        return - torch.matmul(D , self.Phi( torch.matmul(X.transpose(-2,-1).conj(), gradf)  )) - torch.matmul(X , self.Phi( torch.matmul(D.transpose(-2,-1).conj() , gradf)  )) - torch.matmul(gradf , self.Phi( torch.matmul(D.transpose(-2,-1).conj() , X) ))


    def JC(self, X, Lambda):
        return torch.matmul(2*X , self.Phi(Lambda))

    
    

    # def C_quad_penalty(self, X):
    #     CX = self.C(X)
    #     return torch.sum(CX * CX.conj())


    def hess_feas(self, X, D):
        return torch.matmul(4*X , self.Phi( torch.matmul(X.transpose(-2,-1).conj(), D) )  ) + 2* torch.matmul(D , self.C(X))

    



    # def Feas_eval(self, X):
    #     return torch.linalg.norm( self.C(X).flatten() , 2)

    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit = torch.randn(*self.var_shape, device = self.device, dtype = self.dtype)
        else:
            Xinit = Xinit.detach().to(device = self.device, dtype = self.dtype)
        # Xinit = Xinit.to(device = self.device, dtype = self.dtype)

        if self.Feas_eval(Xinit) > 1e-6:
            UX, SX, VX = torch.svd(Xinit)
            Xinit = torch.matmul(UX, VX.transpose(-2,-1).conj())
        
        Xinit.requires_grad = True
        return Xinit

    def Post_process(self,X):
        UX, SX, VX = torch.svd(X)
        return torch.matmul(UX, VX.transpose(-2,-1).conj())



    def generate_cdf_fun(self, obj_fun, beta):
        def local_obj_fun(X):
            CX = self.C(X)
            AX = X - 0.5 * X@ CX
            return (obj_fun(AX) + (beta/2) * torch.sum(CX *CX.conj())).real

        



        return local_obj_fun  




    def generate_cdf_grad(self, obj_grad, beta):
        def local_grad(X):
            CX = self.C(X)
            AX = X - 0.5 * torch.matmul(X,CX)
            gradf = obj_grad(AX)
            XG = self.Phi( torch.matmul(X.transpose(-2,-1).conj() , gradf) )

            # local_JA_gradf = gradf @ (np.eye(self._p) - 0.5 * CX) - X @ XG 
            
            # local_JC_CX = 2 * X @(CX)

            return torch.matmul(gradf , (self.Ip - 0.5 * CX)) +  torch.matmul(X ,  ( 2* beta * CX - XG))

        return local_grad  



    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            CX = self.C(X)
            AX = X - 0.5 *  torch.matmul(X,CX)
            gradf = obj_grad(AX)
            XG = self.Phi( torch.matmul(X.transpose(-2,-1).conj() , gradf) )
            XD = self.Phi( torch.matmul(X.transpose(-2,-1).conj() , D) )

            local_JAT_D = torch.matmul(D , (self.Ip - 0.5 * CX)) - torch.matmul(X , XD)
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ (np.eye(self._p) - 0.5 * CX) -  X @ self.Phi( X.T @ local_objhess_JAT_D )

            # local_hessA_objgrad_D = - D @ XG - X @ self.Phi(D.T @ gradf) - gradf @ XD

            # local_hess_feas = 4*X @ XD + 2*D @ CX
            # return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas

            return (   torch.matmul(local_objhess_JAT_D , (self.Ip - 0.5 * CX)) 
                    -  torch.matmul(X , self.Phi( torch.matmul(X.transpose(-2,-1).conj() , local_objhess_JAT_D) + self.Phi(torch.matmul(D.transpose(-2,-1).conj() , gradf)) - 4* beta * XD) )
                    + torch.matmul(D , (2*beta*CX - XG) - torch.matmul(gradf , XD)  )   )



        return local_hess



    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        def local_hess(X, D):
            CX = self.C(X)
            AX = X - 0.5 *  torch.matmul(X,CX)
            gradf = obj_grad(AX)
            XG = self.Phi( torch.matmul(X.transpose(-2,-1).conj() , gradf) )
            XD = self.Phi( torch.matmul(X.transpose(-2,-1).conj() , D) )

            local_JAT_D = D  - torch.matmul(X , XD) 
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)

            return local_objhess_JAT_D  -  torch.matmul(X , self.Phi( torch.matmul(X.transpose(-2,-1).conj() , local_objhess_JAT_D) + self.Phi(torch.matmul(D.transpose(-2,-1).conj() , gradf)) - 4* beta * XD) ) + torch.matmul(D , (2*beta*CX - XG) - torch.matmul(gradf , XD)  ) 


        return local_hess