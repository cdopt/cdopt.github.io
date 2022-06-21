import numpy as np


from .basic_manifold_np import basic_manifold_np

class sphere_np(basic_manifold_np):
    def __init__(self, var_shape) -> None:
        n = var_shape[0]
        p = var_shape[1]
        self._n = n
        self._p = p


        super().__init__('Sphere',(n,p), (1,))
    


    def A(self, X):
        # X_rvec = torch.sum( X * X, 1 )[:, None]
        return (2*X)/( 1 + np.sum( X**2 ) )


    def JA(self, X, G):
        XG = np.sum(X*G)
        X_rvec = np.sum( X **2  ) +1
        return (2*G - ( (4*XG)*X )/X_rvec )/X_rvec

    def JA_transpose(self, X, D):
        return self.JA(X,D) 


    def hessA(self, X, gradf, D):
        XG = np.sum(X*gradf)
        XD = np.sum(X * D)
        GD = np.sum(gradf * D)
        X_rvec = np.sum( X **2  ) +1
        return -(4/(X_rvec**2))*( gradf * XD + D * XG + X * GD ) + 16/( X_rvec**3 ) * (X * (XG * XD))


    def JC(self, X, Lambda):
        return 2 * X * Lambda

    
    def C(self, X):
        return np.sum( X **2 ) - 1

    def hess_feas(self, X, D):
        return 2 * D * self.C(X) + 4 * X * np.sum(X*D)


    def C_quad_penalty(self, X):
        return np.sum( self.C(X) **2  )

    def Feas_eval(self, X):
        return np.sqrt(self.C_quad_penalty(X))






    def Init_point(self, Xinit = None):
        if Xinit is None:
            Xinit = np.random.randn(self._n, self._p)
        
        X_rvec = np.sqrt(np.sum( Xinit * Xinit ))
        Xinit = Xinit / X_rvec
        
        return Xinit

    def Post_process(self,X):
        X_rvec = np.sqrt(np.sum( X **2  ))
        X = X / X_rvec
        return X
