import numpy as np
from .basic_manifold_np import basic_manifold_np
from scipy.sparse import csr_matrix


class symp_stiefel_np(basic_manifold_np):
    def __init__(self, var_shape) -> None:
        n = var_shape[0]
        p = var_shape[1]
        self._n = n
        self._p = p

        

        self._half_n = int(n/2)

        self._half_p = int(p/2)

        In = np.eye(int(n/2))
        Zero_n = np.zeros((int(n/2),int(n/2)))

        Jn = np.block([[Zero_n,  In], [-1*In, Zero_n]    ])

        Ip = np.eye(int(p/2))
        Zero_p = np.zeros((int(p/2), int(p/2)))
        Jp = np.block([[Zero_p,  Ip], [-1*Ip, Zero_p]    ])

        Jn = csr_matrix(Jn)
        Jp = csr_matrix(Jp)

  

        def JnX(X):
            return Jn@X

        



        def JpM(X):
            return Jp@X

        # In = np.eye(int(n/2))
        # Zero_n = np.zeros((int(n/2),int(n/2)))

        # Jn = np.block([[Zero_n,  In], [-1*In, Zero_n]    ])

        # Ip = np.eye(int(p/2))
        # Zero_p = np.zeros((int(p/2), int(p/2)))
        # Jp = np.block([[Zero_p,  Ip], [-1*Ip, Zero_p]    ])

        # Jn = torch.tensor(Jn)
        # Jp = torch.tensor(Jp)
        # # Jn = torch.tensor(Jn).to_sparse_csr()
        # # Jp = torch.tensor(Jp).to_sparse_csr()
        # if self.device == torch.device('cpu'):
        #     Jn = torch.tensor(Jn).to_sparse()
        #     Jp = torch.tensor(Jp).to_sparse()

        #     Jn = Jn.to(device = self.device, dtype = self.dtype)
        #     Jp = Jp.to(device = self.device, dtype = self.dtype)

        #     def JnX(X):
        #         return torch.matmul(Jn, X)

        #     def JpM(X):
        #         return torch.matmul(Jp, X)


        # # Jn = torch.tensor(Jn, dtype= self.dtype).to_sparse_csr()
        # # Jp = torch.tensor(Jp, dtype= self.dtype).to_sparse_csr()


        # else:
            
        #     Jn = Jn.to(device = self.device, dtype = self.dtype)
        #     Jp = Jp.to(device = self.device, dtype = self.dtype)

        #     def JnX(X):
        #         return torch.cat( (X[self._half_n:, :], -X[:self._half_n,:]), 0 )
        #         # return torch.matmul(Jn, X)

        #     def JpM(X):
        #         return torch.cat( (X[self._half_p:, :], -X[:self._half_p,:]), 0 )
        #         # return torch.matmul(Jp, X)

        self.JnX = JnX
        self.JpM = JpM
        self.Ip = np.eye(self._p)

        super().__init__('Symp_Stiefel',(n,p), (p,p))

        # Clearly, use sparse matrices from scipy.sparse could leads to higher efficiency. 
        # However, autograd package has bad compatiablility with scipy package.
        # That is the reason for using changes of indexes to compute Jn @ X and Jp @ M.



    def Psi(self, M):
        return (M - M.T)/2

    # def Phi(self, M):
    #     return (M + M.T)/2


    def A(self, X):
        JX = self.JnX(X)
        XJX = X.T @ JX 

        return X @ ( self.Ip + 0.5 * (self.JpM( XJX) + self.Ip  )  )



    def JA(self, X, G):
        # JX = self.JnX(X)
        # XJX= X.T @ JX 
        XG = X.T @ G
        return G @ ( 1.5*self.Ip + 0.5 * self.JpM(X.T @ self.JnX(X) )  ).T + self.JnX( X @ self.Psi( self.JpM( XG ) ) )
        # return G @ ( self.Ip + 0.5 * self.C(X)  ).T + self.JnX( X @ self.Psi( self.JpM( XG ) ) )

    
    def JA_transpose(self, X, G):
        return G @ ( 1.5*self.Ip + 0.5 * self.JpM(X.T @ self.JnX(X) )  ) +  X@ self.JpM( self.Psi( X.T @ self.JnX(G) ) ) 


    def hessA(self, X, gradf, D):
        return self.JnX(X) @ self.Psi( self.JpM(D.T @ gradf ) ) + self.JnX(D) @ self.Psi( self.JpM(X.T @ gradf) ) + gradf @ ( self.JpM( self.Psi( D.T @ self.JnX(X) ) ) )


    


    def C(self, X):
        # return self.JpM(X.T @ self.JnX(X) ) + self.Ip
        return self.JpM( X.T @ self.JnX(X) ) + self.Ip

    def C_quad_penalty(self, X):
        return np.sum(self.C(X) **2)



    def JC(self, X, T):
        return 2 * self.JnX( X@self.Psi( self.JpM(T) ) )


    def hess_feas(self, X, D):
        return 2 *  self.JnX(D) @  self.JpM( self.C(X) ) - 4 * self.JnX(X) @ self.Psi( D.T @ self.JnX(X) )



    def hessp(self, X, D, gradf, hessf, beta):
        return self.JA( X, hessf( self.JA_transpose(X,D) ) ) + self.hessA(X, gradf, D) + beta * self.hess_feas(X, D)

    def Feas_eval(self, X):
        return np.linalg.norm( self.C(X) , 'fro')

    def Init_point(self, Xinit = None):
        X = Xinit
        if X is None:
            X = np.random.randn(self._n, self._p)
            X, R = np.linalg.qr(X)

        for jl in range(10):
            X = X - 0.5/( np.linalg.norm(X.T @ X, 2) )   * self.JC(X,self.C(X))
            # X = X - 0.1  *self.JnX(X) @ self.JpM(self.C(X))
            if self.Feas_eval(X) < 1e-3:
                break
        # print(self.Feas_eval(X))
        for jl in range(10):
            JX = self.JnX(X)
            XJX = X.T @ JX 
            feas = np.linalg.norm( self.C(X) , 'fro')
            
            if feas < 1e-2:
                X = self.A(X)
            else:
                TTT = 0.5* self.Ip - 0.5 * self.JpM( XJX) 
                X = np.linalg.solve(TTT.T, X.T ).T
            if feas < 1e-8:
                break


        
        return X

    def Post_process(self,X):
        for jl in range(3):
            JX = self.JnX(X)
            XJX = X.T @ JX 
            feas = np.linalg.norm( self.C(X) , 'fro')
            if feas < 1e-1:
                X = self.A(X)
            else:
                TTT = 0.5*self.Ip - 0.5 * self.JpM( XJX) 
                X = np.linalg.solve(TTT.T, X.T ).T
            if feas < 1e-12:
                break
        return X





    def generate_cdf_fun(self, obj_fun, beta):
        return lambda X: obj_fun(self.A(X)) + (beta/2) * self.C_quad_penalty(X)
        

    # def generate_cdf_grad(self, obj_grad,beta):
    #     return lambda X: self.JA(X, obj_grad(self.A(X))) + beta * self.JC(X, self.C(X))


    # def generate_cdf_grad(self, obj_grad, beta):
    #     def local_grad(X):
    #         CX = self.C(X)
    #         local_JnX = self.JnX(X)
    #         AX = X @ ( self.Ip + 0.5 * CX  )
    #         gradf = obj_grad(AX)


    #         local_JA_gradf = gradf @ ( self.Ip + CX  ).T + local_JnX @ self.Psi( self.JpM( X.T @ gradf ) ) 
    #         local_JC_CX = 2 * local_JnX @ self.Psi( self.JpM(CX) ) 

    #         return local_JA_gradf + beta * local_JC_CX

    #     return local_grad





    def generate_cdf_grad(self, obj_grad, beta):
        def local_grad(X):
            CX = self.C(X)
            local_JnX = self.JnX(X)
            AX = X @ ( self.Ip + 0.5 * CX  )
            gradf = obj_grad(AX)


            # local_JA_gradf = gradf @ ( self.Ip + CX  ).T + local_JnX @ self.Psi( self.JpM( X.T @ gradf ) ) 
            # local_JC_CX = 2 * local_JnX @ self.Psi( self.JpM(CX) ) 


            return gradf @ ( self.Ip + CX  ).T + local_JnX @ self.Psi( self.JpM( X.T @ gradf ) + 2* beta * self.JpM(CX) ) 

            # return local_JA_gradf + beta * local_JC_CX

        return local_grad  








    # def generate_cdf_hess(self, obj_grad, obj_hess, beta):
    #     # return a function that calculates the hessian-matrix product 
    #     def local_hess(X, D):
    #         CX = self.C(X)
    #         local_JnD = self.JnX(D)
    #         local_JnX = self.JnX(X)
    #         AX = X @ ( self.Ip + 0.5 * CX  )
    #         gradf = obj_grad(AX)



    #         # local_JAT_D = G @ ( 1.5*self.Ip + 0.5 * self.JpM(X.T @ self.JnX(X) )  ) +  X@ self.JpM( self.Psi( X.T @ self.JnX(G) ) ) 
    #         local_JAT_D = D @(self.Ip + 0.5 * CX) + X@ self.JpM( self.Psi( X.T @ local_JnD ) ) 
    #         local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
    #         local_JA_objhess_JAT_D = local_objhess_JAT_D @ ( self.Ip + 0.5 * CX  ).T + local_JnX @ self.Psi( self.JpM( X.T @ local_objhess_JAT_D ) )
            
            
    #         # local_JA_objhess_JAT_D = self.JA( X, obj_hess( AX, local_JAT_D ) )
    #         # local_hessA_objgrad_D = self.hessA(X, gradf, D)
    #         local_hessA_objgrad_D = local_JnX @ self.Psi( self.JpM(D.T @ gradf ) ) + local_JnD @ self.Psi( self.JpM(X.T @ gradf) ) + gradf @ ( self.JpM( self.Psi( D.T @ local_JnX ) ) )
            
    #         # local_hess_feas = self.hess_feas(X,D)
    #         local_hess_feas = 2 *  local_JnD @  self.JpM( CX ) - 4 * local_JnX @ self.Psi( D.T @ local_JnX )
            
    #         return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas
        
    #     return local_hess 


    def generate_cdf_hess(self, obj_grad, obj_hess, beta):
        # return a function that calculates the hessian-matrix product 
        def local_hess(X, D):
            CX = self.C(X)
            local_JnD = self.JnX(D)
            local_JnX = self.JnX(X)
            AX = X @ ( self.Ip + 0.5 * CX  )
            gradf = obj_grad(AX)



            # local_JAT_D = G @ ( 1.5*self.Ip + 0.5 * self.JpM(X.T @ self.JnX(X) )  ) +  X@ self.JpM( self.Psi( X.T @ self.JnX(G) ) ) 
            local_JAT_D = D @(self.Ip + 0.5 * CX) + X@ self.JpM( self.Psi( X.T @ local_JnD ) ) 
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            # local_JA_objhess_JAT_D = local_objhess_JAT_D @ ( self.Ip + 0.5 * CX  ).T + local_JnX @ self.Psi( self.JpM( X.T @ local_objhess_JAT_D ) )
            
            
            # local_JA_objhess_JAT_D = self.JA( X, obj_hess( AX, local_JAT_D ) )
            # local_hessA_objgrad_D = self.hessA(X, gradf, D)
            # local_hessA_objgrad_D = local_JnX @ self.Psi( self.JpM(D.T @ gradf ) ) + local_JnD @ self.Psi( self.JpM(X.T @ gradf) ) + gradf @ ( self.JpM( self.Psi( D.T @ local_JnX ) ) )
            
            # local_hess_feas = self.hess_feas(X,D)
            # local_hess_feas = 2 *  local_JnD @  self.JpM( CX ) - 4 * local_JnX @ self.Psi( D.T @ local_JnX )
            

            return local_objhess_JAT_D @ ( self.Ip + 0.5 * CX  ).T + local_JnX @ self.Psi( - 4 * beta * D.T @ local_JnX + self.JpM( X.T @ local_objhess_JAT_D + D.T @ gradf )  ) + local_JnD @ (self.Psi( self.JpM(X.T @ gradf) ) + 2* beta *  self.JpM( CX )) + gradf @ ( self.JpM( self.Psi( D.T @ local_JnX ) ) )


            # return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas
        
        return local_hess


        

    def generate_cdf_hess_approx(self, obj_grad, obj_hess, beta):
        # return a function that calculates the hessian-matrix product 
        # approximated to enhance the performance
        def local_hess(X, D):
            CX = self.C(X)
            local_JnX = self.JnX(X)
            local_JnD = self.JnX(D)
            AX = X @ ( self.Ip + 0.5 * CX  )
            gradf = obj_grad(AX)



            # local_JAT_D = G @ ( 1.5*self.Ip + 0.5 * self.JpM(X.T @ self.JnX(X) )  ) +  X@ self.JpM( self.Psi( X.T @ self.JnX(G) ) ) 
            local_JAT_D = D  + X@ self.JpM( self.Psi( X.T @ local_JnD ) ) 
            local_objhess_JAT_D = obj_hess(AX, local_JAT_D)
            local_JA_objhess_JAT_D = local_objhess_JAT_D  + local_JnX @ self.Psi( self.JpM( X.T @ local_objhess_JAT_D ) )
            
            
            # local_JA_objhess_JAT_D = self.JA( X, obj_hess( AX, local_JAT_D ) )
            # local_hessA_objgrad_D = self.hessA(X, gradf, D)

            local_hessA_objgrad_D = local_JnX @ self.Psi( self.JpM(D.T @ gradf ) ) + local_JnD @ self.Psi( self.JpM(X.T @ gradf) ) + gradf @ ( self.JpM( self.Psi( D.T @ local_JnX ) ) )
            
            # local_hess_feas = self.hess_feas(X,D)
            local_hess_feas = 2 *  local_JnD @  self.JpM( CX ) - 4 * local_JnX @ self.Psi( D.T @ local_JnX )
            
            return local_JA_objhess_JAT_D + local_hessA_objgrad_D + beta * local_hess_feas
        
        return local_hess