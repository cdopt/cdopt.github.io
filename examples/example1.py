import torch
import cdopt 
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.sparse import csr_matrix
import time

# Generating the data
n = 30        
m = 10*n**2   
theta = 0.3   

# set torch device and data  type
device = torch.device('cuda')  
dtype = torch.float64  

Y= torch.as_tensor(norm.ppf(np.random.rand(m,n)) * (norm.ppf(np.random.rand(m,n)) <= theta), 
                    device=device, dtype=dtype)


# Define objective functions and the Stiefel manifold
def obj_fun(X):
    return -torch.sum(torch.matmul(Y, X) **4 )

M = cdopt.manifold_torch.stiefel_torch((n,n), device= device, dtype= dtype)   

# describe the optimization problem and set the penalty parameter \beta.
problem_test = cdopt.core.Problem(M, obj_fun, beta = 500)  

# the vectorized function value, gradient and Hessian-vector product
#  of the constraint dissolving function. Their inputs are numpy 1D 
# array, and their outputs are float or numpy 1D array.
cdf_fun_np = problem_test.cdf_fun_vec_np   
cdf_grad_np = problem_test.cdf_grad_vec_np 
cdf_hvp_np = problem_test.cdf_hvp_vec_np


## Apply limit memory BFGS solver from scipy.minimize 
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_l_bfgs_b, fmin_ncg
Xinit = M.tensor2array(M.Init_point())  # set initial point

# optimize by L-BFGS method
out_msg = sp.optimize.minimize(cdf_fun_np, Xinit.flatten(),
                                method='L-BFGS-B',jac = cdf_grad_np)

