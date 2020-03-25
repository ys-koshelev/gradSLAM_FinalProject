# import open3d as o3d 

import os
import numpy as np
import torch

from pose_est import PoseEst
from LM import PoseFunctionBase, DiffLM, DampingFunction

def check_grad_wrt_y(y):
    # Fit using implemented differentiable LM
    params = torch.DoubleTensor(6).fill_(0)
    x = torch.DoubleTensor(y.shape).fill_(1)
    s = PoseEst(x, y, init_params=params)
    d = DampingFunction(lam_min=0.1, lam_max=1, D=1, sigma=1e-5)
    solver = DiffLM(y=y.flatten(), function=s, decision_function=d, tol=1e-5, max_iter=2)
    f = solver.optimize(verbose=False)
    return f.params.sum()

def check_grad_wrt_x(x):
    # Fit using implemented differentiable LM
    params = torch.DoubleTensor(6).fill_(0)
    y = torch.DoubleTensor(x.shape).fill_(1)
    s = PoseEst(x, y, init_params=params)
    d = DampingFunction(lam_min=0.1, lam_max=1, D=1, sigma=1e-5)
    solver = DiffLM(y=y.flatten(), function=s, decision_function=d, tol=1e-5, max_iter=2)
    f = solver.optimize(verbose=False)
    
x = torch.rand(2, 3).double()
x.requires_grad = True
print("check_grad_wrt_x:", torch.autograd.gradcheck(check_grad_wrt_x, (x)))

y = torch.rand(2, 3).double()
y.requires_grad = True
print("check_grad_wrt_y:", torch.autograd.gradcheck(check_grad_wrt_y, (y)))