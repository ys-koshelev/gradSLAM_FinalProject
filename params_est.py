import open3d as o3d

import torch
import copy

import numpy as np
from utils.cloud import Cloud

path_to_depth = '../stylized-ww01-plane/blender_render_depth4/1.npy'
path_to_intrinsic = '../stylized-ww01-plane/blender_render_depth4/K_1.txt'
path_to_extrinsic = '../stylized-ww01-plane/blender_render_depth4/RT_1.txt'
pc1 = Cloud().from_depth_file(path_to_depth, path_to_intrinsic, path_to_extrinsic, ignore='max')

path_to_depth = '../stylized-ww01-plane/blender_render_depth4/2.npy'
path_to_intrinsic = '../stylized-ww01-plane/blender_render_depth4/K_2.txt'
path_to_extrinsic = '../stylized-ww01-plane/blender_render_depth4/RT_2.txt'
pc2 = Cloud().from_depth_file(path_to_depth, path_to_intrinsic, path_to_extrinsic, ignore='max')

from LM import PoseFunctionBase, DiffLM, DampingFunction
from pose_est import PoseEst, create_rot_from_angle, transform_pc


def align_pc(x, y, params, lam_min=0.1, lam_max=1, D=1, sigma=1e-5):
    """
    Fit using implemented differentiable LM
        Args:
            x (torch.Tensor): point cloud X. Shape: (N, 3)
            y (torch.Tensor): point cloud Y. Shape: (N, 3)
            params (torch.Tensor): initial estimate. Shape: (6)
        Returns:
            torch.Tensor: params (translation + angle) to get y from x. Shape: (6)
    """
    
    s = PoseEst(x, y, init_params=params)
    d = DampingFunction(lam_min, lam_max, D, sigma)
    solver = DiffLM(y=y.flatten(), function=s, decision_function=d, tol=1e-7, max_iter=100)
    f = solver.optimize(verbose=False)
    return f.params



R = pc2.extrinsic@torch.inverse(pc1.extrinsic)
rot = R[:3,:3]
transl = R[:3,-1]

pcA = o3d.geometry.PointCloud()
pcA.points = o3d.utility.Vector3dVector(pc1.points)

pcB = o3d.geometry.PointCloud()
pcB.points = o3d.utility.Vector3dVector(pc2.points)


reg_p2l = o3d.registration.registration_icp(pcA, pcB, 0.2, R)

pc1_idx, pc2_idx = np.asarray(reg_p2l.correspondence_set).T

x = pc1.points[pc1_idx]
y = pc2.points[pc2_idx]

params = torch.DoubleTensor(6).fill_(0)
params = align_pc(x.double(), y.double(), params, lam_min=0.01, lam_max=1, D=1, sigma=1e-5)

