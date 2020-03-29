import torch
import copy

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


def align_pc(x, y, params):
    """
    Fit using implemented differentiable LM
        Args:
            x (torch.Tensor): point cloud X. Shape: (N, 3)
            y (torch.Tensor): point cloud Y. Shape: (N, 3)
            params (torch.Tensor): initial estimate. Shape: (6)
        Returns:
            torch.Tensor: params (translation + angle) to get y from x. Shape: (6)
    """
#     params = torch.DoubleTensor(6).fill_(0)
    s = PoseEst(x, y, init_params=params)
    d = DampingFunction(lam_min=0.1, lam_max=1, D=1, sigma=1e-5)
    solver = DiffLM(y=y.flatten(), function=s, decision_function=d, tol=1e-5, max_iter=3)
    f = solver.optimize(verbose=False)
    return f.params



pc2p = copy.copy(pc2)
for i in range(20):
    print("params", params)
    transl, angle = params.split([3,3], dim=-1)
    rot = create_rot_from_angle(angle.unsqueeze(0)).squeeze(0)
    pc2p.points = transform_pc(pc1.points.double(), rot, transl).squeeze(1)
    pc2p_ind = pc2p.project.round().long()
    pc2p_valid_mask = torch.all((torch.tensor([0, 0]) <= pc2p_ind) & (pc2p_ind < torch.tensor([1920, 1080])), dim=1) # filter out
    pc2p_ind = pc2p_ind[pc2p_valid_mask]
    
    mask = torch.zeros_like(pc2.depth)
    mask[pc2p_ind[:,1],pc2p_ind[:,0]] = 1
    depth2_mask = pc2.mask.reshape(pc2.depth.shape)
    joint_mask = mask.bool()*depth2_mask # this mask is on the depth of camera 2

    x = pc1.points[joint_mask[pc2p_ind[:,1], pc2p_ind[:,0]]] # indexing in mask
    mask_y = pc2p_ind[joint_mask[pc2p_ind[:,1], pc2p_ind[:,0]]]
    y = pc2.unmasked_points.reshape(*pc2.depth.shape,3)[mask_y[:,1], mask_y[:,0]]
    params = align_pc(x.double(), y.double(), params)
