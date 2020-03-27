import numpy as np
import torch

from LM import PoseFunctionBase, DiffLM, DampingFunction


eps=1e-12

# Create a skew-symmetric matrix "S" of size [B x 3 x 3] (passed in) given a [B x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    vec = vector.contiguous().view(N, 3)
    output = vec.new().resize_(N, 3, 3).fill_(0).type_as(vector)
    output[:, 0, 1] = -vec[:, 2]
    output[:, 1, 0] =  vec[:, 2]
    output[:, 0, 2] =  vec[:, 1]
    output[:, 2, 0] = -vec[:, 1]
    output[:, 1, 2] = -vec[:, 0]
    output[:, 2, 1] =  vec[:, 0]
    return output.type_as(vector)

########
#### Translation-Angle (SO(3) + translation) representation helpers

# Compute the rotation matrix R from the angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# From https://github.com/abyravan/se3posenets-pytorch/blob/master/layers/SE3ToRt.py
def create_rot_from_angle(angle):
    # Get the un-normalized axis and angle
    N = angle.size(0)
#     axis = params.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (angle * angle).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle_norm = torch.sqrt(angle2)  # Angle

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(angle)
    K2 = torch.bmm(K, K)  # K * K

    # Compute sines
    S = torch.sin(angle_norm) / angle_norm
    S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

    # Compute cosines
    C = (1 - torch.cos(angle_norm)) / angle2
    C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(angle)  # R = I (avoid use expand as it does not allocate new memory)
    rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
    rot += K2 * C.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2
    return rot


def transform_pc(x, rot, transl):
    """
        Args:
            x (torch.Tensor): point cloud X. Shape: (N, 3)
            rot (torch.Tensor): SO(3) rota. Shape: (3, 3)
            transl (torch.Tensor): translation + SO(3). Shape: (3)
    """
    return torch.bmm(x.unsqueeze(1), rot.T.unsqueeze(0).repeat(x.size(0),1,1)) + transl


def get_J(x, y, params):
    transl, angle = params.split([3,3], dim=-1)
    rot = create_rot_from_angle(angle.unsqueeze(0)).squeeze(0)
    x_t = transform_pc(x, rot, transl)
    x_t.squeeze_(1)
    R = y - x_t
    skew = create_skew_symmetric_matrix(x_t)
    J = torch.cat((torch.eye(skew.size(1)).type_as(x).unsqueeze(0).repeat(skew.size(0),1,1), -skew), dim=-1)
    return J.flatten(start_dim=0, end_dim=1)


class PoseEst(PoseFunctionBase):
    def __init__(self, x, y, init_params):
        """
        Args:
            x (torch.Tensor): point cloud X. Shape: (N, 3)
            y (torch.Tensor): point cloud Y. Shape: (N, 3)
            init_params (torch.Tensor): inital_estimate. Shape: (6). [tranlsation, omega] for translation + SO(3)
        """
        self.x = x
        self.y = y
        if init_params is None:
            self.params = torch.rand()
        else:
            self.params = init_params

    def value(self):
        return self.evaluate(self.x)

    def jacobian(self):
        J = get_J(self.x, self.y, self.params)
        return J

    def evaluate(self, x):
        transl, angle = self.params.split([3,3], dim=-1)
        rot = create_rot_from_angle(angle.unsqueeze(0)).squeeze(0)
        x_t = transform_pc(x, rot, transl)
        x_t.squeeze_(1)
        x_t = x_t.flatten()
        return x_t