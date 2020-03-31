import torch as th
from .io import load_depth_synthetic, load_matrix_synthetic

class Cloud:
    def __init__(self):
        self.points = None
        self.intrinsic = None
        self.depth = None
        self.extrinsic = None
        self.mask = None
        self.unmasked_points = None

    @staticmethod
    def from_depth_file(path_to_depth, path_to_intrinsic, path_to_extrinsic, ignore='max'):
        depth = load_depth_synthetic(path_to_depth)
        intrinsic = load_matrix_synthetic(path_to_intrinsic)
        extrinsic = load_matrix_synthetic(path_to_extrinsic)
        self = Cloud.from_tensors(depth, intrinsic, extrinsic, ignore=ignore)
        return self
    
    def update_data(self, depth, intrinsic, extrinsic, ignore='max', update_mask=True):
        self.depth = depth
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        if update_mask:
            self.unmasked_points, self.mask = self.extrude_cloud(depth, ignore)
        else:
            self.unmasked_points, _ = self.extrude_cloud(depth, ignore)
        self.points = self.unmasked_points[self.mask]
        
    @staticmethod
    def from_tensors(depth, intrinsic, extrinsic, ignore='max'):
        if len(depth.shape) == 2:
            depth = depth.squeeze()
        self = Cloud()
        self.update_data(depth, intrinsic, extrinsic, ignore)
        return self

    def transform(self, T):
        ret = th.cat((self.points, th.ones_like(self.points[:, :1])), dim=-1).unsqueeze(-1)
        T_expanded = T.expand(ret.shape[0], -1, -1)
        ret = th.bmm(T_expanded, ret).squeeze(-1)
        if ret.shape[-1] == 4:
            ret = ret[:, :3]
        return ret
        
    @property
    def align(self):
        return self.transform(th.inverse(self.extrinsic))

    @property
    def project(self):
        assert self.points is not None, 'Cloud is not initialized properly. Consider calling from_depth_file.'
        assert self.intrinsic is not None, 'Intrinsics not found. Guess cloud is not initialized properly. ' \
                                       'Consider calling from_depth_file.'

        projected = th.bmm(self.intrinsic.unsqueeze(0).expand(self.points.shape[0], 3, 3),
                           self.points.unsqueeze(-1)).squeeze(-1)
        projected = projected[:, :2]/projected[:, 2, None]
        return projected

    def extrude_cloud(self, dpth, ignore=None):
        Intr = self.intrinsic
        M = th.inverse(Intr)
        H, W = dpth.shape
        uv = get_coords_list(H, W).to(dpth)
        uv *= dpth[..., None]
        uv = th.cat([uv, dpth[..., None]], dim=-1).flatten(start_dim=0, end_dim=1).unsqueeze(-1)
        M = M.unsqueeze(0).expand(uv.shape[0], 3, 3)
        xyz = th.bmm(M, uv).squeeze(-1)
        mask = None
        if ignore is not None:
            if isinstance(ignore, th.Tensor):
                mask_coeff = ignore.squeeze()
            elif ignore == 'min':
                mask_coeff = xyz[:, 2].min()
            elif ignore == 'max':
                mask_coeff = xyz[:, 2].max()
            else:
                raise ValueError('Unknown ignore type. Only min, max or float number are accepted.')
            mask = (xyz[:, 2] != mask_coeff)

        # return all points, unmasked
        return xyz, mask


def get_coords_list(H, W):
    uv = th.flip(th.stack(th.meshgrid(th.arange(H), th.arange(W)), axis=-1).double(), (-1,))
    return uv