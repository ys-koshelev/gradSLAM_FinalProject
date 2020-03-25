import torch as th
import numpy as np
import json
import cv2

def load_intrinsics_from_file(file_path):
    assert file_path[-4:] == 'json', 'Only json files are allowed.'
    with open(file_path) as json_file:
        data = json.load(json_file)['intrinsic_matrix']
    I = th.zeros(3, 3)
    I[0, 0] = data[0]
    I[1, 1] = data[4]
    I[1, 1] = data[4]
    I[0, 2] = data[6]
    I[1, 2] = data[7]
    I[2, 2] = data[8]

    return I


def get_coords_list(H, W):
    uv = th.flip(th.stack(th.meshgrid(th.arange(H), th.arange(W)), axis=-1).float(), (-1,))
    return uv


def load_depth(dpth_path):
    img = cv2.imread(dpth_path, cv2.IMREAD_GRAYSCALE)
    img = th.FloatTensor(img) / 255
    return img


def get_cloud(dpth, Intr):
    M = th.inverse(Intr)
    H, W = dpth.shape
    uv = get_coords_list(H, W)
    uv *= dpth[..., None]
    uv = th.cat([uv, dpth[..., None]], dim=-1).flatten(start_dim=0, end_dim=1).unsqueeze(-1)
    M = M.unsqueeze(0).expand(uv.shape[0], 3, 3)
    xyz = th.bmm(M, uv).squeeze(-1)
    return xyz



