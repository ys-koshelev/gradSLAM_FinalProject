import cv2
import torch as th
import numpy as np
import json


def load_intrinsics_real(file_path):
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


def load_matrix_synthetic(path):
    ret = np.loadtxt(path)
    ret = th.Tensor(ret)
    return ret


def load_depth_synthetic(depth_path):
    depth = np.load(depth_path)
    return th.Tensor(depth).squeeze()


def load_depth_synthetic_old(depth_path, minmax_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.
    max_depth, min_depth = np.loadtxt(minmax_path)
    depth = depth * (max_depth - min_depth) + min_depth
    return th.Tensor(depth)