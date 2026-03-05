import numpy as np
import torch
from utils.rotation_conversions import (
    rotation_6d_to_axis_angle, 
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle
)
import os

def to_torch(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    else:
        raise TypeError("Input should be a torch.Tensor or a numpy.ndarray")


class InterX2SMPL:
    """ The default input shape is [T, 56, 12] """
    def __init__(self):
        pass

    def interx2smpl(self, interx_motion):
        pass
    
    
def interx2smpl(interx_motion,                             
                name:str,
                folder:str = './',
                gender='neutral', 
                betas=torch.zeros(16,), fps=30, dataset='test'):
    """ The default input shape is [T, 56, 12] with no normalization"""
    interx_motion = to_torch(interx_motion)
    poses1 = rotation_6d_to_axis_angle(interx_motion[:, :-1, :6])
    poses2 = rotation_6d_to_axis_angle(interx_motion[:, :-1, 6:])
    trans1 = interx_motion[:, -1, :3]
    trans2 = interx_motion[:, -1, 6:9]
    
    dict1 = {
        'poses': poses1.numpy().reshape(-1, 3*55),
        'trans': trans1.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    dict2 = {
        'poses': poses2.numpy().reshape(-1, 3*55),
        'trans': trans2.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    np.savez(os.path.join(folder, name+'_p1.npz'), **dict1)
    np.savez(os.path.join(folder, name+'_p2.npz'), **dict2)
    
    
def aa2smpl(aa_motion, 
            name:str, 
            folder:str = './', 
            gender='neutral', 
            betas=torch.zeros(16,), 
            fps=30, 
            dataset='test'):
    interx_motion = to_torch(aa_motion)
    poses1 = (interx_motion[:, :-1, :3])
    poses2 = (interx_motion[:, :-1, 3:])
    trans1 = interx_motion[:, -1, :3]
    trans2 = interx_motion[:, -1, 3:]
    
    dict1 = {
        'poses': poses1.numpy().reshape(-1, 3*55),
        'trans': trans1.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    dict2 = {
        'poses': poses2.numpy().reshape(-1, 3*55),
        'trans': trans2.numpy(),
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset': dataset
    }
    
    np.savez(os.path.join(folder, name+'_p1.npz'), **dict1)
    np.savez(os.path.join(folder, name+'_p2.npz'), **dict2)