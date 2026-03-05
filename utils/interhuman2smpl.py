# This script is used to translate the 262 representation to the smpl 69 representation. 

from utils.body_model import BodyModel
from utils.paramUtil import relax_hand_pose
from utils.joints2smpl import joints2smpl

import numpy as np
import torch

import os

J2S = joints2smpl(cuda=False)


def to_np_array(motion: torch.Tensor):
    if type(motion) is torch.Tensor:
        return motion.detach().cpu().numpy()
    elif type(motion) is np.ndarray:
        return motion
    else:
        raise TypeError

def to_torch(motion: np.ndarray):
    if type(motion) is torch.Tensor:
        return motion.detach().cpu()
    elif type(motion) is np.ndarray:
        return torch.Tensor(motion)
    else:
        raise TypeError    


def interhuman2smpl(motion:torch.Tensor, length:int=-1):
    """
    Args:
        motion (torch.Tensor | np.ndarray): [L, 262]
            3 * 22 +  22 * 3 + 21 * 6 +  + 4
    """
    smpl_data = J2S.joint2smpl(input_joints=motion[:length, :66].detach().cpu())
    return {'trans':smpl_data['trans'], 
            'poses':smpl_data['poses']}
    
    

def dump_interhuman_npz(motion, 
                        name:str,
                        folder:str = './',
                        gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
    smpl_data = interhuman2smpl(motion)
    smpl_data.update({
        'gender': gender,
        'betas': betas,
        'mocap_framerate': fps,
        'dataset':dataset
    })
    np.savez(os.path.join(folder, name+'.npz'), **smpl_data)
    
def dump_interhuman_cat_npzs(motions,
                             name:str,
                             folder:str = './',
                             gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
    """ motions in shape of [L, 262 * 2]"""
    motion1 = motions[:, :262]
    motion2 = motions[:, 262:]
    dump_interhumman_npzs(motion1, motion2, name, folder, gender, betas, fps, dataset)
    
    
def dump_interhumman_npzs(motion1,
                          motion2, 
                          name:str,
                          folder:str = './',
                          gender='neutral', betas=torch.zeros(16,), fps=30, dataset='test'):
    dump_interhuman_npz(motion1, name + 'p1', folder, gender, betas, fps, dataset)
    dump_interhuman_npz(motion2, name + 'p2', folder, gender, betas, fps, dataset)