import torch    
from torch import nn
from utils.body_model import BodyModel
from utils.rotation_conversions import rotation_6d_to_axis_angle


class SMPL_Loss(nn.Module):
    def __init__(self):
        super(SMPL_Loss, self).__init__()
        self.l1_criterion = torch.nn.L1Loss()
        self.bm = BodyModel(bm_fname='./deps/body_models/smplx/neutral/model.npz')
        self.num_joints = 55
        self.num_joints_smpl = 22
        self.num_joints_hands = 30
        
        
        
    def forward_foot_contact(self, joints1: torch.Tensor, joints2: torch.Tensor):
        """ Foot contact loss,
        Args:
            joints1 (torch.Tensor): [B, T, 22, 3]
            joints2 (torch.Tensor): [B, T, 22, 3]
        """
        l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
        relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
        gt_joint_xyz = joints1[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        fc_mask = torch.unsqueeze((gt_joint_vel <= self.vel_threshold), dim=2).repeat(1, 1, 3*self.num_person, 1)
        pred_joint_xyz = joints2[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
        pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
        pred_vel[~fc_mask] = 0
        return self.l1_criterion(pred_vel,
                              torch.zeros(pred_vel.shape, device=pred_vel.device))

    def forward(self, motion1:torch.Tensor, motion2:torch.Tensor, weight_hands=0.5):
        """
        Args:
            motion1 (Tensor): [B, T, 56, 6]
            motion2 (Tensor): [B, T, 56, 6]

        Returns:
            Tensor: loss of 55 point
        """
        aa1 = rotation_6d_to_axis_angle(motion1[:,:,:-1,:6]).reshape(-1, 3*self.num_joints)
        aa2 = rotation_6d_to_axis_angle(motion2[:,:,:-1,:6]).reshape(-1, 3*self.num_joints)
        trans1 = motion1[:,:,-1:,:3].reshape(-1, 3)
        trans2 = motion2[:,:,-1:,:3].reshape(-1, 3)
        
        joints1 = self.bm.forward(
            root_orient=aa1[:,:3*1],
            pose_body=aa1[:,3*1:(3*self.num_joints_smpl)],
            pose_hand=aa1[:,-3*self.num_joints_hands:],
            trans=trans1
        ).Jtr
        joints2 = self.bm.forward(
            root_orient=aa2[:,:3*1],
            pose_body=aa2[:,3*1:(3*self.num_joints_smpl)],
            pose_hand=aa2[:,-3*self.num_joints_hands:],
            trans=trans2
        ).Jtr

        joints1_hands = joints1[:,:self.num_joints_smpl,:]
        joints2_hands = joints2[:,:self.num_joints_smpl,:]
        joints1_body = joints1[:,self.num_joints_smpl:,:]
        joints2_body = joints2[:,self.num_joints_smpl:,:]
        
        
        body_loss = self.l1_criterion.forward(joints1_body, joints2_body)
        hand_loss = self.l1_criterion.forward(joints1_hands, joints2_hands)
        return {'body_loss': body_loss, 'hand_loss': hand_loss}
    