import random

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from models.reactor.mar2dema.encdec import Decoder, Encoder
from models.reactor.mar2dema.marquant2dema import MARQuantizer2DEMA
from omegaconf import OmegaConf
from torch.functional import F
from utils.utils import MotionNormalizerTorch
from models.utils.sparse_loss import masked_l1_loss_2d, masked_smooth_l1_loss_2d, weighted_l1_loss_2d
from copy import deepcopy

""" to make it easy, we use normalize = False for Interhuman dataset, 
    and for its forward_test, we make it back to normlaized result. 
"""

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def backward(self, x):
        return x

class MARVQVAE2D(LightningModule):
    def __init__(self, cfg):
        """ cfg: full cfg, should contains: [vqvae, dataset]"""

        super().__init__()
        
        self.cfg = cfg
        
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        self.code_dim = cfg.vqvae.code_dim
        self.num_code = cfg.vqvae.nb_code 
        self.quant_conv_ks = cfg.vqvae.quant_conv_ks
        
        self.dataset_name = cfg.dataset.name
        self.joints_num = cfg.dataset.joints_num
        
        self.is_normal = cfg.vqvae.is_normal
        self.beta = cfg.vqvae.beta
        self.lambda_vq = cfg.vqvae.lambda_vq
        self.lambda_joint = cfg.vqvae.lambda_joint
        
        self.normalizer = MotionNormalizerTorch() if self.is_normal else IdentityLayer()
        
        self.use_t_first = False if not hasattr(cfg.vqvae, 'use_t_first') else cfg.vqvae.use_t_first
        if self.use_t_first:
            cfg.vqvae.s_scale_numbers, cfg.vqvae.t_scale_numbers = cfg.vqvae.t_scale_numbers, cfg.vqvae.s_scale_numbers
            self.cfg = cfg

        self.encoder = nn.ModuleList([Encoder(cfg.vqvae.input_dim, 
                               cfg.vqvae.output_emb_width, 
                               cfg.vqvae.down_t, 
                               cfg.vqvae.stride_t, 
                               cfg.vqvae.width, 
                               cfg.vqvae.depth,
                               cfg.vqvae.dilation_growth_rate, 
                               activation=cfg.vqvae.activation, 
                               norm=cfg.vqvae.norm, 
                               filter_s=cfg.vqvae.filter_s, 
                               stride_s=cfg.vqvae.stride_s,
                               use_t_first=self.use_t_first) for _ in range(2)])
        self.decoder = nn.ModuleList([Decoder(cfg.vqvae.input_dim, 
                               cfg.vqvae.output_emb_width, 
                               cfg.vqvae.down_t, 
                               cfg.vqvae.stride_t, 
                               cfg.vqvae.width, 
                               cfg.vqvae.depth,
                               cfg.vqvae.dilation_growth_rate, 
                               activation=cfg.vqvae.activation, 
                               norm=cfg.vqvae.norm,
                               spatial_upsample=cfg.vqvae.spatial_upsample,
                               use_t_first=self.use_t_first) for _ in range(2)])
        
        self.quantizer = nn.ModuleList([MARQuantizer2DEMA(cfg = cfg.vqvae) for _ in range(2)]) # for vqvae only.
        
        self.quant_conv = nn.ModuleList([torch.nn.Conv2d(self.code_dim, self.code_dim, 
                                          self.quant_conv_ks, stride=1, padding=self.quant_conv_ks//2) for _ in range(2)])
        self.post_quant_conv = nn.ModuleList([torch.nn.Conv2d(self.code_dim, self.code_dim, 
                                                self.quant_conv_ks, stride=1, padding=self.quant_conv_ks//2) for _ in range(2)])
        
        self.sparse_weight = cfg.vqvae.sparse_weight if hasattr(cfg.vqvae, 'sparse_weight') else 0
        
    
    def preprocess(self, x):
        """ x input is in the shape of B, T, 2D, each D is in the shape 
        262 = [global pos 22*3, global vel 22*3, global rot 21*6, face center 4]
        return : B, D=12, J=22 * 2, T
        """
        if self.dataset_name == "interhuman":
            D = self.joints_num*3 + self.joints_num*3 + (self.joints_num-1)*6 + 4
            x = x[:,:,:D]            
            pos = x[:,:,:3*self.joints_num].reshape([x.shape[0], x.shape[1], -1, 3])
            vel = x[:,:,3*self.joints_num:6*self.joints_num].reshape([x.shape[0], x.shape[1], -1, 3])
            rot = x[:,:,6*self.joints_num:6*self.joints_num+6*(self.joints_num-1)].reshape([x.shape[0], x.shape[1], -1, 6])
            rot = torch.cat([torch.zeros(rot.shape[0], rot.shape[1], 1, 6).to(x.device), rot], dim=2)
            joints = torch.cat([pos, vel, rot], dim=-1) # B, T, J=22, D=12
         
            joints = joints.reshape(joints.shape[0], joints.shape[1], self.joints_num, 12)
        else:
            joints = x
        joints = joints.permute(0, 3, 2, 1).float() # B, D=12, J=22, T 
 
        return joints
    
    

    def postprocess(self, x):
        """ input should be in the shape of B, D, J, T, output should be
        B, T, D; the result will be normalized. 
        """
        x = x.permute(0, 3, 2, 1).float() # to B, T, J=22*2, D=12
        if self.dataset_name == "interhuman":
            pos = x[:,:,:,:3].reshape([x.shape[0], x.shape[1], -1])
            vel = x[:,:,:,3:6].reshape([x.shape[0], x.shape[1], -1])
            rot = x[:,:,1:,6:6+6].reshape([x.shape[0], x.shape[1], -1])
            fc = torch.zeros((x.shape[0], x.shape[1], 4)).to(x.device)
            x = torch.cat([pos, vel, rot, fc], dim=-1)
        elif self.dataset_name == "interx":
            x = x.reshape([x.shape[0], x.shape[1], -1])
        else:
            raise ValueError
        return x
    
    
    
    
    def encode(self, x, idx=0):
        # B, T, D

        with torch.no_grad():
            x_in = self.preprocess(x) # B, D=12, J=22, T || B, J=22xD=12, T
            
        x_encoder = self.encoder[idx](x_in) # B, D=512, J/4, T/4
        x_encoder = x_encoder[idx]
        return x_encoder
    
    
    def decode(self, x, idx=0):
        return self.decoder[idx](self.post_quant_conv[idx](x))
    
    
    def forward(self, inp, length, idx=0):
        """ inp in shape of b, l, 2d length in shape of b, """
        inp = self.preprocess(inp)
        inp_ori = inp.clone()
        encoded = self.quant_conv[idx](self.encoder[idx](inp))
        mask = self.get_motion_mask(length, inp, encoded)
        f_hat, vq_loss, ppl = self.quantizer[idx].forward(encoded, mask=mask)

        return self.decoder[idx](self.post_quant_conv[idx](f_hat)), inp_ori, vq_loss, ppl
    
    
    def get_motion_mask(self, length, inp, enc):
        # return a mask of [b, sn, tn] 1 for valid, 0 for invalid
        if length is None: return length
        b, _, _, l = inp.shape
        b, _, sn, tn = enc.shape
        mask = torch.arange(tn, device=length.device).unsqueeze(0).unsqueeze(1)  # [1, 1, tn]
        down_rate = l // tn
        mask = mask.expand(b, sn, tn) < ((length + down_rate - 1) // down_rate).view(b, 1, 1)
        mask = mask.float()
        return mask
    
    def motion_to_idxBl(self, motions: torch.Tensor, idx=0):
        """
        motions in shape of B, T, D=12, J=22*2
        return List[B,l]
        """
        motions = self.preprocess(motions)
        f = self.quant_conv[idx](self.encoder[idx](motions))
        return self.quantizer[idx].f_to_idxBl_or_fhat(f, to_fhat=False)
    
    def embed_to_motion(self, ms_h_bcjl, all_to_max_scale=True, last_one=True, idx=0):
        if last_one:
            return self.decoder[idx](self.post_quant_conv[idx](
                self.quantizer[idx].embed_to_fhat(ms_h_bcjl, 
                                              all_to_max_scale=all_to_max_scale, 
                                              last_one=True)))
        else:
            return [self.decoder[idx](self.post_quant_conv[idx](f_hat)) \
                for f_hat in self.quantizer[idx].embed_to_fhat(ms_h_bcjl, 
                                                          all_to_max_scale=all_to_max_scale, 
                                                          last_one=False)]
    
    
    def idxBl_to_motion(self, ms_idx_Bl:torch.Tensor, same_shape:bool=True, last_one:bool=True, idx=0):
        """ decode process 
        Args:
            ms_idx_Bl (torch.Tensor): _description_
            same_shape (bool, optional): . Defaults to True.
            last_one (bool, optional): _description_. Defaults to True.

        Returns:
            motion: [B, C, 2J, L]
        """
        B = ms_idx_Bl[0].shape[0]
        ms_h_bcjl = []
        
        lengths = [s * t for s,t in self.quantizer[idx].scale_numbers]
        
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            si = lengths.index(l)
            sn, tn = self.quantizer[idx].scale_numbers[si]
            ms_h_bcjl.append(self.quantizer[idx].embedding(idx_Bl).transpose(1, 2).view(B, self.code_dim, sn, tn))
        return self.embed_to_motion(ms_h_bcjl=ms_h_bcjl, all_to_max_scale=same_shape, last_one=last_one, idx=idx)
    
    
    
    def motion_to_reconstruct(self, inp, length=None, idx=0):
        """ x in shape of [B, L, 2C] unnormed motion """
        self.eval()
        inp = self.preprocess(inp)
        encoded = self.quant_conv[idx](self.encoder[idx](inp))
        mask = self.get_motion_mask(length, inp, encoded)
        f_hat, _, _ = self.quantizer[idx].forward(encoded, mask=mask)
        
        return self.postprocess(self.decoder[idx](self.post_quant_conv[idx](f_hat)))


    def training_step(self, batch):
        batch1 = batch.copy()
        batch2 = batch.copy()
        batch1['motions'] = batch1['motions'][:, :, :262]
        batch2['motions'] = batch2['motions'][:, :, 262:]
        loss1 = self.training_step_idx(batch1, 0)
        loss2 = self.training_step_idx(batch2, 1)
        return (loss1 + loss2) / 2

    def training_step_idx(self, batch, idx=0):
        motions = batch['motions'] # B, T, 2xD
        lengths = batch['motion_lens']
        motion_rec, inp, vq_loss, ppl = self.forward(motions, lengths, idx=idx)
        if self.use_t_first:
            motion_rec = motion_rec.permute(0, 1, 3, 2)
            inp = inp.permute(0, 1, 3, 2)
        inp = inp[:, :, :, :motion_rec.shape[3]]
        rec_loss = weighted_l1_loss_2d(motion_rec, inp, lengths, weights=self.sparse_weight)        
        
        loss = rec_loss + self.lambda_vq * vq_loss
        
        joint_loss = weighted_l1_loss_2d(motion_rec[:, :3, :, :], inp[:, :3, :, :], lengths)
        loss = loss + joint_loss * self.lambda_joint
        
        self.log(f'train_{idx}/joint_loss', joint_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'train_{idx}/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'train_{idx}/rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'train_{idx}/vq_loss', vq_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'train_{idx}/ppl', ppl, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch):
        batch1 = batch.copy()
        batch2 = batch.copy()
        batch1['motions'] = batch1['motions'][:, :, :262]
        batch2['motions'] = batch2['motions'][:, :, 262:]
        loss1 = self.validation_step_idx(batch1, 0)
        loss2 = self.validation_step_idx(batch2, 1)
        return (loss1 + loss2) / 2

    def validation_step_idx(self, batch, idx=0):
        motions = batch['motions']
        lengths = batch['motion_lens']
        motion_rec, inp, vq_loss, ppl = self.forward(motions, lengths, idx=idx)
        if self.use_t_first:
            motion_rec = motion_rec.permute(0, 1, 3, 2)
            inp = inp.permute(0, 1, 3, 2)
        inp = inp[:, :, :, :motion_rec.shape[3]]
        rec_loss = weighted_l1_loss_2d(motion_rec, inp, lengths, weights=self.sparse_weight)
        joint_loss = weighted_l1_loss_2d(motion_rec[:, :3, :, :], inp[:, :3, :, :], lengths, weights=self.sparse_weight)
                  
        loss = rec_loss + self.lambda_vq * vq_loss + joint_loss * self.lambda_joint
        self.log(f'val_{idx}/joint_loss', joint_loss, on_step=False, on_epoch=True)
        self.log(f'val_{idx}/loss', loss, on_step=False, on_epoch=True)
        self.log(f'val_{idx}/rec_loss', rec_loss, on_step=False, on_epoch=True)
        self.log(f'val_{idx}/vq_loss', vq_loss, on_step=False, on_epoch=True)
        self.log(f'val_{idx}/ppl', ppl, on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.vqvae.train.lr)
        
        # Warm-up scheduler for the first 10 epochs 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, total_iters=1
        )
        
        # Exponential decay scheduler to decay LR by 0.5 every 500 epochs
        gamma = 0.5 ** (1 / 500)
        decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )
        
        # Combine schedulers: first warm-up, then exponential decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[10]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/val_loss'
        }