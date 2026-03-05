import random

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from models.reactor.mar2d.vqencoder2d import VQEncoder2D, VQDecoder2D
from models.reactor.mar2d.marquant2d import MARQuantizer2D
from omegaconf import OmegaConf
from torch.functional import F
from utils.utils import MotionNormalizerTorch


""" This is the version that use the vqencoder to encode the motion instead of motion encoder"""

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def backward(self, x):
        return x

class MARVQVAE2D2(LightningModule):
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
        
        self.normalizer = MotionNormalizerTorch() if self.is_normal else IdentityLayer()


        self.encoder = VQEncoder2D(
            d_input=cfg.vqvae.d_input,
            ch=cfg.vqvae.channels,
            ch_mult=cfg.vqvae.ch_mult_encoder,
            num_res_blocks=cfg.vqvae.num_res_blocks,
            dropout=cfg.vqvae.dropout,
            z_channels=cfg.vqvae.z_channels,
            use_attention=cfg.vqvae.use_attention
        )
        self.decoder = VQDecoder2D(
            d_output=cfg.vqvae.d_input,
            ch=cfg.vqvae.channels,
            ch_mult=cfg.vqvae.ch_mult_decoder,
            num_res_blocks=cfg.vqvae.num_res_blocks,
            dropout=cfg.vqvae.dropout,
            z_channels=cfg.vqvae.z_channels,
            use_attention=cfg.vqvae.use_attention
        )
        
        self.quantizer = MARQuantizer2D(cfg = cfg.vqvae) # for vqvae only.
        
        self.quant_conv = torch.nn.Conv2d(self.code_dim, self.code_dim, 
                                          self.quant_conv_ks, stride=1, padding=self.quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.code_dim, self.code_dim, 
                                                self.quant_conv_ks, stride=1, padding=self.quant_conv_ks//2)
        
        
    def preprocess_old(self, x):
        """ x input is in the shape of B, T, 2D, each D is in the shape 
        262 = [global pos 22*3, global vel 22*3, global rot 21*6, face center 4]
        return : B, D=12, J=22 * 2, T
        """
        if self.dataset_name == "interhuman":
            D = self.joints_num*3 + self.joints_num*3 + (self.joints_num-1)*6 + 4
            x1 = self.normalizer.forward(x[:,:,:D])
            x2 = self.normalizer.forward(x[:,:,D:])
            x = torch.cat([x1, x2], dim=-1)
            
            pos_idx = np.r_[0:self.joints_num*3, D: D + self.joints_num*3]
            vel_idx = np.r_[self.joints_num*3: self.joints_num*3*2, 
                            D + self.joints_num*3: D + self.joints_num*3*2]
            rot_idx = np.r_[2 * 3* self.joints_num: 2 *  3 * self.joints_num + (self.joints_num-1)*6, 
                            D + self.joints_num*3*2: D + self.joints_num*3*2 + (self.joints_num-1)*6]
            pos = x[..., pos_idx].reshape([x.shape[0], x.shape[1], self.joints_num, 2, 3])
            vel = x[..., vel_idx].reshape([x.shape[0], x.shape[1], self.joints_num, 2, 3])
            rot = x[..., rot_idx].reshape([x.shape[0], x.shape[1], self.joints_num-1, 2, 6])
            rot = torch.cat([torch.zeros(rot.shape[0], rot.shape[1], 1, 2, 6).to(x.device), rot], dim=2) # B, T, J, 2, 6
                        
            joints = torch.cat([pos, vel, rot], dim=-1) # B, T, J, 2, 12
            joints = joints.reshape(joints.shape[0], joints.shape[1], 2 * self.joints_num, 12)
        else:
            joints = x
        joints = joints.permute(0, 3, 2, 1).float() # B, D=12, J=22 * 2, T 
 
        return joints
    
    def preprocess(self, x):
        """ x input is in the shape of B, T, 2D, each D is in the shape 
        262 = [global pos 22*3, global vel 22*3, global rot 21*6, face center 4]
        return : B, D=12, J=22 * 2, T
        """
        if self.dataset_name == "interhuman":
            D = self.joints_num*3 + self.joints_num*3 + (self.joints_num-1)*6 + 4
            x1 = x[:,:,:D]
            x2 = x[:,:,D:]
            
            pos1 = x1[:,:,:3*self.joints_num].reshape([x.shape[0], x.shape[1], -1, 3])
            vel1 = x1[:,:,3*self.joints_num:6*self.joints_num].reshape([x.shape[0], x.shape[1], -1, 3])
            rot1 = x1[:,:,6*self.joints_num:6*self.joints_num+6*(self.joints_num-1)].reshape([x.shape[0], x.shape[1], -1, 6])
            rot1 = torch.cat([torch.zeros(rot1.shape[0], rot1.shape[1], 1, 6).to(x.device), rot1], dim=2)
            pos2 = x2[:,:,:3*self.joints_num].reshape([x.shape[0], x.shape[1], -1, 3])
            vel2 = x2[:,:,3*self.joints_num:6*self.joints_num].reshape([x.shape[0], x.shape[1], -1, 3])
            rot2 = x2[:,:,6*self.joints_num:6*self.joints_num+6*(self.joints_num-1)].reshape([x.shape[0], x.shape[1], -1, 6])
            rot2 = torch.cat([torch.zeros(rot2.shape[0], rot2.shape[1], 1, 6).to(x.device), rot2], dim=2)
            
            joint1 = torch.cat([pos1, vel1, rot1], dim=-1) # B, T, J=22, D=12
            joint2 = torch.cat([pos2, vel2, rot2], dim=-1)

                        
            joints = torch.cat([joint1, joint2], dim=-2) # B, T, J=22*2, D=12
            joints = joints.reshape(joints.shape[0], joints.shape[1], 2 * self.joints_num, 12)
        else:
            joints = x
        joints = joints.permute(0, 3, 2, 1).float() # B, D=12, J=22 * 2, T 
 
        return joints
    
    

    def postprocess(self, x):
        """ input should be in the shape of B, T, J=22*2, D=12 , output should be
        B, T, D=262 * 2; the result will be normalized. 
        """
        x = x.permute(0, 3, 2, 1).float() # B, T, J=22*2, D=12
        x1, x2 = torch.split(x, [self.joints_num, self.joints_num], dim=-2)
        pos1 = x1[:,:,:,:3].reshape([x.shape[0], x.shape[1], -1])
        vel1 = x1[:,:,:,3:6].reshape([x.shape[0], x.shape[1], -1])
        rot1 = x1[:,:,1:,6:6+6].reshape([x.shape[0], x.shape[1], -1])
        pos2 = x2[:,:,:,:3].reshape([x.shape[0], x.shape[1], -1])
        vel2 = x2[:,:,:,3:6].reshape([x.shape[0], x.shape[1], -1])
        rot2 = x2[:,:,1:,6:6+6].reshape([x.shape[0], x.shape[1], -1])
        fc = torch.zeros((x.shape[0], x.shape[1], 4)).to(x.device)
        x1 = torch.cat([pos1, vel1, rot1, fc], dim=-1)
        x2 = torch.cat([pos2, vel2, rot2, fc], dim=-1)
        x = torch.cat([x1, x2], dim=-1)
        return x

    def encode(self, x):
        # N, T, _, _ = x.shape

        with torch.no_grad():
            x_in = self.preprocess(x) # B, D=12, J=22, T || B, J=22xD=12, T
            
        x_encoder = self.encoder(x_in) # B, D=512, 5, T/2 || B, J=7xD=512, T//4
        # x_encoder = x_encoder if len(encoder_shape) == 3 else x_encoder.reshape(encoder_shape[0], encoder_shape[1], -1)
        # code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True) # B,375,1; 1,B,512,375
        return x_encoder
    
    
    def decode(self, x):
        return self.decoder(self.post_quant_conv(x))
    
    
    def forward(self, inp):   # -> rec_B3HW, idx_N, loss
        inp = self.preprocess(inp)
        encoded = self.quant_conv(self.encoder(inp))
        f_hat, vq_loss, ppl = self.quantizer.forward(encoded)
        decoded = self.decoder(self.post_quant_conv(f_hat))
        return decoded, inp, vq_loss, ppl
    
    
    
    def motion_to_idxBl(self, motions: torch.Tensor):
        """
        motions in shape of B, T, D=12, J=22*2
        return List[Bl]
        """
        f = self.quant_conv(self.encoder(motions))
        return self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=False)
    
    def embed_to_motion(self, ms_h_bcjl, all_to_max_scale=True, last_one=True):
        if last_one:
            return self.decoder(self.post_quant_conv(
                self.quantizer.embed_to_fhat(ms_h_bcjl, 
                                              all_to_max_scale=all_to_max_scale, 
                                              last_one=True)))
        else:
            return [self.decoder(self.post_quant_conv(f_hat)) \
                for f_hat in self.quantizer.embed_to_fhat(ms_h_bcjl, 
                                                          all_to_max_scale=all_to_max_scale, 
                                                          last_one=False)]
    
    
    def idxBl_to_motion(self, ms_idx_Bl:torch.Tensor, same_shape:bool=True, last_one:bool=True):
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
        
        lengths = [s * t for s,t in self.quantizer.scale_numbers]
        
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            si = lengths.index(l)
            sn, tn = self.quantizer.scale_numbers[si]
            ms_h_bcjl.append(self.quantizer.embedding(idx_Bl).transpose(1, 2).view(B, self.code_dim, sn, tn))
        return self.embed_to_motion(ms_h_bcjl=ms_h_bcjl, all_to_max_scale=same_shape, last_one=last_one)
    
    
    
    def motion_to_reconstruct(self, x, last_one=False):
        """ x in shape of [B, L, 2C] unnormed motion """
        x = self.preprocess(x) # b, c, 2j, t
        f = self.quant_conv(self.encoder(x)) # b, c, 1j, 
        ls_f_hat_bcjl = self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=True)
        if last_one:
            x_decoded = self.decoder(self.post_quant_conv(ls_f_hat_bcjl[-1])) # b, c, 2j, t
            x_decoded = self.postprocess(x_decoded) # b, t, 2c
            return x_decoded
        else:
            return [self.postprocess(self.decoder(self.post_quant_conv(f_hat))) \
                    for f_hat in ls_f_hat_bcjl]


    def training_step(self, batch):
        motions = batch['motions']
        motion_rec, inp, vq_loss, ppl = self.forward(motions)
        rec_loss = F.l1_loss(motion_rec, inp) if self.cfg.vqvae.loss_fn == 'l1' else F.smooth_l1_loss(motion_rec, inp)
        
        loss = rec_loss + self.lambda_vq * vq_loss
        
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/vq_loss', vq_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/ppl', ppl, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch):
        motions = batch['motions']
        motion_rec, inp, vq_loss, ppl = self.forward(motions)
        rec_loss = F.l1_loss(motion_rec, inp) if self.cfg.vqvae.loss_fn == 'l1' else F.smooth_l1_loss(motion_rec, inp)
        
        loss = rec_loss + self.lambda_vq * vq_loss
        
        self.log('val/loss', loss)
        self.log('val/rec_loss', rec_loss)
        self.log('val/vq_loss', vq_loss)
        self.log('val/ppl', ppl)
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