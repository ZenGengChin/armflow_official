""" This is the implementation of the ReMeanFlow with AdaLN encoder, no decoder involved """


import torch 
import torch.nn as nn
import omegaconf
import math
from lightning import LightningModule
from diffusers import DDIMScheduler, DDPMScheduler
import torch.nn.functional as F

from models.reactor.armf.adaln import AdaLNTransformer, SkipAdaLNTransformer
from models.reactor.armf.translator import Translator
from models.reactor.armf.loss import SILoss
from models.utils.tools import TimestepEmbedder
from models.reactor.armf.meanflow_sampler import meanflow_sampler
from models.utils.tools import PositionalEncoding, LearnablePositionalEncoding
from models.reactor.armf.textenc import TextEncoder
from models.reactor.vae.klvae import KLVAE
from utils.utils import MotionNormalizerTorch

from eval.interhuman.evaluator import InterHumanEvaluator


    
    
class SOSLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.sos = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        sos = self.sos.repeat(len(x), 1, 1)
        return torch.cat([sos, x], dim=1)
    

from omegaconf import OmegaConf

class ReMDM(LightningModule):
    def __init__(self, cfg:omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
                
        self.load_vae()
        
        self.name = cfg.mf.name
        
        self.is_dim1 = cfg.mf.get('is_dim1', False)
        self.is_dcond = cfg.mf.get('is_dcond', False)
        self.d_cond = cfg.mf.text_encoder.d_cond
        self.d_model = cfg.mf.d_model
        self.is_lpe = cfg.mf.get('is_lpe', False)
        
        self.translator = SkipAdaLNTransformer(
            d_model=cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2,
            nhead=cfg.mf.nhead,
            nlayer=cfg.mf.num_encoder_layers,
            d_ffn=cfg.mf.d_ffn,
            d_cond=self.d_cond if self.is_dcond else cfg.mf.text_encoder.d_model,
            dropout=cfg.mf.dropout,
            shared_aln=False,
            flash_if_available=False
        )
        self.loss_fn = F.mse_loss
        
        self.inference_steps = 10
        self.num_timesteps = 1000
        
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_timesteps,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            prediction_type="epsilon",
            clip_sample=False,
            steps_offset=0
        )
        
        self.scale_factor = cfg.mf.get('scale_factor', 0.1825)
        
        
        if self.is_lpe:
            self.pe = LearnablePositionalEncoding(cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2)
        else:
            self.pe = PositionalEncoding(cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2)
            
        self.t_embedder = TimestepEmbedder(cfg.mf.d_model if not self.is_dcond else self.d_cond,
                                           sequence_pos_encoder=PositionalEncoding(cfg.mf.d_model if not self.is_dcond else self.d_cond))
        
        self.sos = SOSLayer(cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2)
        cfg.mf.text_encoder.is_dmodel = True if not self.is_dcond else False # d_cond overrides d_model
        self.text_encoder = TextEncoder(cfg.mf)
        
        self.normalizer = MotionNormalizerTorch()
        
        self.num_steps = cfg.mf.get('num_steps', 1)
        self.cfg_scale = cfg.mf.get('cfg_scale', 1.0)
        self.cfg_omega = cfg.mf.siloss.cfg_omega
        
        


    def load_vae(self):
        import os
        from omegaconf import OmegaConf
        self.vae_name = self.cfg.vae.name
        print(f'---------Loading vae from {self.vae_name}---------')
        self.vae_path = f'ckpt/{self.vae_name}/last.ckpt'
        version_id = os.listdir(f'ckpt/{self.vae_name}/lightning_logs/')[0]
        self.vae_cfg = f'ckpt/{self.vae_name}/lightning_logs/{version_id}/hparams.yaml'
        self.vae_cfg = OmegaConf.load(self.vae_cfg)
        
        # assert self.latent_size == self.vae_cfg.vae.latent_size, f'latent_size mismatch, {self.latent_size} != {self.vae_cfg.vae.latent_size}'
        # assert self.latent_dim == self.vae_cfg.vae.latent_dim, f'latent_dim mismatch, {self.latent_dim} != {self.vae_cfg.vae.latent_dim}'
        
        self.vae = KLVAE.load_from_checkpoint(self.vae_path, cfg=self.vae_cfg)
        self.vae.eval()
        self.vae.to(self.device)
        self.vae.freeze()
        print(f'---------Loaded vae from {self.vae_path}---------')   
        
        
    
    def forward(self, x:torch.Tensor, t:torch.Tensor, y=None):
        """
        Args:
            x (torch.Tensor): noisy latent motion in shape of [B, T, D]
            t (torch.Tensor): end time in shape of [B]
            y (torch.Tensor, optional): cat [text_emb, [B, T, D]]
        """
        t = self.t_embedder.forward(t) # b, d_model
        
        cond = y[:, 0, :] + t # b, d_model
        
        if not self.is_dim1:
            x = torch.cat([y[:, 1:, :self.d_model], x], dim=-1) # b, t, 2d
            x = self.sos.forward(x) # b, t + 1, 2d
            x = self.pe.forward(x) # b, t + 1, 2d
        else:
            x = torch.cat([y[:, 1:, :self.d_model], x], dim=1) # b, 2t, d_model
            x = self.pe.forward(x) # b, 2t, d_model
        
        output = self.translator.forward(x = x, cond = cond, src_mask=None)
        if not self.is_dim1:
            output = output[:, 1:, self.cfg.mf.d_model:]
        else:
            output = output[:, output.shape[1]//2:, :]
        return output
        
        
        
        
    
    def process_y(self, batch):
        cmotion = batch['motion1']
        text = batch['text']
        text_emb = self.text_encoder.forward(text).unsqueeze(1) # b, 1, d_cond
        cmotion_emb = self.vae.scale_encode(cmotion, scale_factor=self.scale_factor)
        if self.is_dcond:
            y = torch.zeros(cmotion_emb.shape[0], cmotion_emb.shape[1], self.d_cond).to(cmotion_emb.device)
            y[:, :, :cmotion_emb.shape[2]] = cmotion_emb
            return torch.cat([text_emb, y], dim=1) # b, 1 + t, d_cond
        else:
            return torch.cat([text_emb, cmotion_emb], dim=1)
        
        
    def training_step(self, batch):
        motion = batch['motion2']
        motion_emb = self.vae.scale_encode(motion, scale_factor=self.scale_factor)
        y = self.process_y(batch)
        timesteps = torch.randint(0, self.num_timesteps, (motion_emb.shape[0],), device=motion_emb.device)
        noise = torch.randn_like(motion_emb)
        motion_emb = self.scheduler.add_noise(motion_emb, noise, timesteps)
        noise_pred = self.forward(motion_emb, timesteps, y)
        loss = self.loss_fn(noise_pred, noise)
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        motion = batch['motion2']
        motion_emb = self.vae.scale_encode(motion, scale_factor=self.scale_factor)
        y = self.process_y(batch)
        timesteps = torch.randint(0, self.num_timesteps, (motion_emb.shape[0],), device=motion_emb.device)
        noise = torch.randn_like(motion_emb)
        motion_emb = self.scheduler.add_noise(motion_emb, noise, timesteps)
        noise_pred = self.forward(motion_emb, timesteps, y)
        loss = self.loss_fn(noise_pred, noise)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    
    def forward_test(self, batch):
        # input batch is denormalized, but the output should be normalized ones.
        
        motion1 = batch['motions'][:,:,0,:].float().cuda()
        batch['motion1'] = motion1
        latents = self.vae.scale_encode(batch['motion1'], scale_factor=self.scale_factor)
        latents = torch.randn_like(latents).cuda()
        
        self.scheduler.set_timesteps(self.inference_steps)
        timesteps = self.scheduler.timesteps.to(latents.device)
        xt = torch.randn_like(latents).to(latents.device)
        for t in (timesteps):
            x0 = self.forward(xt, t.repeat(xt.shape[0]), y=self.process_y(batch))
            xt = self.scheduler.step(model_output=x0, timestep=t, sample=xt).prev_sample
        
        sample = self.vae.scale_decode(xt, scale_factor=self.scale_factor)
        sample = self.normalizer.forward(sample)
        motion1 = self.normalizer.forward(motion1)
        return {'output': torch.cat([motion1, sample], dim=-1)}
    
        
        
    def on_validation_epoch_end(self):
        self.evaluator = InterHumanEvaluator(model=self)
        result = self.evaluator.evaluation(replication_times=3)
        for k, v in result.items():
            try:
                if len(v) > 1:
                    for i in range(len(v)):
                        self.log(f'eval/{k}_{i+1}', v[i], on_epoch=True, prog_bar=True)
                else:
                    self.log(f'eval/{k}', v, on_epoch=True, prog_bar=True)
            except:
                self.log(f'eval/{k}', v, on_epoch=True, prog_bar=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.mf.train.lr)
        
        # Warm-up scheduler for the first 10 epochs 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, total_iters=1
        )
        
        # Exponential decay scheduler to decay LR by 0.5 every 500 epochs
        gamma = 0.5 ** (1 / 800)
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
            'monitor': 'val/loss'
        }