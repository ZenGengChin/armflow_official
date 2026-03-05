import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from models.utils.sparse_loss import weighted_l1_loss_1d
from models.reactor.vae.encdec import Encoder, Decoder


    
    
from omegaconf import OmegaConf
        
class KLVAE(LightningModule):
    def __init__(self, cfg):
        super(KLVAE, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg))
        self.name = cfg.vae.name
        self.encoder = Encoder(
            input_emb_width=cfg.vae.d_input,
            output_emb_width=cfg.vae.output_emb_width * 2,
            down_t=cfg.vae.down_t,
            stride_t=cfg.vae.stride_t,
            width=cfg.vae.width * 2 if cfg.vae.is_double else cfg.vae.width,
            depth=cfg.vae.depth,
            dilation_growth_rate=cfg.vae.dilation_growth_rate,
            activation=cfg.vae.activation,
            norm=cfg.vae.norm
        )
        self.decoder = Decoder(
            input_emb_width=cfg.vae.d_input,
            output_emb_width=cfg.vae.output_emb_width,
            down_t=cfg.vae.down_t,
            stride_t=cfg.vae.stride_t,
            width=cfg.vae.width,
            depth=cfg.vae.depth,
            dilation_growth_rate=cfg.vae.dilation_growth_rate,
            activation=cfg.vae.activation,
            norm=cfg.vae.norm
        )
        self.d_joint = cfg.vae.d_joint
        self.d_input = cfg.vae.d_input
        self.d_latent = cfg.vae.output_emb_width
        self.latent_size = cfg.vae.latent_size
        
    def forward(self, motion:torch.Tensor, length:torch.Tensor):
        latents = self.encoder.forward(motion)
        sample = self._sample(mu=latents[:, :, :self.d_latent], logvar=latents[:,:,self.d_latent:])
        motion_rec = self.decoder.forward(sample)
        return motion_rec
    
    def forward_test(self, batch):
        device = next(self.parameters()).device
        motion, length = batch['motions'], batch['motion_lens']
        length = length.to(device)
        latents = self.encoder.forward(motion, length)
        x1, x2 = self.decoder.forward(latents, length)
        return {'output': torch.cat([x1, x2], dim=-1)}
    
    def _sample(self, mu:torch.Tensor, logvar:torch.Tensor):
        latent_dist = torch.distributions.Normal(mu, logvar.exp().pow(0.5))
        latent_z = latent_dist.rsample()
        return latent_z
    
    def _kl_div_z(self, mu1:Tensor, logvar1:Tensor, mu2:Tensor, logvar2:Tensor):
        return -0.5 * torch.mean(1 + logvar1 - mu1.pow(2) - logvar1.exp() + \
                                1 + logvar2 - mu2.pow(2) - logvar2.exp())

    def _kl_div(self, mu:torch.Tensor, logvar:torch.Tensor):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def compute_loss(self, motion:torch.Tensor, length:torch.Tensor):
        latents = self.encoder.forward(motion)
        mu = latents[:, :, :self.d_latent]
        logvar = latents[:, :, self.d_latent:]
        latents = self._sample(mu, logvar)
        motion_rec = self.decoder.forward(latents)
        rec_loss = weighted_l1_loss_1d(motion_rec, motion, length)
        joint_loss = weighted_l1_loss_1d(motion_rec[:,:,:self.d_joint], motion[:,:,:self.d_joint], length)
        kl_loss = self._kl_div(mu, logvar)
        return rec_loss, joint_loss, kl_loss
    
    
    def encode(self, motion:torch.Tensor, length:torch.Tensor=None):
        latents = self.encoder.forward(motion)
        mu = latents[:, :, :self.d_latent]
        logvar = latents[:, :, self.d_latent:]
        sample = self._sample(mu, logvar)
        return sample, mu, logvar
    
    def scale_encode(self, motion:torch.Tensor, length:torch.Tensor=None, scale_factor:float=1):
        latents = self.encoder.forward(motion)
        mu = latents[:, :, :self.d_latent]
        logvar = latents[:, :, self.d_latent:]
        sample = self._sample(mu, logvar)
        return sample * scale_factor
    
    def scale_decode(self, latents:torch.Tensor, length:torch.Tensor=None, scale_factor:float=1):
        return self.decoder.forward(latents / scale_factor)
    
    
    def decode(self, latents:torch.Tensor, length:torch.Tensor=None):
        return self.decoder.forward(latents)

    
    
    def training_step(self, batch):
        motion, length = batch['motions'], batch['motion_lens']
        motion = motion.reshape(motion.shape[0], motion.shape[1], 2, -1).permute(0, 2, 1, 3).reshape(motion.shape[0] * 2, motion.shape[1], -1)
        length = length.repeat_interleave(2)
        rec_loss, joint_loss, kl_loss = self.compute_loss(motion, length)
        loss = rec_loss + joint_loss + kl_loss * self.cfg.vae.beta_kl
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/joint_loss', joint_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/kl_loss', kl_loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch):
        motion, length = batch['motions'], batch['motion_lens']
        motion = motion.reshape(motion.shape[0], motion.shape[1], 2, -1).permute(0, 2, 1, 3).reshape(motion.shape[0] * 2, motion.shape[1], -1)
        length = length.repeat_interleave(2)
        rec_loss, joint_loss, kl_loss = self.compute_loss(motion, length)
        loss = rec_loss + joint_loss + kl_loss * self.cfg.vae.beta_kl
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/rec_loss', rec_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/joint_loss', joint_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val/kl_loss', kl_loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        from eval.interhuman.evaluator import InterHumanEvaluator
        self.evaluator = InterHumanEvaluator(model=self, is_mm=True)
        result = self.evaluator.evaluation(replication_times=20)
        output = {}
        for k, v in result.items():
            try:
                if len(v) > 1:
                    for i in range(len(v)):
                        output[f'{k}_{i+1}'] = v[i]
                else:
                    output[f'{k}'] = v
            except:
                output[f'{k}'] = v
        return output
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.vae.train.lr)
        
        # Warm-up scheduler for the first 10 epochs 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, total_iters=1
        )
        
        # Exponential decay scheduler to decay LR by 0.5 every 500 epochs
        gamma = 0.5 ** (1 / 1000)
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