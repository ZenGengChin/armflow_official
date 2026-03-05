import torch 
import torch.nn as nn
import omegaconf
import math
from lightning import LightningModule

from models.reactor.armf.adaln import AdaLNTransformer
from models.reactor.armf.translator import Translator
from models.reactor.armf.loss import SILoss
from models.utils.tools import TimestepEmbedder
from models.reactor.armf.meanflow_sampler import meanflow_sampler
from models.utils.tools import PositionalEncoding
from models.reactor.armf.textenc import TextEncoder
from models.reactor.vae.klvae import KLVAE
from utils.utils import MotionNormalizerTorch

from eval.interhuman.evaluator import InterHumanEvaluator

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
    
    
class TextSOS(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.sos = nn.Parameter(torch.randn(d_model))
        
    def forward(self, text):
        return self.sos.repeat(len(text), 1, 1)
    

from omegaconf import OmegaConf

class ReMeanFlow(LightningModule):
    def __init__(self, cfg:omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
                
        self.load_vae()
        
        self.name = cfg.mf.name
        self.translator = Translator(cfg.mf)
        self.loss_fn = SILoss(**cfg.mf.siloss)
        self.r_embedder = TimestepEmbedder(cfg.mf.d_model)
        self.t_embedder = TimestepEmbedder(cfg.mf.d_model)
        
        self.pe = PositionalEncoding(cfg.mf.d_model)
        
        
        self.is_text_cond = cfg.mf.is_text_cond
        self.is_adaln = cfg.mf.is_adaln
        
        if self.is_text_cond:
            self.sos = TextEncoder(cfg.mf)
        else:
            self.sos = TextSOS(cfg.mf.d_model)
            
        self.normalizer = MotionNormalizerTorch()
        
        self.cfg_scale = cfg.mf.get('cfg_scale', 1.0)
        self.omega = cfg.mf.siloss.cfg_omega
        self.num_steps = cfg.mf.get('num_steps', 1)
        
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
        
        
    
    def forward(self, x:torch.Tensor, r:torch.Tensor, t:torch.Tensor, y=None):
        """
        Args:
            x (torch.Tensor): noisy latent motion in shape of [B, T, D]
            r (torch.Tensor): start time in shape of [B]
            t (torch.Tensor): end time in shape of [B]
            y (torch.Tensor, optional): cat [sos, motion_1]
        """
        r = self.r_embedder.forward(r).unsqueeze(1) # b, 1, d
        t = self.t_embedder.forward(t).unsqueeze(1) # b, 1, d
        
        if not self.is_adaln:
            y = y + r + t
            y = self.pe.forward(y) # b, t + 1, d
        else:
            # Avoid in-place operations during gradient computation
            y_cond = y[:, 0:1, :] + r + t
            y_motion = self.pe.forward(y[:, 1:, :])
            y = torch.cat([y_cond, y_motion], dim=1)
        
        x = self.pe.forward(x) # b, t, d
        
        return self.translator.foward(src=y, tgt=x, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None)
        
    
    def process_y(self, batch):
        cmotion = batch['motion1']
        text = batch['text']
        text_emb = self.sos.forward(text)
        text_emb = text_emb if text_emb.dim() == 3 else text_emb.unsqueeze(1)
        cmotion_emb = self.vae.scale_encode(cmotion)
        return torch.cat([text_emb, cmotion_emb], dim=1)
        
        
    def training_step(self, batch):
        motion = batch['motion2']
        motion_emb = self.vae.scale_encode(motion)
        y = self.process_y(batch)
        
        loss, loss_mean_ref = self.loss_fn(model=self, images=motion_emb, model_kwargs={'y': y})
        self.log('train/loss', loss_mean_ref, on_epoch=True, prog_bar=True)
        return loss_mean_ref
    
    def validation_step(self, batch):
        motion = batch['motion2']
        motion_emb = self.vae.scale_encode(motion)
        y = self.process_y(batch)
        loss, loss_mean_ref = self.loss_fn(model=self, images=motion_emb, model_kwargs={'y': y})
        self.log('val/loss', loss_mean_ref, on_epoch=True, prog_bar=True)
        return loss_mean_ref
    
    
    def forward_test(self, batch):
        # input batch is denormalized, but the output should be normalized ones.
        
        motion1 = batch['motions'][:,:,0,:].float().cuda()
        batch['motion1'] = motion1
        latents = self.vae.scale_encode(batch['motion1'])
        latents = torch.randn_like(latents).cuda()
        
        # if trained with omega > 1, the cfg_scale should be 1. during inference
        # if trained with omega = 1, the cfg_scale can be any, and can be reset during evaluation
        if self.omega > 1:
            self.cfg_scale = 1.
        sample = meanflow_sampler(model=self, latents=latents, y=self.process_y(batch), cfg_scale=self.cfg_scale, num_steps=self.num_steps)
        sample = self.vae.scale_decode(sample)
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
        
    def on_test_epoch_end(self):
        self.evaluator = InterHumanEvaluator(model=self)
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