""" This is the implementation of the ReMeanFlow with AdaLN encoder, no decoder involved """


import torch 
import torch.nn as nn
import omegaconf
import math
from lightning import LightningModule

from models.reactor.armf.adaln import AdaLNTransformer, SkipAdaLNTransformer
from models.reactor.armf.translator import Translator
from models.reactor.armf.loss import SILoss
from models.utils.tools import TimestepEmbedder
from models.reactor.armf.meanflow_sampler import meanflow_sampler
from models.utils.tools import PositionalEncoding, LearnablePositionalEncoding
from models.reactor.armf.textenc import TextEncoder
from models.reactor.vae.klvaex import KLVAE
from utils.utils import MotionNormalizerTorch

from eval.interx.evaluator import InterXEvaluator

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
    
    
class SOSLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.sos = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        sos = self.sos.repeat(len(x), 1, 1)
        return torch.cat([sos, x], dim=1)
    

from omegaconf import OmegaConf

class ReMeanFlow2(LightningModule):
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
        self.loss_fn = SILoss(**cfg.mf.siloss)
        self.r_embedder = TimestepEmbedder(cfg.mf.d_model if not self.is_dcond else self.d_cond)
        self.t_embedder = TimestepEmbedder(cfg.mf.d_model if not self.is_dcond else self.d_cond)
        
        if self.is_lpe:
            self.pe = LearnablePositionalEncoding(cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2)
        else:
            self.pe = PositionalEncoding(cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2)
        
        self.sos = SOSLayer(cfg.mf.d_model if self.is_dim1 else cfg.mf.d_model * 2)
        cfg.mf.text_encoder.is_dmodel = True if not self.is_dcond else False # d_cond overrides d_model
        self.text_encoder = TextEncoder(cfg.mf)
        
        self.normalizer = MotionNormalizerTorch()
        
        self.num_steps = cfg.mf.get('num_steps', 1)
        self.cfg_scale = cfg.mf.get('cfg_scale', 1.0)
        self.cfg_omega = cfg.mf.siloss.cfg_omega
        self.scale_factor = cfg.mf.get('scale_factor', 0.1825)
        
        


    def load_vae(self):
        import os
        from omegaconf import OmegaConf
        self.vae_name = self.cfg.vae.name
        print(f'---------Loading vae from {self.vae_name}---------')
        self.vae_path = f'ckpt/{self.vae_name}/last.ckpt'
        version_id = os.listdir(f'ckpt/{self.vae_name}/lightning_logs/')[0]
        self.vae_cfg = f'ckpt/{self.vae_name}/lightning_logs/{version_id}/hparams.yaml'
        self.vae_cfg = OmegaConf.load(self.vae_cfg)
        
        
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
            y (torch.Tensor, optional): cat [text_emb, [B, T, D]]
        """
        r = self.r_embedder.forward(r) # b, d_model
        t = self.t_embedder.forward(t) # b, d_model
        
        cond = y[:, 0, :] + r + t # b, d_model
        
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
        cmotion = cmotion.reshape(cmotion.shape[0], cmotion.shape[1], -1)
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
        motion = motion.reshape(motion.shape[0], motion.shape[1], -1)
        motion_emb = self.vae.scale_encode(motion, scale_factor=self.scale_factor)
        y = self.process_y(batch)
        
        loss, loss_mean_ref = self.loss_fn(model=self, images=motion_emb, model_kwargs={'y': y})
        self.log('train/loss', loss_mean_ref, on_epoch=True, prog_bar=True)
        return loss_mean_ref
    
    def validation_step(self, batch):
        motion = batch['motion2']
        motion = motion.reshape(motion.shape[0], motion.shape[1], -1)
        motion_emb = self.vae.scale_encode(motion, scale_factor=self.scale_factor)
        y = self.process_y(batch)
        loss, loss_mean_ref = self.loss_fn(model=self, images=motion_emb, model_kwargs={'y': y})
        self.log('val/loss', loss_mean_ref, on_epoch=True, prog_bar=True)
        return loss_mean_ref
    
    
    def forward_test(self, motion:torch.Tensor, text:str, motion_lens:torch.Tensor):
        
        batch = {'motion1': motion[:, :, :, :6], 'text': text}

        motion1 = batch['motion1'] # b, l, j, 6
        latents = self.vae.scale_encode(motion1.reshape(motion1.shape[0], motion1.shape[1], -1), scale_factor=self.scale_factor)
        latents = torch.randn_like(latents).cuda()
        
        if self.cfg_omega > 1:
            self.cfg_scale = 1.0
        
        sample = meanflow_sampler(model=self, latents=latents, y=self.process_y(batch), cfg_scale=self.cfg_scale, num_steps=self.num_steps)
        sample = self.vae.scale_decode(sample, scale_factor=self.scale_factor)
        return torch.cat([motion1, sample], dim=-1) # b, l, j, 12
    
        
        
    def on_validation_epoch_end(self):
        self.evaluator = InterXEvaluator(model=self)
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
        self.evaluator = InterXEvaluator(model=self, is_mm=True)
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