import torch 
import torch.nn as nn
import math
from omegaconf import OmegaConf
from lightning import LightningModule

from models.arrflow.adaln import AdaLNTransformer, SkipAdaLNTransformer
from models.arrflow.mlp import AdaLnMLP
from models.arrflow.textenc import TextEncoder
from models.arrflow.loss import RectifiedFlowLoss
from models.arrflow.arrflow_sampler import arrflow_sampler
from models.arrflow.arrflow import ARRFlow
from utils.utils import MotionNormalizerTorch

from models.reactor.vae.klvae import KLVAE
from eval.interhuman.evaluator import InterHumanEvaluator





class ARRFlowModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.load_vae()
        
        self.name = cfg.mf.name
        self.d_model = cfg.mf.d_model
        self.nhead = cfg.mf.nhead
        self.d_ffn = cfg.mf.d_ffn
        self.d_cond = cfg.mf.text_encoder.d_cond
        self.latent_size = cfg.vae.latent_size
        
        
        self.scale_factor = cfg.mf.scale_factor
        self.num_steps = cfg.mf.num_steps
        self.cfg_scale = cfg.mf.siloss.cfg_omega
        
        
        self.text_encoder = TextEncoder(cfg.mf)
        
        self.loss_fn = RectifiedFlowLoss(**cfg.mf.siloss)
        self.arrflow = ARRFlow(cfg.mf)
        
        
        self.normalizer = MotionNormalizerTorch()
        
        self.regen_ratio = cfg.mf.get('regen_ratio', 0.0)
        self.initial_state = cfg.mf.get('initial_state', False)
        self.double_epoch = cfg.mf.get('double_epoch', 1000)
        self.replace_mask = cfg.mf.get('replace_mask', False)

        
    
    
    def forward(self, x, t, y):
        """ This forward is for the single timestep inference
        Args:
            x_t (Tensor): [B, any, d_model] is the noised latent represnetation with no [s]
            y (Tensor): [B, any, d_model + 2 * dmodel], cat [cmotion, context]
            t (Tensor): [B * any, 1] OR [1] , is the t time step
            
        Returns:
            x (Tensor): [B, any, d_model] is the predicted mean velocity
        """
        cmotion = y[:, :, :self.d_model]
        context = y[:, :, self.d_model:]
        x_t = torch.cat([cmotion, x], dim=-1) # b, any, 2 * d_model
        u = self.arrflow.predict_mean_velocity(x=x_t, context=context, t_timesteps=t)
        return u # b, 1, d_model
    

    def process_y(self, batch):
        """ This is used for training
        Args:
            batch (dict): {
                'cmotion': [B, L, d_joint],
                'text': list[str]
            }
        Returns:
            y (Tensor): [B, 1 + L, d_cond]
            null_y (Tensor): [B, 1 + L, d_cond]
        """
        cmotion = batch['motion1'] # [B, L, d_joint]
        motion = batch['motion2'] # [B, L, d_joint]
        text = batch['text']
        
        cmotion_emb = self.vae.scale_encode(cmotion, scale_factor=self.scale_factor)
        motion_emb = self.vae.scale_encode(motion, scale_factor=self.scale_factor)
        text_emb = self.text_encoder.forward(text).unsqueeze(1) # b, 1, d_cond
        return text_emb, cmotion_emb, motion_emb
    
    def replace_with_gen(self, cmotion_emb, motion_emb, text_emb, repeat_times=1):
        # This is used to replace the motion_emb with generated motion
        # cmotion_emb: [B, L, d_model]
        # motion_emb: [B, L, d_model]
        # text_emb: [B, 1, d_cond]
        # return: [B, L, d_model]
            
        latents = torch.randn_like(cmotion_emb).cuda().reshape(cmotion_emb.shape[0] * cmotion_emb.shape[1], 1, -1)
        with torch.no_grad():
            gen_motion = motion_emb.clone()
            for i in range(repeat_times):
                context = self.arrflow.encode_context(x0=gen_motion[:, :-1, :], 
                                                      cond_B1D=text_emb[:, :1, :], 
                                                      cmotion=cmotion_emb[:, :-1, :],
                                                      mask=self.process_mask() if self.replace_mask else None)
                y = torch.cat([cmotion_emb, context], dim=-1)
                y = y.reshape(y.shape[0] * y.shape[1], 1, -1)
                gen_motion = arrflow_sampler(model=self, 
                                            latents=latents, 
                                            y=y, 
                                            cfg_scale=self.cfg_scale, 
                                            num_steps=self.num_steps)
                gen_motion = gen_motion.reshape(cmotion_emb.shape[0], cmotion_emb.shape[1], -1)
   
            
        if self.regen_ratio > 0:
            B, L, d_model = cmotion_emb.shape
            replace_indices = torch.rand(B, L) < self.regen_ratio
            mixed_motion_emb = motion_emb.clone()
            mixed_motion_emb[replace_indices] = gen_motion[replace_indices]
        else:
            mixed_motion_emb = motion_emb
        return mixed_motion_emb

    def process_mask(self):
        """ This is because that the xformers need last dim to be at least 8"""
        L = self.latent_size
        pad = (L // 8 + 1) * 8
        mask = ~torch.triu(torch.ones(pad, pad, dtype=torch.bool), diagonal=1).cuda()
        mask = mask[:L,:pad].reshape(1, 1, L, pad)
        mask = torch.where(mask, 0, -torch.inf)[:,:, :,:L]
        return mask
        
    
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
        
        
    def encode_null_condition(self, cmotion:torch.Tensor, text:list[str]):
        # encode the null condition
        # cmotion: [B, L(300), d_joint] -> [B, L(75), d_model]
        # text: list[str] -> [B, 1, d_cond]
        # return: padded [B, L + 1, d_cond]
        cz = self.vae.scale_encode(motion = torch.zeros_like(cmotion).to(cmotion.device), 
                                   scale_factor = self.scale_factor)
        bd = self.text_encoder.forward([''] * len(text))
        B, L, d_model = cz.shape
        _, d_cond = bd.shape
        padded = torch.zeros(B, L + 1, d_cond).to(cz.device)
        padded[:, 1:, :d_model] = cz
        padded[:, 0, :d_cond] = bd
        
        return padded
    
    def forward_test(self, batch):
        if not self.initial_state:
            return self.forward_test_from_scratch(batch)
        else:
            return self.forward_test_with_init(batch)
        
    def forward_test_with_init(self, batch):
        
        # batch has only motions and text
        
        
        motion1 = batch['motions'][:,:,0,:].float().cuda()
        motion2 = batch['motions'][:,:,1,:].float().cuda()
        batch['motion1'] = motion1
        cmotion_emb = self.vae.scale_encode(batch['motion1'], scale_factor=self.scale_factor)
        init_emb = self.vae.scale_encode(motion2, scale_factor=self.scale_factor)
        latents = torch.randn_like(cmotion_emb).cuda()
        text_emb = self.text_encoder.forward(batch['text']).unsqueeze(1) # b, 1, d_cond
        
        sequence = [init_emb[:, :1, :]]
        
        for i in range(1, latents.shape[1]):
            motion_prev = torch.cat(sequence, dim=1)
            context = self.arrflow.encode_context(x0=motion_prev, 
                                                  cond_B1D=text_emb[:, :1, :], 
                                                  cmotion=cmotion_emb[:, :i, :])
            y = torch.cat([cmotion_emb[:, i:i+1, :], context[:, -1:, :]], dim=-1)
                
            x_i = arrflow_sampler(model=self, 
                                  latents=latents[:, -1:, :], 
                                  y=y,
                                  cfg_scale=self.cfg_scale, 
                                  num_steps=self.num_steps)
            sequence.append(x_i)
            
        sequence = torch.cat(sequence, dim=1)
        motion2 = self.vae.scale_decode(sequence, scale_factor=self.scale_factor)
        
        motion1 = self.normalizer.forward(motion1)
        motion2 = self.normalizer.forward(motion2)
        return {'output': torch.cat([motion1, motion2], dim=-1)}

    
    def forward_test_from_scratch(self, batch):
        
        # batch has only motions and text
        
        
        motion1 = batch['motions'][:,:,0,:].float().cuda()
        batch['motion1'] = motion1
        cmotion_emb = self.vae.scale_encode(batch['motion1'], scale_factor=self.scale_factor)
        latents = torch.randn_like(cmotion_emb).cuda()
        text_emb = self.text_encoder.forward(batch['text']).unsqueeze(1) # b, 1, d_cond
        
        sequence = []
        
        for i in range(latents.shape[1]):
            if i == 0:
                motion_prev = None
            else:
                motion_prev = torch.cat(sequence, dim=1)
                
            context = self.arrflow.encode_context(x0=motion_prev, 
                                                  cond_B1D=text_emb[:, :1, :], 
                                                  cmotion=cmotion_emb[:, :i, :])
            y = torch.cat([cmotion_emb[:, i:i+1, :], context[:, -1:, :]], dim=-1)
                
            x_i = arrflow_sampler(model=self, 
                                  latents=latents[:, -1:, :], 
                                  y=y,
                                  cfg_scale=self.cfg_scale, 
                                  num_steps=self.num_steps)
            sequence.append(x_i)
            
        sequence = torch.cat(sequence, dim=1)
        motion2 = self.vae.scale_decode(sequence, scale_factor=self.scale_factor)
        
        motion1 = self.normalizer.forward(motion1)
        motion2 = self.normalizer.forward(motion2)
        return {'output': torch.cat([motion1, motion2], dim=-1)}

    
    def training_step(self, batch):
        repeat_times = self.current_epoch // self.double_epoch + 1
        text_emb, cmotion_emb, motion_emb = self.process_y(batch)
        mixed_motion_emb = self.replace_with_gen(cmotion_emb, motion_emb, text_emb, repeat_times=repeat_times)
        context = self.arrflow.encode_context(x0=mixed_motion_emb[:, :-1, :], 
                                              cond_B1D=text_emb[:, 0, :], 
                                              cmotion=cmotion_emb[:, :-1, :], mask=self.process_mask())
        # B, L-1, 2  * d_model
        
        y = torch.cat([cmotion_emb, context], dim=-1)
        
        loss_mean_ref = self.loss_fn(model=self, 
                                           x=motion_emb.reshape(motion_emb.shape[0] * motion_emb.shape[1], 1, -1), 
                                           model_kwargs={'y': y.reshape(y.shape[0] * y.shape[1], 1, -1)})
        self.log('train/loss', loss_mean_ref, on_epoch=True, prog_bar=True)
        return loss_mean_ref
    
    
    def validation_step(self, batch):
        repeat_times = self.current_epoch // self.double_epoch + 1
        text_emb, cmotion_emb, motion_emb = self.process_y(batch)
        mixed_motion_emb = self.replace_with_gen(cmotion_emb, motion_emb, text_emb, repeat_times=repeat_times)
        context = self.arrflow.encode_context(x0=mixed_motion_emb[:, :-1, :], 
                                              cond_B1D=text_emb[:, 0, :], 
                                              cmotion=cmotion_emb[:, :-1, :], mask=self.process_mask())
        # B, L-1, 2  * d_model
        
        y = torch.cat([cmotion_emb, context], dim=-1)
        
        loss_mean_ref = self.loss_fn(model=self, 
                                    x=motion_emb.reshape(motion_emb.shape[0] * motion_emb.shape[1], 1, -1), 
                                    model_kwargs={'y': y.reshape(y.shape[0] * y.shape[1], 1, -1)})
        self.log('val/loss', loss_mean_ref, on_epoch=True, prog_bar=True)
        return loss_mean_ref
    
    
        
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
    
    
    