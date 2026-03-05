from lightning import LightningModule
from omegaconf import DictConfig
        
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


from diffusers import DDIMScheduler, DDPMScheduler
from eval.interhuman.evaluator import InterHumanEvaluator
from models.armflow.textenc import TextEncoder
from utils.utils import MotionNormalizerTorch

class CAMDM(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.camdm.name
        self.is_dim1 = cfg.camdm.get('is_dim1', False)
        self.model = MotionDiffusion(
            input_feats=cfg.camdm.input_feats if self.is_dim1 else cfg.camdm.input_feats * 2,
            njoints=cfg.camdm.njoints,
            nfeats=cfg.camdm.nfeats,
            clip_len=cfg.camdm.clip_len,
            latent_dim=cfg.camdm.latent_dim if self.is_dim1 else cfg.camdm.latent_dim * 2,
            ff_size=cfg.camdm.ff_size if self.is_dim1 else cfg.camdm.ff_size * 2,
            num_layers=cfg.camdm.num_layers,
            num_heads=cfg.camdm.num_heads,
            dropout=cfg.camdm.dropout,
            is_dim1=self.is_dim1,
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=cfg.noise_scheduler.num_train_timesteps,
            beta_schedule=cfg.noise_scheduler.beta_schedule,
            set_alpha_to_one=cfg.noise_scheduler.set_alpha_to_one,
            prediction_type=cfg.noise_scheduler.prediction_type,
            clip_sample=cfg.noise_scheduler.clip_sample,
            steps_offset=cfg.noise_scheduler.steps_offset
        )
        self.clip_len = cfg.camdm.clip_len
        
        self.loss_fn = nn.MSELoss()
        self.text_encoder = TextEncoder(cfg)
        
        self.inference_steps = cfg.noise_scheduler.num_inference_timesteps
        self.cfg_scale = cfg.camdm.cfg_scale
        
        self.normalizer = MotionNormalizerTorch()
        
        self.is_dim1 = cfg.camdm.get('is_dim1', False)
        

    def encode_text(self, text):
        text_emb = self.text_encoder.forward(text) # b, d_cond
        return text_emb


    def training_step(self, batch):
        cmotion = batch['motion1']
        cmotion = cmotion.reshape(cmotion.shape[0], -1, self.clip_len, cmotion.shape[-1])
        motion = batch['motion2']
        motion = motion.reshape(motion.shape[0], -1, self.clip_len, motion.shape[-1])
        
        cmotion_past = cmotion[:, :-1, :, :].reshape(-1, self.clip_len, cmotion.shape[-1])
        cmotion_future = cmotion[:, 1:, :, :].reshape(-1, self.clip_len, cmotion.shape[-1])
        motion_past = motion[:, :-1, :, :].reshape(-1, self.clip_len, motion.shape[-1])
        motion_future = motion[:, 1:, :, :].reshape(-1, self.clip_len, motion.shape[-1])
        
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (cmotion_future.shape[0],), device=cmotion.device)
        noise = torch.randn_like(cmotion_future)
        noisy_motion_future = self.scheduler.add_noise(motion_future.clone(), noise, timesteps)
        
        output = self.model.interface(
            x = torch.cat((cmotion_future, noisy_motion_future), dim=1 if self.is_dim1 else 2),
            timesteps = timesteps,
            text_emb = self.encode_text(batch['text']).repeat_interleave(motion.shape[1]-1, dim=0),
            past_motion = torch.cat((cmotion_past, motion_past), dim=1 if self.is_dim1 else 2)     
        )
        rec_loss = self.loss_fn(output, motion_future)
        joint_loss = self.loss_fn(output[:, :, :66], motion_future[:, :, :66])
        loss = rec_loss + joint_loss
        
        self.log('train/rec_loss', rec_loss, on_epoch=True, prog_bar=True)
        self.log('train/joint_loss', joint_loss, on_epoch=True, prog_bar=True)
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)
        return loss
        

    def validation_step(self, batch):
        cmotion = batch['motion1']
        cmotion = cmotion.reshape(cmotion.shape[0], -1, self.clip_len, cmotion.shape[-1])
        motion = batch['motion2']
        motion = motion.reshape(motion.shape[0], -1, self.clip_len, motion.shape[-1])
        
        cmotion_past = cmotion[:, :-1, :, :].reshape(-1, self.clip_len, cmotion.shape[-1])
        cmotion_future = cmotion[:, 1:, :, :].reshape(-1, self.clip_len, cmotion.shape[-1])
        motion_past = motion[:, :-1, :, :].reshape(-1, self.clip_len, motion.shape[-1])
        motion_future = motion[:, 1:, :, :].reshape(-1, self.clip_len, motion.shape[-1])
        
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (cmotion_future.shape[0],), device=cmotion.device)
        noise = torch.randn_like(cmotion_future)
        noisy_motion_future = self.scheduler.add_noise(motion_future.clone(), noise, timesteps)
        
        output = self.model.interface(
            x = torch.cat((cmotion_future, noisy_motion_future), dim=1 if self.is_dim1 else 2),
            timesteps = timesteps,
            text_emb = self.encode_text(batch['text']).repeat_interleave(motion.shape[1]-1, dim=0),
            past_motion = torch.cat((cmotion_past, motion_past), dim=1 if self.is_dim1 else 2)     
        )
        rec_loss = self.loss_fn(output, motion_future)
        joint_loss = self.loss_fn(output[:, :, :66], motion_future[:, :, :66])
        loss = rec_loss + joint_loss
        
        self.log('val/rec_loss', rec_loss, on_epoch=True, prog_bar=True)
        self.log('val/joint_loss', joint_loss, on_epoch=True, prog_bar=True)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.camdm.train.lr)
        
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
    
    
    
    def forward_test(self, batch):
        cmotion = batch['motions'][:,:,0,:].cuda().float()
        cmotion = cmotion.reshape(cmotion.shape[0], -1, self.clip_len, cmotion.shape[-1])
        motion = batch['motions'][:,:,1,:].cuda().float()
        motion = motion.reshape(motion.shape[0], -1, self.clip_len, motion.shape[-1])
        
        sequence = [motion[:, 0,:, :]] # b, 4, input_feats
        text_emb = self.encode_text(batch['text'])
        
        N = cmotion.shape[1] # 75
        
        
        for i in range(1, N):
            cmotion_cur = cmotion[:,i,:,:] # b, 4, input
            xt = torch.randn_like(cmotion_cur).to(cmotion_cur.device)
            
            self.scheduler.set_timesteps(self.inference_steps)
            timesteps = self.scheduler.timesteps.to(xt.device) # b, 8
            
            null_past = torch.zeros(cmotion_cur.shape[0], 
                                    cmotion_cur.shape[1] * 2 if self.is_dim1 else cmotion_cur.shape[1], 
                                    cmotion_cur.shape[2] if self.is_dim1 else cmotion_cur.shape[2] * 2).to(cmotion_cur.device)
            
            null_current = torch.zeros_like(cmotion_cur).to(cmotion_cur.device)
            
            for t in (timesteps):
                x_input_null = torch.cat([null_current, xt], dim=1 if self.is_dim1 else 2)
                x_input_cond = torch.cat([cmotion_cur, xt], dim=1 if self.is_dim1 else 2)
                x0 = self.model.forward(torch.cat([x_input_null, x_input_cond], dim=0), # 2b, 4, 2 * input 
                                        t.repeat(xt.shape[0] * 2),
                                        past_motion = torch.cat([null_past, torch.cat([cmotion[:,i-1,:],sequence[i-1]], 
                                                                                      dim=1 if self.is_dim1 else 2)], dim=0),
                                        text_emb = torch.cat([text_emb] * 2)) # b, 4, input * 1
                
                uncond_x0, cond_x0 = x0.chunk(2)
                x0 = uncond_x0 + (cond_x0 - uncond_x0) * self.cfg_scale
                
                
                xt = self.scheduler.step(
                    model_output=x0,
                    timestep=t, 
                    sample=xt
                ).prev_sample
                
            sequence.append(xt)
            
        motion_gen = torch.cat(sequence, dim=1)
        
        motion_gen = self.normalizer.forward(motion_gen)
        motion1 = self.normalizer.forward(batch['motions'][:,:,0,:].cuda().float())

        return {'output': torch.cat([motion1, motion_gen], dim=-1)}

        



class MotionDiffusion(nn.Module):
    def __init__(self, input_feats, njoints, nfeats, clip_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.2,
                 activation="gelu", arch='trans_enc', cond_mask_prob=0, d_cond=768, device=None, **kwargs):
        super().__init__()

        self.nfeats = nfeats
        self.njoints = njoints
        self.clip_len = clip_len
        self.input_feats = input_feats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        self.arch = arch
        self.is_dim1 = kwargs.get('is_dim1', False)

        # local conditions
        self.future_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.past_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        # self.traj_trans_process = TrajProcess(2, self.latent_dim)
        # self.traj_pose_process = TrajProcess(6, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # global conditions
        self.embed_style = EmbedStyle(d_cond, self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation,
                                                              batch_first=True)

            self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation,
                                                              batch_first=True)
            self.seqEncoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'gru':
            print("GRU init")
            self.seqEncoder = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
      
        self.output_process = OutputProcess(self.input_feats, self.latent_dim)


    def forward(self, x, timesteps, past_motion, text_emb):
        bs,  nframes, _ = x.shape
        
        time_emb = self.embed_timestep(timesteps)  # [bs, 1, L]
        text_emb = self.embed_style(text_emb).unsqueeze(1)  # [bs, 1, L]
        past_motion_emb = self.past_motion_process(past_motion)  # [bs, past_frames, L] 
        
        future_motion_emb = self.future_motion_process(x) 
        
        
        xseq = torch.cat((time_emb, text_emb, 
                          past_motion_emb, future_motion_emb), dim=1)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[:, -nframes // (2 if self.is_dim1 else 1):, :] 
        output = self.output_process(output)  
        return output if self.is_dim1 else output[:, :, self.input_feats//2:]
        

    def interface(self, x, timesteps, past_motion, text_emb):
        """
            x: [batch_size, frames, input_feats], denoted x_t, [x_a, x_b] in the paper 
            nframe = 4
            timesteps: [batch_size] (int)
            past_motion: [batch_size, frames, input_feats], denoted x_a in the paper
            text_emb: [batch_size, 1, d_cond]
        """
        bs, nframes, nfeats = x.shape
        

        # CFG on past motion
        keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1-self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((bs, 1, 1))
        
        if self.is_dim1:
            x[:,:x.shape[1]//2,:] = x[:,:x.shape[1]//2,:] * keep_batch_idx.view((bs, 1, 1))
            
        else:
            x[:,:, :x.shape[2]//2] = x[:,:,:x.shape[2]//2] * keep_batch_idx.view((bs, 1, 1))
            
        
        return self.forward(x, timesteps, past_motion, text_emb)
    

class MotionProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)  
        return x





class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])# .permute(1, 0, 2)


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  
        return output
    
    
class EmbedStyle(nn.Module):
    def __init__(self, d_cond, latent_dim):
        super().__init__()
        self.action_embedding = nn.Linear(d_cond, latent_dim, bias=False)

    def forward(self, input):
        output = self.action_embedding(input)
        return output