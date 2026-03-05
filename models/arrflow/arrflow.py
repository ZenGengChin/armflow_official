import torch
import torch.nn as nn
from torch import Tensor
import math
from models.arrflow.adaln import SkipAdaLNTransformer
from models.arrflow.mlp import AdaLnMLP
from models.arrflow.textenc import TextEncoder
from models.arrflow.loss import RectifiedFlowLoss
from models.utils.tools import PositionalEncoding
from utils.utils import MotionNormalizerTorch



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


class ARRFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.d_model = cfg.d_model 
        self.d_cond = cfg.text_encoder.d_cond
        self.nhead = cfg.nhead
        self.d_ffn = cfg.d_ffn
        self.nlayer_ar = cfg.nlayer_ar
        self.nlayer_mlp = cfg.nlayer_mlp
        self.dropout = cfg.dropout
        
        
        self.mlp: nn.Module = AdaLnMLP(
            d_cond=self.d_model * 2,
            d_model=self.d_model * 2,
            mlp_ratio=4,
            nlayer=self.nlayer_mlp
        )
        
        self.arnet = SkipAdaLNTransformer(
            d_model=self.d_model * 2,
            nhead=self.nhead,
            nlayer=self.nlayer_ar,
            d_ffn=self.d_ffn,
            d_cond=self.d_cond,
            dropout=self.dropout
        )
        
        # timestep embedding
        self.t_embedder = TimestepEmbedder(self.d_model * 2)
        
        self.start_token = nn.Parameter(torch.randn(1, 1, self.d_model * 2))
                

        self.pe = PositionalEncoding(self.d_model * 2)
        
        
    def forward(self, x, cond_B1D, cmotion, t_timesteps, mask=None):
        """
        Args:
            x (Tensor): [B, latent_size, 2 * d_model] is clean previous latent represnetation with no [s]
            cond_B1D (Tensor): [B, 1, d_cond], is the text condition form CLIP
            cmotion (Tensor): [B, L, d_model], is the cmotion of previous t-1 frames
            t_timesteps (Tensor): [B, 1] OR [1] , is the t time step
            mask (Tensor): [L, L], is the mask for the transformer encoder
        Returns:
            x (Tensor): [B, latent_size, d_model] is the predicted mean velocity
        """
        context = self.encode_context(x, cond_B1D, cmotion, mask)
        x = self.predict_mean_velocity(x, context, t_timesteps)
        return x
        
    def encode_context(self, x0: torch.Tensor, cond_B1D:Tensor, cmotion:Tensor, mask:torch.Tensor=None)->Tensor:
        """
        Args: 
            x0 (Tensor): [B, L, d_model] is the latent represnetation with no [s]
            cond_B1D (Tensor): [B, 1, d_cond], is the text condition form CLIP
            cmotion (Tensor): [B, L, d_model]
            mask (Tensor): [L, L], is the mask for the transformer encoder
        Returns:
            cond_token: [B, 1, d_model]
        """
        start_token = self.start_token.expand(cond_B1D.shape[0], -1, -1)
        if x0 is not None and x0.shape[1] > 0:
            x0 = torch.cat([start_token, torch.cat([cmotion, x0], dim=-1)], dim=1) # [B, latent_size + 1, d_model]
        else:
            x0 = start_token
        x = self.arnet.forward(x=self.pe.forward(x0), cond=cond_B1D, src_mask=mask)
        return x
    
    def predict_mean_velocity(self, x:torch.Tensor, context:Tensor, t_timesteps:Tensor):
        """
        Args:
            x (Tensor): [B, any, 2 * d_model] is the noised latent represnetation cat(cmtoion, x_t)
            context (Tensor): [B, any , 2 * d_model], is the context of the transformer encoder
            t_timesteps (Tensor): [B, 1] OR [1] , is the t time step
        Returns:
            u (Tensor): [B, any, d_model] is the predicted mean velocity
        """
        B = x.shape[0]
        t_emb = self.t_embedder.forward(t_timesteps) # B, d_model
        if t_timesteps.shape[0] != B:
            t_emb = t_emb.expand(B, -1)
        context = context + t_emb.unsqueeze(1) # B, 1, d_model
        x = self.mlp.forward(x=x, cond_B1D=context)

        return x[:,:,self.d_model:]