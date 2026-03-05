from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from omegaconf import DictConfig


from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F



# This is the 2d version of the work 

class MARQuantizer2D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        self.nb_code = cfg.nb_code
        self.dim_code = cfg.code_dim
        
        self.s_scale_numbers = cfg.s_scale_numbers
        self.t_scale_numbers = cfg.t_scale_numbers
        
        self.scale_numbers = list(zip(self.s_scale_numbers, self.t_scale_numbers))
        
        self.znorm = cfg.znorm
        
        
        self.quant_resi = cfg.quant_resi
        self.share_quant_resi = cfg.share_quant_resi


        if self.share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(self.dim_code, self.quant_resi) \
                                             if abs(self.quant_resi) > 1e-6 else nn.Identity()) \
                                            for _ in range(len(self.scale_numbers))])
        elif self.share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(self.dim_code, self.quant_resi) if abs(self.quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(self.dim_code, self.quant_resi)
                                                    if abs(self.quant_resi) > 1e-6 else nn.Identity())
                                                    for _ in range(self.share_quant_resi)]))
        
        
        self.embedding = nn.Embedding(self.nb_code, self.dim_code)
        self.beta = cfg.beta
        

    def forward(self, f_bcjl: torch.Tensor):
        """_summary_
        Args:
            f_bcjl (torch.Tensor): [b, c, j, l] encoded feature. 
        Returns:
            f_hat: estimated f from the VQ.
            mean_vq_loss: loss for the VQ.
        """
        B, C, J, L = f_bcjl.shape
        f_no_grad = f_bcjl.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        
        mean_vq_loss: torch.Tensor = 0.0
        SN = len(self.scale_numbers)
        idx_N_all = []
        
        for si, pn in enumerate(self.scale_numbers):
            sn, tn = pn
            if self.znorm:
                rest_NC = F.interpolate(f_rest, size=(sn, tn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) \
                    if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) 
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                rest_NC = F.interpolate(f_rest, size=(sn, tn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) \
                    if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_N_all.append(idx_N)            
            idx_bl = idx_N.view(B, sn, tn) # b, j, l   
            
            h_bcjl = F.interpolate(self.embedding(idx_bl).permute(0, 3, 1, 2), size=(J,L), mode='bicubic').contiguous() \
                if (si != SN-1) else self.embedding(idx_bl).permute(0, 3, 1, 2).contiguous()
            h_bcjl = self.quant_resi[si/(SN-1)](h_bcjl)
            f_hat = f_hat + h_bcjl
            f_rest -= h_bcjl
            
            mean_vq_loss += F.mse_loss(f_hat.data, f_bcjl).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

        perplexity = self.compute_perplexity(torch.cat(idx_N_all, dim=0))
        mean_vq_loss *= 1. / SN
        f_hat = (f_hat.data - f_no_grad).add_(f_bcjl)
        
        return f_hat, mean_vq_loss, perplexity
        
    
    def embed_to_fhat(self, 
                      ms_h_bcjl: List[torch.Tensor],
                      all_to_max_scale: bool = True,
                      last_one: bool = False
                      ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """ calculate the fhat from the multi-scale hidden states
        Args:
            ms_h_bcjl (List[torch.Tensor]): [1, 2**2, 2**4, 2**6, 2**8, 2**10, 2**12...]
        Returns:
            fhat: [b, c, j, l] or list of K [b, c, j, l] if last_one = False
        """
        ls_f_hat_bcjl = []
        B = ms_h_bcjl[0].shape[0]
        J, L = self.scale_numbers[-1]
        SN = len(self.scale_numbers)
        if all_to_max_scale:
            f_hat = ms_h_bcjl[0].new_zeros(B, self.dim_code, J, L, dtype=torch.float32)
            for si, pn in enumerate(self.scale_numbers): # from small to large
                h_bcjl = ms_h_bcjl[si]
                if si < len(self.scale_numbers) - 1:
                    h_bcjl = F.interpolate(h_bcjl, size=(L,), mode='bicubic')
                h_bcjl = self.quant_resi[si/(SN-1)](h_bcjl)
                f_hat.add_(h_bcjl)
                if last_one: ls_f_hat_bcjl = f_hat
                else: ls_f_hat_bcjl.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max L, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_bcjl[0].new_zeros(B, self.dim_code, self.scale_numbers[0], dtype=torch.float32)
            for si, pn in enumerate(self.scale_numbers): # from small to large
                sn, tn = pn
                f_hat = F.interpolate(f_hat, size=(sn, tn), mode='bicubic')
                h_bcjl = self.quant_resi[si/(SN-1)](ms_h_bcjl[si])
                f_hat.add_(h_bcjl)
                if last_one: ls_f_hat_bcjl = f_hat
                else: ls_f_hat_bcjl.append(f_hat)
        
        return ls_f_hat_bcjl
        
        
    def f_to_idxBl_or_fhat(self, 
                           f_bcjl: torch.Tensor, 
                           to_fhat: bool
                           ) -> List[Union[torch.Tensor, torch.LongTensor]]:  
        """ Do the quantize for f to embedding or index. 
        Args:
            f_bcjl (torch.Tensor): [b, c, j, l] encoded feature. 
            to_fhat (bool): if True, return the fhat, otherwise return the idxBl
        Returns:
            f_hat_or_idx_Bl: list of K [b, c, j, l] or list of K [b, j, l]
        """
        B, C, J, L = f_bcjl.shape
        f_no_grad = f_bcjl.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        SN = len(self.scale_numbers)
        for si, pn in enumerate(self.scale_numbers): # from small to large
            sn, tn = pn
            z_NC = F.interpolate(f_rest, size=(sn, tn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) \
                if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
        if self.znorm:
            z_NC = F.normalize(z_NC, dim=-1)
            idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)
            
        idx_bjl = idx_N.view(B, sn, tn)
        h_bcjl = F.interpolate(self.embedding(idx_bjl).permute(0, 3, 1, 2), size=(J, L), mode='bicubic').contiguous() \
            if (si != SN-1) else self.embedding(idx_bjl).permute(0, 3, 1, 2).contiguous()
        h_bcjl = self.quant_resi[si/(SN-1)](h_bcjl)
        f_hat.add_(h_bcjl)
        f_rest.sub_(h_bcjl)
        f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, sn * tn))
        
        return f_hat_or_idx_Bl
        

    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.dim_code
        J, L = self.scale_numbers[-1]
        SN = len(self.scale_numbers)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, J, L, dtype=torch.float32)
        sn_next, tn_next = self.scale_numbers[0]
        for si in range(SN-1):
            h_bcjl = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, sn_next, tn_next), size=(J, L), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_bcjl))
            sn_next, tn_next = self.scale_numbers[si+1]
            next_scales.append(F.interpolate(f_hat, size=(sn_next, tn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, 
                                      si: int, 
                                      SN: int, 
                                      f_hat: torch.Tensor, 
                                      h_bcjl: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        J, L = self.scale_numbers[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_bcjl, size=(J, L), mode='bicubic')) # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.scale_numbers[si+1][0], self.scale_numbers[si+1][1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_bcjl)
            f_hat.add_(h)
            return f_hat, f_hat
        
        
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity





class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw:torch.Tensor):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())


