import torch
from torch import Tensor
import torch.nn as nn

from models.reactor.armf.net import (
    SkipTransformerDecoder, 
    SkipTransformerEncoder, 
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder
)

from models.reactor.armf.adaln import (
    SkipAdaLNTransformer,
    AdaLNTransformer,
    AdaLNTransformerDecoder,
    SkipAdaLNTransformerDecoder
)



class Translator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.d_model = cfg.d_model
        self.nhead = cfg.nhead
        self.num_encoder_layers = cfg.num_encoder_layers
        self.num_decoder_layers = cfg.num_decoder_layers
        self.d_ffn = cfg.d_ffn
        self.dropout = cfg.dropout
        
        self.is_skip = cfg.is_skip
        self.is_norm_first = cfg.is_norm_first
        self.is_adaln = cfg.is_adaln
        self.is_adaln_dec = cfg.get('is_adaln_dec', False)
        
        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_ffn,
            dropout=self.dropout,
            batch_first=True,
            normalize_before=self.is_norm_first
        )
        
        self.decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_ffn,
            dropout=self.dropout,
            batch_first=True,
            normalize_before=self.is_norm_first
        )
        

            
        
        if self.is_skip:

            
            if self.is_adaln:
                self.encoder = SkipAdaLNTransformer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    nlayer=self.num_encoder_layers,
                    d_ffn=self.d_ffn,
                    d_cond=self.cfg.text_encoder.d_model,
                    dropout=self.dropout,
                    shared_aln=False,
                    flash_if_available=False
                )
            else:
                self.encoder = SkipTransformerEncoder(
                    encoder_layer=self.encoder_layer,
                    num_layers=self.num_encoder_layers
                )
                
                
            if self.is_adaln_dec:
                self.decoder = SkipAdaLNTransformerDecoder(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    nlayer=self.num_decoder_layers,
                    d_ffn=self.d_ffn,
                    d_cond=self.cfg.text_encoder.d_model,
                    dropout=self.dropout,
                    shared_aln=False,
                    flash_if_available=False
                )
            else:
                self.decoder = SkipTransformerDecoder(
                    decoder_layer=self.decoder_layer,
                    num_layers=self.num_decoder_layers
                )
            
        else:
            if self.is_adaln:
                self.encoder = AdaLNTransformer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    nlayer=self.num_encoder_layers,
                    d_ffn=self.d_ffn,
                    d_cond=self.cfg.text_encoder.d_model,
                    dropout=self.dropout,
                    shared_aln=False,
                    flash_if_available=False
                )
            else:
                self.encoder = TransformerEncoder(
                    encoder_layer=self.encoder_layer,
                    num_layers=self.num_encoder_layers
                )
            
            if self.is_adaln_dec:
                self.decoder = AdaLNTransformerDecoder(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    nlayer=self.num_decoder_layers,
                    d_ffn=self.d_ffn,
                    d_cond=self.cfg.text_encoder.d_model,
                    dropout=self.dropout,
                    shared_aln=False,
                    flash_if_available=False
                )
            else:
                self.decoder = TransformerDecoder(
                decoder_layer=self.decoder_layer,
                num_layers=self.num_decoder_layers
            )
            
    def foward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        if self.is_skip:
            if self.is_adaln:
                self.encoder: SkipAdaLNTransformer
                self.decoder: SkipTransformerDecoder
                x = src[:, 1:, :]
                cond = src[:, 0, :]
                memory = self.encoder.forward(x = x, cond=cond, src_mask=src_mask)                
            else:
                self.encoder: SkipTransformerEncoder
                self.decoder: SkipTransformerDecoder
                
                memory = self.encoder.forward(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                
            if self.is_adaln_dec:
                self.decoder: SkipAdaLNTransformerDecoder
                return self.decoder.forward(tgt, memory, cond, tgt_mask)
            else:
                self.decoder: SkipTransformerDecoder
                return self.decoder.forward(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=src_key_padding_mask)
        else:
            
            if self.is_adaln:
                self.encoder: AdaLNTransformer
                x = src[:, 1:, :]
                cond = src[:, 0, :]
                memory = self.encoder.forward(x = x, cond=cond, src_mask=src_mask)
            else:
                self.encoder: TransformerEncoder
                memory = self.encoder.forward(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                
            if self.is_adaln_dec:
                self.decoder: AdaLNTransformerDecoder
                return self.decoder.forward(tgt, memory, cond, tgt_mask)
            else:
                self.decoder: TransformerDecoder
                return self.decoder.forward(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=src_key_padding_mask)
