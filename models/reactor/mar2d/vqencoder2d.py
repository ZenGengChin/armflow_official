import torch
import torch.nn as nn
from typing import List
from models.utils.common import Normalize, nonlinearity


class AttnBlock2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.nin_shortcut = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)


        x = self.nin_shortcut(x)

        return x + h


class Upsample2D(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


        
class Downsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class VQEncoder2D(nn.Module):
    def __init__(self,
                 d_input: int = 12,
                 ch: int = 32,
                 ch_mult: List[int] = [1, 2],
                 num_res_blocks: int = 6,
                 dropout: float = 0.1,
                 z_channels: int = 11,
                 double_z: bool = False,
                 use_attention: bool = True
                 ) -> None:
        super().__init__()

                
        self.d_input = d_input
        self.ch = ch
        self.ch_mult = ch_mult
        self.ch_mult_in = (1,) + tuple(ch_mult)
        
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.double_z = double_z
        
        self.resnet_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        self.input_conv = nn.Conv2d(d_input, ch, kernel_size=3, stride=1, padding='same')

        current_channels_in = ch
        current_channels_out = ch
        for idx, mult in enumerate(ch_mult):
            current_channels_in = self.ch * self.ch_mult_in[idx]
            current_channels_out = self.ch * self.ch_mult[idx]
            for _ in range(num_res_blocks):
                self.resnet_layers.append(
                    ResnetBlock2D(
                        in_channels=current_channels_in,
                        out_channels=current_channels_out,
                        dropout=dropout
                    )
                )
                current_channels_in = current_channels_out
            self.attention_layers.append(
                AttnBlock2D(in_channels=current_channels_out) if use_attention else nn.Identity()
            )
            self.downsample_layers.append(
                Downsample2D(in_channels=current_channels_out)
            )
            
        self.norm_out = Normalize(in_channels=current_channels_out)
        self.output_conv = nn.Conv2d(current_channels_out, 
                                     z_channels * (2 if double_z else 1), 
                                     kernel_size=3, 
                                     stride=1,
                                     padding='same')
        
        self.quant_conv = torch.nn.Conv2d(z_channels * (2 if double_z else 1), 
                                          z_channels, 
                                          kernel_size=1)
        

                
    def forward(self, x:torch.Tensor):
        # x = x.permute(0, 2, 1).contiguous()
        x = self.input_conv(x)
        # To keep track of resnet layers
        for idx in range(len(self.ch_mult)):
            for ires in range(self.num_res_blocks):
                x = self.resnet_layers[idx * self.num_res_blocks + ires](x)
            x = self.attention_layers[idx](x)
            x = self.downsample_layers[idx](x)
        x = self.norm_out(x)
        x = self.output_conv(x)
        x = nonlinearity(x)
        x = self.quant_conv(x)
        return x
        

class VQDecoder2D(nn.Module):
    def __init__(self,
                    d_output: int = 12, 
                    ch: int = 128,
                    ch_mult: List[int] = [3, 1],
                    num_res_blocks: int = 6,
                    dropout: float = 0.1,
                    z_channels: int = 32,
                    use_attention: bool = True
                    ) -> None:
        super().__init__()
                        
        self.d_output = d_output
        self.ch = ch
        self.ch_mult = ch_mult
        self.ch_mult_in = (max(ch_mult),) + tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resnet_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        # Input linear layer to expand channel dimensions
        self.input_conv = nn.Conv2d(z_channels, ch * ch_mult[0], kernel_size=3, stride=1, padding='same')

        
        for idx, mult in enumerate(ch_mult):
            current_channels_in = self.ch * self.ch_mult_in[idx]
            current_channels_out = self.ch * ch_mult[idx]
            for _ in range(num_res_blocks):
                self.resnet_layers.append(
                    ResnetBlock2D(
                        in_channels=current_channels_in,
                        out_channels=current_channels_out,
                        dropout=dropout
                    )
                )
                current_channels_in = current_channels_out
                
            self.attention_layers.append(
                AttnBlock2D(in_channels=current_channels_out) if use_attention else nn.Identity()
            )
            self.upsample_layers.append(
                Upsample2D(in_channels=current_channels_out)
            )
        # Final linear layer to map to desired output dimension
        self.norm_out = Normalize(in_channels=current_channels_out)
        self.output_conv = nn.Conv2d(current_channels_out, self.d_output, kernel_size=3, stride=1, padding='same')
                
    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        resnet_step = 0  # To keep track of resnet layers
        
        for idx in range(len(self.ch_mult)):
            for _ in range(self.num_res_blocks):
                x = self.resnet_layers[resnet_step](x)
                resnet_step += 1
            x = self.attention_layers[idx](x)
            if idx < len(self.upsample_layers):
                x = self.upsample_layers[idx](x)
        x = nonlinearity(x)
        x = self.norm_out(x)
        x = self.output_conv(x)
        return x