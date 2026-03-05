import torch 
import torch.nn as nn

from lightning import LightningModule

from models.reactor.armf.adaln import AdaLNTransformer


class ARMF(nn.Module):
    def __init__(self, cfg):
        pass