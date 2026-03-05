
import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.interx import InterX, interx_collate
from models.reactor.vae.klvaex import KLVAE

from omegaconf import OmegaConf

import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

def main():

    data_cfg = OmegaConf.load('cfg/interx.yaml')
    cfg = OmegaConf.load('cfg/reactor/armfx.yaml')

    import sys
    iargs = sys.argv[1:]
    if iargs:
        cfg = OmegaConf.merge(cfg,  OmegaConf.from_dotlist(iargs))

    train_dataset = InterX(opt=data_cfg, split='train')
    test_dataset = InterX(opt=data_cfg, split='test')

    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.vae.train.batch_size,
        shuffle=True,
        collate_fn=interx_collate,
    )

    # Create DataLoader for the validation set
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.vae.train.batch_size,
        shuffle=False,
        collate_fn=interx_collate
    )
    exp_name = cfg.vae.name
    model = KLVAE(cfg=cfg)
        
    if os.path.exists('ckpt/' + exp_name + '/last.ckpt'):
        model = KLVAE.load_from_checkpoint(
            'ckpt/' + exp_name + '/last.ckpt',
            cfg=cfg
        )

    trainer = Trainer(
        default_root_dir='ckpt/' + exp_name,
        max_epochs=cfg.vae.train.epochs,
        callbacks=[
            ModelCheckpoint(
                save_top_k=3,
                monitor="val/body_loss",
                mode="min",
                dirpath="ckpt/" + exp_name,
                filename='epoch={epoch}-body_loss={val/body_loss:.4f}',
                auto_insert_metric_name=False,
                enable_version_counter=False,
                save_last=True
            )
        ],
        check_val_every_n_epoch=10 if not hasattr(cfg.vae.train, 'check_val_every_n_epoch') else cfg.vae.train.check_val_every_n_epoch,
        accelerator='cuda',
        devices=1
    )

    trainer.fit(model, train_dataloader, val_dataloader)


    
if __name__ == "__main__":
    main()