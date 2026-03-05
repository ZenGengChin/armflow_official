
import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.interx import InterX, interx_collate
from models.regennetx.model.cmdm_module import CMDMModule

from omegaconf import OmegaConf

import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

def main():

    data_cfg = OmegaConf.load('cfg/interx.yaml')
    cfg = OmegaConf.load('cfg/regennet/offlinex.yaml')


    train_dataset = InterX(opt=data_cfg, split='train')
    test_dataset = InterX(opt=data_cfg, split='test')

    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.cmdm.train.batch_size,
        shuffle=True,
        collate_fn=interx_collate,
    )

    # Create DataLoader for the validation set
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.cmdm.train.batch_size,
        shuffle=False,
        collate_fn=interx_collate
    )
    exp_name = cfg.cmdm.name
    model = CMDMModule(cfg=cfg)
        
    if os.path.exists('ckpt/' + exp_name + '/last.ckpt'):
        model = CMDMModule.load_from_checkpoint(
            'ckpt/' + exp_name + '/last.ckpt',
            cfg=cfg
        )

    trainer = Trainer(
        default_root_dir='ckpt/' + exp_name,
        max_epochs=cfg.cmdm.train.epochs,
        callbacks=[
            ModelCheckpoint(
                save_top_k=5,
                monitor="eval/FID",
                mode="min",
                dirpath="ckpt/" + exp_name,
                filename='epoch={epoch}-fid={eval/FID:.4f}',
                auto_insert_metric_name=False,
                enable_version_counter=False,
                save_last=True
            )
        ],
        check_val_every_n_epoch=10 if not hasattr(cfg.cmdm.train, 'check_val_every_n_epoch') else cfg.cmdm.train.check_val_every_n_epoch,
        accelerator='cuda',
        devices=1
    )

    trainer.fit(model, train_dataloader, val_dataloader)

"""
python -m train.regennet.train_cmdm_interx  --setting cmdm --save_dir no  --dataset interx --cond_mask_prob 0 --num_person 2 --layers 8 --num_frames 3
00 --arch online --overwrite --pose_rep rot6d --body_model smplx  --vel_threshold 0.03 --cond_mode text
"""
    
if __name__ == "__main__":
    main()