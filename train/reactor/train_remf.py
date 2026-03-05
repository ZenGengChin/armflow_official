
import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.interhuman import InterHuman, interhuman_collate
from models.reactor.armf.remf import ReMeanFlow

from omegaconf import OmegaConf

import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

def main():

    data_cfg = OmegaConf.load('cfg/interhuman.yaml')
    cfg = OmegaConf.load('cfg/reactor/armf.yaml')

    import sys
    iargs = sys.argv[1:]
    if iargs:
        cfg = OmegaConf.merge(cfg,  OmegaConf.from_dotlist(iargs))

    train_dataset = InterHuman(opt=data_cfg.interhuman, normalize=False)
    test_dataset = InterHuman(opt=data_cfg.interhuman_test, normalize=False)

    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.mf.train.batch_size,
        shuffle=True,
        collate_fn=interhuman_collate,
    )

    # Create DataLoader for the validation set
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.mf.train.batch_size,
        shuffle=False,
        collate_fn=interhuman_collate
    )
    exp_name = cfg.mf.name
    model = ReMeanFlow(cfg=cfg)
        
    if os.path.exists('ckpt/' + exp_name + '/last.ckpt'):
        model = ReMeanFlow.load_from_checkpoint(
            'ckpt/' + exp_name + '/last.ckpt',
            cfg=cfg
        )

    trainer = Trainer(
        default_root_dir='ckpt/' + exp_name,
        max_epochs=cfg.mf.train.epochs,
        callbacks=[
            ModelCheckpoint(
                save_top_k=3,
                monitor="eval/FID",
                mode="min",
                dirpath="ckpt/" + exp_name,
                filename='epoch={epoch}-fid={eval/FID:.4f}',
                auto_insert_metric_name=False,
                enable_version_counter=False,
                save_last=True
            )
        ],
        check_val_every_n_epoch=10 if not hasattr(cfg.mf.train, 'check_val_every_n_epoch') else cfg.mf.train.check_val_every_n_epoch,
        accelerator='cuda',
        devices=1
    )

    trainer.fit(model, train_dataloader, val_dataloader)


    
if __name__ == "__main__":
    main()