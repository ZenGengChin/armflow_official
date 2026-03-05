
import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.interhuman import InterHuman, interhuman_collate
from models.regennet2.model.cmdm_module import CMDMModule
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_API_KEY"] = "f102063b3dba7a0fc298136198afa909015c94c5"

from omegaconf import OmegaConf

import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

def main():

    data_cfg = OmegaConf.load('cfg/interhuman.yaml')
    cfg = OmegaConf.load('cfg/regennet/regennet.yaml')

    import sys
    iargs = sys.argv[1:]


    train_dataset = InterHuman(opt=data_cfg.interhuman, normalize=True)
    test_dataset = InterHuman(opt=data_cfg.interhuman_test, normalize=True)

    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.cmdm.train.batch_size,
        shuffle=True,
        collate_fn=interhuman_collate,
    )

    # Create DataLoader for the validation set
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.cmdm.train.batch_size,
        shuffle=False,
        collate_fn=interhuman_collate
    )
    exp_name = cfg.cmdm.name
    model = CMDMModule()
        
    # wandb_logger = WandbLogger(
    #    project="cmdm2_interhuman",  
    #     name=exp_name,        
    #     save_dir=f'ckpt/{exp_name}/',     
    #  )
    
    tb_logger = TensorBoardLogger(
        save_dir=f'ckpt/{exp_name}',
    )
    
        
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
        check_val_every_n_epoch=20,
        accelerator='cuda',
        devices=1
        # logger=[wandb_logger, tb_logger]
    )

    trainer.fit(model, train_dataloader, val_dataloader)



    """
    python -m train.regennet.train_cmdm2_interhuman --setting cmdm --save_dir save/cmdm/ntu_smplx --dataset interhuman --cond_mask_prob 0 --num_person 2 --layers 8 --num_frames 300 --arch online --overwrite --pose_rep xyz --body_model smplx --data_path PATH/TO/xsub.train.h5 --train_platform_type TensorboardPlatform --vel_threshold 0.03 --unconstrained --use_ddim --timestep_respacing ddim5
    """
    
if __name__ == "__main__":
    main()