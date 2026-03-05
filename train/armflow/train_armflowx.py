
import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.interx import InterX, interx_collate
from models.armflowx.armflow_module import ARMFlowXModule

from omegaconf import OmegaConf

import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'




class MultiMetricCheckpoint(Callback):
    def __init__(self, dirpath, fid_threshold=0.1, mmdist_threshold=3.6):
        super().__init__()
        self.dirpath = dirpath
        self.fid_threshold = fid_threshold
        self.mmdist_threshold = mmdist_threshold
        os.makedirs(dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        fid = metrics.get("eval/FID")
        mmdist = metrics.get("eval/MM Distance")

        if fid is None or mmdist is None:
            print("Missing eval/FID or eval/MM_Dist in metrics.")
            return

        if fid < self.fid_threshold and mmdist < self.mmdist_threshold:
            epoch = trainer.current_epoch
            path = os.path.join(
                self.dirpath, f"my-epoch={epoch}-fid={fid:.4f}-mmd={mmdist:.4f}.ckpt"
            )
            print(f"Saving model (FID={fid:.4f}, MMD={mmdist:.4f}) to {path}")
            trainer.save_checkpoint(path)

class SkipValidationUntil(Callback):
    def __init__(self, start_epoch=180):
        super().__init__()
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        """Disable validation until start_epoch"""
        if trainer.current_epoch < self.start_epoch:
            trainer.should_stop = False  # ensure training continues
            trainer.limit_val_batches = 0  # effectively disable validation
        else:
            trainer.limit_val_batches = 1.0  # restore validation normally



def main():

    data_cfg = OmegaConf.load('cfg/interx.yaml')
    cfg = OmegaConf.load('cfg/armflow/armflowx.yaml')

    import sys
    iargs = sys.argv[1:]
    if iargs:
        cfg = OmegaConf.merge(cfg,  OmegaConf.from_dotlist(iargs))

    train_dataset = InterX(opt=data_cfg, split='train')
    test_dataset = InterX(opt=data_cfg, split='test')

    print(len(train_dataset), len(test_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.mf.train.batch_size,
        shuffle=True,
        collate_fn=interx_collate,
    )

    # Create DataLoader for the validation set
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.mf.train.batch_size,
        shuffle=False,
        collate_fn=interx_collate
    )
    exp_name = cfg.mf.name
    model = ARMFlowXModule(cfg=cfg)
        
    if os.path.exists('ckpt/' + exp_name + '/last.ckpt'):
        model = ARMFlowXModule.load_from_checkpoint(
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
            ),
            SkipValidationUntil(50),
            MultiMetricCheckpoint(
                dirpath="ckpt/" + exp_name,
                fid_threshold=0.1,
                mmdist_threshold=3.60)
        ],
        check_val_every_n_epoch=2,
        accelerator='cuda',
        devices=1
    )

    trainer.fit(model, train_dataloader, val_dataloader)


    
if __name__ == "__main__":
    main()