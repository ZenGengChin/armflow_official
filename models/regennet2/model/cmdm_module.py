from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from models.regennet2.utils.model_util import create_model_and_diffusion
from models.regennet2.utils.parser_util import train_args
from models.regennet2.model.cmdm import CMDM
import torch


def interhuman_cmdm_collate(batch):
    name = [item[0] for item in batch]
    text = [item[1] for item in batch]
    motion1 = torch.stack([torch.from_numpy(item[2]) for item in batch]).float()
    motion2 = torch.stack([torch.from_numpy(item[3]) for item in batch]).float()
    length = torch.Tensor([torch.tensor(item[4]) for item in batch]).long()
    return {'y': {
        'name': name,
        'text': text,
        'motion1': motion1,
        'motion2': motion2,
        'motions': torch.cat([motion1, motion2], dim=-1),
        'motion_lens': length
        },
    }
    
def interx_cmdm_collate(batch):
    word_embeddings = torch.stack([torch.from_numpy(item[0]) for item in batch])
    pos_one_hots = torch.stack([torch.from_numpy(item[1]) for item in batch])
    captions = [item[2] for item in batch]
    sent_lens = torch.tensor([torch.tensor(item[3]) for item in batch], dtype=torch.long)
    motions = torch.stack([(item[4]) for item in batch])
    lengths = torch.tensor([torch.tensor(item[5]) for item in batch], dtype=torch.long)
    return {
        'text': captions,
        'cap_lens': sent_lens,
        'motions': motions,
        'motion_lens': lengths,
        'word_embeddings': word_embeddings,
        'pos_one_hots': pos_one_hots,
    }



class CMDMModule(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.name = 'cmdm'
        
        args = train_args()

        print("creating model...")
        if args.arch == 'trans_enc' or args.arch == 'mlp' or args.arch == 'gru' or args.arch == 'offline':
            arch_mode = 'offline'
        elif args.arch == 'trans_dec' or args.arch == 'online':
            arch_mode = 'online'
        if args.unconstrained:
            action_conditioned = False
        else:
            action_conditioned = True
        print("Setting:", args.setting, "| Dataset:", args.dataset, "| Arch:", arch_mode, "| Action conditioned:", action_conditioned)
        
        self.args = args
        
        from utils.utils import MotionNormalizerTorch   
        self.normalizer = MotionNormalizerTorch()
        
        self.model, self.diffusion = create_model_and_diffusion(args)
    
    
    def training_step(self, batch):
        motion = batch['motion2']
        text = batch['text']
        motion_lens = batch['motion_lens']
        
        timesteps = torch.randint(0, 1000, (motion.shape[0],), device=motion.device)
        
        # print(motion.shape, timesteps, '-------------------------')
        
        noise = torch.randn_like(motion)
        
        terms = self.diffusion.training_losses(self.model, motion, timesteps, model_kwargs=batch, noise=noise, dataset=self.args.dataset)
        for key in terms:
            self.log(f'train/{key}', terms[key], on_epoch=True, prog_bar=True)
        return terms['loss']
    
    
    def validation_step(self, batch):
        cmotion = batch['motion1']
        motion = batch['motion2']
        text = batch['text']
        motion_lens = batch['motion_lens']
        
        timesteps = torch.randint(0, 1000, (motion.shape[0],), device=motion.device)
        noise = torch.randn_like(motion)
        terms = self.diffusion.training_losses(self.model, motion, timesteps, model_kwargs=batch, noise=noise, dataset=self.args.dataset)
        for key in terms:
            self.log(f'val/{key}', terms[key], on_epoch=True, prog_bar=True)
        return terms['loss']


    def create_ddim_diffusion(self, args):
        from models.regennet2.utils.model_util import create_gaussian_diffusion
        setting = args.setting
        args.num_person = 1 # Attention here
        args.use_ddim = True
        args.timestep_respacing = 'ddim5'
        diffusion = create_gaussian_diffusion(args)
        return diffusion
    
    def forward_test(self, batch):
        
        self.ddim_diffusion = self.create_ddim_diffusion(self.args)
        sample_fn = self.ddim_diffusion.p_sample_loop
        
        # self.ddim_diffusion = self.diffusion
        # sample_fn = self.ddim_diffusion.p_sample_loop
        
        # input batch is no normalized motion, but the output should be normalized ones. 
        
        batch['motion1'] = self.normalizer.forward(batch['motions'][:,:,0,:].cuda().float())
        
        
        
        sample = sample_fn(
            self.model,
            batch['motion1'].shape,
            clip_denoised=False,
            model_kwargs=batch,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
            # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
        )
         
        return {'output': torch.cat([batch['motion1'], sample], dim=-1)}
        
        
            
        
    def on_validation_epoch_end(self):
        from eval.interhuman.evaluator import InterHumanEvaluator
        self.evaluator = InterHumanEvaluator(model=self)
        result = self.evaluator.evaluation(replication_times=1)
        for k, v in result.items():
            try:
                if len(v) > 1:
                    for i in range(len(v)):
                        self.log(f'eval/{k}_{i+1}', v[i], on_epoch=True, prog_bar=False)
                else:
                    self.log(f'eval/{k}', v, on_epoch=True, prog_bar=False)
            except:
                self.log(f'eval/{k}', v, on_epoch=True, prog_bar=False)   
                
    def on_test_epoch_end(self):
        from eval.interhuman.evaluator import InterHumanEvaluator
        self.evaluator = InterHumanEvaluator(model=self, is_mm=True)
        result = self.evaluator.evaluation(replication_times=20)
        output = {}
        for k, v in result.items():
            try:
                if len(v) > 1:
                    for i in range(len(v)):
                        output[f'{k}_{i+1}'] = v[i]
                else:
                    output[f'{k}'] = v
            except:
                output[f'{k}'] = v
        return output

            
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        
        # Warm-up scheduler for the first 10 epochs 
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, total_iters=1
        )
        
        # Exponential decay scheduler to decay LR by 0.5 every 500 epochs
        gamma = 0.5 ** (1 / 1000)
        decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )
        
        # Combine schedulers: first warm-up, then exponential decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[10]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
        
from omegaconf import OmegaConf
from datasets.interhuman import InterHuman, interhuman_collate
from torch.utils.data import DataLoader
if __name__ == "__main__":

    data_cfg = OmegaConf.load('cfg/interhuman.yaml')
    data_cfg = data_cfg.interhuman_test
    dataset = InterHuman(data_cfg, normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True, collate_fn=interhuman_collate)
    cfg = OmegaConf.load('cfg/regennet/regennet.yaml')
    model:CMDMModule = CMDMModule().cuda()
    
    
    single_batch = next(iter(dataloader))
    single_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in single_batch.items()}
    single_batch['cmotion'] = single_batch['motion2']
    single_step = model.model.forward(single_batch['motion1'], [0]*32, **single_batch)
    
    # print(single_step.shape)
    
    terms = model.training_step(single_batch)
    print(terms, '-------------------------')
    
    
    output = model.forward_test(single_batch)
    print(output['output'].shape)
