import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

class DiffusionLoss:
    def __init__(
        self,
        label_dropout_prob=0.1,
        is_l2=True,
        num_train_timesteps=1000,
        cfg_omega=2.0,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        time_sampler="uniform",
        prediction_type="epsilon",
    ):
        """
        Diffusion loss using diffusers' add_noise and standard epsilon prediction.
        """
        self.label_dropout_prob = label_dropout_prob
        self.is_l2 = is_l2
        self.time_sampler = time_sampler
        self.prediction_type = prediction_type
        self.cfg_omega = cfg_omega
        # DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.num_train_timesteps = num_train_timesteps

    def sample_time_steps(self, batch_size, device):
        if self.time_sampler == "uniform":
            t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, device=device)
            u = torch.sigmoid(normal_samples)
            t = (u * (self.num_train_timesteps - 1)).long()
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        return t

    def __call__(self, model, x0, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        B = x0.shape[0]
        device = x0.device

        # classifier-free guidance dropout
        if model_kwargs.get("y") is not None and self.label_dropout_prob > 0:
            y = model_kwargs["y"].clone()
            null_cond = torch.zeros_like(y)[0]
            dropout_mask = torch.rand(B, device=device) < self.label_dropout_prob
            y[dropout_mask] = null_cond
            model_kwargs["y"] = y

        # sample timesteps and noise
        t = self.sample_time_steps(B, device)
        noise = torch.randn_like(x0, device=device)

        # add noise
        x_t = self.scheduler.add_noise(x0, noise, t)

        # model predicts epsilon
        pred = model(x_t, t, **model_kwargs)

        # compute loss
        if self.is_l2:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.l1_loss(pred, noise)

        return loss
