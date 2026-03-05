import torch
import torch.nn.functional as F


class RectifiedFlowLoss:
    def __init__(self, time_sampler="uniform", label_dropout_prob=0.1, is_l2=True, **kwargs):
        self.time_sampler = time_sampler
        self.label_dropout_prob = label_dropout_prob
        self.is_l2 = is_l2

    def sample_time_steps(self, batch_size, device):
        if self.time_sampler == "uniform":
            t = torch.rand(batch_size, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, device=device)
            t = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        return t

    def __call__(self, model, x, model_kwargs=None):
        """
        Simple Rectified Flow Loss:
        v_target = x - noise
        z_t = (1 - t)x + t * noise
        """
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        B, L, D = x.shape
        device = x.device

        # Label dropout (optional)
        if model_kwargs.get("y") is not None and self.label_dropout_prob > 0:
            y = model_kwargs["y"].clone()
            null_cond = torch.zeros_like(y)[0]
            dropout_mask = torch.rand(B, device=device) < self.label_dropout_prob
            y[dropout_mask] = null_cond
            model_kwargs["y"] = y

        # Sample time steps and noise
        t = self.sample_time_steps(B, device)
        t_reshaped = t.view(-1, 1, 1)
        noise = torch.randn_like(x)

        # Interpolation and target velocity
        z_t = (1 - t_reshaped) * x + t_reshaped * noise
        v_target = x - noise

        # Model prediction
        v_pred = model(z_t, t, **model_kwargs)

        # Loss
        error = v_pred - v_target.detach()
        if self.is_l2:
            loss = (error ** 2).reshape(B, -1).mean()
        else:
            loss = torch.abs(error).reshape(B, -1).mean()

        return loss
