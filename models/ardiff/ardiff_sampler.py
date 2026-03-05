import torch
from diffusers import DDIMScheduler

@torch.no_grad()
def ardiff_sampler(
    model,
    latents,
    y=None,
    num_steps=5,
    cfg_scale=1.0,
    device=None,
):
    """
    DDIM-based sampler with general CFG using diffusers.DDIMScheduler.
    Args:
        model: the model that takes (latents, t, y) and returns noise or predicted sample
        latents: initial noise tensor, shape (B, L, D)
        y: dict or tensor of class labels (optional)
        num_steps: number of inference steps (default 1000)
        cfg_scale: guidance scale (>1 enables CFG)
        device: torch device
    """
    if device is None:
        device = latents.device
    B, L, D = latents.shape
    # Prepare scheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(num_steps, device=device)
    # initial sample
    sample = latents

    # CFG handling
    do_cfg = (y is not None) and (cfg_scale > 1.0)
    if do_cfg:
        # assume y is tensor of shape (B, …)
        y_cond = y["y"]
        y_uncond = torch.zeros_like(y_cond)

    for t_index, t in enumerate(scheduler.timesteps):
        # scale model input if required
        sample_input = scheduler.scale_model_input(sample, t)

        if do_cfg:
            # prepare cond/uncond batch
            sample_in = torch.cat([sample_input, sample_input], dim=0)
            y_in = torch.cat([y_cond, y_uncond], dim=0)

            t_input = torch.full((2 * B,), t, device=device)            
            
            model_output = model(sample_in, t_input, y=y_in)
            noise_cond, noise_uncond = model_output.chunk(2, dim=0)
            # apply CFG
            model_output = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            t_input = torch.full((B,), t, device=device)            
            model_output = model(sample_input, t_input, y=y)

        # step to previous sample
        sample = scheduler.step(model_output, t, sample)["prev_sample"]

    return sample