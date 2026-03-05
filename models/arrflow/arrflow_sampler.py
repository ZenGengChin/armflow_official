import torch
@torch.no_grad()
def arrflow_sampler(
    model,
    latents,
    y=None,
    num_steps=10,
    cfg_scale=1.0,
    use_heun=True,   # 默认 Heun
):
    """
    Rectified Flow sampler with optional Heun's method (RK2) and general CFG.

    Args:
        model: velocity predictor v_theta(z_t, t, y)
        latents: initial noise, shape (B, L, D)
        y: tensor or dict of conditional inputs (optional, for CFG)
        num_steps: number of integration steps
        cfg_scale: guidance scale (>1 enables CFG)
        use_heun: True -> Heun (RK2), False -> Euler
    """
    B, L, D = latents.shape
    device = latents.device
    z = latents

    # t: 1 -> 0
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    # 是否启用 CFG
    do_cfg = (y is not None) and (cfg_scale is not None) and (cfg_scale > 1.0)

    # ------------ helper functions ------------
    def make_t_batch(t_scalar):
        # 与训练时保持一致的形状；这里使用 (B,)
        return torch.full((B,), float(t_scalar), device=device)

    def zeros_like_y(y_in):
        if isinstance(y_in, dict):
            return {k: torch.zeros_like(v) for k, v in y_in.items()}
        else:
            return torch.zeros_like(y_in)

    def cat_y(y_a, y_b):
        if isinstance(y_a, dict):
            return {k: torch.cat([y_a[k], y_b[k]], dim=0) for k in y_a.keys()}
        else:
            return torch.cat([y_a, y_b], dim=0)

    def predict_v(z_in, t_scalar, y_in):
        """
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        """
        t_batch = make_t_batch(t_scalar)

        if do_cfg:
            y_null = zeros_like_y(y_in)

            # duplicate for cond & uncond
            z_cat = torch.cat([z_in, z_in], dim=0)
            t_cat = torch.cat([t_batch, t_batch], dim=0)
            y_cat = cat_y(y_in, y_null)

            v_cat = model(z_cat, t_cat, y=y_cat)
            v_cond, v_uncond = v_cat[:B], v_cat[B:]
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(z_in, t_batch, y=y_in)
        return v
    # ------------------------------------------

    for i in range(num_steps):
        t_cur = t_steps[i].item()
        t_next = t_steps[i + 1].item()
        dt = t_cur - t_next

        if use_heun:
            # Heun (predictor-corrector)
            v_cur = predict_v(z, t_cur, y)
            z_pred = z - dt * v_cur
            v_next = predict_v(z_pred, t_next, y)
            z = z - dt * 0.5 * (v_cur + v_next)
        else:
            # Euler
            v = predict_v(z, t_cur, y)
            z = z - dt * v
    return z