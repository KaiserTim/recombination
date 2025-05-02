import torch
import torchvision
import numpy as np


#----------------------------------------------------------------------------
# Residual Implementation (Algorithm 1# Residual Implementation (Algorithm 1)
def denoising_error(N, x, n, sigma, alpha=.2, M=1, class_labels=None, avg_func=lambda x : x, **kwargs):
    d_v = torch.zeros_like(x)
    for _ in range(M):
        eta = torch.randn_like(x)                           # [bs, ch, L, L]
        d_n = alpha * sigma * eta                           # [bs, ch, L, L], perturbation noise
        d_eps = x - N(x + d_n, (1+alpha)*avg_func(sigma).reshape(sigma.shape) * torch.ones_like(x), class_labels, force_fp32=True) - n   # [bs, ch, L, L], denoising residual
        d_v += d_eps**2                                     # [bs, ch, L, L]
    return torch.min(torch.sqrt(d_v/M)/alpha, sigma)        # [bs, ch, L, L],  mean error for next denoising step)


def denoising_error_v2(N, x, n, sigma, M=1, class_labels=None, avg_func=lambda x : x, force_fp32=False, **kwargs):
    d_v = torch.zeros_like(x)
    y_hat = x - n
    for _ in range(M):
        n_new = torch.randn_like(sigma) * sigma
        d_eps = y_hat + n_new - N(y_hat + n_new, avg_func(sigma).reshape(sigma.shape) * torch.ones_like(x), class_labels, force_fp32=force_fp32) - n # [bs, ch, L, L], denoising residual
        d_v += d_eps**2                                     # [bs, ch, L, L]
    return torch.min(torch.sqrt(d_v/M), sigma)        # [bs, ch, L, L],  mean error for next denoising step)


def reverse_trajectory(net, latent, sigma_max, alpha, class_tensor, eps=1e-2, ds=1, noise_levels="img",
                       return_history=False, denoising_error_func=denoising_error):
    if noise_levels == "img":
        avg_func = lambda x: torch.mean(x, dim=(1, 2, 3))
    else:
        avg_func = lambda x: x
    sigma = torch.ones_like(latent) if noise_levels == "pixel" else torch.ones(latent.shape[0], 1, 1, 1,
                                                                               device=latent.device)
    sigma *= sigma_max
    x = latent * sigma_max
    latent_history = [(x.cpu(), sigma.cpu())]
    transform = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=1)
    c_steps = 0  # step counter
    with torch.no_grad():
        while True:
            n = x - net(x, avg_func(sigma).reshape(sigma.shape) * torch.ones_like(x), class_tensor, force_fp32=True)
            sigma_new = denoising_error_func(net, x, n, sigma, alpha=alpha, M=1, class_labels=class_tensor, avg_func=avg_func)
            if noise_levels == "img":
                sigma_new = avg_func(sigma_new).reshape(sigma.shape)

            d_sigma = ds * (sigma - sigma_new)
            D_t = d_sigma * sigma_new
            mu = d_sigma / (sigma + 1e-8)
            if sigma_new.max() < eps:
                latent_history.append(((x - mu * n).cpu(), sigma.cpu()))
                if return_history:
                    return x - mu * n, latent_history
                return x - mu * n, c_steps
            else:
                x = x - mu * n + torch.sqrt(D_t) * torch.randn_like(x)
                sigma = torch.sqrt((sigma - d_sigma) ** 2 + D_t)
                if noise_levels == "pixel":
                    sigma = torch.max(transform(sigma), torch.ones_like(sigma) * 1e-3)
                latent_history.append((x.cpu(), sigma.cpu()))
            c_steps += 1

@torch.no_grad()
def table_inverse(func, y, domain, args=None, n_disc=1000, gridtype='lin'):
    """
    Function inversion via table look-up. Given a function, discretize it on the domain [a,b] and compute its image
    of this domain. Then, for each y value, find the two closest matching points in the image and linearly interpolate
    between their corresponding preimages, based on their distance in the image.

    Args:
        func: python func, function to be inverted
        y: tensor, values to evaluate the inverted function at
        domain: tuple, lower and upper bound of the interval where func is inverted
        args: list or tuple, additional function arguments for func
        n_disc: int, number of points in the grid-discretization
        gridtype: str, 'lin' for linspace or 'geom' for geomspace

    Returns:
        x: tensor, values with y /approx func(x)
        y_min: tensor, func(a)
        y_max: tensor, func(b)
    """
    assert gridtype == 'lin' or gridtype == 'geom', f'gridtype must be lin or geom, but is {gridtype}.'
    a, b = domain[0], domain[1]
    if gridtype == 'lin':
        x = torch.linspace(a, b, n_disc, device=y.device)
    else:
        x = torch.tensor(np.geomspace(a, b, n_disc, dtype=np.float32), device=y.device)

    # Create a look-up table for inversion
    table = torch.zeros(2, n_disc, device=y.device)
    table[0] = x
    table[1] = func(x, *args).reshape(-1) if args is not None else func(x).reshape(-1)

    # Find the closest y
    diff = table[1] - y.reshape(-1, 1)
    val, idx = torch.min(diff.abs(), dim=1)
    arange = torch.arange(idx.shape[0])

    # Find the second-closest y and determine if it's to the left or right
    idx2 = idx - torch.sign(diff[arange, idx]).int()  # go on the left or right, depending on the sign of the distance to the closest y
    idx2[idx2 == -1] = 0  # Catch values below a
    idx2[idx2 == n_disc] = n_disc - 1  # Catch values above b

    # Get the distance to the closest and second-closest y
    w1 = diff[arange, idx].abs() + 1e-6  # avoid division by zero below
    w2 = diff[arange, idx2].abs() + 1e-6

    # Compute the weighted average of the two closest x
    x_out = (w1 * table[0, idx] + w2 * table[0, idx2]) / (w1 + w2)

    return x_out

#
# def compute_sigmas_exact_old(net, y, sigmas, M=1, beta=0, labels=None, disable_tqdm=True):
#     """
#     Algorithm 2, 10.05.2023
#     simgas.shape: [bs, t, C, H, W]
#     """
#     for t in tqdm(reversed(range(sigmas.shape[1] - 1)), total=sigmas.shape[1] - 1, disable=disable_tqdm):
#         delta_v = torch.zeros_like(y)
#         for _ in range(M):
#             n = torch.randn_like(y) * sigmas[:, t+1]
#             eps = y - net(y + n, torch.mean(sigmas[:, t+1], dim=(1,2,3)), labels)
#             delta_v += eps**2
#         sigmas[:, t] = torch.sqrt(beta * sigmas[:, t] * sigmas[:, t] + (1-beta) * delta_v / M)  # EMA
#         sigmas[:, t] = torch.min(sigmas[:, t], sigmas[:, t+1])
#     return sigmas
#
#
# def compute_sigmas_exact(net, y, sigmas, t_steps, M=1, beta=0, labels=None, disable_tqdm=True):
#     """
#     Algorithm 2, 12.05.2023
#     net: network
#     y: target
#     sigmas: noise levels per image and timestep, [bs, t_max, C, H, W]
#     t_steps: Which steps to update, [bs, t], each row defines the step k -> k+1 for each image in the batch
#     M: Number of iterations to estimate the expectation of sigma
#     beta: Exponential moving average of sigmas between function calls
#     labels: model condition
#     disable_tqdm: disable progress bar
#     """
#     sigmas = sigmas.clone()  # don't modify input
#     bs, t_max = sigmas.shape[:2]
#     arange = torch.arange(bs, device=y.device)
#     for t in tqdm(t_steps.transpose(0,1), total=t_steps.shape[1], disable=disable_tqdm):  # [bs]
#         delta_v = torch.zeros(bs, 1, 1, 1, device=y.device)
#         for _ in range(M):
#             n = torch.randn_like(y) * sigmas[arange, t+1]
#             eps = y - net(y + n, torch.mean(sigmas[arange, t+1], dim=(1,2,3)), labels)
#             delta_v += (eps**2).mean(dim=(1,2,3), keepdims=True)  # reduce to scalar, [bs, 1, 1, 1]
#         sigmas[arange, t] = torch.sqrt(beta * sigmas[arange, t] * sigmas[arange, t] + (1-beta) * delta_v / M)
#         # w = torch.exp(-(t[:, None] - torch.arange(t_max, device=t.device))**2 / (2 * (t_max / 10) ** 2))  # gaussian kernel smoother, [bs, t_max]
#         # sigmas[arange, t] = ((w[:, :, None, None, None] * sigmas).sum(dim=(1,2,3,4), keepdims=True) / sigmas.sum(dim=(1,2,3,4), keepdims=True)).squeeze(1)  # squeeze t_max dimension
#         # sigmas[arange, t] = torch.min(sigmas[arange, t], sigmas[arange, t+1])
#     return sigmas
#
#
# def compute_sigmas_fast(net, y, S, beta=.5):
#     """Algorithm 3, 10.05.2023"""
#     sigmas = ...
#     return sigmas
#
#
# def denoising_error_sampling(N, x, n, sigma, alpha, t, M, class_labels=None):
#     """Algorithm 1, 10.05.2023"""
#     d_v = 0
#     for _ in range(M):
#         eta = torch.randn_like(x)                           # [bs, ch, L, L]
#         d_n = alpha * sigma * eta                           # [bs, ch, L, L], perturbation noise
#         d_eps = N(x + d_n, (t+alpha)*sigma, class_labels) - n - d_n    # [bs, ch, L, L], denoising residual
#         d_v += d_eps**2                                     # [bs, ch, L, L]
#     return torch.min(torch.sqrt(d_v/M)/alpha, sigma)        # [bs, ch, L, L],  mean error for next denoising step
#
#
# def true_residuals(net, x, y, sigma_avg, t, R=10, class_labels=None):
#     """
#     NOT FUNCTIONAL AND NOT USED, 10.05.2023.
#     Compute the residual given the true target y.
#     Args:
#         net: Denoising model
#         x: Noised data-point (noise can also be 0)
#         y: latents to compare against
#         sigma_avg: Average noise level per time step (avg. over dataset, computed in training)
#         t: Current noise level
#         R: Number of resampling steps
#     Returns:
#         sigma: Denoising error
#     """
#     v = 0
#     for _ in range(R):
#         n = torch.randn_like(x)                             # [bs, ch, L, L]
#         delta_sigma = (t**2 - sigma_avg**2)**0.5            # Correction term for noising x
#         x_K = x + n * delta_sigma                           # (Re-)noised x, see remark 3 on Algorithms
#         N_t = x_K - net(x_K, t, class_labels).to(torch.float64)
#         v += (x_K - N_t - y)**2                             # Denoising residual
#     return (v / R)**0.5
