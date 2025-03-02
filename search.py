from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm


def adam_search(
    x_init: torch.Tensor,
    forward_model: nn.Module,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x_opt_lr: float = 1e-3,
    x_opt_step: int = 100,
) -> torch.Tensor:

    x_res = x_init.clone()

    for i in tqdm(range(len(x_init)), desc="Searching x with Adam"):
        x_i = x_init[i : i + 1].clone()
        x_i.requires_grad = True
        x_opt = torch.optim.Adam(params=[x_i], lr=x_opt_lr)
        opt_step = x_opt_step
        for _ in range(opt_step):
            x_opt.zero_grad()
            y_pred = forward_model(x_i)
            score = -score_fn(y_pred)
            score.backward()
            x_opt.step()

        with torch.no_grad():
            x_res[i] = x_i.data

    return x_res


def grad_search(
    x_init: torch.Tensor,
    forward_model: nn.Module,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x_opt_lr: float = 1e-3,
    x_opt_step: int = 100,
) -> torch.Tensor:

    x_res = x_init.clone()

    for i in tqdm(range(len(x_init)), desc="Searching x with Grad"):
        x_i = x_init[i : i + 1].clone()
        x_i.requires_grad = True
        opt_step = x_opt_step

        for _ in range(opt_step):
            y_pred = forward_model(x_i)
            score = -score_fn(y_pred)
            _grad = torch.autograd.grad(outputs=score, inputs=x_i)[0]
            x_i = x_i + x_opt_lr * _grad

        with torch.no_grad():
            x_res[i] = x_i.data

    return x_res
