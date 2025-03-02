import numpy as np
import torch


def sinkhorn_scaling(mat, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    """
    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-10)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-10)

        if (
            torch.max(torch.abs(mat.sum(dim=2) - 1.0)) < tol
            and torch.max(torch.abs(mat.sum(dim=1) - 1.0)) < tol
        ):
            break

    return mat


def deterministic_neural_sort(s, tau):
    """
    Deterministic neural sort.
    """
    dev = s.device
    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    scaling = torch.arange(n, 0, -1, device=dev).type(torch.float32)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, device, eps=1e-10):
    """
    Sampling from Gumbel distribution.
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(s, n_samples, tau, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort.
    """
    dev = s.device
    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    P_hat = deterministic_neural_sort(s_perturb, tau)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat
