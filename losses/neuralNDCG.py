import torch
import torch.nn as nn

from .base import LossFunction
from .loss_utils import (
    deterministic_neural_sort,
    sinkhorn_scaling,
    stochastic_neural_sort,
)


def dcg(y_true, y_pred, ats=None, gain_function=lambda x: torch.pow(2, x) - 1):
    """
    Discounted Cumulative Gain (DCG) calculation.
    """
    if ats is None:
        ats = [y_true.shape[1]]

    device = y_true.device
    max_ranking_size = max(ats)

    true_sorted = y_true.gather(1, (-y_pred).argsort(dim=1))
    gains = gain_function(true_sorted)
    discounts = torch.log2(
        torch.arange(true_sorted.shape[1], dtype=torch.float, device=device).add(2.0)
    )
    discounted_gains = gains / discounts

    cum_dcg = torch.cumsum(discounted_gains, dim=1)
    dcg_at_ks = [
        cum_dcg[:, at - 1] if at <= cum_dcg.shape[1] else cum_dcg[:, -1] for at in ats
    ]

    return torch.stack(dcg_at_ks, dim=0)


class NeuralNDCGLossFunction(LossFunction):

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        temperature=1.0,
        powered_relevancies=True,
        k=None,
        stochastic=False,
        n_samples=32,
        beta=0.1,
        log_scores=True,
    ) -> torch.Tensor:

        dev = y_pred.device

        if k is None:
            k = y_true.shape[1]

        # Choose the deterministic/stochastic variant
        if stochastic:
            P_hat = stochastic_neural_sort(
                y_pred.unsqueeze(-1),
                n_samples=n_samples,
                tau=temperature,
                beta=beta,
                log_scores=log_scores,
            )
        else:
            P_hat = deterministic_neural_sort(
                y_pred.unsqueeze(-1), tau=temperature
            ).unsqueeze(0)

        # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
        P_hat = sinkhorn_scaling(
            P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3])
        )
        P_hat = P_hat.view(
            int(P_hat.shape[0] / y_pred.shape[0]),
            y_pred.shape[0],
            P_hat.shape[1],
            P_hat.shape[2],
        )

        # Apply to true labels, i.e., approximately sort them
        y_true_expanded = y_true.unsqueeze(-1).unsqueeze(0)
        if powered_relevancies:
            y_true_expanded = torch.pow(2.0, y_true_expanded) - 1.0

        ground_truth = torch.matmul(P_hat, y_true_expanded).squeeze(-1)
        discounts = (
            torch.tensor(1.0)
            / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.0)
        ).to(dev)
        discounted_gains = ground_truth * discounts

        if powered_relevancies:
            ideal_dcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
        else:
            ideal_dcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(
                1, 0
            )

        discounted_gains = discounted_gains[:, :, :k]
        ndcg = discounted_gains.sum(dim=-1) / (ideal_dcg + 1e-10)

        assert (ndcg < 0.0).sum() >= 0, "every ndcg should be non-negative"

        mean_ndcg = ndcg.mean()
        return -1.0 * mean_ndcg  # -1 because we want to maximize NDCG


class StochasticNeuralNDCGLossFunction(NeuralNDCGLossFunction):
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        return super(StochasticNeuralNDCGLossFunction, self).forward(
            y_pred=y_pred, y_true=y_true, stochastic=True
        )
