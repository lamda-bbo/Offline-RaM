from typing import Optional

import torch
import torch.nn as nn

from .base import _TKWARGS, LossFunction


def lambdaRank_scheme(G, D, *args):
    return torch.abs(
        torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
    ) * torch.abs(G[:, :, None] - G[:, None, :])


class LambdaRankLossFunction(LossFunction):

    def __init__(
        self,
        weighing_scheme: Optional[str] = None,
        k: Optional[int] = None,
        sigma: float = 1.0,
        mu: float = 10.0,
        reduction: str = "sum",
        reduction_log: str = "binary",
    ):
        super(LambdaRankLossFunction, self).__init__()

        self.weighing_scheme = weighing_scheme
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # y_pred = y_pred.reshape(1, -1)
        # y_true = y_true.reshape(1, -1)

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if self.weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]),
            device=_TKWARGS["device"],
            dtype=torch.bool,
        )
        ndcg_at_k_mask[: self.k, : self.k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device=_TKWARGS["device"])
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, : self.k], dim=-1
        ).clamp(min=1e-10)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if self.weighing_scheme is None:
            weights = 1.0
        else:
            weights = globals()[self.weighing_scheme](G, D, self.mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(
            min=-1e8, max=1e8
        )
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.0)
        weighted_probas = (
            torch.sigmoid(self.sigma * scores_diffs).clamp(min=1e-10) ** weights
        ).clamp(min=1e-10)
        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if self.reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif self.reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss
