import torch

from .base import _TKWARGS, LossFunction


class ApproxNDCGLossFunction(LossFunction):

    def __init__(self, alpha=1.0):
        super(ApproxNDCGLossFunction, self).__init__()
        self.alpha = alpha

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
        padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(_TKWARGS["device"])
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(
            min=1e-10
        )
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
        scores_diffs = y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]
        scores_diffs[~padded_pairs_mask] = 0.0
        approx_pos = 1.0 + torch.sum(
            padded_pairs_mask.float()
            * (torch.sigmoid(-self.alpha * scores_diffs).clamp(min=1e-10)),
            dim=-1,
        )
        approx_D = torch.log2(1.0 + approx_pos)
        approx_NDCG = torch.sum((G / approx_D), dim=-1)

        return -torch.mean(approx_NDCG)
