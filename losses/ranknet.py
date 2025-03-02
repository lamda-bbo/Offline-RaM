import torch
import torch.nn as nn

from .base import LossFunction


class RankNetLossFunction(LossFunction):

    def __init__(
        self, weight_by_diff: bool = False, weight_by_diff_powed: bool = False
    ) -> None:

        super(RankNetLossFunction, self).__init__()
        self.weight_by_diff = weight_by_diff
        self.weight_by_diff_powed = weight_by_diff_powed

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # y_pred = y_pred.view(-1)
        # y_true = y_true.view(-1)

        indices = torch.arange(y_true.size(0), device=y_true.device).unsqueeze(1)
        pairs_indices = indices.repeat(1, y_true.size(0))

        pairs_true = y_true[pairs_indices], y_true[pairs_indices.T]
        selected_pred = y_pred[pairs_indices], y_pred[pairs_indices.T]

        true_diffs = pairs_true[0] - pairs_true[1]
        pred_diffs = selected_pred[0] - selected_pred[1]

        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
        pred_diffs = pred_diffs[the_mask]

        weight = None
        if self.weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif self.weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[0], 2) - torch.pow(pairs_true[1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

        true_diffs = (true_diffs > 0).float()
        true_diffs = true_diffs[the_mask]

        return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)
