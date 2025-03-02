import torch
import torch.nn as nn

from .base import LossFunction


class ListMLELossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # y_pred = y_pred.reshape(1, -1)
        # y_true = y_true.reshape(1, -1)

        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(
            preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])

        observation_loss = torch.log(cumsums + 1e-10) - preds_sorted_by_true_minus_max

        return torch.mean(torch.sum(observation_loss, dim=1))
