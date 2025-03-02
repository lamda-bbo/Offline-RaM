import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossFunction


class ListNetLossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # y_pred = y_pred.reshape(1, -1)
        # y_true = y_true.reshape(1, -1)

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + 1e-10
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
