import torch
import torch.nn.functional as F

from .base import LossFunction


class BCELossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # return nn.BCEWithLogitsLoss(reduction='sum')(y_pred, y_true)
        y_pred = y_pred.float()
        y_true = y_true.float()

        loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

        return loss
