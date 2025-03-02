import torch
import torch.nn as nn

from .base import LossFunction


class SigmoidCrossEntropyLossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.float()
        y_true = y_true.float()

        loss = (
            torch.clamp(y_pred, min=0)
            - y_pred * y_true
            + torch.log1p(torch.exp(-torch.abs(y_pred)))
        )

        return torch.mean(loss)
