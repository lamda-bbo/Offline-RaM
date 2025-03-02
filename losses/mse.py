import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossFunction


class MSELossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        return F.mse_loss(
            input=y_pred.squeeze(), target=y_true.squeeze(), reduction="mean"
        )
