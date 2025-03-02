import torch
import torch.nn as nn

from .base import LossFunction


class SoftmaxLossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.float()
        y_true = y_true.float()

        exp_pred = torch.exp(y_pred)

        prob = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)

        loss = -torch.sum(y_true * torch.log(prob + 1e-10), dim=1)

        return torch.mean(loss)
