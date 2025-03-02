import torch
import torch.nn as nn

from .base import LossFunction


class RankCosineLossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.float()
        y_true = y_true.float()

        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True)

        y_pred_centered = y_pred - y_pred_mean
        y_true_centered = y_true - y_true_mean

        numerator = torch.sum(y_pred_centered * y_true_centered, dim=1)
        denominator = torch.sqrt(torch.sum(y_pred_centered**2, dim=1)) * torch.sqrt(
            torch.sum(y_true_centered**2, dim=1)
        )

        cosine_similarities = numerator / (denominator + 1e-8)

        cosine_distances = 1 - cosine_similarities

        return torch.mean(cosine_distances)
