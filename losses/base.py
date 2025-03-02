import abc

import torch
import torch.nn as nn

_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


class LossFunction(nn.Module):

    @abc.abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass

    def score(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred
