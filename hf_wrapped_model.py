from typing import List, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class MLPConfig(PretrainedConfig):
    model_type = "mlp"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: List[int] = [2048, 2048],
        output_dim: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


class SimpleMLP(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        layers = []
        layers.append(nn.Linear(config.input_dim, config.hidden_dim[0]))
        layers.append(nn.ReLU())

        for i in range(len(config.hidden_dim) - 1):
            layers.append(nn.Linear(config.hidden_dim[i], config.hidden_dim[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(config.hidden_dim[-1], config.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
