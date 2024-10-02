import torch 
import torch.nn as nn 
from typing import List, Union

class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[List[int], int],
        output_dim: int = 1,
    ) -> None:
        super(SimpleMLP, self).__init__()
        
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        
        layers = [] 
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_dim)-1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)