"""
The code of this folder is partly inherited from 
allRank repository (https://github.com/allegro/allRank/). 
"""

from .base import LossFunction
from .mse import MSELossFunction
from .bce import BCELossFunction
from .lambdaRank import LambdaRankLossFunction
from .listmle import ListMLELossFunction
from .listnet import ListNetLossFunction
from .ranknet import RankNetLossFunction
from .approxNDCG import ApproxNDCGLossFunction
from .rankcosine import RankCosineLossFunction
from .sigmoid_ce import SigmoidCrossEntropyLossFunction
from .softmax import SoftmaxLossFunction
from .neuralNDCG import NeuralNDCGLossFunction, StochasticNeuralNDCGLossFunction

def get_loss_fn(loss_type: str, *args, **kwargs) -> LossFunction:
    loss_type = loss_type
    TYPE2LOSS = {
        "mse": MSELossFunction,
        "lambdarank": LambdaRankLossFunction,
        "listmle": ListMLELossFunction,
        "listnet": ListNetLossFunction,
        "ranknet": RankNetLossFunction,
        "bce": BCELossFunction,
        "approxndcg": ApproxNDCGLossFunction,
        "rankcosine": RankCosineLossFunction,
        "sigmoid_ce": SigmoidCrossEntropyLossFunction,
        "softmax": SoftmaxLossFunction,
        "neuralndcg": NeuralNDCGLossFunction,
        "stochastic_neuralndcg": StochasticNeuralNDCGLossFunction,
    }
    assert loss_type.lower() in TYPE2LOSS.keys(), \
        f"{loss_type} loss not found"
    
    return TYPE2LOSS[loss_type.lower()](*args, **kwargs)