from torch import zeros, eye, Tensor
from torch.nn.modules import Module
from torch.nn.functional import linear

__all__ = [
    "AugmentationLinear",
    "TransposedAugmentationLinear"
]

class AugmentationLinear(Module) :
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""

    in_features: int
    win_size : int

    padding : bool

    def __init__(self, in_features: int, win_size : int) -> None:
        super().__init__()
        self.in_features = in_features
        self.win_size = win_size

        n_win = in_features - win_size + 1
        self.w = zeros((n_win*self.win_size, self.in_features))

        for k in range(n_win) :
            rows = slice(k*win_size, (k+1)*win_size)
            cols = slice(k, win_size+k)
            self.w[rows, cols] = eye(win_size)

    def forward(self, input: Tensor) -> Tensor:
        return linear(input, self.w, None)

class TransposedAugmentationLinear(Module) :
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""

    in_features: int
    win_size : int

    padding : bool

    def __init__(self, in_features: int, win_size : int) -> None:
        super().__init__()
        self.layer = AugmentationLinear(in_features, win_size)

    def forward(self, input: Tensor) -> Tensor:
        return linear(input, self.layer.w.T, None)
