import math

from torch import empty, zeros, Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn.functional import linear
from torch.nn import init

__all__ = ["BlockLinear"]

class BlockLinear(Module):
    in_features: int
    out_features: int
    grad_mask: Tensor
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, in_features_by_win : int,
        out_features_by_win : int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_by_win = in_features_by_win
        self.out_features_by_win = out_features_by_win

        self.weight = Parameter(empty((out_features, in_features)))
        self.grad_mask = zeros((out_features, in_features))
        if bias:
            self.bias = Parameter(empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.grad_mask = zeros((self.out_features, self.in_features))
        w1 = empty((self.out_features, self.in_features))
        init.kaiming_uniform_(self.weight, a = math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        if self.in_features % self.in_features_by_win != 0 :
            raise ValueError('in_features must be a multiple of in_features_by_win')
        n_win = self.in_features // self.in_features_by_win
        w2 = zeros((self.out_features, self.in_features))
        for k in range(n_win) :
            rows = slice(k*self.out_features_by_win, (k+1)*self.out_features_by_win)
            cols = slice(k*self.in_features_by_win, (k+1)*self.in_features_by_win)
            w2[rows, cols] = w1[rows, cols]
            self.grad_mask[rows, cols] = 1.

        self.weight = Parameter(w2)

    def forward(self, input: Tensor) -> Tensor:
        return linear(input, self.weight, self.bias)

    def apply_grad_mask(self) -> None :
        self.weight.grad *= self.grad_mask

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
