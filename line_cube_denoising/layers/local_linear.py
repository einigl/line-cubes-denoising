import math
from abc import ABC, abstractmethod

from torch import Tensor, ones
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.functional import linear
from torch.nn.parameter import Parameter

__all__ = ["LocalLinear", "TransposeLocalLinear"]

class LocalLayer(Module, ABC) :
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features_by_win : int
    out_features: int
    win_size : int
    stride : int

    padding : bool

    def __init__(self, in_features: int, out_features_by_win: int, win_size : int,
        stride : int = 1, padding : bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features_by_win = out_features_by_win
        self.win_size = win_size
        self.stride = stride

        self.padding = padding

        #if win_size % 2 == 0 :
        #    raise ValueError('win_size must be a odd number')

        if padding :
            self.n_win = math.ceil(self.in_features / self.stride)
            self.out_features = self.n_win * self.out_features_by_win
        else :
            self.n_win = math.ceil((self.in_features-self.win_size+1) / self.stride)
            self.out_features = self.n_win * self.out_features_by_win

    @abstractmethod
    def reset_parameters(self) :
        pass

    def create_weight(self, is_transposed : bool = False) -> None:
        W = Tensor(self.out_features, self.in_features)
        b = Tensor(self.out_features if not is_transposed else self.in_features)
        mask = ones((self.out_features, self.in_features))

        init.kaiming_uniform_(W, a = math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(W)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(b, -bound, bound)

        bounds = lambda k : slice(max(k*self.out_features_by_win, 0),\
            min((k+1)*self.out_features_by_win, self.out_features))
        offset = 0 if not self.padding else self.win_size // 2

        for k in range(self.n_win) :
            # Zero the weights outside windows
            W[bounds(k), :max(k*self.stride-offset, 0)] = 0
            W[bounds(k), k*self.stride+self.win_size-offset:] = 0
            # Zero the mask outside windows
            mask[bounds(k), :max(k*self.stride-offset, 0)] = 0
            mask[bounds(k), k*self.stride+self.win_size-offset:] = 0
        
        """import matplotlib.pyplot as plt
        plt.imshow(mask)
        plt.show()"""

        if is_transposed :
            return W.t(), b, mask.t()
        return W, b, mask

    def forward(self, input: Tensor) -> Tensor:
        return linear(input, self.weight, self.bias)

    def apply_grad_mask(self) -> None :
        self.weight.grad *= self.grad_mask

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LocalLinear(LocalLayer):

    weight: Tensor
    grad_mask: Tensor

    def __init__(self, in_features: int, out_features_by_win: int, win_size : int,
        stride : int = 1, bias: bool = True, padding : bool = True) :
        super().__init__(in_features, out_features_by_win, win_size, stride = stride, padding = padding)

        self.weight = Parameter(Tensor(self.out_features, self.in_features))
        self.grad_mask = ones((self.out_features, self.in_features))

        if bias:
            self.bias = Parameter(Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
            self.bias = None
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        W, b, mask = self.create_weight(is_transposed = False)
        self.weight = Parameter(W)
        self.bias = Parameter(b) if self.bias is not None else None
        self.grad_mask = mask

    def transpose(self) -> 'TransposeLocalLinear' :
        return TransposeLocalLinear(self.in_features, self.out_features_by_win, self.win_size,
        stride = self.stride, bias = self.bias is not None, padding = self.padding)


# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _LocalLinearWithBias(LocalLinear):
    bias: Tensor  # type: ignore

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)  # type: ignore


class TransposeLocalLinear(LocalLayer):

    weight: Tensor
    grad_mask: Tensor

    def __init__(self, in_features: int, out_features_by_win: int, win_size : int,
        stride : int = 1, bias: bool = True, padding : bool = True) :
        super().__init__(in_features, out_features_by_win, win_size, stride = stride, padding = padding)

        self.weight = Parameter(Tensor(self.in_features, self.out_features))
        self.grad_mask = ones((self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(Tensor(self.in_features))
        else:
            self.register_parameter('bias', None)
            self.bias = None
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        W, b, mask = self.create_weight(is_transposed = True)
        self.weight = Parameter(W)
        self.bias = Parameter(b) if self.bias is not None else None
        self.grad_mask = mask

    def transpose(self) -> 'LocalLinear' :
        return LocalLinear(self.in_features, self.out_features_by_win, self.win_size,
        stride = self.stride, bias = self.bias is None, padding = self.padding)

# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _TransposeLocalLinearWithBias(TransposeLocalLinear):
    bias: Tensor  # type: ignore

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)  # type: ignore