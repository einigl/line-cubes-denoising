from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn, Tensor

__all__ = ["LossFunction", "ReconstructionLoss", "InformedLoss"]

class LossFunction(nn.Module, ABC) :
    """ Abstract class implementing a loss function """

    def __init__(self) :
        """ Initializer """
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x_hat: Tensor,
        y_hat: Tensor,
        x: Tensor,
        sigma: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, ...] :
        pass

class ReconstructionLoss(LossFunction) :
    """ Computes the reconstruction loss for the autoencoder neural network """

    normalize: bool

    def __init__(
        self,
        normalize: bool=True,
    ) :
        """ Initializer """
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        x_hat: Tensor,
        y_hat: Tensor,
        x: Tensor,
        sigma: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, ...] :
        if not self.normalize:
            sigma = 0. * sigma + 1.
        l = torch.mean( (x_hat - x)**2 / sigma**2 )
        return (l,)

    def __str__(self) :
        """ Returns str(self) """
        return 'Reconstruction loss (normalize: {})'.format(self.normalize)

class InformedLoss(LossFunction) :
    """ Computes an informed reconstruction loss for the autoencoder neural network """

    def __init__(self, q: float=1., normalize: bool=True) :
        """ Initializer """
        super().__init__()
        self.q = q
        self.normalize = normalize

    def forward(
        self,
        x_hat: Tensor,
        y_hat: Tensor,
        x: Tensor,
        sigma: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, ...] :
        n1 = mask.sum()
        n0 = mask.numel() - n1
        if not self.normalize:
            sigma = 0. * sigma + 1.
        l = 1/(n1+1) * (mask*(x_hat-x)/sigma).square().sum()\
             + 1/(n0+1) * ((1-mask)*x_hat/sigma).abs().float_power(self.q).sum()
        return (l,)

    def __str__(self) :
        """ Returns str(self) """
        return 'Informed loss (q: {}, normalize: {})'.format(self.q, self.normalize)
