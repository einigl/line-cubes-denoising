""" Locally connected autoencoder """

from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor

from ..layers import AugmentationLinear, TransposedAugmentationLinear, BlockLinear

from .autoencoder import Autoencoder

__all__ = ["LocalAutoencoder"]

class LocalAutoencoder(Autoencoder) :
    """ Implementation of an local autoencoder artificial neural network """

    def __init__(self,
            input_size: int,
            bottleneck_size: int,
            win_size: int,
            half_description_by_win: List[int],
            activation: nn.Module,
            seed: Optional[int] = None
        ) :
        
        super().__init__(seed)

        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.win_size = win_size
        self.half_description_by_win = half_description_by_win

        self.activation = activation

        self.n_win = input_size - win_size + 1

        self.augmentation_layer = AugmentationLinear(input_size, win_size)
        self.transposed_augmentation_layer = TransposedAugmentationLinear(input_size, win_size)

        self.encoder_module = nn.ModuleList()
        for i in range(1, len(self.half_description_by_win)) :
            self.encoder_module.append(
                BlockLinear(
                    self.n_win*self.half_description_by_win[i-1],
                    self.n_win*self.half_description_by_win[i],
                    self.half_description_by_win[i-1],
                    self.half_description_by_win[i]
                )
            )

        self.decoder_module = nn.ModuleList()
        for i in range(len(self.half_description_by_win)-1, 0, -1) :
            self.decoder_module.append(
                BlockLinear(
                    self.n_win*self.half_description_by_win[i],
                    self.n_win*self.half_description_by_win[i-1],
                    self.half_description_by_win[i],
                    self.half_description_by_win[i-1]
                )
            )

        self.projection_layer = nn.Linear(self.encoder_module[-1].out_features, bottleneck_size)
        self.transposed_projection_layer = nn.Linear(bottleneck_size, self.decoder_module[0].in_features)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor] :
        """
        Computes the output value of the autoencoder for input `x`

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        x_hat : torch.Tensor
            Reconstruction of `x` by the autoencoder
            
        y_hat : torch.Tensor
            Content in the bottleneck latent space
        """
        y_hat = self.augmentation_layer(x)
        y_hat = self.encoder(y_hat)
        y_hat = self.projection_layer(y_hat)

        x_hat = self.transposed_projection_layer(y_hat)
        x_hat = self.decoder(x_hat)
        x_hat = self.transposed_augmentation_layer(x_hat)

        return x_hat, y_hat

    def encoder(self, x: Tensor) -> Tensor:
        """
        Encoder part of the autoencoder 
        
        Parameters
        ----------
        x : torch.Tensor
            Input

        Returns
        -------
        y_hat : torch.Tensor
            Encoded `x`
        """

        y_hat = x

        for layer in self.encoder_module :
            y_hat = layer(y_hat)
            y_hat = self.activation(y_hat)

        return y_hat

    def decoder(self, y_hat: Tensor) -> Tensor:
        """
        Decoder part of the autoencoder 
        
        Parameters
        ----------
        yHat : torch.Tensor
            Encoded input

        Returns
        -------
        x_hat : torch.Tensor
            Decoded `y_hat`
        """
        x_hat = y_hat

        for layer in self.decoder_module[:-1] :
            x_hat = layer(x_hat)
            x_hat = self.activation(x_hat)
        x_hat = self.decoder_module[-1](x_hat)

        return x_hat

    def backward(self,
            loss: torch.Tensor,
            optimizer,
            optimizer_idx: int,
            *args, **kwargs
        ) -> None :
        """ TODO """
        loss.backward()

        for layer in self.encoder_module :
            layer.apply_grad_mask()
        for layer in self.decoder_module :
            layer.apply_grad_mask()

    def __str__(self) -> str :
        """ Returns str(self) """
        return "Local autoencoder artificial neural network"
