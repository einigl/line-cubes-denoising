""" Autoencoder """

from typing import List, Tuple, Optional

import torch
from torch import nn, Tensor

from ..layers import LocalLinear, TransposeLocalLinear

from .autoencoder import Autoencoder

from .autoencoder import Autoencoder

__all__ = ["LocalAutoencoder"]

class LocalAutoencoder(Autoencoder) :
    """ Implementation of an autoencoder artificial neural network """

    def __init__(self, dataloader, loss, input_size, reduction_factor, augmentation_factor, win_size, activation = nn.ReLU(), seed = 0) :
        """
        Initializer
        
        Parameters
        ----------
        dataloader : pytorch_lightning.LightningDataModule
            Data loader

        half_description : list of int
            Contains the number of neurons of each hidden layer of the encoder (decoder is symetric)

        loss : AutoencoderLoss
            Loss

        activation : optional
            Activation function (default: nn.Sigmoid())

        seed : int
            Pseudo-random seed for reproducibility.
        """

        super().__init__(dataloader, seed)

        self.input_size = input_size
        self.hidden_size = augmentation_factor * input_size
        self.bott_size = round(input_size / reduction_factor)

        self.reduction_factor = reduction_factor
        self.augmentation_factor = augmentation_factor
        self.win_size = win_size

        self.activation = activation
        self.loss = loss

        #self.example_input_array = self.dataLoader.train_dataset()[0]['lines']['line']

        out_1 = self.augmentation_factor
        win_size_1 = self.win_size
        stride_1 = 1
        out_2 = 1
        win_size_2 = self.augmentation_factor * self.reduction_factor# * 2
        stride_2 = win_size_2# // 2

        self.encoder_1 = LocalLinear(self.input_size, out_1,
            win_size = win_size_1, stride = stride_1, padding = True)
        self.encoder_2 = LocalLinear(self.hidden_size, out_2,
            win_size = win_size_2, stride = stride_2, padding = False)

        self.decoder_2 = self.encoder_2.transpose()
        self.decoder_1 = self.encoder_1.transpose()

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

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

        y_hat = self.encoder(x)
        x_hat = self.decoder(y_hat)

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
        
        for layer in self.encoder_module[:-1] :
            y_hat = layer(y_hat)
            y_hat = self.activation(y_hat)
        y_hat = self.encoder_module[-1](y_hat)

        return y_hat

    def decoder(self, y_hat) -> Tensor:
        """
        Decoder part of the autoencoder 
        
        Parameters
        ----------
        y_hat : torch.Tensor
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

    def __str__(self) -> str :
        """ Returns str(self) """
        return "Dense autoencoder artificial neural network"


    def backward(self, loss: torch.Tensor, optimizer, optimizer_idx: int, *args, **kwargs) -> None :
        """ TODO """
        loss.backward()
        self.encoder_1.apply_grad_mask()
        self.encoder_2.apply_grad_mask()
        self.decoder_1.apply_grad_mask()
        self.decoder_2.apply_grad_mask()

        #self.encoder_inter_layer.apply_grad_mask()
        #self.decoder_inter_layer.apply_grad_mask()
