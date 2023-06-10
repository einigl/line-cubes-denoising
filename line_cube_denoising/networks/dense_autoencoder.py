""" Fully connected autoencoder """

from typing import List, Tuple, Optional

from torch import nn, Tensor

from .autoencoder import Autoencoder

__all__ = ["DenseAutoencoder"]

class DenseAutoencoder(Autoencoder) :
    """ Implementation of an standard fully connected autoencoder neural network """

    def __init__(self,
        half_description: List[int],
        activation: nn.Module,
        seed: Optional[int]=None
    ) :
        """
        Initializer
        
        Parameters
        ----------

        half_description : list of int
            Contains the number of neurons of each hidden layer of the encoder (decoder is symetric)

        activation : optional
            Activation function (default: nn.Sigmoid())

        seed : int
            Pseudo-random seed for reproducibility.
        """

        super().__init__(seed)

        self.input_size = half_description[0]
        self.bottleneck_size = half_description[-1]
        
        self.half_description = half_description
        self.activation = activation

        self.encoder_module = nn.ModuleList()
        self.decoder_module = nn.ModuleList()
        for i in range(len(self.half_description)-1):
            self.encoder_module.append(
                nn.Linear(
                    self.half_description[i],
                    self.half_description[i+1],
                )
            )
            self.decoder_module.insert(
                0, 
                nn.Linear(
                    self.half_description[i+1],
                    self.half_description[i],
                )
            )

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
