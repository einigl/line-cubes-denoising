""" Autoencoder """

import torch
import torch.nn as nn

from ..layers import LocalLinear, TransposeLocalLinear

from .autoencoder import Autoencoder

from .autoencoder import Autoencoder

__all__ = ["LocalAutoencoder"]

class LocalAutoencoder(Autoencoder) :
    """ Implementation of an autoencoder artificial neural network """

    def __init__(self, dataloader, loss, input_size, reduction_factor, augmentation_factor, win_size,
                 activation = nn.ReLU(), learning_rate = 1e-2, batch_size = 10,
                 optimizer_name = 'SGD', scheduler_name = None, scheduler_params = None,
                 dropout_p = 0, seed = 0) :
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

        learning_rate : float, optional
            Learning rate (default: 1e-2)

        batch_size : int, optional
            Batch size (default: 10)

        optimizer_name : {'SGD', 'adam', 'RMSprop', None}, optional
            Specifies the optimizer the network has to use (default: 'SGD').

        scheduler_name : {'steplr', 'lambdalr', 'multiplicativelr', 'reduce_lr_on_plateau', None}, optional
            Specifies the scheduler the network has to use (default: None)

        scheduler_params : Iterable, optional
            Parameters of chosen scheduler. The needed number of element depends on the scheduler.

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

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.optimizer_name = optimizer_name.strip().lower()
        self.scheduler_name = scheduler_name
        if self.scheduler_name is not None :
            self.scheduler_name = self.scheduler_name.lower()
        self.scheduler_params = scheduler_params

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

        #self.linear_1 = nn.Linear(self.bott_size, self.bott_size // 2)
        #self.linear_2 = nn.Linear(self.bott_size // 2, self.bott_size)

        self.decoder_2 = self.encoder_2.transpose()
        self.decoder_1 = self.encoder_1.transpose()

        #self.encoder_inter_layer = LocalLinear(self.hidden_size, self.augmentation_factor, self.augmentation_factor,
        #    stride = self.augmentation_factor, padding = False)
        #self.decoder_inter_layer = LocalLinear(self.hidden_size, self.augmentation_factor, self.augmentation_factor,
        #    stride = self.augmentation_factor, padding = False)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, inputs) :
        """
        Computes the output value of the autoencoder for input `x`

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        xHat : torch.Tensor
            Reconstruction of `x` by the autoencoder
            
        yHat : torch.Tensor
            Content in the bottleneck latent space
        """

        x = inputs['lines']['line']

        y_hat = self.encoder(x)
        #print('y_hat :', y_hat)
        #print('-----------------')
        x_hat = self.decoder(y_hat)
        #print('x_hat :', x_hat)

        outputs = {
            'lines' : {
                'line' : x_hat
            },
            'bottleneck' : {
                'line' : y_hat
            }
        }

        return outputs

    def encoder(self, x) :
        """
        Encoder part of the autoencoder 
        
        Parameters
        ----------
        x : torch.Tensor
            Input

        Returns
        -------
        yHat : torch.Tensor
            Encoded `x`
        """

        yHat = x

        yHat = self.encoder_1(yHat)
        yHat = self.activation(yHat)

        #yHat = self.encoder_inter_layer(yHat)
        #yHat = self.activation(yHat)

        yHat = self.encoder_2(yHat)
        yHat = self.activation(yHat)

        #yHat = self.linear_1(yHat)

        #yHat = self.sig(yHat)
        #yHat = self.relu(yHat)

        return yHat

    def decoder(self, yHat) :
        """
        Decoder part of the autoencoder 
        
        Parameters
        ----------
        yHat : torch.Tensor
            Encoded input

        Returns
        -------
        xHat : torch.Tensor
            Decoded `yHat`
        """
        xHat = yHat

        #xHat = self.linear_2(xHat)

        xHat = self.decoder_2(xHat)
        xHat = self.activation(xHat)

        #xHat = self.decoder_inter_layer(xHat)
        #xHat = self.activation(xHat)

        xHat = self.decoder_1(xHat)

        #xHat = self.relu(xHat)

        return xHat

    def backward(self, loss: torch.Tensor, optimizer, optimizer_idx: int, *args, **kwargs) -> None :
        """ TODO """
        loss.backward()
        self.encoder_1.apply_grad_mask()
        self.encoder_2.apply_grad_mask()
        self.decoder_1.apply_grad_mask()
        self.decoder_2.apply_grad_mask()

        #self.encoder_inter_layer.apply_grad_mask()
        #self.decoder_inter_layer.apply_grad_mask()

    def training_step(self, batch, _) :
        """
        Computes the output and the loss for a batch of the training set
        
        Parameters
        ----------
        batch : torch.Tensor
            Set of input of size N x M where N is the batch size and M the dimension of input space

        Returns
        -------
        loss : float
            Loss of `batch`
        """
        outputs = self.forward(batch)

        l = self.loss.add_training_batch( batch, outputs )
        return l

    def validation_step(self, batch, _) :
        """
        Computes the output and the loss for a batch of the validation set
        
        Parameters
        ----------
        batch : torch.Tensor
            Set of input of size N x M where N is the batch size and M the dimension of input space

        Returns
        -------
        loss : float
            Loss of `batch`
        """
        outputs = self.forward(batch)

        l = self.loss.add_validation_batch( batch, outputs )
        return l

    def configure_optimizers(self) :
        """
        Chooses what optimizers and learning-rate schedulers to use in your optimization.

        Returns
        -------
        optim : torch.optim.Optimizer
            Optimizer for training step

        sched : torch.optim.lr_scheduler._LRScheduler
            Scheduler for training step
        """

        # Optimizer
        if self.optimizer_name == 'sgd' :
            print('SGD optimizer')
            optim = torch.optim.SGD(self.parameters(), self.learning_rate)
        elif self.optimizer_name == 'adam' :
            print('Adam optimizer')
            optim = torch.optim.Adam(self.parameters(), self.learning_rate)
        elif self.optimizer_name == 'rmsprop' :
            print('RMSprop optimizer')
            optim = torch.optim.RMSprop(self.parameters(), self.learning_rate, alpha = 0.9, eps = 1e-7)
        else :
            print('Unknown optimizer, SGD by default')
            optim = torch.optim.SGD(self.parameters(), self.learning_rate)

        # Scheduler
        if self.scheduler_name is None :
            print('No learning rate scheduler')
            return [optim]
        if self.scheduler_name == 'reduce_lr_on_plateau' :
            print('Reduce learning rate on plateau scheduler')
            sched = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim),
                     'interval': 'step',
                     'monitor': 'train_loss'}
        elif self.scheduler_name == 'steplr' :
            print('StepLR learning rate scheduler')
            sched = torch.optim.lr_scheduler.StepLR(optim, self.scheduler_params[0], self.scheduler_params[1])
        elif self.scheduler_name == 'lambdalr' :
            print('LambdaLR learning rate scheduler')
            sched = torch.optim.lr_scheduler.LambdaLR(optim, self.scheduler_params[0])
        elif self.scheduler_name == 'multiplicativelr' :
            print('MultiplicativeLR learning rate scheduler')
            sched = torch.optim.lr_scheduler.MultiplicativeLR(optim, self.scheduler_params[0])
        else :
            print('Unknown scheduler, None by default')
            return [optim]

        return [optim], [sched]
        
    def on_epoch_end(self) :
        """ Hook which is called at the end of every epochs """

        # Compute mean losses
        self.loss.epoch_end()

    def __str__(self) -> str :
        """ Returns str(self) """

        return f"""Autoencoder artificial neural network.

Architecture :
    Input size : ' {self.input_size}
    Bottleneck size : {self.bott_size}
    Augmentation factor : {self.augmentation_factor}
    Activation function : {self.activation}
    Seed : {self.seed}

Training parameters :
    Learning rate : {self.learning_rate}
    BatchSize : {self.batch_size}
    Optimizer : {self.optimizer_name}
    Scheduler : {self.scheduler_name}
    Scheduler parameters : {self.scheduler_params}

Loss function : {self.loss}

Data loader : {self.dataloader}"""
# add printable for scheduler parameters