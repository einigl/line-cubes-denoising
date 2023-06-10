""" Autoencoder """

import torch
import torch.nn as nn

from ..layers import AugmentationLinear, BlockLinear

from .autoencoder import Autoencoder

__all__ = ["LocalAutoencoder"]

class LocalAutoencoder(Autoencoder) :
    """ Implementation of an autoencoder artificial neural network """

    def __init__(self, dataloader, loss, input_size, win_size, half_description_by_win,
                 activation = nn.ReLU(), learning_rate = 1e-2, batch_size = 10,
                 optimizer_name = 'SGD', scheduler_name = None, scheduler_params = None,
                 seed = 0) :
        
        super().__init__(dataloader, seed)

        self.input_size = input_size
        self.win_size = win_size
        self.half_description_by_win = half_description_by_win

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

        self.n_win = input_size - win_size + 1

        self.augmentation_layer = AugmentationLinear(input_size, win_size)

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

        self.combination_layer = nn.Linear(win_size * self.n_win, input_size)

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

        x_aug = self.augmentation(x)

        y_hat = self.encoder(x_aug)
        x_aug_hat = self.decoder(y_hat)

        #x_hat = self.combination(x_aug_hat.detach())
        x_hat = self.combination(x_aug_hat)

        outputs = {
            'lines' : {
                'line' : x_hat
            },
            'augmented input' : {
                'line' : x_aug,
            },
            'augmented output' : {
                'line' : x_aug_hat,
            },
            'bottleneck' : {
                'line' : y_hat
            }
        }

        return outputs

    def augmentation(self, x) :
        """ TODO """
        return self.augmentation_layer(x)

    def encoder(self, x_aug) :
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

        y_hat = x_aug

        for layer in self.encoder_module :
            y_hat = layer(y_hat)
            y_hat = self.activation(y_hat)

        #yHat = self.sig(yHat)
        #yHat = self.relu(yHat)

        return y_hat

    def decoder(self, y_hat) :
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
        x_aug_hat = y_hat

        for layer in self.decoder_module[:-1] :
            x_aug_hat = layer(x_aug_hat)
            x_aug_hat = self.activation(x_aug_hat)
        x_aug_hat = self.decoder_module[-1](x_aug_hat)

        #xHat = self.relu(xHat)

        return x_aug_hat

    def combination(self, x_aug_hat) :
        """ TODO """
        return self.combination_layer(x_aug_hat)

    def backward(self, loss: torch.Tensor, optimizer, optimizer_idx: int, *args, **kwargs) -> None :
        """ TODO """
        loss.backward()

        for layer in self.encoder_module :
            layer.apply_grad_mask()
        for layer in self.decoder_module :
            layer.apply_grad_mask()

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