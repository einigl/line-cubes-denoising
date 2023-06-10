import datetime
import random
from math import log
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..dataset import CubeDataset, CubeSubset
from ..networks import Autoencoder

from .loss_functions import LossFunction

__all__ = [
    "LearningParameters",
    "learning_procedure",
]

LOG10 = log(10)


class LearningParameters:
    """
    Description.

    Attributes
    ----------
    att : type
        Description.

    Methods
    -------
    meth()
        Description
    """

    loss_fun: LossFunction
    epochs: int
    batch_size: Optional[int]
    optimizer: Optimizer
    scheduler: _LRScheduler

    def __init__(
        self,
        loss_fun: LossFunction,
        epochs: int,
        batch_size: Optional[int],
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
    ):
        """
        Description.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        type
            Description.
        """
        self.loss_fun = loss_fun
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        if scheduler is None:
            self.scheduler = ConstantLR(optimizer, 1.0)
        else:
            self.scheduler = scheduler

    def __str__(self):
        """Returns str(self)"""
        s = "Learning parameters:\n"
        s += f"\tLoss function: {self.loss_fun}\n"
        s += f"\tEpochs: {self.epochs}\n"
        s += f"\tBatch size: {self.batch_size}\n"
        s += f"\tOptimizer: {self.optimizer}\n"
        s += f"\tScheduler: {self.scheduler}"
        return s


def learning_procedure(
    model: Autoencoder,
    dataset: Union[CubeDataset, Tuple[CubeDataset, CubeDataset]],
    learning_parameters: Union[LearningParameters, List[LearningParameters]],
    train_samples: Optional[Sequence] = None,
    val_samples: Optional[Sequence] = None,
    val_frac: Optional[float] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    max_iter_no_improve: Optional[int] = None,
) -> Dict[str, object]:
    """
    Description.

    Parameters
    ----------
    param : type
        Description.

    Returns
    -------
    type
        Description.
    """

    # Start counter
    tic = datetime.datetime.now()

    if isinstance(dataset, CubeDataset):
        pass
    elif (
        isinstance(dataset, Sequence)
        and len(dataset) == 2
        and isinstance(dataset[0], CubeDataset)
        and isinstance(dataset[1], CubeDataset)
    ):
        pass
    else:
        raise TypeError(
            f"dataset must be an instance of RegressionDataset or a tuple of two RegressionDataset, not {type(dataset)}"
        )

    if max_iter_no_improve is not None:
        assert isinstance(max_iter_no_improve, int)
        assert max_iter_no_improve >= 1

    if seed is not None:
        random.seed(seed)

    if isinstance(dataset, CubeDataset):

        if train_samples is not None and val_samples is not None:
            pass

        if train_samples is not None and val_samples is None:
            pass

        if train_samples is None and val_samples is None and val_frac is not None:
            n_val = round(val_frac * len(dataset))
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            train_samples, val_samples = indices[n_val:], indices[:n_val]

        if train_samples is None and val_samples is None and val_frac is None:
            train_samples, val_samples = list(range(len(dataset))), None

        if train_samples is not None and val_samples is not None:
            intersect = set(train_samples) & set(val_samples)
            if len(intersect) > 0:
                raise ValueError(
                    "Train dataset and validation dataset must not share samples, here {intersect}"
                )

        train_set = CubeSubset(dataset, train_samples)
        val_set = (
            CubeSubset(dataset, val_samples) if val_samples is not None else None
        )

    else:
        train_set = dataset[0]
        val_set = dataset[1]

    # Training loop

    if verbose:
        count = model.count_parameters()
        size, unit = model.count_bytes()
        print("Training initiated")
        print(
            f"{model}: {count:,} learnable parameters ({size:.2f} {unit})", end="\n\n"
        )

    if isinstance(learning_parameters, LearningParameters):
        learning_parameters = [learning_parameters]
    elif not isinstance(learning_parameters, (list, tuple)):
        raise ValueError(
            "learning_parameters must be an instance of LearningParameters or a list of Learning Parameters"
        )
    elif any(not isinstance(p, LearningParameters) for p in learning_parameters):
        raise ValueError(
            "learning_parameters must be an instance of LearningParameters or a list of Learning Parameters"
        )

    train_loss = []
    val_loss = []
    lr = []

    for learning_parameter in learning_parameters:

        epochs = learning_parameter.epochs
        batch_size = learning_parameter.batch_size
        if batch_size is None:
            batch_size = len(train_set)
        optimizer = learning_parameter.optimizer
        scheduler = learning_parameter.scheduler

        loss_fun = learning_parameter.loss_fun

        # Dataloaders
        
        dataloader_train = DataLoader(
            train_set,
            batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        dataloader_train_eval = DataLoader(
            train_set,
            len(train_set),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        dataloader_val_eval = DataLoader(
            val_set,
            len(val_set),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        n_batchs_train = len(dataloader_train)
        n_batchs_train_eval = 1
        n_batchs_val_eval = 1

        pbar_epoch = tqdm(range(epochs), disable=not verbose)
        pbar_epoch.set_description("Epoch")

        for epoch in pbar_epoch:

            print("x", end=" ")

            # Dataloader            
            n_batchs_train = len(dataloader_train)

            lr.append(optimizer.param_groups[0]["lr"])

            # Training
            model.train()

            pbar_batch = tqdm(
                enumerate(dataloader_train),
                leave=False,
                total=n_batchs_train,
                disable=not verbose,
            )
            pbar_batch.set_description("Batch (training)")
            for _, batch in pbar_batch:
                optimizer.zero_grad(set_to_none=True)

                loss = _batch_processing(
                    model,
                    batch,
                    loss_fun,
                )[0]

                loss.backward()
                optimizer.step()

                pbar_batch.set_postfix({"loss": loss.item()})

            model.eval()

            # Evaluation on train set

            sizes = []
            memory_loss = []

            pbar_batch = tqdm(
                enumerate(dataloader_train_eval),
                leave=False,
                total=n_batchs_train_eval,
                disable=not verbose,
            )
            pbar_batch.set_description("Batch (model eval)")
            for _, batch in pbar_batch:
                sizes.append(batch[0].size(0))

                with torch.no_grad():
                    loss = _batch_processing(
                        model,
                        batch,
                        loss_fun,
                    )[0]

                memory_loss.append(loss.item())

            n_tot = sum(sizes)
            train_loss.append(sum([s / n_tot * l for l, s in zip(memory_loss, sizes)]))

            # Evaluation on validation set

            sizes = []
            memory_loss = []

            pbar_batch = tqdm(
                enumerate(dataloader_val_eval),
                leave=False,
                total=n_batchs_val_eval,
                disable=not verbose,
            )
            pbar_batch.set_description("Batch (validation)")

            for _, batch in pbar_batch:
                sizes.append(batch[0].size(0))

                with torch.no_grad():
                    loss = _batch_processing(
                        model,
                        batch,
                        loss_fun,
                    )[0]

                memory_loss.append(loss.item())

            n_tot = sum(sizes)
            val_loss.append(sum([s / n_tot * l for l, s in zip(memory_loss, sizes)]))

            # End of epoch
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_loss[-1])
            else:
                scheduler.step()

            pbar_epoch.set_postfix(
                {
                    "train loss": train_loss[-1],
                    "val loss": val_loss[-1],
                }
            )

            # Early stopping
            if (
                max_iter_no_improve is not None
                and epoch > max_iter_no_improve
                and np.min(train_loss[-max_iter_no_improve:]) > np.min(train_loss)
            ):
                break

    model.eval()

    toc = datetime.datetime.now()

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_set": train_set,
        "val_set": val_set,
        "lr": lr,
        "duration": toc - tic,
    }


def _batch_processing(
    model: Autoencoder,
    batch: Optional[torch.Tensor],
    loss_fun: LossFunction,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    x, m, sigma = batch
    x_hat, y_hat = model.forward(x)

    return loss_fun.forward(x_hat, y_hat, x, sigma, m)
