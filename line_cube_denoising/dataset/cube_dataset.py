""" Dataset """

from typing import Optional, Sequence, Tuple

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = [
    "CubeDataset",
    "CubeSubset",
]

class CubeDataset(Dataset) :
    """ TODO """
    
    def __init__(self, x : np.ndarray, mask : Optional[np.ndarray] = None,
                 sigma : Optional[np.ndarray] = None) :
        """ TODO """
        super().__init__()

        self.nz, self.ny, self.nx = x.shape
        self.n_samples = self.nx * self.ny
        self.n_features = self.nz

        self.x = torch.from_numpy( x.reshape((self.n_features, self.n_samples)).T ).float()

        if mask is None :
            self.mask = torch.ones_like(self.x).float()
        else :
            self.mask = torch.from_numpy( mask.reshape((self.n_features, self.n_samples)).T ).float()
        
        if sigma is None :
            self.sigma = torch.ones_like(self.x[0]).float()
        else :
            self.sigma = torch.from_numpy( sigma.reshape((1, self.n_samples)).T ).float()

        print(f'Dataset created: {self.n_features} features, {self.n_samples} samples')

    def __len__(self) -> int:
        """ Returns len(self) """
        return self.n_samples

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor] :
        """ Returns self[index] """
        return self.x[index, :], self.mask[index, :], self.sigma[index, :]

    def cube_from_samples(self, samples : np.ndarray, nz : Optional[int] = None) :
        """ TODO """
        if nz is None :
            return np.moveaxis(samples.reshape(self.ny, self.nx, self.nz), -1, 0)
        return np.moveaxis(samples.reshape(self.ny, self.nx, nz), -1, 0)

    def map_from_samples(self, samples : np.ndarray) :
        """ TODO """
        return samples.reshape(self.ny, self.nx)

class CubeSubset(CubeDataset) :
    """ TODO """

    def __init__(self, dataset : CubeDataset, indices : Sequence[int]) -> None :
        """ Initializer """
        self.dataset : CubeDataset = dataset
        self.indices : Sequence[int] = indices

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)