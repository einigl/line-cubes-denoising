""" Astro autoencoder abstract class """

import os
import time
import shutil
import json
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Literal
from inspect import signature
from collections import OrderedDict

import torch
from torch import nn, Tensor

MAX_BATCH_SIZE = 10_000

__all__ = ["Autoencoder"]

class Autoencoder(nn.Module, ABC) :
    """ Implementation of an autoencoder artificial neural network """

    def __init__(self, seed: Optional[int]=None) :
        """
        Initializer
        
        Parameters
        ----------
        """

        super().__init__()
        
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the output value and the bottleneck of the autoencoder for input `x`.

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
        pass

    def count_parameters(self, learnable_only: bool = True) -> int:
        """
        Returns the number of parameters of the module.
        If `learnable_only` is True, then this function returns the number of parameters whose has a `requires_grad = True` property.
        If `learnable_only` is False, then this function returns the number of parameters, independently to their `requires_grad` property.

        Parameters
        ----------
        learnable_only : bool, optional
            Indicates the the type of parameter to count. Defaults to True.

        Returns
        -------
        int
            Number of parameters.
        """
        self.train()
        if learnable_only:
            count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        count = sum(p.numel() for p in self.parameters())
        self.eval()
        return count

    def count_bytes(
        self,
        learnable_only: bool = True,
        display: bool = False,
    ) -> Union[Tuple[int, Literal["b", "kb", "Mb", "Gb", "Tb", "Pb"]], str]:
        """
        Returns the number of parameters of the module.
        If `learnable_only` is True, then this function returns the number of parameters whose has a `requires_grad = True` property.
        If `learnable_only` is False, then this function returns the number of parameters, independently to their `requires_grad` property.

        Parameters
        ----------
            learnable_only (bool, optional): Indicates the the type of parameter to count. Defaults to True.

        Returns
        -------
        int
            Number of parameters.
        str
            Unit ('b', 'kb', 'Mb', 'Gb', 'Tb')
        """
        self.train()
        size = 0.0
        for p in self.parameters():
            if p.requires_grad or not learnable_only:
                size += (
                    p.numel() * p.element_size()
                )  # Does not take into consideration the Python object size which can be obtain using sys.getsizeof()
        self.eval()

        for (v, u) in [(1e0, "B"), (1e3, "kB"), (1e6, "MB"), (1e9, "GB"), (1e12, "TB")]:
            if size < 1e3 * v:
                if display:
                    return f"{size / v:.2f} {u}"
                else:
                    return (size / v, u)

    def time(self, n: int, repeat: int) -> Tuple[float, float, float]:
        """
        Compute the evaluation time of the model for a batch of `n` inputs. Returns the average, min and max durations (in sec) over `repeat` iterations.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, not {type(n)}")
        if not isinstance(repeat, int):
            raise TypeError(f"repeat must be an integer, not {type(repeat)}")
        times = []
        for it in range(repeat):
            x = torch.normal(0., 1., size=(n, self.input_features))
            tic = time.time()
            self.forward(x)
            toc = time.time()
            times.append(toc-tic)
        return sum(times)/repeat, min(times), max(times)
    
    def save(self,
        module_name: str,
        module_path: Optional[str] = None,
        overwrite: bool = True
    ) -> None:
        if module_path is None:
            path = module_name
        else:
            path = os.path.join(module_path, module_name)

        if not os.path.exists(path):
            pass
        elif not overwrite:
            raise FileExistsError(f'{path} already exists')
        else:
            for _, __, f in os.walk(path):
                if not all(os.path.isdir(file) or file.endswith(('.json', '.pkl', '.pth')) for file in f):
                    raise ValueError(f"{path} directory cannot be overwritten because it doesn't seem to be a NeuralNetwork save directory")
            shutil.rmtree(path)
        
        Autoencoder._recursive_save(self, path)

    @staticmethod
    def _recursive_save(
        module: nn.Module,
        path: str,
    ) -> List[str]:
        os.mkdir(path)
        template = os.path.join(path, '{}')

        args = list(signature(module.__init__).parameters)
        with open(template.format("init.pkl"), "wb") as f:
            pickle.dump((type(module), args), f)

        keys = []
        for arg in args:
            obj = getattr(module, arg)
            if Autoencoder._needs_recursion(obj):
                new_keys = Autoencoder._recursive_save(obj, os.path.join(path, arg))
                keys.extend(new_keys)
            else:
                if Autoencoder._needs_json(obj):
                    with open(template.format(f"{arg}.json"), "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=4)
                else :
                    with open(template.format(f"{arg}.pkl"), "wb") as f:
                        pickle.dump(obj, f)

        sd = module.state_dict()
        sd = OrderedDict([(key, val) for key, val in sd.items() if not key in keys])

        torch.save(sd, template.format('state_dict.pth'))

        return keys

    @staticmethod
    def _needs_recursion(obj: object) -> bool:
        """ Returns True of obj is an object that need to be saved recursively, else False """
        return isinstance(obj, Autoencoder)

    @staticmethod
    def _needs_json(obj: object) -> bool:
        """ Returns True if the object `obj` must be saved in a JSON file. """
        if isinstance(obj, (bool, int, float, complex, str)):
            return True
        if isinstance(obj, (list, tuple)):
            return all(Autoencoder._needs_json(v) for v in obj)
        return False
        
    @classmethod
    def load(self, module_name: str, module_path: Optional[str] = None) -> "Autoencoder":
        if module_path is None:
            path = module_name
        else:
            path = os.path.join(module_path, module_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} directory not exist")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory")

        return Autoencoder._recursive_load(path)

    @staticmethod
    def _recursive_load(path: str) -> "Autoencoder":
        template = os.path.join(path, '{}')

        with open(template.format("init.pkl"), "rb") as f:
            type_module, args = pickle.load(f)

        d = {}
        for arg in args:
            if os.path.exists(template.format(f"{arg}.json")):
                with open(template.format(f"{arg}.json"), "r") as f:
                    d.update({arg: json.load(f)})
            elif os.path.exists(template.format(f"{arg}.pkl")):
                with open(template.format(f"{arg}.pkl"), "rb") as f:
                    d.update({arg: pickle.load(f)})
            elif os.path.isdir(template.format(arg)):
                d.update({arg: Autoencoder._recursive_load(template.format(arg))})
            else:
                raise RuntimeError("Should never been here.")

        module: nn.Module = type_module(**d)

        sd = module.state_dict()
        sd.update(torch.load(template.format('state_dict.pth')))
        module.load_state_dict(sd)

        return module

    def copy(self):
        """ TODO """
        d = {name: getattr(self, name) for name in list(signature(self.__init__).parameters)}
        return type(self)(**d) # TODO: state dict

    def __str__(self) -> str:
        d = list(signature(self.__init__).parameters)
        descr = f'{type(self).__name__}:\n'
        for arg in d:
            obj = getattr(self, arg)
            if isinstance(obj, list) and len(obj) > 6:
                obj = obj[:6] + ['...']
            elif isinstance(obj, tuple):
                obj = obj[:6] + ('...', )
            descr += f'\t{arg}: {obj}\n'
        return descr
    
    @abstractmethod
    def __str__(self) -> str :
        """ Returns str(self) """
        pass
