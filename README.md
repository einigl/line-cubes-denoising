# Deep learning denoising by dimension reduction of molecular line cubes.

## About

Line cube denoising `line_cube_denoising` is a Python package that handle the creation and the training of deep neural networks to increase the signal-to-noise ratio (SNR) of molecular line cubes.

For tips on how to get started with ``line_cube_denoising``, see the section [Getting started](#gettingstarted) further below.


## Installation

### Dependencies

You will need the following packages to import and use `line_cube_denoising`. We list the version of each package which we know to be compatible with `line_cube_denoising`.

* [python 3.9.0](https://www.python.org/)
* [pytorch 2.0.0](https://pytorch.org/)
* Other dependencies will be specified later.

If you do not have a Python environment compatible with the above dependencies, we advise you to create a specific conda environment to use this code (https://conda.io/projects/conda/en/latest/user-guide/).

### Download

You can download this code using the following command `$ git clone git@github.com:einigl/line-cubes-denoising.git`. Alternatively, you can also download a zip file.

### Installing Dependencies on Windows

Coming soon.

### Installing Dependencies on Linux

Coming soon.

### Installing Dependencies on OSX

Coming soon.

## Getting started

Line cubes must be placed in the `cubes` directory. ORION-B cubes are currently not available but they will be released in the coming weeks.

The `line_cube_denoising` library must be loaded without any particular installation. Just consider adding the containing folder to your Python path (temporarily or permanently).

The library can be loaded using the following command
```python
import line_cube_denoising as lcd
```

Alternatively, to load only an autoencoder architecture for instance, consider using the following command
```python
from line_cube_denoising import DenseAutoencoder
```
Example notebooks are available in the `notebooks` directory.

