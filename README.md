<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/dolomite-matrix.svg?branch=main)](https://cirrus-ci.com/github/<USER>/dolomite-matrix)
[![ReadTheDocs](https://readthedocs.org/projects/dolomite-matrix/badge/?version=latest)](https://dolomite-matrix.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/dolomite-matrix/main.svg)](https://coveralls.io/r/<USER>/dolomite-matrix)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/dolomite-matrix.svg)](https://anaconda.org/conda-forge/dolomite-matrix)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/dolomite-matrix)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/dolomite-matrix.svg)](https://pypi.org/project/dolomite-matrix/)
[![Monthly Downloads](https://pepy.tech/badge/dolomite-matrix/month)](https://pepy.tech/project/dolomite-matrix)
![Unit tests](https://github.com/ArtifactDB/dolomite-matrix/actions/workflows/run-tests.yml/badge.svg)

# Read and save matrices in Python

## Introduction

The **dolomite-matrix** package is the Python counterpart to the [**alabaster.matrix**](https://github.com/ArtifactDB/alabaster.matrix) R package,
providing methods for saving/reading arrays and matrices within the [**dolomite** framework](https://github.com/ArtifactDB/dolomite-base).
Dense arrays are stored in the usual HDF5 dataset, while sparse matrices are saved inside a HDF5 file in compressed sparse format.

## Quick start

Let's save a dense matrix to a HDF5 file with some accompanying metadata:

```python
import numpy
x = numpy.random.rand(1000, 200) 

import os
import tempfile
dir = os.path.join(tempfile.mkdtemp(), "whee")

import dolomite_base
import dolomite_matrix
dolomite_base.save_object(x, dir)
```

Now we can transfer the directory and reload the matrix in a new session.
This constructs a HDF5-backed dense array that can be used for block processing or realized into the usual NumPy array.

```python
import dolomite_base
obj = dolomite_base.read_object(dir)
## <1000 x 200> ReloadedArray object of type 'float64'
## [[0.58444226, 0.82595149, 0.7214525 , ..., 0.32493652, 0.58206044,
##   0.73770346],
##  [0.96398317, 0.73200292, 0.16410134, ..., 0.31626547, 0.11499628,
##   0.19768697],
##  [0.82350911, 0.48012452, 0.65221052, ..., 0.94989611, 0.15422992,
##   0.77173718],
##  ...,
##  [0.71715436, 0.19266116, 0.52316388, ..., 0.23104537, 0.935654  ,
##   0.51663007],
##  [0.38585049, 0.26709808, 0.70358993, ..., 0.91822795, 0.66144925,
##   0.42465112],
##  [0.08535589, 0.00144712, 0.51411921, ..., 0.84546122, 0.35001404,
##   0.53644868]]
```

## Sparse matrices

We can also save and load a sparse matrix from a HDF5 file:

```python
import scipy 
import numpy
x = scipy.sparse.random(1000, 200, 0.2, dtype=numpy.int16, format="csc")

import os
import tempfile
dir = os.path.join(tempfile.mkdtemp(), "stuff")

import dolomite_base
import dolomite_matrix
dolomite_base.save_object(x, dir)
```

And again, loading it back in a new session.
This constructs a HDF5-backed sparse array that can be used for block processing or realized into the usual NumPy array.

```python
import dolomite_base
obj = dolomite_base.read_object(dir)
## <1000 x 200> sparse ReloadedArray object of type 'int16'
## [[     0,      0, -28638, ...,      0,      0,  26194],
##  [     0,      0,      0, ...,      0, -30829,      0],
##  [     0,      0,      0, ...,      0,      0,      0],
##  ...,
##  [ 10895,      0,      0, ...,      0,      0,      0],
##  [     0,  32539,      0, ...,      0,   2780, -12106],
##  [     0,      0,      0, ...,   1452,      0, -26314]]
```
