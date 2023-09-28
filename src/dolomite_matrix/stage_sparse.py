from dolomite_base import stage_object
import os

from .write_sparse_matrix import write_sparse_matrix
from ._utils import _translate_array_type


has_scipy = False
try:
    import scipy.sparse
    has_scipy = True
except:
    pass


def _stage_sparse_matrix(x, dir: str, path: str, is_child: bool, **kwargs):
    os.mkdir(os.path.join(dir, path))
    newpath = path + "/matrix.h5"
    fullpath = os.path.join(dir, newpath)
    write_sparse_matrix(x, fullpath, "data", **kwargs)
    return {
        "$schema": "hdf5_sparse_matrix/v1.json",
        "path": newpath,
        "array": {
            "type": _translate_array_type(x.dtype),
            "dimensions": list(x.shape),
        },
        "hdf5_sparse_matrix": {
            "format": "tenx_matrix",
            "group": "data"
        }
    }


if has_scipy:
    @stage_object.register
    def stage_scipy_csc_matrix(x: scipy.sparse.csc_matrix, dir: str, path: str, is_child: bool = False, **kwargs):
        """Method for saving :py:class:`~scipy.sparse.csc_matrix` objects to
        file, see :py:meth:`~dolomite_base.stage_object.stage_object` for
        details.

        Args:
            x: Matrix to be saved.

            dir: Staging directory.

            path: Relative path inside ``dir`` to save the object.

            is_child: Is ``x`` a child of another object?

            kwargs: 
                Further arguments to pass to
                :py:meth:`~dolomite_matrix.write_sparse_matrix.write_sparse_matrix`.

        Returns:
            Metadata that can be edited by calling methods and then saved with 
            :py:meth:`~dolomite_base.write_metadata.write_metadata`.
        """
        return _stage_sparse_matrix(x, dir, path, is_child, **kwargs)


    @stage_object.register
    def stage_scipy_csr_matrix(x: scipy.sparse.csr_matrix, dir: str, path: str, is_child: bool = False, **kwargs):
        """Method for saving :py:class:`~scipy.sparse.csr_matrix` objects to
        file, see :py:meth:`~dolomite_base.stage_object.stage_object` for
        details.

        Args:
            x: Matrix to be saved.

            dir: Staging directory.

            path: Relative path inside ``dir`` to save the object.

            is_child: Is ``x`` a child of another object?

            kwargs: 
                Further arguments to pass to
                :py:meth:`~dolomite_matrix.write_sparse_matrix.write_sparse_matrix`.

        Returns:
            Metadata that can be edited by calling methods and then saved with 
            :py:meth:`~dolomite_base.write_metadata.write_metadata`.
        """
        return _stage_sparse_matrix(x, dir, path, is_child, **kwargs)


    @stage_object.register
    def stage_scipy_coo_matrix(x: scipy.sparse.coo_matrix, dir: str, path: str, is_child: bool = False, **kwargs):
        """Method for saving :py:class:`~scipy.sparse.coo_matrix` objects to
        file, see :py:meth:`~dolomite_base.stage_object.stage_object` for
        details.

        Args:
            x: Matrix to be saved.

            dir: Staging directory.

            path: Relative path inside ``dir`` to save the object.

            is_child: Is ``x`` a child of another object?

            kwargs: 
                Further arguments to pass to
                :py:meth:`~dolomite_matrix.write_sparse_matrix.write_sparse_matrix`.

        Returns:
            Metadata that can be edited by calling methods and then saved with 
            :py:meth:`~dolomite_base.write_metadata.write_metadata`.
        """
        return _stage_sparse_matrix(x, dir, path, is_child, **kwargs)
