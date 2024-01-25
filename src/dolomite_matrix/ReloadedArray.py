import delayedarray
from dolomite_base import save_object
import os
import shutil

from .WrapperArraySeed import WrapperArraySeed
from .save_compressed_sparse_matrix import _save_compressed_sparse_matrix
from .save_dense_array import _save_dense_array


class ReloadedArraySeed(WrapperArraySeed):
    """
    Seed for the :py:class:`~ReloadedArray` class. This is a subclass
    of :py:class:`~dolomite_matrix.WrapperArraySeed.WrapperArraySeed`.
    """

    def __init__(self, seed, path: str):
        """
        Args:
            seed: The contents of the reloaded array.
            path: Path to the directory containing the on-disk representation.
        """
        super(ReloadedArraySeed, self).__init__(seed)
        self._path = path

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the directory containing the on-disk representation.
        """
        return self._path


class ReloadedArray(delayedarray.DelayedArray):
    """
    An array that was reloaded from disk by the
    :py:func:`~dolomite_base.read_object.read_object` function, and remembers
    the path from which it was loaded.  This class allows methods to refer to
    the existing on-disk representation by inspecting the path.  For example,
    :py:func:`~dolomite_base.save_object.save_object` can just copy/link to the
    existing files instead of repeating the saving process.
    """

    def __init__(self, seed, path: str):
        """
        To construct a ``ReloadedArray`` from an existing
        :py:class:`~ReloadedArraySeed`, use :py:meth:`~delayedarray.wrap.wrap`
        instead.

        Args:
            seed: The contents of the reloaded array.
            path: Path to the directory containing the on-disk representation.
        """
        if not isinstance(seed, ReloadedArraySeed):
            seed = ReloadedArraySeed(seed, path)
        super(ReloadedArray, self).__init__(seed)

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the directory containing the on-disk representation.
        """
        return self.seed._path


@delayedarray.wrap.register
def wrap_ReloadedArraySeed(x: ReloadedArraySeed) -> ReloadedArray:
    """See :py:func:`~delayedarray.wrap.wrap`."""
    return ReloadedArray(x)


@save_object.register
def save_object_ReloadedArray(x: ReloadedArray, path: str, reloaded_array_reuse_mode: str = "link", **kwargs):
    """
    Method for saving :py:class:`~ReloadedArray.ReloadedArray` objects to disk,
    see :py:meth:`~dolomite_base.save_object.save_object` for details.

    Args:
        x: Object to be saved.

        path: Path to a directory to save ``x``.

        reloaded_array_reuse_mode:
            How the files in ``x.path`` should be re-used when populating
            ``path``.  This can be ``"link"``, to create a hard link to each
            file; ``"symlink"``, to create a symbolic link to each file;
            ``"copy"``, to create a copy of each file; or ``"none"``, to
            perform a fresh save of ``x`` without relying on ``x.path``.

        kwargs: Further arguments, ignored.

    Returns:
        ``x`` is saved to ``path``. 
    """
    if reloaded_array_reuse_mode == "none": 
        if delayedarray.is_sparse(x):
            return _save_compressed_sparse_matrix(x, path, **kwargs)
        else:
            return _save_dense_array(x, path, **kwargs)

    if reloaded_array_reuse_mode == "link":
        def FUN(src, dest):
            try:
                os.link(src, dest)
            except Exception as _:
                shutil.copyfile(src, dest)
    elif reloaded_array_reuse_mode == "symlink":
        def FUN(src, dest):
            try:
                os.symlink(src, dest)
            except Exception as _:
                shutil.copyfile(src, dest)
    elif reloaded_array_reuse_mode == "copy":
        FUN = shutil.copyfile
    else:
        raise ValueError("invalid reuse mode '" + reloaded_array_reuse_mode + "'")

    for root, dirs, files in os.walk(x.path):
        newpath = os.path.join(path, os.path.relpath(root, x.path))
        os.makedirs(newpath)
        for f in files:
            FUN(os.path.join(root, f), os.path.join(newpath, f))
