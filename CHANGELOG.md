# Changelog

## Version 0.4.1

- Added preliminary support for custom variable length string (VLS) arrays, which compress the heap for more efficient storage than HDF5's VLS implementation.
This is enabled via the `dense_array_string_vls=` option in the `save_object()` method for NumPy arrays.

## Version 0.4.0

- chore: Remove Python 3.8 (EOL).
- Add support for Python 3.13.
- precommit: Replace docformatter with ruff's formatter.

## Version 0.0.1

- Initial release.
