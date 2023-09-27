from numpy import issubdtype, integer, floating, bool_


def _translate_array_type(dtype):
    if issubdtype(dtype, integer):
        return "integer"
    if issubdtype(dtype, floating):
        return "number"
    if dtype == bool_:
        return "boolean"
    raise NotImplementedError("staging of '" + str(type(dtype)) + "' arrays not yet supported")
