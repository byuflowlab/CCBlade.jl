import numpy as np


def get_rows_cols(ss_sizes, of_ss, wrt_ss):

    # Get the shape of the output variable (the "of") and the input variable
    # (the "wrt").
    of_shape = [ss_sizes[s] for s in of_ss]
    wrt_shape = [ss_sizes[s] for s in wrt_ss]

    # Get the output subscript, which will start with the of_ss, then the
    # wrt_ss with the subscripts common to both removed.
    deriv_ss = of_ss + "".join(set(wrt_ss) - set(of_ss))

    # Get the complete subscript that will be passed to numpy.einsum.
    ss = f"{of_ss},{wrt_ss}->{deriv_ss}"

    # Shamelessly stolen from John Hwang's OpenBEMT code.
    a = np.arange(np.prod(of_shape)).reshape(of_shape)
    b = np.ones(wrt_shape, dtype=int)
    rows = np.einsum(ss, a, b).flatten()

    a = np.ones(of_shape, dtype=int)
    b = np.arange(np.prod(wrt_shape)).reshape(wrt_shape)
    cols = np.einsum(ss, a, b).flatten()

    return rows, cols
