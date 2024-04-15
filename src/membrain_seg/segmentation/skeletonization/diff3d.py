import numpy as np


def diff3d(T: np.ndarray, k: int) -> np.ndarray:
    """
    Calculate the partial derivative of a 3D tomogram along a specified dimension.

    Parameters
    ----------
    T : np.ndarray
        The input 3D tomogram as a numpy array, where each dimension
        corresponds to spatial dimensions.
    k : int
        The axis along which to compute the derivative.
        Set k=0 for the x-dimension, k=1 for the y-dimension,
        and any other value for the z-dimension.

    Returns
    -------
    np.ndarray
        The output tomogram,
        which represents the partial derivatives along the specified axis.
        This output has the same shape as the input array.

    Notes
    -----
    The function computes the centered difference in the specified dimension.
    The boundaries are handled by padding the last slice with the value from
    the second to last slice, ensuring smooth derivative values at the edges
    of the tomogram.

    Examples
    --------
    Create a sample 3D array and compute the partial derivative along the x-axis (k=0):

    >>> T = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> diff3d(T, 0)
    array([[[ 4.,  4.],
            [ 4.,  4.]],

           [[ 0.,  0.],
            [ 0.,  0.]]])
    """
    # Get the size of the input tomogram
    Nx, Ny, Nz = T.shape

    # Initialize arrays for forward and backward differences
    Idp = np.zeros((Nx, Ny, Nz), dtype="float64")
    Idn = np.zeros((Nx, Ny, Nz), dtype="float64")

    # Calculate partial derivatives along the specified dimension (k)
    if k == 0:
        Idp[0 : Nx - 1, :, :] = T[1:Nx, :, :]
        Idn[1:Nx, :, :] = T[0 : Nx - 1, :, :]
        # Pad extremes
        Idp[Nx - 1, :, :] = Idp[Nx - 2, :, :]
        Idn[0, :, :] = Idn[1, :, :]
    elif k == 1:
        Idp[:, 0 : Ny - 1, :] = T[:, 1:Ny, :]
        Idn[:, 1:Ny, :] = T[:, 0 : Ny - 1, :]
        # Pad extremes
        Idp[:, Ny - 1, :] = Idp[:, Ny - 2, :]
        Idn[:, 0, :] = Idn[:, 1, :]
    else:
        Idp[:, :, 0 : Nz - 1] = T[:, :, 1:Nz]
        Idn[:, :, 1:Nz] = T[:, :, 0 : Nz - 1]
        # Pad extremes
        Idp[:, :, Nz - 1] = Idp[:, :, Nz - 2]
        Idn[:, :, 0] = Idn[:, :, 1]

    # Calculate the output tomogram
    output = (Idp - Idn) * 0.5

    return output
