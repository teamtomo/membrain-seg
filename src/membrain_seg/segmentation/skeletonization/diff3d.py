import numpy as np


def diff3d(T, k):
    """
    Calculates partial derivative along any dimension in a tomogram.
    INPUT:
        T - Input tomogram
        k - 1: x-dimension, 2: y-dimension and otherwise: z-dimension
    OUTPUT:
    D - Output tomgram
    """
    # Get the size of the input tomogram
    Nx, Ny, Nz = T.shape

    # Initialize Idp and Idn
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

    # Calculate the output tomogram (D)
    D = (Idp - Idn) * 0.5

    return D
