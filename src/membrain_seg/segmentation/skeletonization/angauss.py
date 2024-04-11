import numpy as np
from scipy.fftpack import fftn, fftshift, ifftn


def angauss(I, s, r):
    """
    ANGAUSS  Anisotropic gaussian filtering.

    INPUT:
      Is - Input tomogram
       s - Gaussian standard deviation
       r - Aspect ration in Z axis, if 1 isotropic
    OUTPUT:
    S - Filtered output
    """
    # Initialization
    Nx, Ny, Nz = I.shape
    Nx2, Ny2, Nz2 = int(Nx / 2), int(Ny / 2), int(Nz / 2)

    Vnx = np.arange(-Nx2, Nx2 + 1) if Nx % 2 != 0 else np.arange(-Nx2, Nx2)
    Vny = np.arange(-Ny2, Ny2 + 1) if Ny % 2 != 0 else np.arange(-Ny2, Ny2)
    Vnz = np.arange(-Nz2, Nz2 + 1) if Nz % 2 != 0 else np.arange(-Nz2, Nz2)

    X, Y, Z = np.meshgrid(Vny, Vnx, Vnz)
    A = 1 / (s**2 * np.sqrt(r * (2 * np.pi) ** 3))
    a = 1 / (2 * s**2)
    b = a / r

    # Kernel
    K = A * np.exp(-a * (X**2 + Y**2) - b * Z**2)

    # Convolution in Fourier domain
    F = fftshift(ifftn(fftn(I) * fftn(K)))

    return F
