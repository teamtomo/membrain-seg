import numpy as np
from scipy.fftpack import fftn, fftshift, ifftn


def angauss(T: np.ndarray, std: float, r: float) -> np.ndarray:
    """
    Apply anisotropic Gaussian filtering to a 3D tomogram.

    This function performs a convolution of the input 3D tomogram with an anisotropic
    Gaussian kernel. The anisotropy is defined by the aspect ratio 'r' in the Z-axis,
    allowing different standard deviations in the XY-plane and Z-axis. The Gaussian
    kernel used for filtering has the formula:

        G(x, y, z) = A * exp(-(a*(x^2 + y^2) + b*z^2))

    where:
    - A is the normalization factor `1 / (std^2 * sqrt(r * (2*pi)^3))`
    - a = `1 / (2 * std^2)` controls the spread in the x and y dimensions
    - b = `a / r` controls the spread in the z dimension

    Parameters
    ----------
    T : np.ndarray
        The input 3D tomogram as a numpy array.
    std : float
        The Gaussian standard deviation in the XY-plane.
    r : float
        The aspect ratio for the Gaussian standard deviation in the Z-axis. If `r` is 1,
        the filtering is isotropic.

    Returns
    -------
    np.ndarray
        The filtered tomogram as a numpy array, with the same shape as the input array.

    Notes
    -----
    The function uses the Fourier domain for convolution, which involves:
    1. Creating a Gaussian kernel that is anisotropic, with standard deviations
       determined by 'std' and 'r'.
    2. Computing the Fourier transform of both the input tomogram
       and the Gaussian kernel.
    3. Multiplying these transforms to apply the convolution.
    4. Computing the inverse Fourier transform of the result
       to retrieve the filtered tomogram.

    Examples
    --------
    Create a sample 3D array and apply anisotropic Gaussian filtering:

    >>> T = np.random.rand(10, 10, 10)  # Example input tomogram
        # Anisotropic filtering with half aspect ratio in Z
    >>> filtered_tomogram = angauss(T, std=1.0, r=0.5)
    """
    # Initialization of variables and computation of grid indices
    Nx, Ny, Nz = T.shape
    Nx2, Ny2, Nz2 = int(Nx / 2), int(Ny / 2), int(Nz / 2)

    # Generate grid vectors for each dimension considering odd/even size
    Vnx = np.arange(-Nx2, Nx2 + 1) if Nx % 2 != 0 else np.arange(-Nx2, Nx2)
    Vny = np.arange(-Ny2, Ny2 + 1) if Ny % 2 != 0 else np.arange(-Ny2, Ny2)
    Vnz = np.arange(-Nz2, Nz2 + 1) if Nz % 2 != 0 else np.arange(-Nz2, Nz2)

    # Create meshgrid for kernel calculation: X, Y, Z correspond to axes Y, X, Z
    X, Y, Z = np.meshgrid(Vny, Vnx, Vnz, indexing="ij")
    # Compute Gaussian normalization factor and exponents for the kernel
    A = 1 / (std**2 * np.sqrt(r * (2 * np.pi) ** 3))
    a = 1 / (2 * std**2)
    b = a / r

    # Create the Gaussian kernel in spatial domain
    gaussian_kernel = A * np.exp(-a * (X**2 + Y**2) - b * Z**2)

    # Perform Fourier convolution by shifting, multiplying, and inverse shifting
    filtered_tomogram = fftshift(ifftn(fftn(T) * fftn(gaussian_kernel)))

    return filtered_tomogram
