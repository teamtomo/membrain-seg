# ---------------------------------------------------------------------------------
# DISCLAIMER: This code is adapted from the MATLAB and C++ implementations provided
# in the paper titled "Robust membrane detection based on tensor voting for electron
# tomography" by Antonio Martinez-Sanchez, Inmaculada Garcia, Shoh Asano, Vladan Lucic,
# and Jose-Jesus Fernandez, published in the Journal of Structural Biology,
# Volume 186, Issue 1, 2014, Pages 49-61. The original work can be accessed via
# https://doi.org/10.1016/j.jsb.2014.02.015 and is used under conditions that adhere
# to the original licensing agreements. For details on the original license, refer to
# the publication: https://www.sciencedirect.com/science/article/pii/S1047847714000495.
# ---------------------------------------------------------------------------------
import numpy as np


def calculate_derivative_3d(tomogram: np.ndarray, axis: int) -> np.ndarray:
    """
    Calculate the partial derivative of a 3D tomogram along a specified dimension.

    Parameters
    ----------
    tomogram : np.ndarray
        The input 3D tomogram as a numpy array, where each dimension
        corresponds to spatial dimensions.
    axis : int
        The axis along which to compute the derivative.
        Set axis=0 for the x-dimension, axis=1 for the y-dimension,
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
    Create a sample 3D array and compute the partial derivative
    along the x-axis (axis=0):

    >>> tomogram = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> calculate_derivative_3d(tomogram, 0)
    array([[[ 4.,  4.],
            [ 4.,  4.]],

           [[ 0.,  0.],
            [ 0.,  0.]]])
    """
    # Get the size of the input tomogram
    num_x, num_y, num_z = tomogram.shape

    # Initialize arrays for forward and backward differences
    forward_difference = np.zeros((num_x, num_y, num_z), dtype="float64")
    backward_difference = np.zeros((num_x, num_y, num_z), dtype="float64")

    # Calculate partial derivatives along the specified dimension (axis)
    if axis == 0:
        forward_difference[0 : num_x - 1, :, :] = tomogram[1:num_x, :, :]
        backward_difference[1:num_x, :, :] = tomogram[0 : num_x - 1, :, :]
        # Pad extremes
        forward_difference[num_x - 1, :, :] = forward_difference[num_x - 2, :, :]
        backward_difference[0, :, :] = backward_difference[1, :, :]
    elif axis == 1:
        forward_difference[:, 0 : num_y - 1, :] = tomogram[:, 1:num_y, :]
        backward_difference[:, 1:num_y, :] = tomogram[:, 0 : num_y - 1, :]
        # Pad extremes
        forward_difference[:, num_y - 1, :] = forward_difference[:, num_y - 2, :]
        backward_difference[:, 0, :] = backward_difference[:, 1, :]
    else:
        forward_difference[:, :, 0 : num_z - 1] = tomogram[:, :, 1:num_z]
        backward_difference[:, :, 1:num_z] = tomogram[:, :, 0 : num_z - 1]
        # Pad extremes
        forward_difference[:, :, num_z - 1] = forward_difference[:, :, num_z - 2]
        backward_difference[:, :, 0] = backward_difference[:, :, 1]

    # Calculate the output tomogram
    derivative_tomogram = (forward_difference - backward_difference) * 0.5

    return derivative_tomogram


def compute_gradients(tomogram: np.ndarray) -> tuple:
    """
    Computes the gradients along each spatial dimension of a 3D tomogram.

    Parameters
    ----------
    tomogram : np.ndarray
        The input 3D tomogram as a numpy array.

    Returns
    -------
    tuple
        A tuple containing the gradient components (gradX, gradY, gradZ).

    Notes
    -----
    This function calculates the partial derivatives of the tomogram along the x, y,
    and z dimensions, respectively. These derivatives represent the gradient components
    along each dimension.
    """
    gradX = calculate_derivative_3d(tomogram, 0)
    gradY = calculate_derivative_3d(tomogram, 1)
    gradZ = calculate_derivative_3d(tomogram, 2)

    return gradX, gradY, gradZ


def compute_hessian(gradX: np.ndarray, gradY: np.ndarray, gradZ: np.ndarray) -> tuple:
    """
    Computes the Hessian tensor components for a 3D tomogram from its gradients.

    Parameters
    ----------
    gradX : np.ndarray
        Gradient of the tomogram along the x-axis.
    gradY : np.ndarray
        Gradient of the tomogram along the y-axis.
    gradZ : np.ndarray
        Gradient of the tomogram along the z-axis.

    Returns
    -------
    tuple
        A tuple containing the Hessian tensor components (hessianXX, hessianYY,
        hessianZZ, hessianXY, hessianXZ, hessianYZ).

    Notes
    -----
    This function computes the second derivatives of the tomogram along each dimension.
    These second derivatives form the components of the Hessian tensor, providing
    information about the curvature of the tomogram.
    """
    hessianXX = calculate_derivative_3d(gradX, 0)
    hessianYY = calculate_derivative_3d(gradY, 1)
    hessianZZ = calculate_derivative_3d(gradZ, 2)
    hessianXY = calculate_derivative_3d(gradX, 1)
    hessianXZ = calculate_derivative_3d(gradX, 2)
    hessianYZ = calculate_derivative_3d(gradY, 2)

    return hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ
