from typing import Tuple, Union

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import distance_transform_edt


def smooth_cosine_dropoff(mask: np.ndarray, decay_width: float) -> np.ndarray:
    """Smooth a given mask.

    Given a binary (1-0) mask, the mask is extended by a smooth drop-off
    using cosine decay.

    Parameters
    ----------
    mask : np.ndarray
        A binary (1-0) numpy array representing the mask.
    decay_width : float
        The width of the smooth drop-off region in the mask.

    Returns
    -------
    np.ndarray
        A numpy array representing the smoothed mask.
    """
    # Create a distance map based on the given mask
    distance_map = distance_transform_edt(1 - mask)

    # Apply a smooth cosine drop-off to the distance map
    dropoff = np.zeros_like(distance_map)
    edge_indices = np.where((distance_map > 0) & (distance_map <= decay_width))
    dropoff[edge_indices] = 0.5 * (
        1 + np.cos(np.pi * (distance_map[edge_indices]) / decay_width)
    )

    # Combine the original mask with the smooth drop-off
    result = mask + dropoff

    # Normalize the result to ensure the mask values are between 0 and 1
    result /= np.max(result)

    return result


def cosine_ellipsoid_filter(
    image_shape: Tuple[int, int, int], radius_factors: Tuple[float, float, float]
) -> np.ndarray:
    """
    Ellipsoid masking with smooth cosine edges.

    An ellipsoid is created that extends maximally in the image shape with
    a border of 12.5 voxels. Then a Cosine decay is applied to extend the
    mask by a smooth drop-off towards the image boundaries (reaches 0 at Nyquist).

    Parameters
    ----------
    image_shape : Tuple[int, int, int]
        The shape of the input image as a tuple (x, y, z).
    radius_factors : Tuple[float, float, float]
        The factors for each axis (x, y, z) to determine the ellipsoid size.
        (0.5, 0.5, 0.5) means that the ellipse extends fully to its maximum extent,
        i.e., to a distance of 12.5 voxels to the image border.


    Returns
    -------
    np.ndarray
        A numpy array representing the ellipsoid filter with smooth cosine edges.
    """
    z_len, y_len, x_len = image_shape
    z, y, x = np.meshgrid(
        np.arange(z_len), np.arange(y_len), np.arange(x_len), indexing="ij"
    )

    # Compute normalized distance from the center of the ellipsoid
    z_center, y_center, x_center = z_len // 2, y_len // 2, x_len // 2
    z_normalized = ((z - z_center) / ((z_len - 25) * radius_factors[0])) ** 2
    y_normalized = ((y - y_center) / ((y_len - 25) * radius_factors[1])) ** 2
    x_normalized = ((x - x_center) / ((x_len - 25) * radius_factors[2])) ** 2

    distance = np.sqrt(z_normalized + y_normalized + x_normalized)

    # Compute the cosine edge filter
    filter_mask = np.zeros(image_shape)
    ellipse_indices = np.where(distance <= 1)
    filter_mask[ellipse_indices] = 1
    filter_mask = smooth_cosine_dropoff(filter_mask, decay_width=12)

    return filter_mask


def fourier_cropping(
    data: np.ndarray, new_shape: Tuple[int, int, int], smoothing: bool
) -> np.ndarray:
    """
    Fourier cropping in case the new shape is smaller than the original shape.

    The data's FFT is computed, cropped to the new image size, and transformed back
    to real space.
    If specified (smoothing), en ellipsoid mask with cosine decay towards the edges
    is applied to avoid artifacts.

    Parameters
    ----------
    data : ndarray
        The input data as a 3D numpy array.
    new_shape : Tuple[int, int, int]
        The target shape for the cropped data as a tuple (x, y, z).
    smoothing : bool
        If True, apply a smoothing filter to the cropped data; otherwise, skip the
        smoothing step.

    Returns
    -------
    np.ndarray
        A tuple containing the resized data.
    """
    # Calculate the FFT of the input data
    data_fft = fftn(data)
    data_fft = np.fft.fftshift(data_fft)

    # Calculate the cropping indices
    original_shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    start_indices = (original_shape - new_shape) // 2
    end_indices = start_indices + new_shape

    # Crop the filtered FFT data
    cropped_fft = data_fft[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ]

    # filtered_fft = gaussian_filter(cropped_fft, sigma=sigma)
    if smoothing:
        exponential_filter = cosine_ellipsoid_filter(
            cropped_fft.shape, radius_factors=(0.5, 0.5, 0.5)
        )
        filtered_fft = cropped_fft * exponential_filter
    else:
        filtered_fft = cropped_fft
    unshifted_cropped_fft = np.fft.ifftshift(filtered_fft)

    # Calculate the inverse FFT of the cropped data and normalize
    resized_data = np.real(ifftn(unshifted_cropped_fft))

    return resized_data


def fourier_extend(
    data: np.ndarray, new_shape: Tuple[int, int, int], smoothing: bool
) -> np.ndarray:
    """
    Fourier padding in case the new shape is larger than the original shape.

    The data's FFT is computed, padded with zeros to the new image shape,
    and transformed back to real space.
    If specified (smoothing), en ellipsoid mask with cosine decay towards the edges
    is applied to avoid artifacts (before padding).

    Parameters
    ----------
    data : np.ndarray
        The input data as a 3D numpy array.
    new_shape : Tuple[int, int, int]
        The target shape for the extended data as a tuple (x, y, z).
    smoothing : bool
        If True, apply a smoothing filter to the data; otherwise, skip the
        smoothing step.

    Returns
    -------
    ndarray
        The resized data as a 3D numpy array.
    """
    data_fft = fftn(data)
    data_fft = np.fft.fftshift(data_fft)

    if smoothing:
        smoothing_mask = cosine_ellipsoid_filter(
            data_fft.shape, radius_factors=(0.5, 0.5, 0.5)
        )
        data_fft = data_fft * smoothing_mask

    padding = [
        (new_dim - old_dim) // 2 for old_dim, new_dim in zip(data.shape, new_shape)
    ]
    padded_fft = np.pad(data_fft, [(pad, pad) for pad in padding], mode="constant")

    unshifted_padded_fft = np.fft.ifftshift(padded_fft)

    # Calculate the inverse FFT of the cropped data and normalize
    resized_data = np.real(ifftn(unshifted_padded_fft))
    return resized_data


def determine_output_shape(
    pixel_size_in: Union[float, int],
    pixel_size_out: Union[float, int],
    orig_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Determine the new output shape given in/out pixel sizes & original shape.

    Parameters
    ----------
    pixel_size_in : Union[float, int]
        The pixel size of the input data.
    pixel_size_out : Union[float, int]
        The target pixel size.
    orig_shape : Tuple[int, int, int]
        The original shape of the data as a tuple (x, y, z).

    Returns
    -------
    Tuple[int, int, int]
        The new output shape as a tuple (x, y, z).
    """
    output_shape = np.array(orig_shape) * (pixel_size_in / pixel_size_out)
    output_shape = np.round(output_shape)
    if output_shape[0] % 2 != 0:
        output_shape[0] += 1
    if output_shape[1] % 2 != 0:
        output_shape[1] += 1
    if output_shape[2] % 2 != 0:
        output_shape[2] += 1
    output_shape = np.array(output_shape, dtype=int)
    return output_shape
