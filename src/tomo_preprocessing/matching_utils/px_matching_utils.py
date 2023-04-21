import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import distance_transform_edt


def smooth_cosine_dropoff(mask, decay_width):
    # Create a distance map based on the given mask
    distance_map = distance_transform_edt(1-mask)

    # Apply a smooth cosine drop-off to the distance map
    dropoff = np.zeros_like(distance_map)
    edge_indices = np.where((distance_map > 0) & (distance_map <= decay_width))
    dropoff[edge_indices] = 0.5 * (1 + np.cos(np.pi * (distance_map[edge_indices]) / decay_width))

    # Combine the original mask with the smooth drop-off
    result = mask + dropoff

    # Normalize the result to ensure the mask values are between 0 and 1
    result /= np.max(result)

    return result


def cosine_ellipsoid_filter(image_shape, radius_factors):
    z_len, y_len, x_len = image_shape
    z, y, x = np.meshgrid(np.arange(z_len), np.arange(y_len), np.arange(x_len), indexing='ij')

    # Compute normalized distance from the center of the ellipsoid
    z_center, y_center, x_center = z_len // 2, y_len // 2, x_len // 2
    z_normalized = ((z - z_center) / ((z_len-25) * radius_factors[0])) ** 2
    y_normalized = ((y - y_center) / ((y_len-25) * radius_factors[1])) ** 2
    x_normalized = ((x - x_center) / ((x_len-25) * radius_factors[2])) ** 2

    distance = np.sqrt(z_normalized + y_normalized + x_normalized)

    # Compute the cosine edge filter
    filter_mask = np.zeros(image_shape)
    ellipse_indices = np.where((distance <= 1))
    filter_mask[ellipse_indices] = 1
    filter_mask = smooth_cosine_dropoff(filter_mask, decay_width=12)

    return filter_mask


def fourier_cropping(data, new_shape, smoothing):
    """ Fourier cropping in case the new shape is smaller than the original shape. """
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
        start_indices[0]:end_indices[0],
        start_indices[1]:end_indices[1],
        start_indices[2]:end_indices[2]
    ]

    # filtered_fft = gaussian_filter(cropped_fft, sigma=sigma)
    if smoothing:
        exponential_filter = cosine_ellipsoid_filter(cropped_fft.shape, radius_factors=(0.5, 0.5, 0.5))
        filtered_fft = cropped_fft * exponential_filter
    else:
        filtered_fft = cropped_fft
    unshifted_cropped_fft = np.fft.ifftshift(filtered_fft)

    # Calculate the inverse FFT of the cropped data and normalize
    resized_data = np.real(ifftn(unshifted_cropped_fft))

    return resized_data, exponential_filter


def fourier_extend(data, new_shape, smoothing):
    data_fft = fftn(data)
    data_fft = np.fft.fftshift(data_fft)

    if smoothing:
        smoothing_mask = cosine_ellipsoid_filter(data_fft.shape, radius_factors=(0.5, 0.5, 0.5))
        data_fft = data_fft * smoothing_mask

    padding = [(new_dim - old_dim) // 2 for old_dim, new_dim in zip(data.shape, new_shape)]
    padded_fft = np.pad(data_fft, [(pad, pad) for pad in padding], mode='constant')

    unshifted_padded_fft = np.fft.ifftshift(padded_fft)

    # Calculate the inverse FFT of the cropped data and normalize
    resized_data = np.real(ifftn(unshifted_padded_fft))
    return resized_data


def determine_output_shape(pixel_size_in, pixel_size_out, orig_shape):
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
