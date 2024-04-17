import torch

from membrain_seg.segmentation.training.surface_dice import (
    apply_gaussian_filter,
)


def process_hessian_tensors(
    hessian_components: list, kernel_size: int = 9, sigma: float = 1.0
) -> list:
    """
    Processes a list of Hessian component matrices by applying a Gaussian filter to each.

    This function takes a list of numpy arrays representing different components of the Hessian matrix
    (e.g., HessianXX, HessianYY, etc.), converts them to PyTorch tensors, applies a Gaussian filter to each,
    and returns the filtered matrices as numpy arrays. The function automatically handles device selection
    (CPU or GPU) based on availability and uses a dictionary to store and reuse Gaussian kernels for efficiency.

    Parameters
    ----------
    hessian_components : list of np.ndarray
        List of numpy arrays, where each array is a component of the Hessian matrix.
    kernel_size : int, optional
        The size of the Gaussian kernel used for filtering, default is 9.
    sigma : float, optional
        The standard deviation of the Gaussian kernel used for filtering, default is 1.0.

    Returns
    -------
    list of np.ndarray
        List of numpy arrays containing the filtered versions of the input Hessian components.

    Examples
    --------
    >>> hessian_components = [np.random.rand(10, 10, 10) for _ in range(6)]
    >>> filtered_components = process_hessian_tensors(hessian_components)
    >>> len(filtered_components)
    6
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tensor_components = [
        torch.from_numpy(comp).float().to(device).unsqueeze(0).unsqueeze(0)
        for comp in hessian_components
    ]

    filtered_tensors = [
        apply_gaussian_filter(tensor, kernel_size, sigma)
        for tensor in tensor_components
    ]
    filtered_arrays = [tensor.cpu().numpy().squeeze() for tensor in filtered_tensors]

    return filtered_arrays
