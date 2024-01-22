"""
Surface Dice implementation.

Adapted from: clDice - A Novel Topology-Preserving Loss Function for Tubular 
Structure Segmentation
Original Authors: Johannes C. Paetzold and Suprosanna Shit
Sources: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/
        soft_skeleton.py
         https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py 
License: MIT License.

The following code is a modification of the original clDice implementation.
Modifications were made to include additional functionality and integrate 
with new project requirements. The original license and copyright notice are 
provided below.

MIT License

Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.functional import sigmoid
from torch.nn.modules.loss import _Loss


def soft_erode(img: torch.Tensor, separate_pool: bool = False) -> torch.Tensor:
    """
    Apply soft erosion operation to the input image.

    Soft erosion is achieved by applying a min-pooling operation to the input image.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor with shape (B, C, D, H, W)
    separate_pool : bool, optional
        If True, perform separate 3D max-pooling operations along different axes.
        Default is False.

    Returns
    -------
    torch.Tensor
        Eroded image tensor with the same shape as the input.

    Raises
    ------
    ValueError
        If the input tensor has an unsupported number of dimensions.

    Notes
    -----
    - The soft erosion can be performed with separate 3D min-pooling operations
            along different axes if separate_pool is True, or with a single
            3D min-pooling operation with a kernel of size (3, 3, 3) if
            separate_pool is False.
    """
    assert len(img.shape) == 5
    if separate_pool:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    p4 = -F.max_pool3d(-img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    return p4


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Apply soft dilation operation to the input image.

    Soft dilation is achieved by applying a max-pooling operation to the input image.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor with shape (B, C, D, H, W).

    Returns
    -------
    torch.Tensor
        Dilated image tensor with the same shape as the input.

    Raises
    ------
    ValueError
        If the input tensor has an unsupported number of dimensions.

    Notes
    -----
    - For 5D input, the soft dilation is performed using a 3D max-pooling operation
            with a kernel of size (3, 3, 3).
    """
    assert len(img.shape) == 5
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor, separate_pool: bool = False) -> torch.Tensor:
    """
    Apply soft opening operation to the input image.

    Soft opening is achieved by applying soft erosion followed by soft dilation.
    The intention of soft opening is to remove thin membranes from the segmentation.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor with shape (B, C, D, H, W).
    separate_pool : bool, optional
        If True, perform separate erosion and dilation operations. Default is False.

    Returns
    -------
    torch.Tensor
        Opened image tensor with the same shape as the input.

    Notes
    -----
    - Soft opening is performed by applying soft erosion followed by soft dilation
            to the input image.
    - For 5D input, separate erosion and dilation can be performed if separate_pool
            is True.
    """
    return soft_dilate(soft_erode(img, separate_pool=separate_pool))


def soft_skel(
    img: torch.Tensor, iter_: int, separate_pool: bool = False
) -> torch.Tensor:
    """
    Compute the soft skeleton of the input image.

    The skeleton is computed by applying soft erosion iteratively to the input image.
    In each iteration, the difference between the input image and the "opened" image is
    computed and added to the skeleton.

    Reasoning: if there is a difference between the input image and the "opened" image,
    there must be a thin membrane skeleton in the input image that was removed by the
    opening operation.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor with shape (B, C, D, H, W).
    iter_ : int
        Number of iterations for skeletonization.
    separate_pool : bool, optional
        If True, perform separate erosion and dilation operations.
        Default is False.

    Returns
    -------
    torch.Tensor
        Soft skeleton image tensor with the same shape as the input.

    Notes
    -----
    - Separate erosion can be performed if separate_pool is True.
    """
    img1 = soft_open(img, separate_pool=separate_pool)
    skel = F.relu(img - img1)
    for _j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img, separate_pool=separate_pool)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 3D Gaussian kernel using the specified size and sigma.

    Parameters
    ----------
    size : int
        The size of the Gaussian kernel. It determines the length of
        each dimension of the cube.
    sigma : float
        The standard deviation of the Gaussian kernel. It controls
        the spread of the Gaussian.

    Returns
    -------
    torch.Tensor
        A 3D tensor representing the Gaussian kernel.

    Notes
    -----
    The function creates a Gaussian kernel, which is essentially a
    cube of dimensions [size, size, size]. Each entry in the cube is
    computed using the Gaussian function based on its distance from the center.
    The kernel is normalized so that its total sum equals 1.
    """
    # Define a coordinate grid centered at (0,0,0)
    grid = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    # Create a 3D meshgrid
    x, y, z = torch.meshgrid(grid, grid, grid)
    xyz_grid = torch.stack([x, y, z], dim=-1)

    # Calculate the 3D Gaussian kernel
    gaussian_kernel = torch.exp(-torch.sum(xyz_grid**2, dim=-1) / (2 * sigma**2))
    gaussian_kernel /= (2 * math.pi * sigma**2) ** (3 / 2)  # Normalize

    # Ensure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


gaussian_kernel_dict = {}
""" Not sure why, but moving the gaussian kernel to GPU takes surprisingly long.+
So we precompute it, store it on GPU, and reuse it.
"""


def apply_gaussian_filter(
    seg: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    """
    Apply a Gaussian filter to a segmentation tensor using PyTorch.

    This function convolves the input tensor with a Gaussian kernel.
    The function creates or retrieves a Gaussian kernel based on the
    specified size and standard deviation, and applies 3D convolution to each
    channel of each batch item with appropriate padding to maintain spatial
    dimensions.

    Parameters
    ----------
    seg : torch.Tensor
        The input segmentation tensor of shape (batch, channel, X, Y, Z).
    kernel_size : int
        The size of the Gaussian kernel, determining the length of each
        dimension of the cube.
    sigma : float
        The standard deviation of the Gaussian kernel, controlling the spread.

    Returns
    -------
    torch.Tensor
        The filtered segmentation tensor of the same shape as input.

    Notes
    -----
    This function uses a precomputed dictionary to enhance performance by
    storing Gaussian kernels. If a kernel with the specified size and standard
    deviation does not exist in the dictionary, it is created and added. The
    function assumes the input tensor is a 5D tensor, applies 3D convolution
    using the Gaussian kernel with padding to maintain spatial dimensions, and
    it performs the operation separately for each channel of each batch item.
    """
    # Create the Gaussian kernel or load it from the dictionary
    if (kernel_size, sigma) not in gaussian_kernel_dict.keys():
        gaussian_kernel_dict[(kernel_size, sigma)] = gaussian_kernel(
            kernel_size, sigma
        ).to(seg.device)
    g_kernel = gaussian_kernel_dict[(kernel_size, sigma)]

    # Add batch and channel dimensions
    g_kernel = g_kernel.view(1, 1, *g_kernel.size())
    # Apply the Gaussian filter to each channel
    padding = kernel_size // 2

    # Move the kernel to the same device as the segmentation tensor
    g_kernel = g_kernel.to(seg.device)

    # Apply the Gaussian filter
    filtered_seg = F.conv3d(seg, g_kernel, padding=padding, groups=seg.shape[1])
    return filtered_seg


def get_GT_skeleton(gt_seg: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """
    Generate the skeleton of a ground truth segmentation.

    This function takes a ground truth segmentation `gt_seg`, smooths it using a
    Gaussian filter, and then computes its soft skeleton using the `soft_skel` function.

    Intention: When using the binary ground truth segmentation for skeletonization,
    the resulting skeleton is very patchy and not smooth. When using the smoothed
    ground truth segmentation, the resulting skeleton is much smoother and more
    accurate.

    Parameters
    ----------
    gt_seg : torch.Tensor
        A torch.Tensor representing the ground truth segmentation.
        Shape: (B, C, D, H, W)
    iterations : int, optional
        The number of iterations for skeletonization. Default is 5.

    Returns
    -------
    torch.Tensor
        A torch.Tensor representing the skeleton of the ground truth segmentation.

    Notes
    -----
    - The input `gt_seg` should be a binary segmentation tensor where 1 represents the
        object of interest.
    - The function first smooths the `gt_seg` using a Gaussian filter to enhance the
        object's structure.
    - The skeletonization process is performed using the `soft_skel` function with the
        specified number of iterations.
    - The resulting skeleton is returned as a binary torch.Tensor where 1 indicates the
        skeleton points.
    """
    gt_smooth = (
        apply_gaussian_filter((gt_seg == 1) * 1.0, kernel_size=15, sigma=2.0) * 1.5
    )
    skel_gt = soft_skel(gt_smooth, iter_=iterations)
    return skel_gt


def masked_surface_dice(
    data: torch.Tensor,
    target: torch.Tensor,
    ignore_label: int = 2,
    soft_skel_iterations: int = 3,
    smooth: float = 3.0,
    binary_prediction: bool = False,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute the surface Dice loss with masking for ignore labels.

    The surface Dice loss measures the similarity between the predicted segmentation's
    skeleton and the ground truth segmentation (and vice versa). Labels annotated with
    "ignore_label" are ignored.

    Parameters
    ----------
    data : torch.Tensor
        Tensor of model outputs representing the predicted segmentation.
        Expected shape: (B, C, D, H, W)
    target : torch.Tensor
        Tensor of target labels representing the ground truth segmentation.
        Expected shape: (B, 1, D, H, W)
    ignore_label : int
        The label value to be ignored when computing the loss.
    soft_skel_iterations : int
        Number of iterations for skeletonization in the underlying operations.
    smooth : float
        Smoothing factor to avoid division by zero.
    binary_prediction : bool
        If True, the predicted segmentation is assumed to be binary. Default is False.
    reduction : str
        Specifies the reduction to apply to the output. Default is "none".

    Returns
    -------
    torch.Tensor
        The calculated surface Dice loss.
    """
    # Create a mask to ignore the specified label in the target
    data = sigmoid(data)
    mask = target != ignore_label

    # Compute soft skeletonization
    if binary_prediction:
        skel_pred = get_GT_skeleton(data.clone(), soft_skel_iterations)
    else:
        skel_pred = soft_skel(data.clone(), soft_skel_iterations, separate_pool=False)
    skel_true = get_GT_skeleton(target.clone(), soft_skel_iterations)

    # Mask out ignore labels
    skel_pred[~mask] = 0
    skel_true[~mask] = 0

    # compute surface dice loss
    tprec = (
        torch.sum(torch.multiply(skel_pred, target), dim=(1, 2, 3, 4)) + smooth
    ) / (torch.sum(skel_pred, dim=(1, 2, 3, 4)) + smooth)
    tsens = (torch.sum(torch.multiply(skel_true, data), dim=(1, 2, 3, 4)) + smooth) / (
        torch.sum(skel_true, dim=(1, 2, 3, 4)) + smooth
    )
    surf_dice_loss = 2.0 * (tprec * tsens) / (tprec + tsens)
    if reduction == "none":
        return surf_dice_loss
    elif reduction == "mean":
        return torch.mean(surf_dice_loss)


class IgnoreLabelSurfaceDiceLoss(_Loss):
    """
    Surface Dice loss, adding ignore labels.

    Parameters
    ----------
    ignore_label : int
        The label to ignore when calculating the loss.
    reduction : str, optional
        Specifies the reduction to apply to the output, by default "mean".
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        ignore_label: int,
        soft_skel_iterations: int = 3,
        smooth: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.soft_skel_iterations = soft_skel_iterations
        self.smooth = smooth

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of model outputs.
            Expected shape: (B, C, D, H, W)
        target : torch.Tensor
            Tensor of target labels.
            Expected shape: (B, 1, D, H, W)

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        # Create a mask to ignore the specified label in the target
        surf_dice_score = masked_surface_dice(
            data=data,
            target=target,
            ignore_label=self.ignore_label,
            soft_skel_iterations=self.soft_skel_iterations,
            smooth=self.smooth,
        )
        surf_dice_loss = 1.0 - surf_dice_score
        return surf_dice_loss
