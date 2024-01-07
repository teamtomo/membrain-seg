from typing import Optional

import torch


def masked_accuracy(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    ignore_label: Optional[int] = None,
    threshold_value: float = 0.0,
    data_channel: int = 0,
) -> torch.Tensor:
    """
    Computes fraction of correctly predicted voxels after thresholding.

    If an ignore label is provided, accuracy is only computed for voxels
    where the GT label is NOT equal to the ignore label.

    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor representing the model's predictions.
    y_gt : torch.Tensor
        Tensor representing the ground truth labels.
    ignore_label : Optional[int], optional
        The label to ignore when calculating accuracy, by default None
    threshold_value : float, optional
        The threshold value to convert predictions into binary values, by default 0.0
    data_channel : int, optional
        The channel of the data to use for computing accuracy, by default 0

    Returns
    -------
    torch.Tensor
        The computed accuracy of the model's predictions.
    """
    y_pred = y_pred[:, data_channel : data_channel + 1, ...]
    mask = (
        y_gt == ignore_label
        if ignore_label is not None
        else torch.zeros_like(y_gt).bool()
    )
    acc = (threshold_function(y_pred, threshold_value=threshold_value) == y_gt).float()
    acc[mask] = 0.0
    acc = acc.sum()
    acc /= (~mask).sum() + 1e-3
    return acc


def masked_mse(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    y_normals_gt: torch.Tensor,
    ignore_label: Optional[int] = None,
    data_channels: tuple = (1, 2, 3),
) -> torch.Tensor:
    """
    Computes the mean squared error between the model's predictions and ground truth.

    If an ignore label is provided, MSE is only computed for voxels
    where the GT label is NOT equal to the ignore label.

    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor representing the model's predictions.
    y_gt : torch.Tensor
        Tensor representing the ground truth labels.
    y_normals_gt : torch.Tensor
        Tensor representing the ground truth normals.
    ignore_label : Optional[int], optional
        The label to ignore when calculating MSE, by default None
    data_channels : int, optional
        The channel of the data to use for computing MSE, by default 0

    Returns
    -------
    torch.Tensor
        The computed MSE of the model's predictions.
    """
    y_pred = y_pred[:, data_channels, ...]
    mask = (
        y_gt == ignore_label
        if ignore_label is not None
        else torch.zeros_like(y_gt).bool()
    )
    mask = torch.cat([mask, mask, mask], dim=1)

    mse = (y_pred - y_normals_gt) ** 2
    mse[mask] = 0.0
    mse = mse.sum()
    mse /= (~mask).sum() + 1e-3
    return mse


def threshold_function(
    predictions: torch.Tensor, threshold_value: float = 0.0
) -> torch.Tensor:
    """
    Return 0-1 Array from thresholding.

    Parameters
    ----------
    predictions : torch.Tensor
        The tensor containing prediction values.
    threshold_value : float, optional
        The threshold value for converting predictions into binary values,
        by default 0.0

    Returns
    -------
    torch.Tensor
        The binary tensor obtained after thresholding the predictions.
    """
    binary_mask = (predictions >= threshold_value).float()
    return binary_mask
