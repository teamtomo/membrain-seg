import torch


def masked_accuracy(y_pred, y_gt, ignore_label=None, threshold_value=0.0):
    """Computes fraction of correctly predicted voxels after thresholding.

    If an ignore label is provided, accuracy is only computed for voxels
    where the GT label is NOT equal to the ignore label.
    """
    mask = (
        y_gt == ignore_label
        if ignore_label is not None
        else torch.ones_like(y_gt).bool()
    )
    acc = (threshold_function(y_pred, threshold_value=threshold_value) == y_gt).float()
    acc[mask] = 0.0
    acc = acc.sum()
    acc /= (~mask).sum()
    return acc


def threshold_function(predictions, threshold_value=0.0):
    """Return 0-1 Array from thresholding."""
    binary_mask = (predictions >= threshold_value).float()
    return binary_mask
