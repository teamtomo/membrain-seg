import numpy as np


def masked_surface_dice(
    pred_skel: np.ndarray,
    gt_skel: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute surface dice score for two skeletons.

    Parameters
    ----------
    pred_skel : np.ndarray
        Skeleton of the prediction.
    gt_skel : np.ndarray
        Skeleton of the ground truth.
    pred : np.ndarray
        Prediction.
    gt : np.ndarray
        Ground truth.
    mask : np.ndarray
        Mask to ignore certain labels.

    Returns
    -------
    float
        Surface dice score.
    dict
        Dictionary containing the following keys:
        - tp_pred: True positives in the prediction.
        - tp_gt: True positives in the ground truth.
        - all_pred: All positives in the prediction.
        - all_gt: All positives in the ground truth.
    """
    # Mask out ignore labels
    pred_skel[~mask] = 0
    gt_skel[~mask] = 0

    tp_pred = np.sum(np.multiply(pred_skel, gt))
    tp_gt = np.sum(np.multiply(gt_skel, pred))
    all_pred = np.sum(pred_skel)
    all_gt = np.sum(gt_skel)

    tprec = tp_pred / (all_pred + 1e-6)
    tsens = tp_gt / (all_gt + 1e-6)

    surf_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
    return surf_dice, {
        "tp_pred": tp_pred,
        "tp_gt": tp_gt,
        "all_pred": all_pred,
        "all_gt": all_gt,
    }


def masked_dice(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Compute dice score for two segmentations.

    Parameters
    ----------
    pred : np.ndarray
        Prediction.
    gt : np.ndarray
        Ground truth.
    mask : np.ndarray
        Mask to ignore certain labels.

    Returns
    -------
    float
        Dice score.
    dict
        Dictionary containing the following
        keys:
        - tp: True positives.
        - fp: False positives.
        - fn: False negatives.

    """
    pred[~mask] = 0
    gt[~mask] = 0
    tp = np.sum(np.logical_and(pred == 1, gt == 1))
    fp = np.sum(np.logical_and(pred == 1, gt == 0))
    fn = np.sum(np.logical_and(pred == 0, gt == 1))
    tprec = tp / (tp + fp + 1e-6)
    tsens = tp / (tp + fn + 1e-6)
    # return also dict
    out_dict = {"tp": tp, "fp": fp, "fn": fn}
    dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
    return dice, out_dict
