import torch
# from monai.metrics import CumulativeIterationMetric, MetricReduction, do_metric_reduction
from monai.utils import MetricReduction
from monai.metrics import DiceHelper
from typing import Union
from .surface_dice_loss import masked_surface_dice
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction
from monai.metrics.metric import CumulativeIterationMetric


class SurfaceDiceMetric(CumulativeIterationMetric):
    """
    Compute average Surface Dice score for a set of pairs of prediction-groundtruth segmentations.

    It supports both multi-classes and multi-labels tasks.
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_label: int = 2.,
        soft_skel_iterations: int = 3,
        smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_label = ignore_label
        self.soft_skel_iterations = soft_skel_iterations
        self.smooth = smooth


    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
            y: ground truth to compute mean Dice metric. `y` can be single-channel class indices or
                in the one-hot format.
        """
        surf_dice = masked_surface_dice(data=y_pred, target=y, ignore_label=self.ignore_label, 
            soft_skel_iterations=self.soft_skel_iterations, smooth=self.smooth, reduction=self.reduction)
        
        return surf_dice

    def aggregate(
        self, reduction: Union[MetricReduction, str] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Execute reduction and aggregation logic for the output of `masked_surface_dice`.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f