import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel
import torch
from monai.losses import DiceLoss, MaskedLoss
from monai.networks.nets import DynUNet
from monai.utils import LossReduction
from torch.nn.functional import binary_cross_entropy, sigmoid
from torch.nn.modules.loss import _Loss

from matplotlib import pyplot as plt

def masked_surface_dice(data: torch.Tensor, target: torch.Tensor, ignore_label, soft_skel_iterations, smooth) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of model outputs.
        target : torch.Tensor
            Tensor of target labels.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        # Create a mask to ignore the specified label in the target
        data = sigmoid(data)
        mask = target != ignore_label

        # Compute soft skeletonization
        skel_pred = soft_skel(data.clone(), soft_skel_iterations, separate_pool=False)
        skel_true = soft_skel(target.clone(), soft_skel_iterations, separate_pool=False)


        
        # Mask out ignore labels
        skel_pred[~mask] = 0
        skel_true[~mask] = 0


        # compute surface dice loss
        tprec = (torch.sum(torch.multiply(skel_pred, target))+smooth)/(torch.sum(skel_pred)+smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, data))+smooth)/(torch.sum(skel_true)+smooth)    
        surf_dice_loss = 2.0*(tprec*tsens)/(tprec+tsens)
        return surf_dice_loss

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
        smooth: float = 3.,
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
        target : torch.Tensor
            Tensor of target labels.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        # Create a mask to ignore the specified label in the target
        surf_dice_score = masked_surface_dice(data=data, target=target, ignore_label=self.ignore_label, soft_skel_iterations=self.soft_skel_iterations, smooth=self.smooth)
        surf_dice_loss = 1. - surf_dice_score

        return surf_dice_loss