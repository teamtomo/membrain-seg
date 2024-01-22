import torch
from monai.losses import DiceLoss, MaskedLoss
from monai.networks.nets import DynUNet
from monai.utils import LossReduction
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    sigmoid,
)
from torch.nn.modules.loss import _Loss


class DynUNetDirectDeepSupervision(DynUNet):
    """Adjusted DynUNet outputting low-resolution deep supervision images.

    This is in contrast to the original DynUNet implementation: Here, images
    from lower stages are first upsampled, and then compared to the original
    resolution GT image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Forward pass."""
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(feature_map)
            return out_all
        return out


class IgnoreLabelDiceCELoss(_Loss):
    """
    Mix of Dice & Cross-entropy loss, adding ignore labels.

    Parameters
    ----------
    ignore_label : int
        The label to ignore when calculating the loss.
    reduction : str, optional
        Specifies the reduction to apply to the output, by default "mean".
    lambda_dice : float, optional
        The weight for the Dice loss, by default 1.0.
    lambda_ce : float, optional
        The weight for the Cross-Entropy loss, by default 1.0.
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        ignore_label: int,
        reduction: str = "none",
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.ignore_label = ignore_label
        self.dice_loss = MaskedLoss(DiceLoss, reduction=reduction, **kwargs)
        self.reduction = reduction
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

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
        orig_data = data.clone()
        data = sigmoid(data)
        mask = target != self.ignore_label

        # Compute the cross entropy loss while ignoring the ignore_label
        target_comp = target.clone()
        target_comp[target == self.ignore_label] = 0
        target_tensor = torch.tensor(target_comp, dtype=data.dtype, device=data.device)

        bce_loss = binary_cross_entropy_with_logits(
            orig_data, target_tensor, reduction="none"
        )
        bce_loss[~mask] = 0.0
        # TODO: Check if this is correct: I adjusted the loss to be
        # computed per batch element
        bce_loss = torch.sum(bce_loss, dim=(1, 2, 3, 4)) / torch.sum(
            mask, dim=(1, 2, 3, 4)
        )
        # Compute Dice loss separately for each batch element
        dice_loss = torch.zeros_like(bce_loss)
        for batch_idx in range(data.shape[0]):
            dice_loss[batch_idx] = self.dice_loss(
                data[batch_idx].unsqueeze(0),
                target[batch_idx].unsqueeze(0),
                mask[batch_idx].unsqueeze(0),
            )

        # Combine the Dice and Cross Entropy losses
        combined_loss = self.lambda_dice * dice_loss + self.lambda_ce * bce_loss
        if self.reduction == "mean":
            combined_loss = combined_loss.mean()
        elif self.reduction == "sum":
            combined_loss = combined_loss.sum()
        else:
            raise ValueError(
                f"Invalid reduction type {self.reduction}. "
                "Valid options are 'mean' and 'sum'."
            )
        return combined_loss


class DeepSuperVisionLoss(_Loss):
    """
    Deep Supervision loss using downsampled GT and low-res outputs.

    Implementation based on nnU-Net's implementation with downsampled images.
    Reference: Zeng, Guodong, et al. "3D U-net with multi-level deep supervision:
    fully automatic segmentation of proximal femur in 3D MR images." Machine Learning
    in Medical Imaging: 8th International Workshop, MLMI 2017, Held in Conjunction with
    MICCAI 2017, Quebec City, QC, Canada, September 10, 2017, Proceedings 8. Springer
    International Publishing, 2017.

    Parameters
    ----------
    loss_fn : _Loss, optional
        The loss function to use, by default IgnoreLabelDiceCELoss.
    weights : list, optional
        List of weights for each input/target pair, by default None.
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        loss_fn: _Loss = IgnoreLabelDiceCELoss,
        weights=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, inputs: list, targets: list, ds_labels: list) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        inputs : list
            List of tensors of model outputs.
        targets : list
            List of tensors of target labels.
        ds_labels : list
            List of dataset labels for each batch element.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        loss = 0.0
        ds_labels_loop = [ds_labels] * 5
        for weight, data, target, ds_label in zip(
            self.weights, inputs, targets, ds_labels_loop
        ):
            loss += weight * self.loss_fn(data, target, ds_label)
        return loss


class CombinedLoss(_Loss):
    """
    Combine multiple loss functions into a single one.

    Parameters
    ----------
    losses : List[Callable]
        A list of loss function instances.
    weights : List[float]
        List of weights corresponding to each loss function (must
        be of same length as losses).
    loss_inclusion_tokens : List[List[str]]
        A list of lists containing tokens for each loss function.
        Each sublist corresponds to a loss function and contains
        tokens for which the loss should be included.
        If the list contains "all", then the loss will be included
        for all cases.

    Notes
    -----
    IMPORTANT: Loss functions need to return a tensors containing the
    loss for each batch element.

    The loss_exclusion_tokens parameter is used to exclude certain
    cases from the loss calculation. For example, if the loss_exclusion_tokens
    parameter is [["ds1", "ds2"], ["ds1"]], then the first loss function
    will be excluded for cases where the dataset label is "ds1" or "ds2",
    and the second loss function will be excluded for cases where the
    dataset label is "ds1".
    """

    def __init__(
        self,
        losses: list,
        weights: list,
        loss_inclusion_tokens: list,
        **kwargs,
    ) -> None:
        super().__init__()
        self.losses = losses
        self.weights = weights
        self.loss_inclusion_tokens = loss_inclusion_tokens

    def forward(
        self, data: torch.Tensor, target: torch.Tensor, ds_label: list
    ) -> torch.Tensor:
        """
        Compute the combined loss.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of model outputs.
        target : torch.Tensor
            Tensor of target labels.
        ds_label : List[str]
            List of dataset labels for each batch element.

        Returns
        -------
        torch.Tensor
            The calculated combined loss.
        """
        loss = 0.0
        for loss_idx, (cur_loss, cur_weight) in enumerate(
            zip(self.losses, self.weights)
        ):
            cur_loss_val = cur_loss(data, target)

            # Zero out losses for excluded cases
            for batch_idx, ds_lab in enumerate(ds_label):
                if (
                    "all" in self.loss_inclusion_tokens[loss_idx]
                    or ds_lab in self.loss_inclusion_tokens[loss_idx]
                ):
                    continue
                cur_loss_val[batch_idx] = 0.0

            # Aggregate loss
            cur_loss_val = cur_loss_val.sum() / ((cur_loss_val != 0.0).sum() + 1e-3)
            loss += cur_weight * cur_loss_val

        # Normalize loss
        loss = loss / sum(self.weights)
        return loss
