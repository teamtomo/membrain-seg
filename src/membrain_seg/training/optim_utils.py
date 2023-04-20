import torch
from monai.losses import DiceLoss, MaskedLoss
from monai.networks.nets import DynUNet
from monai.utils import LossReduction
from torch.nn.functional import binary_cross_entropy, sigmoid
from torch.nn.modules.loss import _Loss


class DynUNetDirectDeepSupervision(DynUNet):
    """Adjusted DynUNet outputting low-resolution deep supervision images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Forward pass."""
        out = self.skip_layers(x)
        out = self.output_block(out)
        # if self.training and self.deep_supervision:
        # #TODO: Should this only be used for training or also validation?
        if self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(feature_map)
            return out_all
        return out


class IgnoreLabelDiceCELoss(_Loss):
    """Mix of Dice & Cross-entropy loss, adding ignore labels."""

    def __init__(
        self,
        ignore_label: int,
        reduction: str = "mean",
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
        """Compute the loss."""
        # Create a mask to ignore the specified label in the target
        data = sigmoid(data)
        mask = target != self.ignore_label

        # Compute the cross entropy loss while ignoring the ignore_label
        target_comp = target.clone()
        target_comp[target == self.ignore_label] = 0
        target_tensor = torch.tensor(target_comp, dtype=data.dtype, device=data.device)
        bce_loss = binary_cross_entropy(data, target_tensor, reduction="none")
        bce_loss[~mask] = 0.0
        bce_loss = torch.sum(bce_loss) / torch.sum(mask)
        dice_loss = self.dice_loss(data, target, mask)

        # Combine the Dice and Cross Entropy losses
        combined_loss = self.lambda_dice * dice_loss + self.lambda_ce * bce_loss
        return combined_loss


class DeepSuperVisionLoss(_Loss):
    """Deep Supervision loss using downsampled GT and low-res outputs."""

    def __init__(
        self,
        loss_fn: _Loss = IgnoreLabelDiceCELoss,
        weights=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, inputs: list, targets: list) -> torch.Tensor:
        """Compute the loss."""
        loss = 0.0
        for weight, data, target in zip(self.weights, inputs, targets):
            loss += weight * self.loss_fn(data, target)
        return loss
