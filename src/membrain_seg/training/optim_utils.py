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
        # bce_loss = binary_cross_entropy(input, torch.FloatTensor(target_comp),
        # reduction='none')
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


def test_loss_fn():
    """Test function for above defined loss functions.

    Several volumes are generated with different values in the ignore regions.
    They all should evaluate to the same loss value.
    """

    def create_pred_labels():
        return [torch.zeros((2, 1, 160, 160, 160)) for _ in range(5)]

    def create_gt_labels():
        gt_labels = torch.zeros((2, 1, 160, 160, 160))
        gt_labels[:, :, :80, :, :] = 1
        gt_labels[:, :, 80:120, :, :] = 2
        return gt_labels

    def set_pred_labels(pred_labels, values):
        for idx, value_info in enumerate(values):
            for sub_value_info in value_info:
                start_end = sub_value_info[0]
                value = sub_value_info[1]
                pred_labels[idx][
                    :,
                    :,
                    (0 if isinstance(start_end, int) else start_end[0]) : (
                        start_end if isinstance(start_end, int) else start_end[1]
                    ),
                    :,
                    :,
                ] = value
        return pred_labels

    def test_ignore_dice_loss(ignore_dice_loss, pred_labels, gt_labels):
        losses = [
            ignore_dice_loss(input=pl, target=gt_labels.clone()) for pl in pred_labels
        ]
        assert losses[0] == losses[1] == losses[2] == losses[4] != losses[3]
        return losses

    def extend_labels(labels):
        return [
            labels,
            labels[:, :, 40:120, 40:120, 40:120],
            labels[:, :, 60:100, 60:100, 60:100],
            labels[:, :, 70:90, 70:90, 70:90],
            labels[:, :, 75:85, 75:85, 75:85],
        ]

    pred_labels = create_pred_labels()
    gt_labels = create_gt_labels()

    values = [
        [[70, 0.8], [(80, 120), 0]],
        [[70, 0.8], [(80, 120), 1]],
        [[70, 0.8], [(80, 120), torch.randn_like(pred_labels[0][:, :, 80:120, :, :])]],
        [[80, 0.8], [(80, 120), 1]],
        [[70, 0.8], [(80, 120), 0.5]],
    ]

    pred_labels = set_pred_labels(pred_labels, values)
    pred_labels[2][pred_labels[2] > 1.0] = 1.0
    pred_labels[2][pred_labels[2] < 0.0] = 0.0

    ignore_dice_loss = IgnoreLabelDiceCELoss(ignore_label=2, reduction="mean")
    losses = test_ignore_dice_loss(ignore_dice_loss, pred_labels, gt_labels)
    assert losses[0] == losses[1] == losses[2] == losses[4] != losses[3]

    deep_supervision_loss = DeepSuperVisionLoss(
        ignore_dice_loss, weights=[1.0, 0.5, 0.25, 0.125, 0.0675]
    )
    gt_labels_ds = extend_labels(gt_labels)

    ds_losses = []
    for pred_label in pred_labels:
        pred_labels_ds = extend_labels(pred_label)
        ds_losses.append(deep_supervision_loss(pred_labels_ds, gt_labels_ds))
    assert ds_losses[0] == ds_losses[1] == ds_losses[2] == ds_losses[4] != ds_losses[3]
    print("All ignore loss assertions passed.")


if __name__ == "__main__":
    test_loss_fn()
