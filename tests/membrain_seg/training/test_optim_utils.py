import pytest


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_loss_fn_correctness():
    """Test function for above defined loss functions.

    Several volumes are generated with different values in the ignore regions.
    They all should evaluate to the same loss value.
    """

    import torch
    from membrain_seg.segmentation.training.optim_utils import (
        CombinedLoss,
        DeepSuperVisionLoss,
        IgnoreLabelDiceCELoss,
    )

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
            ignore_dice_loss(data=pl, target=gt_labels.clone()) for pl in pred_labels
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
    combined_loss = CombinedLoss(
        losses=[ignore_dice_loss], weights=[1.0], loss_inclusion_tokens=["ds1"]
    )
    losses = test_ignore_dice_loss(ignore_dice_loss, pred_labels, gt_labels)
    assert losses[0] == losses[1] == losses[2] == losses[4] != losses[3]

    deep_supervision_loss = DeepSuperVisionLoss(
        combined_loss, weights=[1.0, 0.5, 0.25, 0.125, 0.0675]
    )
    gt_labels_ds = extend_labels(gt_labels)

    ds_losses = []
    for pred_label in pred_labels:
        pred_labels_ds = extend_labels(pred_label)
        ds_losses.append(
            deep_supervision_loss(
                pred_labels_ds, gt_labels_ds, ["ds1"] * len(gt_labels_ds)
            )
        )
    assert ds_losses[0] == ds_losses[1] == ds_losses[2] == ds_losses[4] != ds_losses[3]
    print("All ignore loss assertions passed.")
