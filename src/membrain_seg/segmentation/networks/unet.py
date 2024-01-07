from functools import partial
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType, Lambda

from ..training.metric_utils import masked_accuracy, masked_mse, threshold_function
from ..training.optim_utils import (
    CombinedLoss,
    DeepSuperVisionLoss,
    DynUNetDirectDeepSupervision,
    IgnoreLabelDiceCELoss,
    MaskedNormalMSELoss,
)
from ..training.surface_dice import IgnoreLabelSurfaceDiceLoss, masked_surface_dice


class SemanticSegmentationUnet(pl.LightningModule):
    """Implementation of a Unet for semantic segmentation.

    This is uses the monai Unet. See the monai docs for more details
    on the parameters.
    https://docs.monai.io/en/stable/networks.html#unet

    Parameters
    ----------
    spatial_dims : int
        The number of spatial dimensions is in the data.
    in_channels : int
        The number of channels in the input tensor.
        The default value is 1.
    out_channels : int
        The number of channels in the output tensor.
        The default value is 1.
    channels : Tuple[int, ...]
        The number of channels in each layer of the encoder/decoder.
        default=[32, 64, 128, 256, 512, 1024]
    strides : Tuple[int, ...], default=(1, 2, 2, 2, 2, 2)
        The strides for the convolutions. Must have len(channels - 1) elements.
    learning_rate : float, default=1e-2
        The learning rate for the Adam optimizer.
    min_learning_rate : float, default=1e-6
        The minimum learning rate.
    batch_size : int, default=32
        The batch size for training.
    image_key : str
        The value in the batch data dictionary containing the input image.
        Default value is "image".
    label_key : str
        The value in the batch data dictionary containing the labels.
        Default value is "label"
    roi_size : Tuple[int, ...]
        The size of the sliding window for the validation inference.
        Default value is (160, 160, 160).
    max_epochs : int, default=1000
        The maximum number of epochs for training.
    use_deep_supervision : bool, default=False
        Whether to use deep supervision.
    use_surf_dice : bool, default=False
        Whether to use surface dice loss.
    surf_dice_weight : float, default=1.0
        The weight for the surface dice loss.
    surf_dice_tokens : list, default=[]
        The tokens for which to compute the surface dice loss.

    """

    def __init__(
        self,
        channels: Tuple[int, ...] = [32, 64, 128, 256, 512, 1024],
        strides: Tuple[int, ...] = (1, 2, 2, 2, 2, 2),
        learning_rate: float = 1e-2,
        min_learning_rate: float = 1e-6,
        batch_size: int = 32,
        image_key: str = "image",
        label_key: str = "label",
        roi_size: Tuple[int, ...] = (160, 160, 160),
        max_epochs: int = 1000,
        use_deep_supervision: bool = True,
        use_surf_dice: bool = False,
        surf_dice_weight: float = 1.0,
        surf_dice_tokens: list = None,
        compute_normal_vectors: bool = False,
        normals_loss_weight: float = 1.0,
        normal_loss_tokens: list = None,
        dropout=None,
    ):
        super().__init__()

        # store parameters
        self.spatial_dims = 3
        self.in_channels = 1
        self.out_channels = 1 if not compute_normal_vectors else 4
        self.channels = channels
        self.strides = strides
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.batch_size = batch_size
        self.image_key = image_key
        self.label_key = label_key
        self.roi_size = roi_size
        self.max_epochs = max_epochs
        self.use_deep_supervision = use_deep_supervision
        self.use_surf_dice = use_surf_dice
        self.surf_dice_weight = surf_dice_weight
        self.surf_dice_tokens = surf_dice_tokens
        self.compute_normal_vectors = compute_normal_vectors
        self.normals_loss_weight = normals_loss_weight
        self.normal_loss_tokens = normal_loss_tokens

        self.build_model()
        self.configure_losses()
        self.setup_metrics()
        self.setup_post_transforms()
        self.init_metrics_storage()

    def build_model(self):
        """Builds the model architecture."""
        self._model = DynUNetDirectDeepSupervision(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3, 3, 3, 3, 3),
            strides=self.strides,
            upsample_kernel_size=(1, 2, 2, 2, 2, 2),
            filters=self.channels,
            res_block=True,
            deep_supervision=True,
            deep_supr_num=2,
        )

    def configure_losses(self):
        """Configures the loss function."""
        ### Build up loss function
        losses = []
        weights = []
        loss_inclusion_tokens = []

        # Dice loss (always used)
        ignore_dice_loss = IgnoreLabelDiceCELoss(ignore_label=2, reduction="none")
        losses.append(ignore_dice_loss)
        weights.append(1.0)
        loss_inclusion_tokens.append(["all"])

        # Surface dice loss (optional)
        if self.use_surf_dice:
            ignore_surf_dice_loss = IgnoreLabelSurfaceDiceLoss(
                ignore_label=2, soft_skel_iterations=5
            )
            losses.append(ignore_surf_dice_loss)
            weights.append(self.surf_dice_weight)
            if self.surf_dice_tokens is None:
                surf_dice_tokens = ["all"]
            loss_inclusion_tokens.append(surf_dice_tokens)

        # Normal vectors loss (optional)
        if self.compute_normal_vectors:
            normal_vectors_loss = MaskedNormalMSELoss(
                ignore_label=2,
            )
            losses.append(normal_vectors_loss)
            weights.append(self.normals_loss_weight)
            if self.normal_loss_tokens is None:
                normal_loss_tokens = ["all"]
            loss_inclusion_tokens.append(normal_loss_tokens)

        # Scale weights to sum to 1
        scaled_weights = [entry / sum(weights) for entry in weights]

        # Combine losses
        loss_function = CombinedLoss(
            losses=losses,
            weights=scaled_weights,
            loss_inclusion_tokens=loss_inclusion_tokens,
        )

        # Apply deep supervision if desired
        self.loss_function = DeepSuperVisionLoss(
            loss_function,
            weights=[1.0, 0.5, 0.25, 0.125, 0.0675]
            if self.use_deep_supervision
            else [1.0, 0.0, 0.0, 0.0, 0.0],
        )

    def setup_metrics(self):
        """Set up the metrics for validation."""
        # validation metric
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

    def setup_post_transforms(self):
        """Set up post-processing transforms for validation metric calculation."""
        # transforms for val metric calculation
        threshold_function_partial = partial(threshold_function, threshold_value=0.0)
        self.post_pred = Compose(
            [
                EnsureType("tensor", device="cpu"),
                Lambda(threshold_function_partial),
                AsDiscrete(
                    to_onehot=2
                ),  # No argmax needed, since only 1 channel output
            ]
        )
        self.post_label = Compose(
            [
                EnsureType("tensor", device="cpu"),
                AsDiscrete(to_onehot=2),
            ]
        )

    def init_metrics_storage(self):
        """Initialize storage for outputs and running metrics."""
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.running_train_acc = 0.0
        self.running_train_surf_dice = 0.0
        self.running_val_acc = 0.0
        self.running_val_surf_dice = 0.0
        self.running_train_normals_mse = 0.0
        self.running_val_normals_mse = 0.0

    def forward(self, x) -> torch.Tensor:
        """Implementation of the forward pass.

        See the pytorch-lightning module documentation for details.
        """
        out = self._model(x)
        return out

    def configure_optimizers(self):
        """Set up the Adam optimzier.

        See the pytorch-lightning module documentation for details.
        """
        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self.learning_rate,
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )  # SGD as in nnUNet
        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=self.max_epochs, eta_min=self.min_learning_rate
        # )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1 - epoch / self.max_epochs) ** 0.9
        )  # PolyLR from nnUNet
        return [optimizer], [scheduler]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single training step."""
        # Extracting images, labels, and dataset labels from the batch
        images, labels, ds_label, vec_GTs = (
            batch["image"],
            batch["label"],
            batch["dataset"],
            None,
        )
        if self.compute_normal_vectors:
            vec_GTs = batch["vectors"]

        # Forward pass through the model
        output = self.forward(images)

        # Compute loss
        loss = self.loss_function(output, labels, ds_label, vec_GTs=vec_GTs)

        # Update statistics for training
        stats_dict = {"train_loss": loss, "train_number": output[0].shape[0]}
        self.training_step_outputs.append(stats_dict)

        # Update running metrics for accuracy and surface dice
        self.running_train_acc += (
            masked_accuracy(output[0], labels[0], ignore_label=2.0, threshold_value=0.0)
            * output[0].shape[0]
        )
        self.running_train_surf_dice += (
            masked_surface_dice(
                data=output[0].detach(),
                target=labels[0].detach(),
                ignore_label=2.0,
                soft_skel_iterations=5,
                smooth=1.0,
                reduction="mean",
            )
            * output[0].shape[0]
        )

        if self.compute_normal_vectors:
            self.running_train_normals_mse += (
                masked_mse(
                    y_pred=output[0].detach(),
                    y_gt=labels[0].detach(),
                    y_normals_gt=vec_GTs[0],
                    ignore_label=2.0,
                    data_channels=(1, 2, 3),
                )
                * output[0].shape[0]
            )

        return {"loss": loss}  # Returning loss as part of a dictionary

    def on_train_epoch_end(self):
        """What happens after each training epoch?.

        Learning rate scheduler makes one step.
        Then training loss is logged.
        """
        outputs = self.training_step_outputs
        train_loss, num_items = 0, 0
        for output in outputs:
            train_loss += output["train_loss"].sum().item() * output["train_number"]
            num_items += output["train_number"]
        mean_train_loss = torch.tensor(train_loss / num_items)

        mean_train_acc = self.running_train_acc / num_items
        mean_train_surf_dice = self.running_train_surf_dice / num_items
        self.running_train_acc = 0.0
        self.running_train_surf_dice = 0.0
        self.log("train_loss", mean_train_loss)
        self.log("train_acc", mean_train_acc)
        self.log("train_surf_dice", mean_train_surf_dice)

        self.training_step_outputs = []
        print("EPOCH Training loss", mean_train_loss.item())
        print("EPOCH Training acc", mean_train_acc.item())
        print("EPOCH Training surface dice", mean_train_surf_dice.item())

        if self.compute_normal_vectors:
            mean_train_normals_mse = self.running_train_normals_mse / num_items
            self.running_train_normals_mse = 0.0
            self.log("train_normals_mse", mean_train_normals_mse)
            print("EPOCH Training normals mse", mean_train_normals_mse.item())

        return {"train_loss": mean_train_loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.

        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        images, labels, ds_label, vec_GTs = (
            batch["image"],
            batch["label"],
            batch["dataset"],
            None,
        )
        if self.compute_normal_vectors:
            vec_GTs = batch["vectors"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels, ds_label, vec_GTs=vec_GTs)
        print(loss, "in validation_step")

        # Cloning and adjusting preds & labels for Dice.
        # Could also use the same labels, but maybe we want to
        # compute more stats?
        outputs4dice = outputs[0][:, :1, ...].clone()
        labels4dice = labels[0].clone()
        outputs4dice[
            labels4dice == 2
        ] = -1.0  # Setting to -1 here leads to 0-labels after thresholding
        labels4dice[labels4dice == 2] = 0  # Need to set to zero before post_label
        # Otherwise we have 3 classes
        outputs4dice = [self.post_pred(i) for i in decollate_batch(outputs4dice)]
        labels4dice = [self.post_label(i) for i in decollate_batch(labels4dice)]
        self.dice_metric(y_pred=outputs4dice, y=labels4dice)

        stats_dict = {"val_loss": loss, "val_number": outputs[0].shape[0]}
        self.validation_step_outputs.append(stats_dict)
        self.running_val_acc += (
            masked_accuracy(
                outputs[0], labels[0], ignore_label=2.0, threshold_value=0.0
            )
            * outputs[0].shape[0]
        )

        self.running_val_surf_dice += (
            masked_surface_dice(
                data=outputs[0].detach(),
                target=labels[0].detach(),
                ignore_label=2.0,
                soft_skel_iterations=5,
                smooth=1.0,
                reduction="mean",
            )
            * outputs[0].shape[0]
        )

        if self.compute_normal_vectors:
            self.running_val_normals_mse += (
                masked_mse(
                    y_pred=outputs[0].detach(),
                    y_gt=labels[0].detach(),
                    y_normals_gt=vec_GTs[0],
                    ignore_label=2.0,
                    data_channels=(1, 2, 3),
                )
                * outputs[0].shape[0]
            )
        return stats_dict

    def on_validation_epoch_end(self):
        """Calculate validation loss/metric summary."""
        outputs = self.validation_step_outputs
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item() * output["val_number"]
            # Need to multiply by output["val_number"] because it's later normalized.

            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        mean_val_acc = self.running_val_acc / num_items
        mean_val_surf_dice = self.running_val_surf_dice / num_items
        self.running_val_acc = 0.0
        self.running_val_surf_dice = 0.0
        self.log("val_loss", mean_val_loss),
        self.log("val_dice", mean_val_dice)
        self.log("val_surf_dice", mean_val_surf_dice)
        self.log("val_accuracy", mean_val_acc)

        self.validation_step_outputs = []
        print("EPOCH Validation loss", mean_val_loss.item())
        print("EPOCH Validation dice", mean_val_dice)
        print("EPOCH Validation surface dice", mean_val_surf_dice.item())
        print("EPOCH Validation acc", mean_val_acc.item())

        if self.compute_normal_vectors:
            mean_val_normals_mse = self.running_val_normals_mse / num_items
            self.running_val_normals_mse = 0.0
            self.log("val_normals_mse", mean_val_normals_mse)
            print("EPOCH Validation normals mse", mean_val_normals_mse.item())
        return {"val_loss": mean_val_loss, "val_metric": mean_val_dice}
