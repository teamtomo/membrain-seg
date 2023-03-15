from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet as MonaiUnet
from monai.transforms import AsDiscrete, Compose, EnsureType


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
        Default value is (16, 32, 64, 128, 256)
    strides : Tuple[int, ...]
        The strides for the convolutions. Must have len(channels - 1) elements.
        Default value is (2, 2, 2, 2)
    num_res_units : int
        The number of residual subunits.
    learning_rate : float
        The learning rate for the Adam optimizer.
        Default value is 1e-4.
    image_key : str
        The value in the batch data dictionary containing the input image.
        Default value is "image".
    label_key : str
        The value in the batch data dictionary containing the labels.
        Default value is "label"
    roi_size : Tuple[int, ...]
        The size of the sliding window for the validation inference.
        Default value is (160, 160, 160).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        learning_rate: float = 1e-4,
        image_key: str = "image",
        label_key: str = "label",
        roi_size: Tuple[int, ...] = (160, 160, 160),
    ):
        super().__init__()

        # store parameters
        self.learning_rate = learning_rate
        self.image_key = image_key
        self.label_key = label_key
        self.roi_size = roi_size

        # make the network
        self._model = MonaiUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceFocalLoss()

        # validation metric
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

        # transforms for val metric calculation
        self.post_pred = Compose(
            [
                EnsureType("tensor", device="cpu"),
                AsDiscrete(argmax=True, to_onehot=out_channels),
            ]
        )
        self.post_label = Compose(
            [
                EnsureType("tensor", device="cpu"),
                AsDiscrete(to_onehot=out_channels),
            ]
        )

    def forward(self, x) -> torch.Tensor:
        """Implementation of the forward pass.

        See the pytorch-lightning module documentation for details.
        """
        return self._model(x)

    def configure_optimizers(self):
        """Set up the Adam optimzier.

        See the pytorch-lightning module documentation for details.
        """
        optimizer = torch.optim.Adam(self._model.parameters(), self.learning_rate)
        return optimizer

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Implementation of one training step.

        See the pytorch-lightning module documentation for details.
        """
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)

        self.log("training_loss", loss, batch_size=len(images))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.

        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        images, labels = batch[self.image_key], batch[self.label_key]
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, self.roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        val_metric = self.dice_metric(y_pred=outputs, y=labels)
        self.log("val_loss", loss, batch_size=len(images))
        self.log("val_metric", val_metric, batch_size=len(images))
        return {"val_loss": loss, "val_number": len(outputs)}
