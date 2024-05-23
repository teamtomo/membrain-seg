from typing import Tuple

import torch
import torch.nn.functional as F

from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    fourier_cropping_torch,
    fourier_extend_torch,
)


def rescale_tensor(
    sample: torch.Tensor, target_size: tuple, mode="trilinear"
) -> torch.Tensor:
    """
    Rescales the input tensor by given factors using interpolation.

    Parameters
    ----------
    sample : torch.Tensor
        The input data as a torch tensor.
    target_size : tuple
        The target size of the rescaled tensor.
    mode : str, optional
        The mode of interpolation ('nearest', 'linear', 'bilinear',
          'bicubic', or 'trilinear'). Default is 'trilinear'.

    Returns
    -------
    torch.Tensor
        The rescaled tensor.
    """
    # Add batch and channel dimensions
    sample = sample.unsqueeze(0).unsqueeze(0)

    # Apply interpolation
    rescaled_sample = F.interpolate(
        sample, size=target_size, mode=mode, align_corners=False
    )

    return rescaled_sample.squeeze(0).squeeze(0)


class PreprocessedSemanticSegmentationUnet(SemanticSegmentationUnet):
    """U-Net with rescaling preprocessing.

    This class extends the SemanticSegmentationUnet class by adding
    preprocessing and postprocessing steps. The preprocessing step
    rescales the input to the target shape, and the postprocessing
    step rescales the output to the original shape.
    All of this is done on the GPU if available.
    """

    def __init__(
        self,
        *args,
        rescale_patches: bool = False,  # Should patches be rescaled?
        target_shape: Tuple[int, int, int] = (160, 160, 160),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the preprocessing parameters
        self.rescale_patches = rescale_patches
        self.target_shape = target_shape

    def preprocess(self, x):
        """Preprocess the input to the network.

        In this case, we rescale the input to the target shape.
        """
        rescaled_samples = []
        for sample in x:
            sample = sample[0]  # only use the first channel
            if self.rescale_patches:
                if sample.shape[0] > self.target_shape[0]:
                    sample = fourier_cropping_torch(
                        data=sample, new_shape=self.target_shape, device=self.device
                    )
                elif sample.shape[0] < self.target_shape[0]:
                    sample = fourier_extend_torch(
                        data=sample, new_shape=self.target_shape, device=self.device
                    )
            rescaled_samples.append(sample.unsqueeze(0))
        rescaled_samples = torch.stack(rescaled_samples, dim=0)
        return rescaled_samples

    def postprocess(self, x, orig_shape):
        """Postprocess the output of the network.

        In this case, we rescale the output to the original shape.
        """
        rescaled_samples = []
        for sample in x:
            sample = sample[0]  # only use first channel
            if self.rescale_patches:
                sample = rescale_tensor(sample, orig_shape, mode="trilinear")
            rescaled_samples.append(sample.unsqueeze(0))
        rescaled_samples = torch.stack(rescaled_samples, dim=0)
        return rescaled_samples

    def forward(self, x):
        """Forward pass through the network."""
        orig_shape = x.shape[2:]
        preprocessed_x = self.preprocess(x)
        predicted = super().forward(preprocessed_x)
        postprocessed_predicted = self.postprocess(predicted[0], orig_shape)
        # Return list to be compatible with deep supervision outputs
        return [postprocessed_predicted]
