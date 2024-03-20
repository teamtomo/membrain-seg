from typing import Tuple

from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    fourier_cropping,
    fourier_extend,
)

from scipy import ndimage
import torch

class PreprocessedSemanticSegmentationUnet(SemanticSegmentationUnet):
    def __init__(
        self,
        *args,
        rescale_patches: bool = False, # Should patches be rescaled?
        target_shape: Tuple[int, int, int] = (160, 160, 160),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the preprocessing parameters
        self.rescale_patches = rescale_patches
        self.target_shape = target_shape
    
    def preprocess(self, x):
        rescaled_samples = []
        for sample in x:
            if self.rescale_patches:
                sample = sample.numpy()[0]
                if sample.shape[0] > self.target_shape[0]:
                    sample = fourier_cropping(sample, self.target_shape, smoothing=False)
                elif sample.shape[0] < self.target_shape[0]:
                    print("Entering")
                    sample = fourier_extend(sample, self.target_shape, smoothing=False)
                    print("Exiting")
                sample = torch.from_numpy(sample)
            rescaled_samples.append(sample.unsqueeze(0))
        rescaled_samples = torch.stack(rescaled_samples, dim=0)
        return rescaled_samples

    def postprocess(self, x, orig_shape):
        cur_shape = x.shape[2:]
        rescale_factors = [
            target_dim / original_dim
            for target_dim, original_dim in zip(orig_shape, cur_shape)
        ]
        rescaled_samples = []
        for sample in x:
            if self.rescale_patches:
                sample = sample.numpy()[0]
                sample = ndimage.zoom(sample, rescale_factors, order=0, prefilter=False)
                sample = torch.from_numpy(sample)
                rescaled_samples.append(sample.unsqueeze(0))
        rescaled_samples = torch.stack(rescaled_samples, dim=0)
        return rescaled_samples

    def forward(self, x):
        orig_shape = x.shape[2:]
        preprocessed_x = self.preprocess(x)
        predicted = super().forward(preprocessed_x)
        postprocessed_predicted = self.postprocess(predicted[0], orig_shape)
        # Return list to be compatible with deep supervision outputs
        return [postprocessed_predicted]