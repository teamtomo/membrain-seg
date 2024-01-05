from time import time
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.transforms import (
    Compose,
    MapTransform,
    OneOf,
    RandAdjustContrastd,
    RandAxisFlipd,
    Randomizable,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    Resize,
    Transform,
    Transposed,
    Zoomd,
)
from monai.transforms.spatial.dictionary import (
    DtypeLike,
    GridSampleMode,
    GridSamplePadMode,
)
from monai.transforms.utils import _create_rotate
from monai.utils import convert_to_tensor
from monai.utils.enums import TraceKeys
from scipy.ndimage import convolve, median_filter


class RandApplyTransform(Randomizable, Transform):
    """Randomly apply given MONAI transform with given probability.

    This class is a wrapper around a MONAI transform. It extends the
    Randomizable and Transform classes provided by MONAI. The purpose of
    this class is to randomly apply a specific MONAI transform to the data
    based on a given probability. For example, if the given probability
    is 0.5, then the wrapped transform will be applied to the data 50%
    of the time when called.
    """

    def __init__(
        self,
        transform: Transform,
        prob: float = 0.5,
        do_transform: Optional[bool] = None,
    ):
        if prob < 0.0 or prob > 1.0:
            raise ValueError("Probability must be a value between 0 and 1.")
        self.transform = transform
        self.prob = prob
        self.do_transform = do_transform

    def randomize(self, data: None) -> None:
        """Roll a dice and decide whether we actuall do the augmentation."""
        if self.do_transform is None:
            self._do_transform = self.R.random() < self.prob
        else:
            self._do_transform = self.do_transform

    def __call__(self, data):
        """Apply transform only with given probability."""
        self.randomize(None)
        if self._do_transform:
            return self.transform(data)
        return data


class MedianFilterd(Transform):
    """Median filter from batchgenerators package reimplemented.

    This class extends the Transform class provided by MONAI and implements a median
    filter operation for image data. A median filter is a digital filtering technique,
    used to remove noise from an image or signal. The class allows the radius of
    the median filter to be either a specific integer or a range from which an integer
    can be randomly selected each time the transform is called.
    """

    def __init__(self, keys, radius=1):
        self.keys = keys
        self.radius_range = radius

    def randomize(self, data: None) -> None:
        """Randomly draw median filter range."""
        if isinstance(self.radius_range, int):
            self.radius = self.radius_range
        elif isinstance(self.radius_range, tuple):
            self.radius = np.random.randint(self.radius_range[0], self.radius_range[1])

    def __call__(self, data):
        """Apply median filter."""
        self.randomize(None)
        for key in self.keys:
            for c in range(data[key].shape[0]):
                data[key][c] = torch.from_numpy(
                    median_filter(data[key][c], size=self.radius, mode="reflect")
                )  # TODO: Is this very inefficient?
        return data


class RandomBrightnessTransformd(Transform):
    """Brightness transform from batchgenerators reimplemented.

    This class extends the Transform class provided by MONAI and is used to
    apply a random brightness shift to the image data. The extent of the
    brightness shift follows a Gaussian (normal) distribution defined by the
    given mean (mu) and standard deviation (sigma). The transform is applied based
    on a given probability. For example, if the given probability is 0.7, the brightness
    shift will be applied to the image 70% of the time when the transform is called.
    """

    def __init__(self, keys, mu, sigma, prob=1.0):
        super().__init__()
        self.keys = keys
        self.mu = mu
        self.sigma = sigma
        self.prob = prob

    def __call__(self, data):
        """Apply brightness transform."""
        if np.random.rand() < self.prob:
            for key in self.keys:
                add_const = np.random.normal(loc=self.mu, scale=self.sigma)
                data[key] = data[key] + add_const
        return data


class RandomContrastTransformd(Transform):
    """Randomly scale the data with a contrast factor.

    This class extends the Transform class provided by MONAI. It's designed to randomly
    scale the contrast of an image. It allows to specify the keys to the images to be
    transformed, a range for contrast scaling, and whether the original range of the
    data should be preserved after the transformation.
    The probability of applying this transformation can also be set.
    """

    def __init__(self, keys, contrast_range, preserve_range=False, prob=1.0):
        super().__init__()
        self.keys = keys
        self.contrast_range = contrast_range
        self.prob = prob
        self.preserve_range = preserve_range

    def __call__(self, data):
        """Apply the transform."""
        if np.random.rand() < self.prob:
            for key in self.keys:
                # Choose a random contrast factor from the given range
                contrast_factor = np.random.uniform(
                    self.contrast_range[0], self.contrast_range[1]
                )

                # Apply the contrast transformation
                mean = np.mean(data[key])
                if self.preserve_range:
                    minval = data[key].min()
                    maxval = data[key].max()
                data[key] = mean + contrast_factor * (data[key] - mean)
                if self.preserve_range:
                    data[key][data[key] < minval] = minval
                    data[key][data[key] > maxval] = maxval

        return data


class RandAdjustContrastWithInversionAndStats(RandAdjustContrastd):
    """
    Adjusted version of RandAdjustContrastd from MONAI.

    After the normal function call, the original mean and std are preserved
    and the result is multiplied by -1. It is therefore recommended to
    apply the function 2 times.
    (Without two time application and negation, the transform is non-symmetric)
    """

    def __call__(self, data):
        """Apply the transform."""
        for key in self.keys:
            if self._do_transform:
                img = data[key]
                mean_before = img.mean()
                std_before = img.std()

                img = super().__call__({key: img})[key]

                mean_after = img.mean()
                std_after = img.std()

                img = (img - mean_after) / std_after * std_before + mean_before
                img *= -1
                data[key] = img

        return data


class SimulateLowResolutionTransform(Randomizable, Transform):
    """
    Batchgenerators' transform for simulating low resolution upscaling.

    The image is downsampled using nearest neighbor interpolation using a
    random scale factor. Afterwards, it is upsampled again with trilinear
    interpolation, imitating the upscaling of low-resolution images.
    """

    def __init__(
        self,
        keys,
        downscale_factor_range,
        upscale_mode="trilinear",
        downscale_mode="nearest",
    ):
        self.keys = keys
        self.downscale_factor_range = downscale_factor_range
        self.downscale_factor = None
        self.upscale_mode = upscale_mode
        self.downscale_mode = downscale_mode

    def randomize(self, data):
        """Randomly choose the downsample factor."""
        self.downscale_factor = self.R.uniform(*self.downscale_factor_range)

    def __call__(self, data):
        """Apply the transform."""
        self.randomize(data)
        downscale_transform = Zoomd(
            self.keys, zoom=self.downscale_factor, mode=self.downscale_mode
        )
        upscale_transform = Zoomd(
            self.keys, zoom=1.0 / self.downscale_factor, mode=self.upscale_mode
        )

        data = downscale_transform(data)
        data = upscale_transform(data)
        return data


class BlankCuboidTransform(Randomizable, MapTransform):
    """
    Randomly blank out cuboids from the image.

    It's designed to randomly blank out one or more cuboid regions in an image.
    The number and size of the cuboids are random, with the sizes drawn from a specified
    range. The class also provides options to replace the blanked regions with
    either the mean value of the image or zeros.
    """

    def __init__(
        self,
        keys,
        prob=0.5,
        cuboid_area=(10, 100),
        is_3d=False,
        max_cuboids=1,
        replace_with="mean",
    ):
        super().__init__(keys)
        self.prob = prob
        self.cuboid_area = cuboid_area
        self.is_3d = is_3d
        self.max_cuboids = max_cuboids
        self.replace_with = replace_with

    def randomize(self, data=None):
        """Randomly choose the number of cuboids & their sizes."""
        self.do_transform = self.R.random() < self.prob
        if self.do_transform:
            self.num_cuboids = self.R.randint(1, self.max_cuboids + 1)
            self.cuboids = []
            for _ in range(self.num_cuboids):
                height = self.R.randint(self.cuboid_area[0], self.cuboid_area[1] + 1)
                width = self.R.randint(self.cuboid_area[0], self.cuboid_area[1] + 1)
                depth = (
                    self.R.randint(self.cuboid_area[0], self.cuboid_area[1] + 1)
                    if self.is_3d
                    else 1
                )
                self.cuboids.append((height, width, depth))

    def __call__(self, data):
        """Apply the transform."""
        self.randomize()
        if not self.do_transform:
            return data

        d = dict(data)
        for key in self.keys:
            image = d[key]
            for height, width, depth in self.cuboids:
                if self.is_3d:
                    z_max, y_max, x_max = (
                        image.shape[-3],
                        image.shape[-2],
                        image.shape[-1],
                    )
                    z = self.R.randint(0, z_max - depth)
                else:
                    y_max, x_max = image.shape[-2], image.shape[-1]
                    z, depth = 0, 1
                y = self.R.randint(0, y_max - height)
                x = self.R.randint(0, x_max - width)
                if self.replace_with == "mean":
                    image[
                        ..., z : z + depth, y : y + height, x : x + width
                    ] = torch.mean(torch.Tensor(image))
                elif self.replace_with == 0.0:
                    image[..., z : z + depth, y : y + height, x : x + width] = 0.0
            d[key] = image

        return d


def sample_scalar(value: Union[Tuple, List, Callable, Any], *args: Any) -> Any:
    """
    Implementation from the batchgenerators package.

    Function to sample a scalar from a specified range, or compute it using
    a provided function.

    Args:
        value: The value to be sampled. It can be a tuple/list defining a range,
               a callable function, or any other value. If it's a tuple/list,
               a random value within the range is returned. If it's a function, it is
               called with the arguments supplied in *args. If it's any other value,
               it is returned as is.
        *args: Additional arguments to be passed to the callable 'value', if it is a
                function.

    Returns
    -------
        A sampled or computed scalar value.
    """
    if isinstance(value, (tuple, list)):
        return np.random.uniform(value[0], value[1])
    elif callable(value):
        # return value(image, kernel)
        return value(*args)
    else:
        return value


class BrightnessGradientAdditiveTransform(Randomizable, MapTransform):
    """
    Add a brightness gradient to the image (also from batchgenerators).

    A scaled Gaussian kernel is added to the image. The center of the
    Gaussian kernel is randomly drawn an can also be outside of the image.
    """

    def __init__(
        self,
        keys,
        scale,
        loc=(-1, 2),
        max_strength=1.0,
        mean_centered=True,
    ):
        super().__init__(keys)
        self.scale = scale
        self.loc = loc
        self.max_strength = max_strength
        self.mean_centered = mean_centered

    def _generate_kernel(self, img_shape):
        img_shape = img_shape[2:]
        scale = [sample_scalar(self.scale, img_shape, i) for i in range(len(img_shape))]
        loc = [sample_scalar(self.loc, img_shape, i) for i in range(len(img_shape))]
        coords = [
            np.linspace(-loc[i], img_shape[i] - loc[i], img_shape[i])
            for i in range(len(img_shape))
        ]
        meshgrid = np.meshgrid(*coords, indexing="ij")
        kernel = np.exp(
            -0.5 * sum([(meshgrid[i] / scale[i]) ** 2 for i in range(len(img_shape))])
        )
        kernel = np.expand_dims(kernel, 0)
        return kernel

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            image = d[key]
            img_shape = image.shape
            kernel = self._generate_kernel(img_shape)
            if self.mean_centered:
                kernel -= kernel.mean()
            max_kernel_val = max(np.max(np.abs(kernel)), 1e-8)
            strength = sample_scalar(self.max_strength, image, kernel)
            kernel = kernel / max_kernel_val * strength
            image += kernel

            d[key] = image

        return d


class LocalGammaTransform(Randomizable, MapTransform):
    """Locally adjusts the Gamma value using Gaussian kernel.(from batchgenerators).

    This transform adjusts the Gamma value of the images locally. The Gamma
    correction is a type of power-law transformation that manipulates the
    brightness of an image. It's applied locally based on a generated Gaussian
    kernel. This class is an implementation from the batchgenerators package.

    """

    def __init__(self, keys, scale, loc=(-1, 2), gamma=(0.5, 1)):
        super().__init__(keys)
        self.scale = scale
        self.loc = loc
        self.gamma = gamma

    def _generate_kernel(self, img_shape):
        img_shape = img_shape[2:]
        scale = [sample_scalar(self.scale, img_shape, i) for i in range(len(img_shape))]
        loc = [sample_scalar(self.loc, img_shape, i) for i in range(len(img_shape))]
        coords = [
            np.linspace(-loc[i], img_shape[i] - loc[i], img_shape[i])
            for i in range(len(img_shape))
        ]
        meshgrid = np.meshgrid(*coords, indexing="ij")
        kernel = np.exp(
            -0.5 * sum([(meshgrid[i] / scale[i]) ** 2 for i in range(len(img_shape))])
        )
        kernel = np.expand_dims(kernel, 0)
        return kernel

    def _apply_gamma_gradient(self, img, kernel):
        mn, mx = img.min(), img.max()
        img = (img - mn) / (max(mx - mn, 1e-8))

        gamma = sample_scalar(self.gamma)
        img_modified = np.power(img, gamma)

        return self.run_interpolation(img, img_modified, kernel) * (mx - mn) + mn

    def run_interpolation(self, img, img_modified, kernel):
        """Interpolate between image and modified image, weighted with kernel."""
        return img + kernel * (img_modified - img)

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            image = d[key]
            img_shape = image.shape
            kernel = self._generate_kernel(img_shape)
            d[key] = self._apply_gamma_gradient(image, kernel)
        return d


class SharpeningTransformMONAI(MapTransform):
    """Laplacian Sharpening transform. (from batchgenerators).

    This transform applies a Laplacian sharpening filter to the image, enhancing
    details and emphasizing edges. The strength of sharpening can be controlled
    and can be set to be the same or different for each image channel. This class
    is an implementation from the batchgenerators package.

    """

    filter_2d = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filter_3d = np.array(
        [
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        ]
    )

    def __init__(
        self,
        strength: Union[float, Tuple[float, float]] = 0.2,
        same_for_each_channel: bool = False,
        p_per_channel: float = 0.5,
        keys: str = "image",
    ):
        super().__init__(keys)
        self.strength = strength
        self.same_for_each_channel = same_for_each_channel
        self.p_per_channel = p_per_channel

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
            if self.same_for_each_channel:
                strength_here = (
                    self.strength
                    if isinstance(self.strength, float)
                    else np.random.uniform(*self.strength)
                )
                filter_here = (
                    self.filter_2d if img.ndim == 3 else self.filter_3d
                ) * strength_here
                filter_here[tuple([1] * (img.ndim - 1))] += 1
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        img[c] = torch.from_numpy(
                            convolve(img[c], filter_here, mode="reflect")
                        )
                        img[c] = np.clip(img[c], img[c].min(), img[c].max())
            else:
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        strength_here = (
                            self.strength
                            if isinstance(self.strength, float)
                            else np.random.uniform(*self.strength)
                        )
                        filter_here = (
                            self.filter_2d if img.ndim == 3 else self.filter_3d
                        ) * strength_here
                        filter_here[tuple([1] * (img.ndim - 1))] += 1
                        img[c] = torch.from_numpy(
                            convolve(img[c], filter_here, mode="reflect")
                        )
                        img[c] = torch.from_numpy(
                            np.clip(img[c], img[c].min(), img[c].max())
                        )
            d[key] = img.squeeze() if len(img.shape) == 5 and img.shape[0] == 1 else img
        return d


class DownsampleSegForDeepSupervisionTransform(MapTransform):
    """Downsample labels for deep supervision (from nnUNet).

    This transform downsamples the given labels for the purpose of deep supervision
    during model training. Deep supervision involves making predictions at
    multiple scales or layers, which can help improve performance. This class
    is an implementation from the nnUNet package.
    """

    def __init__(
        self,
        keys: Union[str, Tuple[str]],
        ds_scales: Tuple[float, float, float] = (1, 0.5, 0.25),
        order: tuple = ("nearest"),
        axes: Union[None, Tuple[int, int, int]] = None,
    ):
        super().__init__(keys)
        self.axes = axes
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, data_dict):
        """Apply the transform."""
        for key, cur_order in zip(self.keys, self.order):
            data_dict[key] = downsample_seg_for_ds_transform(
                data_dict[key], self.ds_scales, cur_order, self.axes
            )
        return data_dict


def downsample_seg_for_ds_transform(
    seg: np.ndarray,
    ds_scales: Tuple[float, float, float] = (1, 0.5, 0.25),
    order: str = "nearest",
    axes: Union[None, Tuple[int, int, int]] = None,
):
    """
    Function for downsampling segmentations based on provided downscaling factors.

    For each downscaling factor in 'ds_scales', the input segmentation ('seg') is
    resized along the specified 'axes'. If 'axes' is None, then all axes excluding
    the first one (typically the channel axis) are resized. The interpolation mode
    for resizing is specified by 'order', which defaults to 'nearest'.

    Parameters
    ----------
    seg : np.ndarray
        Input segmentation to be downsampled.
    ds_scales : Tuple[float, float, float], optional
        Downscaling factors for each axis, by default (1, 0.5, 0.25).
    order : str, optional
        Interpolation mode to be used while resizing, by default 'nearest'.
    axes : Union[None, Tuple[int, int, int]], optional
        Axes along which the input is resized. If None, all axes excluding the
        first one are resized, by default None.

    Returns
    -------
    list
        A list of downsampled segmentations corresponding to each scale in 'ds_scales'.
    """
    if axes is None:
        axes = list(range(1, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes[:3]):  # only iterate to axis 3 (spatial dims)
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            resize_transform = Resize(
                new_shape[1:4],
                mode=order,
                align_corners=(None if order == "nearest" else False),
            )
            out_seg = np.zeros(new_shape, dtype=np.array(seg).dtype)
            if len(seg.shape) == 4:
                out_seg = resize_transform(seg)
            elif len(seg.shape) == 5:
                for c in range(seg.shape[-1]):
                    out_seg[:, :, :, :, c] = resize_transform(seg[:, :, :, :, c])
            output.append(out_seg)
    return output


AxesSwap = OneOf(
    [
        Transposed(keys=("image", "label"), indices=(0, 2, 1, 3)),
        Transposed(keys=("image", "label"), indices=(0, 1, 3, 2)),
        Transposed(keys=("image", "label"), indices=(0, 3, 2, 1)),
    ]
)
AxesShuffle = Compose(
    [
        RandApplyTransform(transform=AxesSwap, prob=0.5),
        RandApplyTransform(transform=AxesSwap, prob=0.5),
        RandApplyTransform(transform=AxesSwap, prob=0.5),
    ]
)


def adjust_normals_for_axes_shuffle(
    normals: np.ndarray, shuffle_indices: List[int]
) -> np.ndarray:
    """
    Adjust normal vectors to be correctly directed after axes shuffling.

    Parameters
    ----------
    normals : ndarray
        A 5-dimensional array of shape (batch_size, depth, height, width, 3)
        representing the normal vectors.
    shuffle_indices : List[int]
        A list of indices representing the order of shuffled axes.

    Returns
    -------
    ndarray
        A 5-dimensional array containing the adjusted normals.

    Notes
    -----
    This function assumes that the input normal vectors are in a 5-dimensional
    array with the last dimension representing the 3 components of each normal vector.
    It rearranges the normals based on the new order of axes provided by
    shuffle_indices.
    The shuffle_indices should be a permutation of [1, 2, 3] representing the new
    order of the x, y, and z axes after shuffling.
    """
    # Create a mapping of the new axes order
    new_axes_order = {shuffle_indices[i]: i for i in range(len(shuffle_indices))}
    # Rearrange the normal vectors using the new axes order
    adjusted_normals = normals[
        :,
        :,
        :,
        :,
        [new_axes_order[1] - 1, new_axes_order[2] - 1, new_axes_order[3] - 1],
    ]
    return adjusted_normals


class NormalsAxesShuffle(Randomizable, Transform):
    """
    A transform for shuffling axes, also adjusting normals accordingly if provided.

    Parameters
    ----------
    keys : List[str]
        Keys to the data items to be transformed.
    vector_key : Optional[str]
        Key to the vector data items to be adjusted.

    Attributes
    ----------
    indices : tuple
        The order of the indices after shuffling.

    Methods
    -------
    randomize(data=None)
        Randomly determine the axes order for shuffling.
    __call__(data)
        Apply the axis shuffling and adjust normals if vector_key is provided.
    """

    def __init__(self, keys, vector_key=None):
        self.keys = keys
        self.vector_key = vector_key

    def randomize(self, data=None):
        """Randomly choose the number of cuboids & their sizes."""
        if self.R.random() < 1.0 / 3:
            self.indices = (0, 2, 1, 3, 4)
        elif self.R.random() < 2.0 / 3:
            self.indices = (0, 1, 3, 2, 4)
        else:
            self.indices = (0, 3, 2, 1, 4)

    def __call__(self, data):
        """Apply transform."""
        self.randomize(None)
        shuffled_data = Transposed(keys=self.keys, indices=self.indices[:4])(data)
        if self.vector_key is not None:
            shuffled_vectors_data = Transposed(
                keys=[self.vector_key], indices=self.indices
            )(data)
            shuffled_data["vectors"] = adjust_normals_for_axes_shuffle(
                shuffled_vectors_data["vectors"], self.indices
            )
        return shuffled_data


def create_axes_shuffle_with_vectors(vector_key: Optional[str] = None) -> Compose:
    """
    Axes shuffle with for image and vectors.

    Create a composite transform for 3-fold axes shuffling with optional vector
    adjustment.

    Parameters
    ----------
    vector_key : Optional[str], default=None
        Key to the vector data items to be adjusted.

    Returns
    -------
    Compose
        A composite transform comprising three random applications of
        AdjustedAxesShuffle.

    Notes
    -----
    This function returns a Compose object of MONAI transforms that applies
    three instances of AdjustedAxesShuffle, each with a probability of 0.5,
    to shuffle axes for 'image' and 'label' keys and adjust vectors if a
    vector_key is provided.
    """
    return Compose(
        [
            RandApplyTransform(
                transform=NormalsAxesShuffle(
                    keys=("image", "label"), vector_key=vector_key
                ),
                prob=0.5,
            ),
            RandApplyTransform(
                transform=NormalsAxesShuffle(
                    keys=("image", "label"), vector_key=vector_key
                ),
                prob=0.5,
            ),
            RandApplyTransform(
                transform=NormalsAxesShuffle(
                    keys=("image", "label"), vector_key=vector_key
                ),
                prob=0.5,
            ),
        ]
    )


class RandRotate90dWithVectors(RandRotate90d):
    """
    Rotate image and vectors by 90 degrees.

    Extend RandRotate90d to include the rotation of vector maps,
    specifically normal vectors.

    Parameters
    ----------
    keys : KeysCollection
        Keys to the data items to be transformed.
    vector_key : Hashable
        Key to the vector data items to be adjusted.
    prob : float, optional
        Probability of applying the transformation. Default is 0.1.
    max_k : int, optional
        Maximum number of rotations to apply. Default is 3.
    spatial_axes : Tuple[int, int], optional
        Axes along which to rotate the image. Default is (0, 1).
    allow_missing_keys : bool, optional
        Do not raise exception if key is missing. Default is False.

    Methods
    -------
    rotate_vectors(vector_map: torch.Tensor, k: int) -> torch.Tensor
        Rotate the vector map after rotating the intensity map.
    __call__(data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]
        Apply the transform to the input data.
    """

    def __init__(
        self,
        keys: KeysCollection,
        vector_key: Hashable,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: Tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, prob, max_k, spatial_axes, allow_missing_keys)
        self.vector_key = vector_key

    def rotate_vectors(self, vector_map: torch.Tensor, k: int) -> torch.Tensor:
        """After rotating the intensity maps, also rotate each normal."""
        # Rotate the vector_map by k * 90 degrees
        vector_map_rotated = torch.rot90(
            vector_map, k, (self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)
        )
        # Rotate each vector in its new position to its correct orientation
        for _ in range(k):
            dummy = vector_map_rotated.clone()
            vector_map_rotated[..., self.spatial_axes[0]] = -vector_map_rotated[
                ..., self.spatial_axes[1]
            ]
            vector_map_rotated[..., self.spatial_axes[1]] = dummy[
                ..., self.spatial_axes[0]
            ]

        return vector_map_rotated

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool = False
    ) -> Mapping[Hashable, torch.Tensor]:
        """Apply the transform."""
        # Apply the original RandRotate90d transformation (not on vectors map)
        d = super().__call__(data, lazy=lazy)
        lazy_ = self.lazy if lazy is None else lazy
        if self.vector_key is not None:
            if self._do_transform:
                # Rotate the vectors within the vector mask
                d[self.vector_key] = self.rotate_vectors(
                    d[self.vector_key], self._rand_k
                )
                self.push_transform(d[self.vector_key], replace=True, lazy=lazy_)

        return d


class RandRotatedWithVectors(RandRotated):
    """
    Rotate image and vectors.

    Extend RandRotated to include rotation of vector maps with arbitrary angles,
    adjusting normal vectors.

    Parameters
    ----------
    keys : KeysCollection
        Keys to the data items to be transformed.
    vector_key : Hashable
        Key to the vector data items to be adjusted.
    range_x, range_y, range_z : Union[Tuple[float, float], float], optional
        Range of rotation angles for the x, y, and z axes respectively.
        Default is 0.0.
    prob : float, optional
        Probability of applying the transformation. Default is 0.1.
    keep_size : bool, optional
        Keep the original size of the image. Default is True.
    mode : GridSampleMode, optional
        Interpolation mode to calculate output values. Default is
        GridSampleMode.BILINEAR.
    padding_mode : GridSamplePadMode, optional
        Padding mode for outside grid values. Default is GridSamplePadMode.BORDER.
    align_corners : Union[Sequence[bool], bool], optional
        Align the corners of the input and output. Default is False.
    dtype : Union[Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype],
        optional
        Data type of the output data. Default is np.float32.
    allow_missing_keys : bool, optional
        Do not raise exception if key is missing. Default is False.

    Methods
    -------
    get_rotation_matrix() -> np.ndarray
        Compute and return the rotation matrix for the transform.
    rotate_vectors(vector_map: torch.Tensor, rot_matrix: np.ndarray) -> torch.Tensor
        Rotate the vectors after their maps are rotated.
    __call__(data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]
        Apply the transform to the input data.
    """

    def __init__(
        self,
        keys: KeysCollection,
        vector_key: Hashable,
        range_x: Union[Tuple[float, float], float] = 0.0,
        range_y: Union[Tuple[float, float], float] = 0.0,
        range_z: Union[Tuple[float, float], float] = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: GridSampleMode = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadMode = GridSamplePadMode.ZEROS,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Union[
            Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype
        ] = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys,
            range_x,
            range_y,
            range_z,
            prob,
            keep_size,
            mode,
            padding_mode,
            align_corners,
            dtype,
            allow_missing_keys,
        )
        self.vector_key = vector_key

    def get_rotation_matrix(self) -> np.ndarray:
        """Compute the rotation matrix used for the transform."""
        radians = (self.rand_rotate.x, self.rand_rotate.y, self.rand_rotate.z)
        rotation_matrix = _create_rotate(spatial_dims=3, radians=radians)
        return rotation_matrix[:3, :3].T

    def rotate_vectors(
        self, vector_map: torch.Tensor, rot_matrix: np.ndarray
    ) -> torch.Tensor:
        """Rotate the vectors after their maps are rotated."""
        vector_map_rotated = vector_map.clone()
        vector_map_rotated_np = vector_map_rotated.numpy()
        vector_map_rotated_np = np.einsum(
            "ijklm,nm->ijkln", vector_map_rotated_np, rot_matrix
        )
        to_torch = torch.from_numpy(vector_map_rotated_np)
        return to_torch

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool = False
    ) -> Dict[Hashable, torch.Tensor]:
        """Apply the transform."""
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        self.randomize(None)
        # all the keys share the same random rotate angle
        self.rand_rotate.randomize()
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                if key == "vectors":
                    # Split the channels
                    channels = torch.split(d[key], 1, -1)
                    # Rotate the channels separately
                    rotated_channels = []
                    for channel in channels:
                        squeezed_channel = torch.squeeze(channel, -1)
                        rotated_squeezed_channel = self.rand_rotate(
                            squeezed_channel,
                            mode=mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                            dtype=dtype,
                            randomize=False,
                        )
                        rotated_channel = torch.unsqueeze(rotated_squeezed_channel, -1)
                        rotated_channels.append(rotated_channel)

                    # Merge the channels
                    d[key] = torch.cat(rotated_channels, dim=-1)
                else:
                    d[key] = self.rand_rotate(
                        d[key],
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                        dtype=dtype,
                        randomize=False,
                    )
            else:
                d[key] = convert_to_tensor(
                    d[key], track_meta=get_track_meta(), dtype=torch.float32
                )

            if get_track_meta():
                rot_info = (
                    self.pop_transform(d[key], check=False)
                    if self._do_transform
                    else {}
                )
                self.push_transform(d[key], extra_info=rot_info, lazy=lazy_)

        if self._do_transform:
            # Get the rotation matrix for the applied rotation
            rot_matrix = self.get_rotation_matrix()
            # Rotate the vectors within the vector mask
            if self.vector_key is not None:
                d[self.vector_key] = self.rotate_vectors(d[self.vector_key], rot_matrix)

        return d


class RandZoomdWithChannels(RandZoomd):
    """
    Extends RandZoomd for better handling of multi-channel data.

    This class specifically adjusts the random zooming to handle cases where
    the input data might have multiple channels (e.g., color images / normals).

    Methods
    -------
    __call__(data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]
        Apply the random zoom transform to each channel of the input data.
    """

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool = False
    ) -> Dict[Hashable, torch.Tensor]:
        """Apply the transform."""
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: Dict[Hashable, torch.Tensor] = convert_to_tensor(
                d, track_meta=get_track_meta()
            )
            return out

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])

        for key, mode, padding_mode, align_corners in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners
        ):
            if self._do_transform:
                # Check if the input has 3 channels
                if d[key].shape[-1] == 3:
                    transformed_channels = []
                    # Process each channel separately
                    for c in range(3):
                        channel_data = d[key][..., c]
                        transformed_channel = self.rand_zoom(
                            channel_data,
                            mode=mode,
                            padding_mode=padding_mode,
                            align_corners=align_corners,
                            randomize=False,
                        )
                        transformed_channels.append(transformed_channel)
                    # Combine the transformed channels
                    d[key] = torch.stack(transformed_channels, dim=-1)
                else:
                    d[key] = self.rand_zoom(
                        d[key],
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                        randomize=False,
                    )
            else:
                d[key] = convert_to_tensor(
                    d[key], track_meta=get_track_meta(), dtype=torch.float32
                )
            if get_track_meta():
                xform = (
                    self.pop_transform(d[key], check=False)
                    if self._do_transform
                    else {}
                )
                self.push_transform(d[key], extra_info=xform, lazy=lazy_)
        return d


class RandAxisFlipdWithNormals(RandAxisFlipd):
    """
    Extends RandAxisFlipd to handle normal vectors during random axis flipping.

    This class adjusts the axis flipping transform to correctly flip normal vectors
    when flipping other associated data.

    Parameters
    ----------
    keys : KeysCollection
        Keys to the data items to be transformed.
    normal_keys : KeysCollection
        Keys to the normal vector data items to be adjusted.
    prob : float, optional
        Probability of applying the transformation. Default is 0.1.
    allow_missing_keys : bool, optional
        Do not raise exception if key is missing. Default is False.

    Methods
    -------
    __call__(data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]
        Apply the transform to the input data and adjust normal vectors accordingly.
    inverse(data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]
        Get the inverse transform, adjusting the normals back to their original
        orientation.
    """

    def __init__(
        self,
        keys: KeysCollection,
        normal_keys: KeysCollection,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, prob, allow_missing_keys)
        self.normal_keys = normal_keys

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool = False
    ) -> Dict[Hashable, torch.Tensor]:
        """Apply the transform."""
        d = super().__call__(data, lazy=lazy)
        lazy_ = self.lazy if lazy is None else lazy
        if self.normal_keys is not None:
            for key in self.normal_keys:
                if key in d and self._do_transform:
                    # Flip the corresponding axis of the normal vector
                    d[key][:, :, :, :, self.flipper._axis] *= -1
                    self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        """Get the inverse transform."""
        d = super().inverse(data)
        for key in self.normal_keys:
            if key in d:
                # Get the flip axis from the first transformed key
                first_key: Hashable = self.first_key(data)
                xform = self.pop_transform(d[first_key])

                if xform[TraceKeys.DO_TRANSFORM]:
                    d[key][xform[TraceKeys.EXTRA_INFO]["axis"]] *= -1

        return d


class TimingTransform:
    """Transform for stopping the time of each module."""

    def __init__(self, transform, name=None):
        self.transform = transform
        self.name = name if name else str(transform)
        self.history = []

    def __call__(self, data):
        """Apply the transform."""
        start_time = time()
        transformed_data = self.transform(data)
        end_time = time()
        elapsed_time = end_time - start_time

        self.history.append(elapsed_time)
        print(f"{self.name} took on average {np.mean(self.history):.3f} seconds.")
        return transformed_data
