import logging
import os
from typing import List

from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli

logging.basicConfig(level=logging.INFO)


@cli.command(name="segment", no_args_is_help=True)
def segment(
    tomogram_path: str = Option(  # noqa: B008
        ..., help="Path to the tomogram to be segmented", **PKWARGS
    ),
    ckpt_path: str = Option(  # noqa: B008
        ...,
        help="Path to the pre-trained model checkpoint that should be used.",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions", help="Path to the folder where segmentations should be stored."
    ),
    rescale_patches: bool = Option(  # noqa: B008
        False, help="Should patches be rescaled on-the-fly during inference?"
    ),
    in_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram in Angstrom. \
            (default: 10 Angstrom)",
    ),
    out_pixel_size: float = Option(  # noqa: B008
        10.0,
        help="Pixel size of the output segmentation in Angstrom. \
            (default: 10 Angstrom; should normally stay at 10 Angstrom)",
    ),
    store_probabilities: bool = Option(  # noqa: B008
        False, help="Should probability maps be output in addition to segmentations?"
    ),
    store_connected_components: bool = Option(  # noqa: B008
        False, help="Should connected components of the segmentation be computed?"
    ),
    connected_component_thres: int = Option(  # noqa: B008
        None,
        help="Threshold for connected components. Components smaller than this will \
            be removed from the segmentation.",
    ),
    test_time_augmentation: bool = Option(  # noqa: B008
        True,
        help="Use 8-fold test time augmentation (TTA)? TTA improves segmentation \
        quality slightly, but also increases runtime.",
    ),
    store_uncertainty_map: bool = Option(
        False,
        help="If True, store an uncertainty map based on the voxel-wise variance "
            "across the TTA predictions. Requires --test-time-augmentation=True.",
    ),
    segmentation_threshold: float = Option(  # noqa: B008
        0.0,
        help="Threshold for the membrane segmentation. Only voxels with a \
            membrane score higher than this threshold will be segmented. \
                (default: 0.0)",
    ),
    sliding_window_size: int = Option(  # noqa: B008
        160,
        help="Sliding window size used for inference. Smaller values than 160 \
            consume less GPU, but also lead to worse segmentation results!",
    ),
):
    """Segment tomograms using a trained model.

    Example
    -------
    membrain segment --tomogram-path <path-to-your-tomo>
    --ckpt-path <path-to-your-model>
    """
    from membrain_seg.segmentation.segment import segment as _segment

    print("Segmenting tomogram", tomogram_path)
    print("")
    print(
        "This can take several minutes. If you are bored, why not learn about \
            what's happening under the hood by reading the MemBrain v2 preprint?"
    )
    print(
        "MemBrain v2: an end-to-end tool for the analysis of membranes in \
            cryo-electron tomography"
    )
    print("https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1")
    print("")
    _segment(
        tomogram_path=tomogram_path,
        ckpt_path=ckpt_path,
        out_folder=out_folder,
        rescale_patches=rescale_patches,
        in_pixel_size=in_pixel_size,
        out_pixel_size=out_pixel_size,
        store_probabilities=store_probabilities,
        store_connected_components=store_connected_components,
        connected_component_thres=connected_component_thres,
        sw_roi_size=sliding_window_size,
        test_time_augmentation=test_time_augmentation,
        store_uncertainty_map=store_uncertainty_map,
        segmentation_threshold=segmentation_threshold,
    )


@cli.command(name="components", no_args_is_help=True)
def components(
    segmentation_path: str = Option(  # noqa: B008
        ..., help="Path to the membrane segmentation to be processed.", **PKWARGS
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions", help="Path to the folder where segmentations should be stored."
    ),
    connected_component_thres: int = Option(  # noqa: B008
        None,
        help="Threshold for connected components. Components smaller than this will \
            be removed from the segmentation.",
    ),
):
    """Compute connected components of your segmented tomogram.

    This will annotate the connected components of your segmentation
    with integer values from 1 to [number of membranes].

    Example
    -------
    membrain components --tomogram-path <path-to-your-tomo>
    --connected-component-thres 5
    """
    from membrain_seg.segmentation.connected_components import (
        connected_components as _connected_components,
    )
    from membrain_seg.segmentation.dataloading.data_utils import (
        load_tomogram,
        store_tomogram,
    )

    segmentation = load_tomogram(segmentation_path)
    conn_comps = _connected_components(
        binary_seg=segmentation.data, size_thres=connected_component_thres
    )
    segmentation.data = conn_comps
    out_file = os.path.join(
        out_folder,
        os.path.splitext(os.path.basename(segmentation_path))[0] + "_components.mrc",
    )
    store_tomogram(filename=out_file, tomogram=segmentation)


@cli.command(name="thresholds", no_args_is_help=True)
def thresholds(
    scoremap_path: str = Option(  # noqa: B008
        ..., help="Path to the membrane scoremap to be processed.", **PKWARGS
    ),
    thresholds: List[float] = Option(  # noqa: B008
        ...,
        help="List of thresholds. Provide multiple by repeating the option.",
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions",
        help="Path to the folder where thresholdedsegmentations \
            should be stored.",
    ),
):
    """Process the provided scoremap using given thresholds.

    Given a membrane scoremap, this function thresholds the scoremap data
    using the provided threshold(s). The thresholded scoremaps are then stored
    in the specified output folder. If multiple thresholds are provided,
    separate thresholded scoremaps will be generated for each threshold.

    Example
    -------
    membrain thresholds --scoremap-path <path-to-scoremap>
        --thresholds -1.5 --thresholds -0.5 --thresholds 0.0 --thresholds 0.5

    This will generate thresholded scoremaps for the provided scoremap at
    thresholds -1.5, -0.5, 0.0 and 0.5.The results will be saved with filenames
    indicating the threshold values in the default 'predictions' folder or
    in the folder specified by the user.
    """
    from membrain_seg.segmentation.dataloading.data_utils import (
        load_tomogram,
        store_tomogram,
    )

    scoremap = load_tomogram(scoremap_path)
    score_data = scoremap.data
    if not isinstance(thresholds, list):
        thresholds = [thresholds]
    for threshold in thresholds:
        logging.info("Thresholding at" + str(threshold))
        thresholded_data = score_data > threshold
        segmentation = scoremap
        segmentation.data = thresholded_data
        out_file = os.path.join(
            out_folder,
            os.path.splitext(os.path.basename(scoremap_path))[0]
            + f"_threshold_{threshold}.mrc",
        )
        store_tomogram(filename=out_file, tomogram=segmentation)
        logging.info("Saved thresholded scoremap to " + out_file)
