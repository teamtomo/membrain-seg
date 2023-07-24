import os

from typer import Option

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)

from ..connected_components import connected_components as _connected_components
from ..segment import segment as _segment
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="segment", no_args_is_help=True)
def segment(
    tomogram_path: str = Option(  # noqa: B008
        help="Path to the tomogram to be segmented", **PKWARGS
    ),
    ckpt_path: str = Option(  # noqa: B008
        help="Path to the pre-trained model checkpoint that should be used.", **PKWARGS
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions", help="Path to the folder where segmentations should be stored."
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
    _segment(
        tomogram_path=tomogram_path,
        ckpt_path=ckpt_path,
        out_folder=out_folder,
        store_probabilities=store_probabilities,
        store_connected_components=store_connected_components,
        connected_component_thres=connected_component_thres,
        sw_roi_size=sliding_window_size,
    )


@cli.command(name="components", no_args_is_help=True)
def components(
    segmentation_path: str = Option(  # noqa: B008
        help="Path to the membrane segmentation to be processed.", **PKWARGS
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
    data, header = load_tomogram(segmentation_path, return_header=True)
    conn_comps = _connected_components(
        binary_seg=data, size_thres=connected_component_thres
    )
    out_file = os.path.join(
        out_folder,
        os.path.splitext(os.path.basename(segmentation_path))[0] + "_components.mrc",
    )
    store_tomogram(filename=out_file, tomogram=conn_comps, header=header)
