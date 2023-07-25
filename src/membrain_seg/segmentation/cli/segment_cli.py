from typer import Option

from ..segment import segment as _segment
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


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
    store_probabilities: bool = Option(  # noqa: B008
        False, help="Should probability maps be output in addition to segmentations?"
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
    # Your segmenting logic here
    _segment(
        tomogram_path=tomogram_path,
        ckpt_path=ckpt_path,
        out_folder=out_folder,
        store_probabilities=store_probabilities,
        sw_roi_size=sliding_window_size,
    )
