from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli
from .segment import segment as _segment


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
):
    """Segment tomograms using a trained model.

    Parameters
    ----------
    tomogram_path : str
        Path to the tomogram that is to be segmented.
    ckpt_path : str
        Path to the pre-trained model checkpoint to use for segmentation.
    out_folder : str, optional
        Path to the folder where segmentation results should be stored.
        By default, results are stored in './predictions'.
    store_probabilities : bool, optional
        If set to True, probability maps will be output in addition to segmentations.
        By default, this is set to False.
    """
    # Your segmenting logic here
    _segment(
        tomogram_path=tomogram_path,
        ckpt_path=ckpt_path,
        out_folder=out_folder,
        store_probabilities=store_probabilities,
    )
