from typer import Option

from ..NMS import skeletonization as _skeletonization
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="skeletonize", no_args_is_help=True)
def skeletonize(
    label_path: str = Option(
        ..., help="Path to the label to be skeletonized", **PKWARGS
    ),
):
    """
    Skeletonize tomogram labels using nonmax-suppression.

    Example
    -------
    membrain skeletonize --label-path <path-to-your-label>
    """
    _skeletonization(label_path=label_path)
