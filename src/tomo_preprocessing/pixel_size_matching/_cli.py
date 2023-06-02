from typer import Option

from ..cli import OPTION_PROMPT_KWARGS as PKWARGS
from ..cli import cli
from .match_pixel_size import match_pixel_size as _match_pixel_size
from .match_pixel_size_seg import match_segmentation_pixel_size_to_tomo


@cli.command(name="match_pixel_size", no_args_is_help=True)
def match_pixel_size(
    input_tomogram: str = Option(  # noqa: B008
        ..., help="Path to the input tomogram", **PKWARGS
    ),
    output_path: str = Option(  # noqa: B008
        ..., help="Path to store the output files", **PKWARGS
    ),
    pixel_size_out: float = Option(  # noqa: B008
        10.0, help="Target pixel size (default: 10.0)"
    ),
    pixel_size_in: float = Option(  # noqa: B008
        None,
        help="Input pixel size (optional). If not \
            specified, it will be read from the tomogram's header. ATTENTION: This can \
            lead to severe errors if the header pixel size is not correct.",
    ),
    disable_smooth: bool = Option(  # noqa: B008
        False,
        help="Disable smoothing (ellipsoid mask + \
            cosine decay). Disable if causing problems or for speed up",
    ),
):
    _match_pixel_size(
        input_tomogram, output_path, pixel_size_in, pixel_size_out, disable_smooth
    )


@cli.command(name="match_seg_to_tomo", no_args_is_help=True)
def match_seg_to_tomo(
    seg_path: str = Option(  # noqa: B008
        ..., help="Path to the segmentation", **PKWARGS
    ),
    orig_tomo_path: str = Option(  # noqa: B008
        ...,
        help="Path to the tomogram the segmentation should be matched to.",
        **PKWARGS,
    ),
    output_path: str = Option(  # noqa: B008
        ...,
        help="Path to the where the resized segmentation should be stored.",
        **PKWARGS,
    ),
):
    # Call your function here
    match_segmentation_pixel_size_to_tomo(seg_path, orig_tomo_path, output_path)
