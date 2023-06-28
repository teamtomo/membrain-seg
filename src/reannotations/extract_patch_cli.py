import numpy as np
from membrain_seg.dataloading.data_utils import get_csv_data
from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli
from .extract_patches import extract_patches as _extract_patches


@cli.command(name="extract_patches", no_args_is_help=True)
def extract_patches(
    tomogram_path: str = Option(  # noqa: B008
        help="Path to the tomogram to extract patches from.", **PKWARGS
    ),
    segmentation_path: str = Option(  # noqa: B008
        help="Path to the corresponding segmentation file.", **PKWARGS
    ),
    out_folder: str = Option(  # noqa: B008
        help="Path to the folder where extracted patches should be stored. \
            (subdirectories will be created)"
    ),
    coords_file: str = Option(  # noqa: B008
        None,
        help="Path to a file containing coordinates for patch extraction. The file \
        should \
        contain one set of coordinates per line, formatted as x,y,z. \
        If this option is not provided, x, y, z must be.",
    ),
    x: int = Option(  # noqa: B008
        None,
        help="X coordinate for patch extraction. Used only if coords_file is not \
        provided.",
    ),
    y: int = Option(  # noqa: B008
        None,
        help="Y coordinate for patch extraction. Used only if coords_file is not \
        provided.",
    ),
    z: int = Option(  # noqa: B008
        None,
        help="Z coordinate for patch extraction. Used only if coords_file is not \
        provided.",
    ),
    token: str = Option(  # noqa: B008
        None,
        help="Short token of the tomogram to name extracted patches. If not \
            provided, the \
            full tomogram filename\
            will be used.",
    ),
    idx_add: int = Option(  # noqa: B008
        0,
        help="Constant to be added to patch indices to distinguish between \
            different annotation rounds.",
    ),
):
    r"""
    Extract patches from a tomogram using provided coordinates and save them.\n.

    You can provide coordinates in two different ways: \n
    - a path to a .csv file containing a list of coordinates. Here, each row
    should represent one x,y,z-tuple of coordinates, split by comma:\n
    e.g.: \n
    135,445,98\n
    456, 324, 134\n
    ...\n
    - 1 tuple of 3D coordinates via the "x", "y" and "z" components\n

    Returns\n
    -------\n
    None\n

    Raises\n
    ------\n
    InvalidCoordinatesError\n
        If a patch cannot be extracted due to the provided coordinates being too close
        to the border of the tomogram.\n


    """
    if coords_file is not None:
        coords = get_csv_data(coords_file)
    else:
        coords = [np.array((x, y, z))]
    _extract_patches(
        tomogram_path, segmentation_path, coords, out_folder, idx_add, token
    )
