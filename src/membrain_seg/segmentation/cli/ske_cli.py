import os

from typer import Option

from membrain_seg.segmentation.dataloading.data_utils import store_tomogram

from ..skeletonize import skeletonization as _skeletonization
from .cli import cli


@cli.command(name="skeletonize", no_args_is_help=True)
def skeletonize(
    label_path: str = Option(..., help="Specifies the path for skeletonization."),
    out_folder: str = Option(
        "./predictions", help="Directory to save the resulting skeletons."
    ),
    batch_size: int = Option(
        None,
        help="Optional batch size for processing the tomogram. If not specified, "
        "the entire volume is processed at once. If operating with limited GPU "
        "resources, a batch size of 1,000,000 is recommended.",
    ),
):
    """
    Perform skeletonization on labeled tomograms using nonmax-suppression technique.

    This function reads a labeled tomogram, applies skeletonization using a specified
    batch size, and stores the results in an MRC file in the specified output directory.
    If batch_size is set to None, the entire tomogram is processed at once, which might
    require significant memory. It is recommended to specify a batch size if memory
    constraints are a concern. The maximum possible batch size is the product of the
    tomogram's dimensions (Nx * Ny * Nz).


    Parameters
    ----------
    label_path : str
        File path to the tomogram to be skeletonized.
    out_folder : str
        Output folder path for the skeletonized tomogram.
    batch_size : int, optional
        The size of the batch to process the tomogram. Defaults to None, which processes
        the entire volume at once. For large volumes, consider setting it to a specific
        value like 1,000,000 for efficient processing without exceeding memory limits.


    Examples
    --------
    membrain skeletonize --label-path <path> --out-folder <output-directory>
    --batch-size <batch-size>
    """
    # Assuming _skeletonization function is already defined and can handle batch_size
    ske = _skeletonization(label_path=label_path, batch_size=batch_size)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_file = os.path.join(
        out_folder,
        os.path.splitext(os.path.basename(label_path))[0] + "_skel.mrc",
    )

    store_tomogram(filename=out_file, tomogram=ske)
    print("Skeleton saved to ", out_file)
