import logging
import os

from typer import Option

from .cli import cli

logging.basicConfig(level=logging.INFO)


@cli.command(name="skeletonize", no_args_is_help=True)
def skeletonize(
    label_path: str = Option(  # noqa: B008
        ..., help="Specifies the path for skeletonization."
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions", help="Directory to save the resulting skeletons."
    ),
    batch_size: int = Option(  # noqa: B008
        None,
        help="Optional batch size for processing the tomogram. If not specified, "
        "the entire volume is processed at once. If operating with limited GPU "
        "resources, a batch size of 1,000,000 is recommended.",
    ),
    device: str = Option(  # noqa: B008
        None,
        help="Device to use for processing. If not specified, the default device "
        "will be used. Options are 'cpu' or 'cuda'.",
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
    from membrain_seg.segmentation.dataloading.data_utils import (
        load_tomogram,
        store_tomogram,
    )

    from ..skeletonize import skeletonization as _skeletonization

    print("Skeletonizing the segmentation")
    print("")
    print(
        "This can take several minutes. If you are bored, why not learn about what's \
            happening under the hood by reading the MemBrain v2 preprint?"
    )
    print(
        "MemBrain v2: an end-to-end tool for the analysis of membranes in \
            cryo-electron tomography"
    )
    print("https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1")
    print("")

    segmentation = load_tomogram(label_path)
    ske = _skeletonization(segmentation=segmentation.data, batch_size=batch_size, device=device)

    # Update the segmentation data with the skeletonized output while preserving the
    # original header and voxel_size
    segmentation.data = ske

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_file = os.path.join(
        out_folder,
        os.path.splitext(os.path.basename(label_path))[0] + "_skel.mrc",
    )

    store_tomogram(filename=out_file, tomogram=segmentation)
    logging.info("Skeleton saved to " + out_file)
