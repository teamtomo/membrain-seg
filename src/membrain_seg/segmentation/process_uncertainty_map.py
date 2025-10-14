from scipy.ndimage import binary_erosion, distance_transform_edt, median_filter
from .dataloading.data_utils import load_tomogram, store_tomogram


def process_uncertainty_map(
    prediction_path: str,
    uncertainty_map_path: str,
):
    """
    Process a TTA uncertainty map by filling eroded foreground regions
    and applying median smoothing, then save the processed map as MRC.

    Parameters
    ----------
    tomogram_path : str
        Path to the original tomogram (.map or .mrc)
    prediction_path : str
        Path to the full prediction mask (.mrc)
    skeleton_path : str
        Path to the skeleton mask (.mrc)
    uncertainty_map_path : str
        Path to the raw uncertainty map (.mrc)
    """

    # --- Load data ---
    full_pred = load_tomogram(prediction_path).data
    uncertainty_map = load_tomogram(uncertainty_map_path).data

    fg_mask = full_pred > 0.5

    # --- Post-process uncertainty map ---
    eroded_mask = binary_erosion(fg_mask, iterations=1)
    _, nearest = distance_transform_edt(
        ~eroded_mask, return_distances=True, return_indices=True
    )
    nn_uncertainty_map = uncertainty_map[tuple(nearest)]

    combined_uncertainty_map = uncertainty_map.copy()
    combined_uncertainty_map[fg_mask] = nn_uncertainty_map[fg_mask]

    # Median filter (smooth XY plane only)
    smoothed_uncertainty_map = median_filter(combined_uncertainty_map, size=(1, 3, 3))

    # --- Save processed uncertainty map ---
    processed_uncertainty_map_path = uncertainty_map_path.replace(".mrc", "_processed.mrc")
    store_tomogram(processed_uncertainty_map_path, smoothed_uncertainty_map)

    print(f"✅ Saved processed uncertainty map: {processed_uncertainty_map_path}")
    print("🎉 Done processing uncertainty map.")
