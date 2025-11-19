# MemBrain-seg: `segment()` Function Reference

The `segment()` function is the main entry point for membrane segmentation using a trained MemBrain-seg model. It supports inference with sliding windows and test-time augmentation. Results are written to disk and can be directly used for downstream analysis (e.g. MemBrain-pick, MemBrain-stats, anything else you would like to analyze with it).

If you plan to use this function in your own workflow and need support or adjustments of the API for additional use cases, feel free to open an issue on [GitHub](https://github.com/teamtomo/membrain-seg/issues).

---

## Function Signature

```python
segment(
    tomogram_path,
    ckpt_path,
    out_folder,
    rescale_patches=False,
    in_pixel_size=None,
    out_pixel_size=10.0,
    store_probabilities=False,
    sw_roi_size=160,
    store_connected_components=False,
    connected_component_thres=None,
    test_time_augmentation=True,
    store_uncertainty_map=False,
    segmentation_threshold=0.0,
) -> str
```

---

## Parameters

**tomogram_path** (`str`)  
Path to the input tomogram. Supported formats are common cryo-et formats: `.mrc`, `.rec`, `.map`.

**ckpt_path** (`str`)  
Path to the `.ckpt` file of the trained MemBrain-seg model.

**out_folder** (`str`)  
Directory where the segmentation results will be saved.

**rescale_patches** (`bool`, default: `False`)  
If `True`, rescale input patches to match the network's expected pixel size.

**in_pixel_size** (`float` or `None`, default: `None`)  
Pixel size (in Ångström) of the input tomogram. Required if rescaling is enabled or the header is invalid.

**out_pixel_size** (`float`, default: `10.0`)  
Target pixel size used during model training. Used for patch rescaling.

**store_probabilities** (`bool`, default: `False`)  
If `True`, saves the raw membrane probability map alongside the binary segmentation.

**sw_roi_size** (`int`, default: `160`)  
Patch size used for sliding window inference. Must be divisible by 32.

**store_connected_components** (`bool`, default: `False`)  
If `True`, extracts and stores connected membrane components.

**connected_component_thres** (`int` or `None`, default: `None`)  
If set, removes connected components smaller than the threshold (in voxel count).

**test_time_augmentation** (`bool`, default: `True`)  
Apply 8-fold test-time augmentation (TTA) by mirroring input volumes.

**store_uncertainty_map** (`bool`, default: `False`)  
If `True`, computes and saves voxel-wise prediction variance from TTA.

**segmentation_threshold** (`float`, default: `0.0`)  
Threshold used to binarize the predicted membrane probability map.

---

## Returns

**segmentation_file** (`str`)  
Path to the final saved segmentation file (`*_seg.mrc`).

Additional outputs may be saved to `out_folder`, depending on selected options:
- `*_scores.mrc`: membrane probability map (if `store_probabilities=True`)
- `*_uncertainty_map.mrc`: voxel-wise variance map (if `store_uncertainty_map=True`)
- `*_components.mrc`: connected components mask (if `store_connected_components=True`)

---

## Example

```python
from membrain_seg.segmentation.segment import segment

segment(
    tomogram_path="data/tomo_01.mrc",
    ckpt_path="models/membrain_seg.ckpt",
    out_folder="results/",
    in_pixel_size=13.2,
    rescale_patches=True,
    store_probabilities=True,
    test_time_augmentation=True,
    store_uncertainty_map=True,
    segmentation_threshold=0.5,
)
```

---

## Batch Example

```python
import glob
from membrain_seg.segmentation.segment import segment

tomos = glob.glob("data/*.mrc")

for tomo in tomos:
    segment(
        tomogram_path=tomo,
        ckpt_path="models/model.ckpt",
        out_folder="results/"
    )
```

---

## Integration Notes

All outputs are written to disk and referenced by path, making downstream integration seamless.

For the latest updates, issues, or to contribute, visit:  
https://github.com/teamtomo/membrain-seg
