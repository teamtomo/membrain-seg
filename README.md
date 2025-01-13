# MemBrain-Seg

[![License](https://img.shields.io/pypi/l/membrain-seg.svg?color=green)](https://github.com/teamtomo/membrain-seg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/membrain-seg.svg?color=green)](https://pypi.org/project/membrain-seg)
[![Python Version](https://img.shields.io/pypi/pyversions/membrain-seg.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/membrain-seg/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/membrain-seg/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/membrain-seg/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/membrain-seg)


Membrain-Seg<sup>1</sup> is a Python project developed by [teamtomo](https://github.com/teamtomo) for membrane segmentation in 3D for cryo-electron tomography (cryo-ET). This tool aims to provide researchers with an efficient and reliable method for segmenting membranes in 3D microscopic images. Membrain-Seg is currently under early development, so we may make breaking changes between releases.

## Example notebook
For a quick start, you can walk through our example notebook. You can easily run it on Google Colab by clicking on the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CellArchLab/membrain_tutorial_scripts/blob/main/MemBrain_seg_tutorial.ipynb)


## Publication: 
Membrain-seg's current functionalities are described on more detail in our [preprint](https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1).


<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/34575029/248259282-ee622267-77fa-4c88-ad38-ad0cfd76b810.png">
</p>

Membrain-Seg is currently under early development, so we may make breaking changes between releases.

# Version Updates
For a detailed history of changes and updates, please refer to our [CHANGELOG.md](./CHANGELOG.md).


# Overview
MemBrain-seg is a practical tool for membrane segmentation in cryo-electron tomograms. It's built on the U-Net architecture and makes use of a pre-trained model for efficient performance.
The U-Net architecture and training parameters are largely inspired by nnUNet<sup>2</sup>.


Our current best model is available for download [here](https://drive.google.com/file/d/1tSQIz_UCsQZNfyHg0RxD-4meFgolszo8/view?usp=sharing). Please let us know how it works for you.
If the given model does not work properly, you may want to try one of our previous versions:

Other (older) model versions:
- [v9 -- best model until 10th Aug 2023](https://drive.google.com/file/d/15ZL5Ao7EnPwMHa8yq5CIkanuNyENrDeK/view?usp=sharing)
- [v9b -- model for non-denoised data until 10th Aug 2023](https://drive.google.com/file/d/1TGpQ1WyLHgXQIdZ8w4KFZo_Kkoj0vIt7/view?usp=sharing)

If you wish, you can also train a new model using your own data, or combine it with our (soon to come!) publicly-available dataset. 

To enhance segmentation, MemBrain-seg includes preprocessing functions. These help to adjust your tomograms so they're similar to the data our network was trained on, making the process smoother and more efficient.

Explore MemBrain-seg, use it for your needs, and let us know how it works for you!


Preliminary [documentation](https://teamtomo.github.io/membrain-seg/) is available, but far from perfect. Please let us know if you encounter any issues, and we are more than happy to help (and get feedback what does not work yet).

```
[1] Lamm, L., Zufferey, S., Righetto, R.D., Wietrzynski, W., Yamauchi, K.A., Burt, A., Liu, Y., Zhang, H., Martinez-Sanchez, A., Ziegler, S., Isensee, F., Schnabel, J.A., Engel, B.D., and Peng, T, 2024. MemBrain v2: an end-to-end tool for the analysis of membranes in cryo-electron tomography. bioRxiv, https://doi.org/10.1101/2024.01.05.574336

[2] Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H., 2021. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203-211. https://doi.org/10.1038/s41592-020-01008-z
```

# Installation
For detailed installation instructions, please look [here](https://teamtomo.github.io/membrain-seg/installation/).

# Features
## Segmentation
Segmenting the membranes in your tomograms is the main feature of this repository. 
Please find more detailed instructions [here](https://teamtomo.github.io/membrain-seg/Usage/Segmentation/).

## Preprocessing
Currently, we provide the following two [preprocessing](https://github.com/teamtomo/membrain-seg/tree/main/src/membrain_seg/tomo_preprocessing) options:
- Pixel size matching: Rescale your tomogram to match the training pixel sizes
- Fourier amplitude matching: Scale Fourier components to match the "style" of different tomograms
- Deconvolution: denoises the tomogram by applying the deconvolution filter from Warp

For more information, see the [Preprocessing](https://teamtomo.github.io/membrain-seg/Usage/Preprocessing/) subsection.

## Model training
It is also possible to use this package to train your own model. Instructions can be found [here](https://teamtomo.github.io/membrain-seg/Usage/Training/).

## Patch annotations
In case you would like to train a model that works better for your tomograms, it may be beneficial to add some more patches from your tomograms to the training dataset. 
Recommendations on how to to this can be found [here](https://teamtomo.github.io/membrain-seg/Usage/Annotations/).

