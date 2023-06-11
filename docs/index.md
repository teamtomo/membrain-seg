# Membrain-Seg
[Membrain-Seg](https://github.com/teamtomo/membrain-seg/) is a Python project developed by [teamtomo](https://github.com/teamtomo) for membrane segmentation in 3D in 3D for cryo-electron tomography (cryo-ET). This tool aims to provide researchers with an efficient and reliable method for segmenting membranes in 3D microscopic images. Membrain-Seg is currently under early development, so we may make breaking changes between releases.

# Overview
MemBrain-seg is a practical tool for membrane segmentation in cryo-electron tomograms. It's built on U-Net architecture and makes use of a pre-trained model for efficient performance.

If you wish, you can also train a new model using your own data, or combine it with our available public dataset.

To enhance segmentation, MemBrain-seg includes preprocessing functions. These help to adjust your tomograms so they're similar to the data our network was trained on, making the process smoother and more efficient.

Explore MemBrain-seg, use it for your needs, and let us know how it works for you!

# Features
## Segmentation
... explain segmentation ...

## Model training
... explain training ...

## Preprocessing
Currently, we provide the following two [preprocessing](https://github.com/teamtomo/membrain-seg/tree/add_docs/src/tomo_preprocessing) options:
- pixel size matching: Rescale your tomogram to match the training pixel sizes
- Fourier amplitude matching: Scale Fourier components to match the "style" of different tomograms

For more information, see the [Preprocessing](Usage/Preprocessing.md) subsection.

