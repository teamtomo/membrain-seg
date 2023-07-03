# Membrain-Seg
[Membrain-Seg](https://github.com/teamtomo/membrain-seg/) is a Python project developed by [teamtomo](https://github.com/teamtomo) for membrane segmentation in 3D for cryo-electron tomography (cryo-ET). This tool aims to provide researchers with an efficient and reliable method for segmenting membranes in 3D microscopic images. Membrain-Seg is currently under early development, so we may make breaking changes between releases.

<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/34575029/248259282-ee622267-77fa-4c88-ad38-ad0cfd76b810.png">
</p>

# Overview
MemBrain-seg is a practical tool for membrane segmentation in cryo-electron tomograms. It's built on the U-Net architecture and makes use of a pre-trained model for efficient performance.


If you wish, you can also train a new model using your own data, or combine it with our available public dataset. (soon to come!)

To enhance segmentation, MemBrain-seg includes preprocessing functions. These help to adjust your tomograms so they're similar to the data our network was trained on, making the process smoother and more efficient.

Explore MemBrain-seg, use it for your needs, and let us know how it works for you!

# Installation
For detailed installation instructions, please look [here](./installation.md).

# Features
## Segmentation
Segmenting the membranes in your tomograms is the main feature of this repository. 
Please find more detailed instructions [here](./Usage/Segmentation.md).

## Model training
It is also possible to use this package to train your own model. Instructions can be found [here](./Usage/Training.md).


## Preprocessing
Currently, we provide the following two [preprocessing](https://github.com/teamtomo/membrain-seg/tree/main/src/tomo_preprocessing) options:
- pixel size matching: Rescale your tomogram to match the training pixel sizes
- Fourier amplitude matching: Scale Fourier components to match the "style" of different tomograms

For more information, see the [Preprocessing](Usage/Preprocessing.md) subsection.

