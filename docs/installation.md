# Installation

These installation instructions are very preliminary, and surely will not work on all systems.
But if any problems come up, do not hesitate to contact us (lorenz.lamm@helmholtz-munich.de).

## Step 1: Create a virtual environment
Before running any scripts, you should create a virtual Python environment.
In these instructions, we use Miniconda for managing your virtual environments,
but any alternative like Conda, Mamba, virtualenv, venv, ... should be fine.

If you don't have any, you could install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

Now you can create a new virtual environment using
```shell
conda create --name <env_name> python=3.9
```

In order to use it, you need to activate the environment:
```shell
conda activate <env_name>
```


## Step 2: Install membrain-seg via PyPI

**New:** MemBrain-seg is now pip-installable. <br>

That means, you can install membrain-seg by typing
```shell
pip install membrain-seg
```
This will install MemBrain-seg and all dependencies required for segmenting your tomograms.

## Step 3: Validate installation
As a first check whether the installation was successful, you can run
```shell
membrain
```
This should display the different options you can choose from MemBrain, like "segment" and "train", similar to the screenshot below:


<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250504233-d7e49aef-e783-45fb-b04e-8736b1af7d6e.png">
</p>

## Step 4: Download pre-trained segmentation model (optional)
We recommend to use denoised (ideally Cryo-CARE<sup>1</sup>) tomograms for segmentation. Our current best model is available for download [here](https://drive.google.com/file/d/1hruug1GbO4V8C4bkE5DZJeybDyOxZ7PX/view?usp=sharing). Please let us know how it works for you.
If the given model does not work properly, you may want to try one of our experimental or previous versions:

Experimental models:
- [v10_beta_FAaug -- model traing with Fourier amplitude augmentation for better generalization](https://drive.google.com/file/d/1kaN9ihB62OfHLFnyI2_t6Ya3kJm7Wun9/view?usp=sharing)
- [v10_beta_MWaug -- model traing with missing wedge augmentation for better missing wedge restoration](https://drive.google.com/file/d/1-i836rU-wfuClsqPRbKqJ-eW2jCUlwJm/view?usp=sharing)

Other (older) model versions:
- [v10_alpha -- standard model until 24th April 2025](https://drive.google.com/file/d/1tSQIz_UCsQZNfyHg0RxD-4meFgolszo8/view?usp=sharing)
- [v9 -- best model until 10th Aug 2023](https://drive.google.com/file/d/15ZL5Ao7EnPwMHa8yq5CIkanuNyENrDeK/view?usp=sharing)
- [v9b -- model for non-denoised data until 10th Aug 2023](https://drive.google.com/file/d/1TGpQ1WyLHgXQIdZ8w4KFZo_Kkoj0vIt7/view?usp=sharing)

NOTE: Previous model files are not compatible with MONAI v1.3.0 or higher. So if you're using v1.3.0 or higher, consider downgrading to MONAI v1.2.0 or downloading this [adapted version](https://drive.google.com/file/d/1Tfg2Ju-cgSj_71_b1gVMnjqNYea7L1Hm/view?usp=sharing) of our most recent model file. 


Once downloaded, you can use it in MemBrain-seg's [Segmentation](./Usage/Segmentation.md) functionality to segment your tomograms.


```
[1] T. -O. Buchholz, M. Jordan, G. Pigino and F. Jug, "Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data," 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), Venice, Italy, 2019, pp. 502-506, doi: 10.1109/ISBI.2019.8759519.
```


# Troubleshooting
Here is a collection of common issues and how to fix them:

- `RuntimeError: The NVIDIA driver on your system is too old (found version 11070). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has 
been compiled with your version of the CUDA driver.` 

  The latest Pytorch versions require higher CUDA versions that may not be installed on your system yet. You can either install the new CUDA version or (maybe easier) downgrade Pytorch to a version that is compatible:

  `pip uninstall torch`

  `pip install torch==2.0.1`