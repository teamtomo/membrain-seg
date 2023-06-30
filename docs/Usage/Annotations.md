# How to create training annotations from your own dataset?

This is an example guide on how to create training data in order to improve MemBrain-seg's performance on your tomograms. Credits for this annotation strategy also go to [Fabian Isensee](https://helmholtz-imaging.de/person/dr-rer-nat-fabian-isensee/) and [Sebastian Ziegler](https://modalities.helmholtz-imaging.de/expert/46) from [Helmholtz Imaging](https://helmholtz-imaging.de/).

**Important note:** While this guide describes how to imporve the performance for your own tomograms, we highly encourage you to also share your generated training patches with us and the community, so that everybody can benefit from a more generalized performance of the model.

# General idea
Nobody would like (or has the time) to segment membranes a whole tomogram manually from scratch. Therefore, our approach is to extract small patches (160^3) from the tomogram an create manual annotations for these.

For this, we do not start from scratch, but use the prediction of the currently best MemBrain-seg model. Ideally, this will already segment most areas well and minimize the workload for corrections.

# Workflow
The steps described in this tutorial are:

1. [Which software to use](#software)
2. [Extraction of patches for correction](#patch-extraction)
3. [Performing the corrections](#corrections)

# Software
You will need software to inspect your tomograms and MemBrain-seg's segmentations, as well as to perform the corrections. For both of these tasks, we use [MITK Workbench](https://docs.mitk.org/nightly/MITKWorkbenchManualPage.html), but any software with these functionalities will do, e.g. Amira or Napari.

# Patch extraction
The idea is simple: Find regions in your tomogram, where the segmentation performance is poor, but the tomogram clearly shows that there is or is not a membrane.

# Corrections