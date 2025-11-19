# Training
MemBrain-seg is designed to work out-of-the-box and ideally will not require training your own model.

However, in some cases, your tomograms may be too far out of the distribution of our training images. In this case,
it can make sense to [annotate](./Annotations.md) several patches extracted from your tomogram, and re-train the model
using your corrected data, together with our main training dataset.

Our main training dataset is publicly accessible on Zenodo: https://zenodo.org/records/15089686
You can also download it conveniently using the command line interface of MemBrain-seg (see [here](./Usage/Benchmarking.md#downloading-the-dataset)).

Here are some steps you can follow in order to re-train MemBrain-seg:

# Step 1: Prepare your training dataset
MemBrain-seg assumes a specific data structure for creating the training dataloaders:

```bash
data_dir/
├── imagesTr/       # Directory containing training images
│   ├── img1.nii.gz    # Image file (currently requires nii.gz format)
│   ├── img2.nii.gz    # Image file
│   └── ...
├── imagesVal/      # Directory containing validation images
│   ├── img3.nii.gz    # Image file
│   ├── img4.nii.gz    # Image file
│   └── ...
├── labelsTr/       # Directory containing training labels
│   ├── img1.nii.gz  # Label file (currently requires nii.gz format)
│   ├── img2.nii.gz  # Label file
│   └── ...
└── labelsVal/      # Directory containing validation labels
    ├── img3.nii.gz  # Label file
    ├── img4.nii.gz  # Label file
    └── ...
```

The data_dir argument is then passed to the training procedure (see [Step 2](#step-2-perform-training)).

In addition to our main training dataset, you may want to add some corrected patches from your own tomograms to improve the network's performance on these.

You can find some instructions here: [How to create training annotations from your own tomogram?](./Annotations.md)

# Step 2: Perform training
Performing the training is simple. After activating your virtual Python environment, you can type:
```
membrain train
```
to receive help with the input arguments. You will see that the only parameter you need to provide is the --data-dir argument:

```
membrain train --data-dir <path-to-your-training-data>
```
This is exactly the folder you prepared in [Step 1](#step-1-prepare-your-training-dataset). 

Running this command should start the training and store the fully trained model in the ./checkpoint folder.

**Note:** Training can take up to several days. We therefore recommend that you perform training on a device with a CUDA-enabled GPU.


# Advanced settings
In case you feel fancy and would like to adjust some of the default settings of MemBrain-seg, you can also use the following command to get access to more customizable options:
```
membrain train_advanced
````
This will display all options that can be activated / deactivated. For more in-depth adjustments, you will need to dig into MemBrain-seg's code or contact us.


# Contact
If there are any problems coming up when running the code or anything else is unclear, do not hesitate to contact us (Lorenz.Lamm@helmholtz-munich.de). We are more than happy to help.

