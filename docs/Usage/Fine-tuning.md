# Fine-tuning
MemBrain-seg is built to function optimally out-of-the-box, eliminating the need for most users to train the model themselves.

However, if your tomograms differ significantly from the images used in our training dataset, fine-tuning the model on your own data may enhance performance. In this case, it can make sense to [annotate](./Annotations.md) several patches extracted from your tomogram, and fine-tune the pretrained MemBrain-seg model using your corrected data.

Here are some steps you can follow in order to fine-tune MemBrain-seg:

# Step 1: Prepare your fine-tuning dataset
MemBrain-seg assumes a specific data structure for creating the fine-tuning dataloaders, which can be a smaller or corrected version of your tomograms:

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

The data_dir argument is then passed to the fine-tuning procedure (see [Step 2](#step-2-perform-fine-tuning)).

To fine-tune the pretrained model on your own tomograms, you need to add some corrected patches from your own tomograms to improve the network's performance on these.

You can find some instructions here: [How to create training annotations from your own tomogram?](./Annotations.md)

# Step 2: Perform fine-tuning
Fine-tuning starts from a pretrained model checkpoint. After activating your virtual Python environment, you can type:
```
membrain finetune
```
to receive help with the input arguments. You will see that the two parameters you need to provide are the --pretrained-checkpoint-path and the --data-dir argument:

```
membrain finetune --pretrained-checkpoint-path <path-to-the-pretrained-checkpoint> --finetune-data-dir <path-to-your-finetuning-data>
```
This command fine-tunes the pretrained MemBrain-seg model using your fine-tuning dataset. Be sure to point to the correct checkpoint path containing the pretrained weights, as well as the fine-tuning data directory.

This is exactly the folder you prepared in [Step 1](#step-1-prepare-your-fine-tuning-dataset).

Running this command should start the fine-tuning process and store the fine-tuned model in the ./finetuned_checkpoints folder.

**Note:** Fine-tuning can take up to 24 hours. We therefore recommend that you perform training on a device with a CUDA-enabled GPU.


# Advanced settings
In case you feel fancy and would like to adjust some of the default settings of MemBrain-seg, you can also use the following command to get access to more customizable options:
```
membrain finetune_advanced
````
This will display all available options that can be activated or deactivated. For example, when fine-tuning, you might want to lower the learning rate compared to training from scratch to prevent the model from "forgetting" the knowledge it learned during pretraining. For more in-depth adjustments, you will need to dig into MemBrain-seg's code or contact us.


# Contact
If there are any problems coming up when running the code or anything else is unclear, do not hesitate to contact us (Lorenz.Lamm@helmholtz-munich.de). We are more than happy to help.
