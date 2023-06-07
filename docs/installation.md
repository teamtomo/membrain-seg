# Installation

These installation instructions are very preliminary, and surely will not work on all systems.
But if any problems come up, do not hesitate to contact us (lorenz.lamm@helmholtz-munich.de).

## Step 1: Clone repository

Make sure to have git installed, then run
```shell
git clone -b training_docs https://github.com/teamtomo/membrain-seg.git
```

## Step 2: Create a virtual environment
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

## Step 3: Install MemBrain-seg and its dependencies
Move to the folder "membrain-seg" (from the cloned repository in Step 1) that contains the "src" folder.
Here, run

```shell
pip install -e .
```

This will install MemBrain-seg and all dependencies required for segmenting your tomograms.

## Step 4: Validate installation
As a first check whether the installation was successful, you can run
```shell
membrain
```
This should display the different options you can choose from MemBrain, like "segment" and "train".

