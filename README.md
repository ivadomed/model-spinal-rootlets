# Automatic Segmentation of Spinal Nerve Rootlets 

This repository contains the code for deep learning-based segmentation of the spinal nerve rootlets. 
The code is based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).

## Model Overview

The model was trained on T2-weighted images and provides semantic (i.e., level-specific) segmentation of the dorsal 
spinal nerve rootlets.

![sub-barcelona01](https://github.com/ivadomed/model-spinal-rootlets/assets/39456460/0315228f-a3c5-4aca-80ce-c00fd13a5fc9)

## Getting started

### Dependencies

- [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/user_section/installation.html)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python

### Step 1: Cloning the Repository

Open a terminal and clone the repository using the following command:

~~~
git clone https://github.com/ivadomed/model-spinal-rootlets
~~~

### Step 2: Setting up the Environment

The following commands show how to set up the environment. 
Note that the documentation assumes that the user has `conda` installed on their system. 
Instructions on installing `conda` can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a conda environment with the following command:
```
conda create -n venv_nnunet python=3.9
```

2. Activate the environment with the following command:
```
conda activate venv_nnunet
```

3. Install the required packages with the following command:
```
cd model-spinal-rootlets
pip install -r packaging/requirements.txt
```
 
### Step 3: Getting the Predictions

ℹ️ To temporarily suppress warnings raised by the nnUNet, you can run the following three commands in the same terminal session as the above command:

```bash
export nnUNet_raw="${HOME}/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/nnUNet_preprocessed"
export nnUNet_results="${HOME}/nnUNet_results"
```

To segment a single image using the trained model, run the following command from the terminal. 

This assumes that the model has been downloaded, unzipped (`unzip model-spinal-rootlets_2023-08-15_fold1.zip`) and is available locally.

```bash
python packaging/run_inference_single_subject.py -i <INPUT> -o <OUTPUT> -path-model <PATH_TO_MODEL_FOLDER>
```

For example:

```bash
python packaging/run_inference_single_subject.py -i sub-001_T2w.nii.gz -o sub-001_T2w_label-rootlet.nii.gz -path-model model-spinal-rootlets_2023-08-15_fold1
```

ℹ️ The script also supports getting segmentations on a GPU. To do so, simply add the flag `--use-gpu` at the end of the above commands. By default, the inference is run on the CPU. It is useful to note that obtaining the predictions from the GPU is significantly faster than the CPU.

