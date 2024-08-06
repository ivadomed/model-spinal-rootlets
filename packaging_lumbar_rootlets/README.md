## Getting started

> [!IMPORTANT]
> This README provides instructions on how to use the model for **_lumbar_** rootlets. 
> Please note that this model is still under development and is not yet available in the Spinal Cord Toolbox (SCT).

> [!NOTE]
> If you would like to use the model for _**dorsal**_ cervical rootlets only, use SCT v6.2 or higher (please refer to 
> this [README](..%2FREADME.md)).

### Dependencies

- [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/user_section/installation.html) 6.0 or higher
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python

### Step 1: Cloning the Repository

Open a terminal and clone the repository using the following command:

```
git clone https://github.com/ivadomed/model-spinal-rootlets
```

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
pip install -r packaging_lumbar_rootlets/requirements.txt
```
 
### Step 3: Getting the Predictions

> [!NOTE]  
> To temporarily suppress warnings raised by the nnUNet, you can run the following three commands in the same terminal session as the above command:
>
> ```bash
> export nnUNet_raw="${HOME}/nnUNet_raw"
> export nnUNet_preprocessed="${HOME}/nnUNet_preprocessed"
> export nnUNet_results="${HOME}/nnUNet_results"
> ```

To segment a single image using the trained model, run the following command from the terminal. 

This assumes that the lumbar model has been downloaded and unzipped (`unzip Dataset202_LumbarRootlets_r20240527.zip` or `unzip Dataset302_LumbarRootlets_r20240723.zip`).

```bash
python packaging_lumbar_rootlets/run_inference_single_subject.py -i <INPUT> -o <OUTPUT> -path-model <PATH_TO_MODEL_FOLDER> -fold <FOLD>
```

For example:

```bash
python packaging_lumbar_rootlets/run_inference_single_subject.py -i sub-001_T2w.nii.gz -o sub-001_T2w_label-rootlets_dseg.nii.gz -path-model ~/Downloads/Dataset202_LumbarRootlets_r20240527 -fold 0
```

If the model folder contains also trainer subfolders (e.g., `nnUNetTrainer__nnUNetPlans__3d_fullres`, `nnUNetTrainerDA5__nnUNetPlans__3d_fullres`, ...), specify the trainer folder as well:

```bash
python packaging_lumbar_rootlets/run_inference_single_subject.py -i sub-001_T2w.nii.gz -o sub-001_T2w_label-rootlets_dseg.nii.gz -path-model ~/Downloads/Dataset322_LumbarRootlets/nnUNetTrainerDA5__nnUNetPlans__3d_fullres -fold 0
```

> [!NOTE] 
> Note that some models, for example, `Dataset312_LumbarRootlets` and `Dataset322_LumbarRootlets`, were trained on images cropped around the spinal cord.
> This means that also the input image for inference needs to be cropped around the spinal cord.
> You can use the following command to crop the image:
> ```bash
> file=sub-001_T2w
> # Segment the spinal cord using the contrast-agnostic model
> sct_deepseg -i ${file}.nii.gz -o ${file}_seg.nii.gz -task seg_sc_contrast_agnostic -qc ../qc -qc-subject ${file}
> # Crop the image around the spinal cord
> sct_crop_image -i ${file}.nii.gz -m ${file}_seg.nii.gz -dilate 64x64x64 -o ${file}_crop.nii.gz
> # Now you can use the cropped image for inference
> ```


> [!NOTE] 
> The script also supports getting segmentations on a GPU. To do so, simply add the flag `--use-gpu` at the end of the above commands. 
> By default, the inference is run on the CPU. It is useful to note that obtaining the predictions from the GPU is significantly faster than the CPU.
