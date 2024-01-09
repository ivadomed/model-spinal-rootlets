## Evaluation of the nnUNet model across different resolutions

A single subject (`sub-010_ses-headUp`) from the Spinal Cord Head Position MRI dataset 
(https://openneuro.org/datasets/ds004507/versions/1.0.1) was used. The original 0.6mm iso T2w image  was resampled to 
0.8mm, 1.0mm, 1.2mm, 1.4mm, and 1.5mm iso resolution using the `sct_resample` function to evaluate the performance
of our nnUNet model on different resolutions.

```commandline
sct_resample -i ses-headUp06/anat/sub-010_ses-headUp_T2w.nii.gz -mm 0.8x0.8x0.8 -o ses-headUp08/anat/sub-010_ses-headUp_T2w.nii.gz
sct_resample -i ses-headUp06/anat/sub-010_ses-headUp_T2w.nii.gz -mm 1.0x1.0x1.0 -o ses-headUp10/anat/sub-010_ses-headUp_T2w.nii.gz
sct_resample -i ses-headUp06/anat/sub-010_ses-headUp_T2w.nii.gz -mm 1.2x1.2x1.2 -o ses-headUp12/anat/sub-010_ses-headUp_T2w.nii.gz
sct_resample -i ses-headUp06/anat/sub-010_ses-headUp_T2w.nii.gz -mm 1.4x1.4x1.4 -o ses-headUp14/anat/sub-010_ses-headUp_T2w.nii.gz
sct_resample -i ses-headUp06/anat/sub-010_ses-headUp_T2w.nii.gz -mm 1.5x1.5x1.5 -o ses-headUp15/anat/sub-010_ses-headUp_T2w.nii.gz
```