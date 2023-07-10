# model-spinal-rootlets

## 1) Project Overview

The goal of this project is to develop a method based on the DL technique to segment and locate the spinal roots.
The localization of these nerves is especially important for the electrode-based treatment of spinal cord injuries.

## 2) Work done

After [reviewing](https://github.com/ivadomed/model-spinal-rootlets/issues/1#issue-1706345176) the available datasets,
it was determined that none of them were suitable.
Therefore, a new dataset was used, and each sample was labeled with a binary value representing the presence of
spinal rootlets (0: False, 1: True).

The original dataset can be
found [here](https://openneuro.org/datasets/ds004507/versions/1.0.1). [review](https://github.com/ivadomed/model-spinal-rootlets/issues/1#issue-1706345176)
An initial labeling of 12 subjects was performed to train one fold of the nnUNetV2 3d_fullres model (refer
to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5)).
The resulting model achieved a Dice score of approximately 0.52.

After conducting inference on the entire dataset and performing manual corrections, the dataset was expanded to include
20 subjects. However, two subjects were subsequently excluded due to unsatisfactory label quality (
see [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)).
A full training on 5 folds with 250 epoch resulted to a Dice score of approximately 0.58$ .

Inference was further conducted on 267 subjects from
the [spinegeneric](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject)
dataset, which had an isotropic voxel size of 0.8mm. Some of the predictions from this inference were manually
corrected, and the resulting dataset was merged with the previous dataset, resulting in a final dataset containing 38
subjects.

## 3) Used tools 
