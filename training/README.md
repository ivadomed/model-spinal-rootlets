## `r20250318` training data

The following datasets were used to train the [r20250318](https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20250318) model. 
The model provides semantic (i.e., level-specific) segmentation of dorsal and ventral nerve rootlets C2-T1 from MP2RAGE (INV1, INV2, UNIT1) and T2w data. 
The orientation of the input image is expected to be RPI (right-posterior-inferior).

- open-access ds004507: [https://openneuro.org/datasets/ds004507/versions/1.1.1](https://openneuro.org/datasets/ds004507/versions/1.1.1)
- open-access spine-generic/data-multi-subject: [https://github.com/spine-generic/data-multi-subject/tree/r20250314](https://github.com/spine-generic/data-multi-subject/tree/r20250314)
- private MP2RAGE dataset (`data.neuro.polymtl.ca/hc-leipzig-7t-mp2rage`)

Exact train/val/test splits can be found in the [training/hc-leipzig-7t-mp2rage](https://github.com/ivadomed/model-spinal-rootlets/tree/main/training/hc-leipzig-7t-mp2rage) folder.

The [r20250318](https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20250318) model is accesible via `sct_deepseg rootlets -i < MRI-scan >` command as part of the Spinal Cord Toolbox (SCT) [v7.0](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/7.0) and higher.

Publication: [Krejčí et al., 2026, Scientific Reports](https://www.nature.com/articles/s41598-026-49164-0)
