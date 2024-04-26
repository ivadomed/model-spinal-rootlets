## PAM50 rootlets

The `template/PAM50_rootlets.nii.gz` file contains dorsal cervical (C2-C8) rootlets and was added to the PAM50 template 
(and SCT) in [PR #29](https://github.com/spinalcordtoolbox/PAM50/pull/29) on [2024-02-15](https://github.com/spinalcordtoolbox/PAM50/blob/master/CHANGES.md#2024-02-15-jv).
The `template/PAM50_rootlets.nii.gz` was automatically segmented using the [r20240129 rootlets model](https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240129). 
Then, the right side was manually adjusted (removing/adding a single voxel for some levels), and rootlets were symmetrized 
using [scripts/symmetrize_cord_segmentation.py](https://github.com/spinalcordtoolbox/PAM50/blob/master/scripts/symmetrize_cord_segmentation.py).

### Rootlets to spinal levels

`template/PAM50_rootlets.nii.gz` can be used to obtain spinal levels from the rootlets using the 
[02a_rootlets_to_spinal_levels.py](../inter-rater_variability/02a_rootlets_to_spinal_levels.py) script. These spinal 
levels were compared to the spinal levels based on Frostell et al. 2016; see Fig 10 in https://arxiv.org/abs/2402.00724. 

### Comparison with the spinal levels based on Frostell et al. 2016

The [pam50_levels_overlap.py](pam50_levels_overlap.py) script is used to compute percentage overlap between levels in 
obtained using the proposed nnUNet method and PAM50 spinal levels included in SCT 
(`$SCT_DIR/PAM50/template/PAM50_spinal_levels.nii.gz`) based on Frostell et al. 2016.

Example:

```bash
python pam50_levels_overlap.py 
-nnunet
PAM50_t2_label-rootlet_spinal_levels.nii.gz
-sct
${SCT_DIR}/PAM50/template/PAM50_spinal_levels.nii.gz
```