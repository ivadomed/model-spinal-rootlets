#!/bin/bash
#
# Bring the PAM50 spinal levels (PAM50_spinal_levels.nii.gz) to the subject native space
# Context: https://github.com/ivadomed/model-spinal-rootlets/issues/13
#
# Usage:
#   ./bring_PAM50_spinal_levels_to_native_space.sh <file_t2>
#
# The script expects that the following files are present in the current directory:
#   - <file_t2>.nii.gz: T2-weighted image
#   - <file_t2>_seg.nii.gz: spinal cord segmentation
#   - PAM50_t2_label-rootlet.nii.gz: PAM50 rootlets segmentation (obtained by running our nnUNet model on PAM50_t2.nii.gz)
#
# The script does the following steps:
#   1. Segment spinal cord from the native image
#   2. Create mid-vertebral labels in the cord for vertebrae C3 and C5
#   3. Register the native image to the PAM50 template using the C3 and C5 labels
#   4. Bring rootlets segmentation from native space to PAM50 space
#   5. Run label-wise Tz-only registration between PAM50 rootlets (fixed) and subject rootlets (moving)
#   6. Concatenate transformations
#   7a. Bring PAM50 spinal levels (PAM50_spinal_levels.nii.gz)
#   7b. Bring PAM50 spinal rootlets to subject native space
#
# Author: Jan Valosek, Julien Cohen-Adad
#

# Immediately exit if error
set -e -o pipefail

# Check if input parameters are provided
if [[ $# -eq 0 ]]; then
  echo "ERROR: No input parameters provided. Exiting script."
  exit 1
fi

# Remove .nii.gz extension from the input file if present
file_t2=${1/.nii.gz/}

# 1. Segment spinal cord
sct_deepseg_sc -i ${file_t2}.nii.gz -c t2 -qc qc -qc-subject ${file_t2}

# 2. Create mid-vertebral labels in the cord for vertebrae C3 and C5
sct_label_vertebrae -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -c t2 -qc qc -qc-subject ${file_t2}
sct_label_utils -i ${file_t2}_seg_labeled.nii.gz -vert-body 3,7 -o ${file_t2}_seg_labeled_vertbody_35.nii.gz -qc qc -qc-subject ${file_t2}

# 3. Register T2-w image to PAM50 template
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -l ${file_t2}_seg_labeled_vertbody_35.nii.gz -c t2 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3 -qc qc -qc-subject ${file_t2}
# Rename output for clarity
mv anat2template.nii.gz ${file_t2}_reg.nii.gz

# 4. Bring rootlets segmentation from native space to PAM50 space
# Note: Nearest-neighbor interpolation (-x nn) must be used to preserve the level-wise rootlets segmentation values
sct_apply_transfo -i ${file_t2}_label-rootlet_staple.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_anat2template.nii.gz -x nn

# 5. Run label-wise Tz-only registration between PAM50 rootlets (fixed) and subject rootlets (moving)
$SCT_DIR/bin/isct_antsRegistration --dimensionality 3 --float 0 \
--output [registration2_,${file_t2}_label-rootlet_staple_reg_reg.nii.gz] --interpolation nearestNeighbor --verbose 1 \
--transform Affine[5] --metric MeanSquares[PAM50_t2_label-rootlet.nii.gz,${file_t2}_label-rootlet_staple_reg.nii.gz,1,32] --convergence 20x10 --shrink-factors 2x1 --smoothing-sigmas 0x0mm --shrink-factors 1x1 --restrict-deformation 0x0x1 \
--transform BSplineSyN[0.1,3,0] --metric MeanSquares[PAM50_t2_label-rootlet.nii.gz,${file_t2}_label-rootlet_staple_reg.nii.gz,1,32] --convergence 10x5 --shrink-factors 2x1 --smoothing-sigmas 0x0mm --shrink-factors 1x1 --restrict-deformation 0x0x1
# Help (`$SCT_DIR/bin/isct_antsRegistration --help`):
#     --metric 'MeanSquares[fixedImage,movingImage,....
#	    --output '[outputTransformPrefix,<outputWarpedImage>,.....
#
# Outputs are:
#     registration2_1InverseWarp.nii.gz       # result of BSplineSyN (inverse)
#     registration2_1Warp.nii.gz              # result of BSplineSyN
#     registration2_0GenericAffine.mat        # result of Affine

# 6. Concatenate transformations:
#   1. Inverse affine transformation (inverse of registration2_0GenericAffine.mat)
#   2. BSplineSyN transformation (registration2_1InverseWarp.nii.gz)
#   3. warp_template2anat.nii.gz (obtained by running `sct_warp_template`)
# Note that `-winv` is used to invert the affine transformation listed in flag `-w`
sct_concat_transfo -d ${file_t2}.nii.gz -w registration2_0GenericAffine.mat registration2_1InverseWarp.nii.gz warp_template2anat.nii.gz -o warp_final.nii.gz -winv registration2_0GenericAffine.mat

# 7a. Bring PAM50 spinal levels (template/PAM50_spinal_levels.nii.gz) from PAM50 template space to subject native space
# using the concatenated transformation
# Note that the following command requires the jca/16-spinal-levels branch of the PAM50 repository:
#   cd ~/code/PAM50
#   git pull
#   git checkout jca/16-spinal-levels
sct_apply_transfo -i ~/code/PAM50/template/PAM50_spinal_levels.nii.gz -d ${file_t2}.nii.gz -w warp_final.nii.gz -x nn -o PAM50_spinal_levels_reg.nii.gz

# 7b. Bring PAM50 spinal rootlets from PAM50 template space to subject native space using the concatenated transformation
sct_apply_transfo -i PAM50_t2_label-rootlet.nii.gz -d ${file_t2}.nii.gz -w warp_final.nii.gz -x nn -o PAM50_t2_label-rootlets_reg.nii.gz

echo "----------------------------------------------"
echo "Created: PAM50_spinal_levels_reg.nii.gz"
echo "Created: PAM50_t2_label-rootlets_reg.nii.gz"
echo "----------------------------------------------"
echo "label-wise Tz-only registration in PAM50 space:"
echo "    fsleyes ../PAM50_t2.nii.gz PAM50_t2_label-rootlet.nii.gz -cm red-yellow -dr 0 20  ${file_t2}_label-rootlet_staple_reg.nii.gz -cm blue-lightblue -dr 0 20  ${file_t2}_label-rootlet_staple_reg_reg.nii.gz -dr 0 20 -cm green"
echo "PAM50 spinal levels and rootlets in subject space:"
echo "    fsleyes ${file_t2}.nii.gz ${file_t2}_label-rootlet_staple.nii.gz -cm red-yellow -dr 0 20 PAM50_t2_label-rootlets_reg.nii.gz -cm blue-lightblue -dr 0 20 PAM50_spinal_levels_reg.nii.gz -cm subcortical -dr 0 20"
echo "----------------------------------------------"