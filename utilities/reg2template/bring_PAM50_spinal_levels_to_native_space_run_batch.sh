#!/bin/bash
#
# Bring the PAM50 spinal levels (PAM50_spinal_levels.nii.gz) to the subject native space
# Context: https://github.com/ivadomed/model-spinal-rootlets/issues/13
#
# Usage:
#   ./bring_PAM50_spinal_levels_to_native_space.sh <file_t2> <file_rootlets>
#
# The script expects that the following files are present in the current directory:
#   - <file_t2>.nii.gz: T2-weighted image
#   - <file_t2>_seg.nii.gz: spinal cord segmentation
#   - <file_rootlets>.nii.gz: spinal rootlets discrete segmentation
#   - PAM50_rootlet.nii.gz: PAM50 rootlets segmentation (in PAM50 template)
#
# The script does the following steps:
#   1. Segment spinal cord from the native image
#   2. Create mid-vertebral labels in the cord for vertebrae C3 and C7
#   3. Register the native image to the PAM50 template using the C3 and C7 labels
#   4. Bring rootlets segmentation from native space to PAM50 space
#   5. Run label-wise Tz-only registration between PAM50 rootlets (fixed) and subject rootlets (moving)
#   6. Concatenate transformations
#   7a. Bring PAM50 spinal levels (PAM50_spinal_levels.nii.gz)
#   7b. Bring PAM50 spinal rootlets to subject native space
#   8. Straighten spinal cord (to look at the results)
# Author: Jan Valosek, Julien Cohen-Adad, Sandrine Bédard
#

# Immediately exit if error
set -e -o pipefail

# get starting time:

start_all=`date +%s`
# Remove .nii.gz extension from the input file if present

# Retrieve input params
SUBJECT=$1
ALL_DISC=$2

# Save script path
PATH_SCRIPT=$PWD

# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t2
  fi
}

# Check if manual segmentation already exists. If it does, copy it locally. If it does not, perform segmentation.
segment_if_does_not_exist(){
  local file="$1"
  folder_contrast="anat"

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg -i ${file}.nii.gz -task seg_sc_contrast_agnostic -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if PMJ label exists. If it does not, perform automatic detection.
detect_pmj_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-pmj"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${file}_pmj.nii.gz
    sct_qc -i ${file}.nii.gz -s ${file}_pmj.nii.gz -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}

  else
    echo "Not found. Proceeding with automatic labeling."
    # Detect PMJ
    sct_detect_pmj -i ${file}.nii.gz -c t2 -qc ${PATH_QC}

  fi
}

# Check if PMJ label exists. If it does not, perform automatic detection.
segment_rootlets_if_does_not_exist(){
  local file="$1"
  # Update global variable with segmentation file name
  FILELABEL="${file}_rootlets"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/*rootlets*.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual rootlets labels."
    rsync -avzh $FILELABELMANUAL ${file}_rootlets.nii.gz
  else
    echo "Not found. Proceeding with automatic labeling."
    # Detect PMJ
    sct_deepseg -i ${file}.nii.gz -task seg_spinal_rootlets_t2w -qc ${PATH_QC} -o ${file}_rootlets.nii.gz

  fi
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

SUBJECT_ID=$(dirname "$SUBJECT")
SES=$(basename "$SUBJECT")
# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/
file_t2="${SUBJECT_ID}_${SES}_T2w"


#file_t2=${1/.nii.gz/}
file_rootlets=${2/.nii.gz/}

# 1. Segment spinal cord
segment_if_does_not_exist $file_t2
# 2. Create mid-vertebral labels in the cord for vertebrae C2 and C7
label_if_does_not_exist $file_t2 ${file_t2}_seg
sct_label_utils -i ${file_t2}_seg_labeled.nii.gz -vert-body 2,7 -o ${file_t2}_seg_labeled_vertbody_27.nii.gz -qc qc -qc-subject ${file_t2}

# 3. Register T2-w image to PAM50 template
if [[ $ALL_DISC ]]; then
  sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -l ${file_t2}_seg_labeled_vertbody_27.nii.gz -c t2 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3 -qc qc -qc-subject ${file_t2}
else
  sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -ldisc ${file_t2}_seg_labeled_discs.nii.gz -c t2 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3 -qc qc -qc-subject ${file_t2}
fi
# Rename output for clarity
mv anat2template.nii.gz ${file_t2}_reg.nii.gz

# 4. Segment nerve rootlets
segment_rootlets_if_does_not_exist $file_t2
# 4. Bring rootlets segmentation from native space to PAM50 space
# Note: Nearest-neighbor interpolation (-x nn) must be used to preserve the level-wise rootlets segmentation values

sct_apply_transfo -i ${file_rootlets}.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_anat2template.nii.gz -x nn

start=`date +%s`
# 5. Run label-wise Tz-only registration between PAM50 rootlets (fixed) and subject rootlets (moving)
$SCT_DIR/bin/isct_antsRegistration --dimensionality 3 --float 0 \
--output [registration2_,${file_rootlets}_reg_reg.nii.gz] --interpolation nearestNeighbor --verbose 1 \
--transform Affine[5] --metric MeanSquares[$SCT_DIR/data/PAM50/template/PAM50_rootlets.nii.gz,${file_rootlets}_reg.nii.gz,1,32] --convergence 15x7x5 --shrink-factors 4x2x1 --smoothing-sigmas 0x0x0mm --restrict-deformation 0x0x1 \
--transform BSplineSyN[0.1,3,0] --metric MeanSquares[$SCT_DIR/data/PAM50/template/PAM50_rootlets.nii.gz,${file_rootlets}_reg.nii.gz,1,32] --convergence 5x4x2 --shrink-factors 4x2x1 --smoothing-sigmas 1x0x0mm --restrict-deformation 0x0x1
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
end=`date +%s`
runtime=$((end-start))
echo "+++++++++++ TIME: Duration of of rootlet z reg:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"

sct_concat_transfo -d ${file_t2}.nii.gz -w registration2_0GenericAffine.mat registration2_1InverseWarp.nii.gz warp_template2anat.nii.gz -o warp_final.nii.gz -winv registration2_0GenericAffine.mat

# 7a. Bring PAM50 spinal levels (template/PAM50_spinal_levels.nii.gz) from PAM50 template space to subject native space
# using the concatenated transformation
# Note that the following command requires the jca/16-spinal-levels branch of the PAM50 repository:
#   cd ~/code/PAM50
#   git pull
#   git checkout jca/16-spinal-levels
sct_apply_transfo -i $SCT_DIR/data/PAM50/template/PAM50_spinal_levels.nii.gz -d ${file_t2}.nii.gz -w warp_final.nii.gz -x nn -o PAM50_spinal_levels_reg.nii.gz

# 7b. Bring PAM50 spinal rootlets from PAM50 template space to subject native space using the concatenated transformation
sct_apply_transfo -i $SCT_DIR/data/PAM50/template/PAM50_rootlets.nii.gz -d ${file_t2}.nii.gz -w warp_final.nii.gz -x nn -o PAM50_rootlet_reg.nii.gz

# 8. Straignten spinal cord and spinal levels:
sct_straighten_spinalcord -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz
sct_apply_transfo -i PAM50_spinal_levels_reg.nii.gz -d ${file_t2}_straight.nii.gz -w warp_curve2straight.nii.gz -x nn -o PAM50_spinal_levels_reg_straight.nii.gz
sct_apply_transfo -i ${file_rootlets}_reg_reg.nii.gz -d ${file_t2}_straight.nii.gz -w warp_curve2straight.nii.gz -x nn -o ${file_rootlets}_reg_reg_straight.nii.gz

# 9.Get mid-points of reg PAM50 spinal levels
sct_label_utils -i PAM50_spinal_levels_reg.nii.gz -cubic-to-point -o PAM50_spinal_levels_reg_midpoints.nii.gz

# 10.Regsiter to template using new levels:
# TODO: consider removing C1 label
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -lspinal PAM50_spinal_levels_reg_midpoints.nii.gz -c t2 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3 -qc qc -qc-subject ${file_t2} -ofolder anat2template_spinal

# 11.Get spinal levels directly from nerve rootlets
source ${SCT_DIR}/python/etc/profile.d/conda.sh
conda activate venv_sct
python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${file_rootlets}.nii.gz -s ${file_t2}_seg.nii.gz -pmj ${file_t2}_labels-pmj-manual.nii.gz
sct_label_utils -i ${file_rootlets}_spinal_levels.nii.gz -cubic-to-point -o ${file_rootlets}_spinal_levels_midpoints.nii.gz
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -lspinal ${file_rootlets}_spinal_levels_midpoints.nii.gz -c t2 -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,slicewise=1,smooth=0,iter=5:step=3,type=im,algo=syn,slicewise=1,smooth=0,iter=3 -qc qc -qc-subject ${file_t2} -ofolder anat2template_spinal_NOREG

# Compute distance between mid-point of spinal level estimated with rootlet-informed reg to template
python ~/code/model-spinal-rootlets/utilities/get_distance_pmj_disc.py -centerline ${file_t2}_seg_centerline_extrapolated.csv -spinalroots PAM50_spinal_levels_reg_midpoints.nii.gz -o ${file_t2}_PAM50_spinal_levels_reg_midpoints_2PMJ.csv -subject $file_t2 -disclabel ${file_t2}_seg_labeled_discs.nii.gz
# Compute distance between spinal levels estimated directly from nerve rootlets
python ~/code/model-spinal-rootlets/utilities/get_distance_pmj_disc.py -centerline ${file_t2}_seg_centerline_extrapolated.csv -spinalroots ${file_rootlets}_spinal_levels_midpoints.nii.gz -o ${file_rootlets}_spinal_levels_midpoints_2PMJ.csv -subject $file_t2 -disclabel ${file_t2}_seg_labeled_discs.nii.gz
# Compute distance between intervertebral discs and PMJ 
python ~/code/model-spinal-rootlets/utilities/get_distance_pmj_disc.py -centerline ${file_t2}_seg_centerline_extrapolated.csv -spinalroots ${file_t2}_seg_labeled_discs.nii.gz -o ${file_t2}_seg_labeled_discs_2PMJ.csv -subject $file_t2 -disclabel ${file_t2}_seg_labeled_discs.nii.gz

echo "----------------------------------------------"
echo "Created: PAM50_spinal_levels_reg.nii.gz"
echo "Created: PAM50_t2_label-rootlets_reg.nii.gz"
echo "----------------------------------------------"
echo "label-wise Tz-only registration in PAM50 space:"
echo "    fsleyes ../PAM50_t2.nii.gz PAM50_rootlets.nii.gz -cm red-yellow -dr 0 20  ${file_t2}_label-rootlet_staple_reg.nii.gz -cm blue-lightblue -dr 0 20  ${file_t2}_label-rootlet_staple_reg_reg.nii.gz -dr 0 20 -cm green"
echo "PAM50 spinal levels and rootlets in subject space:"
echo "    fsleyes ${file_t2}.nii.gz ${file_t2}_label-rootlet_staple.nii.gz -cm red-yellow -dr 0 20 PAM50_t2_label-rootlets_reg.nii.gz -cm blue-lightblue -dr 0 20 PAM50_spinal_levels_reg.nii.gz -cm subcortical -dr 0 20"
echo "----------------------------------------------"
end_all=`date +%s`
runtime=$((end_all-start_all))
echo "+++++++++++ TIME: Duration of of rootlet z reg:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
