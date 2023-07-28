import nibabel as nib
import numpy as np
#sct_maths -i sub-brnoUhb01_085_0000_seg.nii.gz -o sub-brnoUhb01_085_0000_dil.nii.gz -dilate 2
root = nib.load("sub-brnoUhb01_085.nii.gz")
sc = nib.load("sub-brnoUhb01_085_0000_dil.nii.gz")
root_arr = root.get_fdata().astype(np.int16)
sc_arr = sc.get_fdata().astype(np.int16)
res = root_arr * sc_arr

nib.save(nib.Nifti1Image(res, affine=root.affine, header=root.header), "cuted.nii.gz")
