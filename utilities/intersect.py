import nibabel as nib
import numpy as np
#sct_maths -i sub-brnoUhb01_085_0000_seg.nii.gz -o sub-brnoUhb01_085_0000_dil.nii.gz -dilate 2
import argparse

def get_parser():
    parser = argparse.ArgumentParser("get intersection of 2 nifti files")
    parser.add_argument("-i1", "--image1", help="path to image 1", required=True)
    parser.add_argument("-i2", "--image2", help="path to image 2", required=True)
    parser.add_argument("-o", help="path out file", required=True)

def main():
    parser = get_parser()
    args = parser.parse_args()
    image1 = args.image1
    image2 = args.image2
    out = args.o
    root = nib.load(image1)
    sc = nib.load(image2)
    root_arr = root.get_fdata().astype(np.int16)
    sc_arr = sc.get_fdata().astype(np.int16)
    res = root_arr * sc_arr
    nib.save(nib.Nifti1Image(res, affine=root.affine, header=root.header), out)

if __name__ == "__main__":
    main()
