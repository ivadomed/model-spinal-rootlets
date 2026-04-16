import os
import shutil
import argparse
import subprocess
from argparse import RawTextHelpFormatter


def get_parser():
    parser = argparse.ArgumentParser(
        description='Convert BIDS datastructure to nnU-Netv2 datastructure.',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )

    parser.add_argument('-i', required=True, help='Path to data_processed folder.')
    parser.add_argument('-o', required=True, help='Path to output folder.')
    parser.add_argument('-dataset-name', required=True, help='nnUNet dataset name.')
    parser.add_argument('-contrast-to-move', required=True, nargs='+',
        choices=['UNIT1', 'UNIT1_neg', 'inv-1_part-mag_MP2RAGE', 'inv-1_part-mag_MP2RAGE_neg',
            'inv-2_part-mag_MP2RAGE', 'inv-2_part-mag_MP2RAGE_neg', 'T2w'], help='Contrasts to include.')
    parser.add_argument('-t', required=True, help='Training or testing folds.', choices=['train', 'test'])

    return parser


def create_nnunet_datastructure_files(base_path, dataset_name, train_test):
    nnunet_root = os.path.join(base_path, "nnUNet_raw", dataset_name)

    if train_test == 'train':
        images = os.path.join(nnunet_root, "imagesTr")
        labels = os.path.join(nnunet_root, "labelsTr")
    elif train_test == 'test':
        images = os.path.join(nnunet_root, "imagesTs")
        labels = os.path.join(nnunet_root, "labelsTs")

    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)

    return images, labels


CONTRAST_MAP = {
    "UNIT1": "UNIT1.nii.gz",
    "UNIT1_neg": "UNIT1_neg.nii.gz",
    "inv-1_part-mag_MP2RAGE": "inv-1_part-mag_MP2RAGE.nii.gz",
    "inv-1_part-mag_MP2RAGE_neg": "inv-1_part-mag_MP2RAGE_neg.nii.gz",
    "inv-2_part-mag_MP2RAGE": "inv-2_part-mag_MP2RAGE.nii.gz",
    "inv-2_part-mag_MP2RAGE_neg": "inv-2_part-mag_MP2RAGE_neg.nii.gz",
    "T2w": "T2w"
}


def convert_bids_to_nnunet_structure(bids_dir, images_out, labels_out, contrasts):

    for subject in os.listdir(bids_dir):
        subj_path = os.path.join(bids_dir, subject)
        if os.path.isdir(subj_path) and subject.startswith("sub"):
            anat_path = os.path.join(subj_path, "anat")
            if not os.path.exists(anat_path):
                continue

            for file in os.listdir(anat_path):
                file_path = os.path.join(anat_path, file)

                if not os.path.isfile(file_path):
                    continue

                if not file.startswith("sub") or not file.endswith(".nii.gz"):
                    continue

                for contrast in contrasts:
                    pattern = CONTRAST_MAP[contrast]
                    if contrast == "T2w":
                        match = "T2w" in file
                    else:
                        match = file.endswith(pattern)

                    if not match:
                        continue

                    file_number = file[8:10]

                    if contrast == "T2w":
                        out_name = f"{file[:-7]}_0000.nii.gz"
                    else:
                        out_name = f"SCDATA_{file_number}_{contrast}_0000.nii.gz"

                    dst_file = os.path.join(images_out, out_name)
                    shutil.copy(file_path, dst_file)

                    # reorientation to the RPI
                    subprocess.run(["sct_image", "-i", dst_file, "-setorient", "RPI"], check=True)
                    subprocess.run(["sct_image", "-i", dst_file, "-getorient"], check=True)

        if subject.startswith("derivatives"):
            deriv_path = os.path.join(bids_dir, subject, 'labels')
            for sub in os.listdir(deriv_path):

                if not sub.startswith("sub"):
                    continue

                anat_path = os.path.join(deriv_path, sub, "anat")

                if not os.path.exists(anat_path):
                    continue

                for file in os.listdir(anat_path):
                    if not file.endswith("UNIT1_label-rootlets_dseg.nii.gz"):
                        continue

                    gt_path = os.path.join(anat_path, file)
                    file_number = file[8:10]

                    for contrast in contrasts:
                        out_name = f"SCDATA_{file_number}_{contrast}.nii.gz"
                        dst_file = os.path.join(labels_out, out_name)
                        shutil.copy(gt_path, dst_file)

                        # reorient labels to RPI
                        subprocess.run(["sct_image", "-i", dst_file, "-setorient", "RPI"], check=True)
                        subprocess.run(["sct_image", "-i", dst_file, "-getorient"], check=True)

    return "Datastructure for nnU-Net has been stored!"


def main():
    parser = get_parser()
    args = parser.parse_args()

    images_out, labels_out = create_nnunet_datastructure_files(args.o, args.dataset_name, args.t)
    convert_bids_to_nnunet_structure(args.i, images_out, labels_out, args.contrast_to_move)

if __name__ == "__main__":
    main()
