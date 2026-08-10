"""Tests for the local dorsal/ventral cluster-labeling application."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.labeling_app_core import (
    EXPERT_CLASSES,
    annotate_record,
    discover_cases,
    discover_nnunet_label_cases,
    discover_reference_label_cases,
    export_annotation_dataset,
    export_nnunet_case,
    load_case,
    merge_saved_annotations,
    read_review_csv,
    write_nnunet_dataset_json,
    write_review_csv,
)


def _case_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (31, 31, 14)
    anatomy = np.zeros(shape, dtype=np.float32)
    cord = np.zeros(shape, dtype=np.uint8)
    rootlets = np.zeros(shape, dtype=np.int16)
    xx, yy = np.ogrid[: shape[0], : shape[1]]
    cord_xy = ((xx - 15) / 3.0) ** 2 + ((yy - 15) / 4.0) ** 2 <= 1
    cord[:, :, :] = cord_xy[:, :, None]
    anatomy[:] = np.linspace(0, 1, shape[1])[None, :, None]

    rootlets[19:27, 10:12, 3:7] = 2
    rootlets[19:27, 19:21, 3:7] = 2
    rootlets[4:12, 10:12, 3:7] = 2
    rootlets[4:12, 19:21, 3:7] = 2
    return anatomy, rootlets, cord


class LabelingAppTest(unittest.TestCase):
    def test_discovery_finds_complete_triplet_across_output_folders(self) -> None:
        anatomy, rootlets, cord = _case_arrays()
        affine = np.eye(4)
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            inputs = directory / "inputs"
            derived = directory / "derived" / "sub-test" / "anat"
            inputs.mkdir(parents=True)
            derived.mkdir(parents=True)
            anatomy_path = inputs / "sub-test_T2w.nii.gz"
            rootlets_path = derived / "sub-test_label-rootlets_dseg.nii.gz"
            cord_path = derived / "sub-test_label-SC_seg.nii.gz"
            nib.save(nib.Nifti1Image(anatomy, affine), anatomy_path)
            nib.save(nib.Nifti1Image(rootlets, affine), rootlets_path)
            nib.save(nib.Nifti1Image(cord, affine), cord_path)

            cases = discover_cases(directory)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "sub-test")
            self.assertEqual(cases[0].anatomy_path, anatomy_path.resolve())
            self.assertEqual(cases[0].rootlets_path, rootlets_path.resolve())
            self.assertEqual(cases[0].cord_path, cord_path.resolve())

    def test_load_case_identifies_a_folder_instead_of_a_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            with self.assertRaisesRegex(
                ValueError, "Anatomical image must be a NIfTI file, not a folder"
            ):
                load_case(directory, directory, directory, case_id="sub-test")

    def test_reference_label_case_needs_no_cord_and_exports_nnunet(self) -> None:
        anatomy, rootlets, _cord = _case_arrays()
        affine = np.diag([0.8, 0.8, 0.8, 1.0])
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            raw = directory / "sub-test" / "anat"
            labels = directory / "derivatives" / "labels" / "sub-test" / "anat"
            raw.mkdir(parents=True)
            labels.mkdir(parents=True)
            anatomy_path = raw / "sub-test_T2w.nii.gz"
            rootlets_path = labels / "sub-test_T2w_label-rootlets_dseg.nii.gz"
            described_path = labels / "sub-test_T2w_desc-rater1_label-rootlets_dseg.nii.gz"
            nib.save(nib.Nifti1Image(anatomy, affine), anatomy_path)
            nib.save(nib.Nifti1Image(rootlets, affine), rootlets_path)
            nib.save(nib.Nifti1Image(rootlets, affine), described_path)

            cases = discover_reference_label_cases(directory)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "sub-test_T2w")
            self.assertIsNone(cases[0].cord_path)
            case = load_case(
                cases[0].anatomy_path,
                cases[0].rootlets_path,
                cases[0].cord_path,
                case_id=cases[0].case_id,
            )
            self.assertEqual(len(case.records), 4)
            self.assertEqual({record["side"] for record in case.records}, {"unassigned"})
            for index, record in enumerate(case.records):
                annotate_record(
                    record,
                    "dorsal" if index % 2 == 0 else "ventral",
                    reviewer_id="reviewer-a",
                    visible="yes",
                    confidence="high",
                    notes="",
                )

            dataset_directory = directory / "Dataset901_RootletDorsalVentral"
            exported = export_nnunet_case(case, dataset_directory, case.records)
            dataset_json = write_nnunet_dataset_json(dataset_directory)
            target = np.asanyarray(nib.load(exported["target"]).dataobj)

            self.assertEqual(set(np.unique(target)), {0, 1, 2})
            self.assertEqual(target.shape, rootlets.shape)
            self.assertEqual(
                nib.aff2axcodes(nib.load(exported["target"]).affine),
                ("R", "P", "I"),
            )
            self.assertTrue((dataset_directory / "imagesTr" / "sub-test_T2w_0000.nii.gz").is_file())
            self.assertTrue((dataset_directory / "imagesTr" / "sub-test_T2w_0001.nii.gz").is_file())
            self.assertEqual(
                json.loads(dataset_json.read_text())["labels"],
                {"background": 0, "dorsal": 1, "ventral": 2},
            )

    def test_nnunet_discovery_finds_image_and_rootlet_label_pairs(self) -> None:
        anatomy, rootlets, _cord = _case_arrays()
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            images = directory / "imagesTr"
            labels = directory / "labelsTr"
            images.mkdir()
            labels.mkdir()
            nib.save(
                nib.Nifti1Image(anatomy, np.eye(4)),
                images / "case-001_0000.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(rootlets, np.eye(4)), labels / "case-001.nii.gz"
            )

            cases = discover_nnunet_label_cases(directory)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "case-001")
            self.assertIsNone(cases[0].cord_path)

    def test_case_resume_and_dataset_export_preserve_cluster_support(self) -> None:
        anatomy, rootlets, cord = _case_arrays()
        affine = np.diag([0.8, 0.8, 0.8, 1.0])
        with tempfile.TemporaryDirectory() as directory_value:
            directory = Path(directory_value)
            anatomy_path = directory / "sub-test_T2w.nii.gz"
            rootlets_path = directory / "sub-test_label-rootlets_dseg.nii.gz"
            cord_path = directory / "sub-test_label-SC_seg.nii.gz"
            nib.save(nib.Nifti1Image(anatomy, affine), anatomy_path)
            nib.save(nib.Nifti1Image(rootlets, affine), rootlets_path)
            nib.save(nib.Nifti1Image(cord, affine), cord_path)

            case = load_case(
                anatomy_path, rootlets_path, cord_path, case_id="sub-test"
            )
            self.assertEqual(len(case.records), 4)
            self.assertEqual(
                {record["side"] for record in case.records}, {"left", "right"}
            )

            annotate_record(
                case.records[0],
                "dorsal",
                reviewer_id="reviewer-a",
                visible="yes",
                confidence="high",
                notes="clear posterior attachment",
            )
            annotate_record(
                case.records[1],
                "ventral",
                reviewer_id="reviewer-a",
                visible="yes",
                confidence="medium",
                notes="",
            )
            annotate_record(
                case.records[2],
                "mixed",
                reviewer_id="reviewer-a",
                visible="yes",
                confidence="low",
                notes="joined branches",
            )
            review_path = directory / "review.csv"
            write_review_csv(review_path, case.records)
            saved = read_review_csv(review_path)
            resumed = merge_saved_annotations(case.records, saved)
            self.assertEqual(resumed[0]["expert_class"], "dorsal")
            self.assertEqual(resumed[2]["notes"], "joined branches")

            export_directory = directory / "export"
            manifest = export_annotation_dataset(
                case, export_directory, case.records
            )
            self.assertEqual(manifest["class_counts"]["dorsal"], 1)
            self.assertEqual(manifest["class_counts"]["ventral"], 1)
            self.assertEqual(manifest["class_counts"]["mixed"], 1)
            self.assertEqual(manifest["class_counts"]["unreviewed"], 1)

            partition = np.zeros(rootlets.shape, dtype=np.int16)
            for label in (*EXPERT_CLASSES, "unreviewed"):
                image = nib.load(manifest["outputs"][label])
                self.assertEqual(nib.aff2axcodes(image.affine), ("R", "P", "I"))
                partition += np.asanyarray(image.dataobj).astype(np.int16)
            np.testing.assert_array_equal(partition, case.rootlets_rpi)

    def test_resume_rejects_a_different_cluster_inventory(self) -> None:
        current = [
            {
                "cluster_key": "L02-left-C01",
                "cluster_fingerprint": "aaa",
                "expert_class": "",
            }
        ]
        saved = [
            {
                "cluster_key": "L03-left-C01",
                "cluster_fingerprint": "bbb",
                "expert_class": "dorsal",
            }
        ]
        with self.assertRaisesRegex(ValueError, "do not match"):
            merge_saved_annotations(current, saved)

    def test_resume_rejects_changed_geometry_with_the_same_cluster_key(self) -> None:
        current = [
            {
                "cluster_key": "L02-left-C01",
                "cluster_fingerprint": "aaa",
                "expert_class": "",
            }
        ]
        saved = [
            {
                "cluster_key": "L02-left-C01",
                "cluster_fingerprint": "bbb",
                "expert_class": "dorsal",
            }
        ]
        with self.assertRaisesRegex(ValueError, "geometry"):
            merge_saved_annotations(current, saved)


if __name__ == "__main__":
    unittest.main()
