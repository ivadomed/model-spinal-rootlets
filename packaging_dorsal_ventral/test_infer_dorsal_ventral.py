from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from packaging_dorsal_ventral.infer_dorsal_ventral import _partition, _save_native


class DorsalVentralPackageTest(unittest.TestCase):
    def test_partition_clips_prediction_and_preserves_levels(self) -> None:
        rootlets = np.zeros((8, 10, 4), dtype=np.int16)
        rootlets[1:3, 2:5, 1:3] = 2
        rootlets[5:7, 6:9, 1:3] = 7
        segmentation = np.zeros(rootlets.shape, dtype=np.uint8)
        segmentation[rootlets == 2] = 1
        probabilities = np.zeros((3, *rootlets.shape), dtype=np.float32)
        probabilities[2, rootlets == 7] = 0.9

        class_map, dorsal, ventral = _partition(
            rootlets, segmentation, probabilities
        )

        np.testing.assert_array_equal(dorsal + ventral, rootlets)
        self.assertTrue(np.all(dorsal[rootlets == 2] == 2))
        self.assertTrue(np.all(ventral[rootlets == 7] == 7))
        self.assertTrue(np.all(class_map[rootlets == 0] == 0))

    def test_save_native_restores_original_orientation(self) -> None:
        shape = (5, 6, 7)
        affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        original = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)
        original_rpi = original.as_reoriented(
            nib.orientations.ornt_transform(
                nib.orientations.io_orientation(original.affine),
                nib.orientations.axcodes2ornt(("R", "P", "I")),
            )
        )
        data_rpi = np.zeros(original_rpi.shape, dtype=np.uint8)
        data_rpi[1, 2, 3] = 2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.nii.gz"
            _save_native(data_rpi, original, output, np.uint8)
            saved = nib.load(output)
            saved_values = np.asanyarray(saved.dataobj)

        self.assertEqual(saved.shape, original.shape)
        np.testing.assert_allclose(saved.affine, original.affine)
        self.assertEqual(int(np.count_nonzero(saved_values)), 1)


if __name__ == "__main__":
    unittest.main()
