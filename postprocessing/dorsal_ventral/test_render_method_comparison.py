"""Tests for qualitative deterministic-method comparison rendering."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.render_method_comparison import run


class RenderMethodComparisonTest(unittest.TestCase):
    def test_renders_multiple_exact_partitions_on_shared_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            affine = np.diag([0.8, 0.8, 0.8, 1.0])
            image = np.linspace(0, 1, 11 * 13 * 7, dtype=np.float32).reshape(11, 13, 7)
            combined = np.zeros((11, 13, 7), dtype=np.int16)
            combined[2:9, 3:5, 1:6] = 2
            combined[2:9, 8:10, 1:6] = 2
            dorsal_a = np.zeros_like(combined)
            dorsal_a[2:9, 3:5, 1:6] = 2
            ventral_a = combined - dorsal_a
            dorsal_b = np.zeros_like(combined)
            dorsal_b[2:5, 3:5, 1:6] = 2
            dorsal_b[2:5, 8:10, 1:6] = 2
            ventral_b = combined - dorsal_b
            for name, data in {
                "image": image,
                "combined": combined,
                "dorsal-a": dorsal_a,
                "ventral-a": ventral_a,
                "dorsal-b": dorsal_b,
                "ventral-b": ventral_b,
            }.items():
                nib.save(nib.Nifti1Image(data, affine), directory / f"{name}.nii.gz")

            manifest = directory / "comparison.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "subject",
                        "session",
                        "image",
                        "combined",
                        "method",
                        "dorsal",
                        "ventral",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "subject": "sub-01",
                            "session": "ses-01",
                            "image": "image.nii.gz",
                            "combined": "combined.nii.gz",
                            "method": "v3",
                            "dorsal": "dorsal-a.nii.gz",
                            "ventral": "ventral-a.nii.gz",
                        },
                        {
                            "subject": "sub-01",
                            "session": "ses-01",
                            "image": "image.nii.gz",
                            "combined": "combined.nii.gz",
                            "method": "v4",
                            "dorsal": "dorsal-b.nii.gz",
                            "ventral": "ventral-b.nii.gz",
                        },
                    ]
                )

            outputs = run(manifest, directory / "rendered", panels=3)
            self.assertEqual(outputs, [directory / "rendered" / "sub-01_ses-01_method-comparison.png"])
            self.assertTrue(outputs[0].is_file())
            self.assertGreater(outputs[0].stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
