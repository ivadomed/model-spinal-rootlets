"""End-to-end test for the guarded and resumable Marseille batch runner."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


class MarseilleRunnerTest(unittest.TestCase):
    def test_worker_runs_in_slot_and_resumes_without_resegmenting(self) -> None:
        runner = Path(__file__).with_name("run_marseille_romane.sh").resolve()
        code_root = runner.parents[2]
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            dataset = directory / "dataset"
            output = directory / "output"
            sct_directory = directory / "sct"
            fake_bin = directory / "bin"
            sct_directory.joinpath("bin").mkdir(parents=True)
            fake_bin.mkdir()
            affine = np.diag([0.8, 0.8, 0.8, 1.0])
            for session in ("ses-01", "ses-02"):
                input_directory = dataset / "sub-01" / session / "anat"
                input_directory.mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(np.zeros((41, 41, 25), dtype=np.int16), affine),
                    input_directory / f"sub-01_{session}_T2w.nii.gz",
                )

            slot_log = directory / "slot.log"
            set_slot = fake_bin / "set_slot"
            set_slot.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    slot="$1"
                    shift
                    printf '%s\n' "$slot" >> "$FAKE_SLOT_LOG"
                    exec "$@"
                    """
                )
            )
            set_slot.chmod(0o755)

            fake_sct = sct_directory / "bin" / "sct_deepseg"
            fake_sct.write_text(
                f"#!{sys.executable}\n"
                + textwrap.dedent(
                    """
                    import os
                    import sys
                    from pathlib import Path

                    import nibabel as nib
                    import numpy as np

                    task = sys.argv[1]
                    input_path = Path(sys.argv[sys.argv.index("-i") + 1])
                    output_path = Path(sys.argv[sys.argv.index("-o") + 1])
                    image = nib.load(input_path)
                    shape = image.shape
                    data = np.zeros(shape, dtype=np.int16)
                    if task == "spinalcord":
                        xx, yy = np.ogrid[:shape[0], :shape[1]]
                        cord = ((xx - 20) / 3.0) ** 2 + ((yy - 20) / 4.0) ** 2 <= 1
                        data[:, :, :] = cord[:, :, None]
                    elif task == "rootlets":
                        data[24:35, 15:17, 5:9] = 2
                        data[24:35, 24:26, 5:9] = 2
                        data[6:17, 15:17, 5:9] = 2
                        data[6:17, 24:26, 5:9] = 2
                        data[24:34, 15:17, 15:19] = 3
                        data[24:34, 24:26, 15:19] = 3
                        data[7:17, 15:17, 15:19] = 3
                        data[7:17, 24:26, 15:19] = 3
                    else:
                        raise SystemExit(f"unexpected task: {task}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(data, image.affine, image.header), output_path)
                    with (Path(__file__).parents[1] / "calls.log").open("a") as stream:
                        stream.write(f"{task},{os.environ.get('CUDA_VISIBLE_DEVICES')},"
                                     f"{os.environ.get('SCT_USE_GPU')}\\n")
                    print(f"sct_deepseg {task} -i {input_path} -o {output_path}")
                    print(f"Total runtime; {1.25 if task == 'spinalcord' else 2.5} seconds.")
                    """
                )
            )
            fake_sct.chmod(0o755)
            fake_sct_python = sct_directory / "fake_sct_python"
            fake_sct_python.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    test "${CUDA_VISIBLE_DEVICES:-}" = "0"
                    test "${SCT_USE_GPU:-}" = "1"
                    exit 0
                    """
                )
            )
            fake_sct_python.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                    "RESOURCE_SLOT_BOOKED": "1",
                    "FAKE_SLOT_LOG": str(slot_log),
                    "DATASET_ROOT": str(dataset),
                    "OUTPUT_ROOT": str(output),
                    "CODE_ROOT": str(code_root),
                    "SCT_DIR": str(sct_directory),
                    "PYTHON_BIN": sys.executable,
                    "SCT_PYTHON_BIN": str(fake_sct_python),
                }
            )
            command = [
                str(runner),
                "--run",
                "--compute",
                "gpu",
                "--slot",
                "0",
                "--cuda-device",
                "0",
            ]
            first = subprocess.run(
                command,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)

            calls_path = sct_directory / "calls.log"
            calls = calls_path.read_text().splitlines()
            self.assertEqual(len(calls), 4)
            self.assertEqual(set(calls), {"spinalcord,0,1", "rootlets,0,1"})
            self.assertEqual(slot_log.read_text().splitlines(), ["0"])

            with (output / "manifest.csv").open(newline="") as stream:
                manifest_rows = list(csv.DictReader(stream))
            self.assertEqual(len(manifest_rows), 2)
            summary = json.loads((output / "session_consistency_summary.json").read_text())
            self.assertEqual(summary["complete_subject_pairs"], 1)
            self.assertEqual(summary["abs_dorsal_fraction_difference"]["median"], 0.0)
            self.assertTrue((output / "session_qc" / "sub-01_paired-session_qc.png").is_file())
            runtime = json.loads((output / "inference_runtime_summary.json").read_text())
            self.assertEqual(runtime["scans_with_complete_timings"], 2)
            self.assertEqual(runtime["compute_modes"], ["gpu"])
            self.assertEqual(runtime["rootlets_seconds"]["median"], 2.5)

            second = subprocess.run(
                command,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(calls_path.read_text().splitlines(), calls)
            self.assertEqual(slot_log.read_text().splitlines(), ["0", "0"])

            fake_sct_python.write_text("#!/usr/bin/env bash\nexit 1\n")
            failed = subprocess.run(
                command,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Refusing silent CPU fallback", failed.stderr)

            cpu_output = directory / "output-cpu"
            environment["OUTPUT_ROOT"] = str(cpu_output)
            cpu_command = [
                str(runner),
                "--run",
                "--compute",
                "cpu",
                "--slot",
                "1",
            ]
            cpu_run = subprocess.run(
                cpu_command,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cpu_run.returncode, 0, cpu_run.stdout + cpu_run.stderr)
            self.assertEqual(
                set(calls_path.read_text().splitlines()[-4:]),
                {"spinalcord,None,None", "rootlets,None,None"},
            )
            self.assertEqual(slot_log.read_text().splitlines(), ["0", "0", "0", "1"])


if __name__ == "__main__":
    unittest.main()
