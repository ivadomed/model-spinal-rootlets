#!/bin/bash
# Install the repository-owned custom trainer into the pinned Romane environment.

set -euo pipefail

readonly PROJECT=/home/kuanyiw/projects/rootlets/model-spinal-rootlets
readonly PYTHON=/home/kuanyiw/.conda/envs/rootlets-romane/bin/python
readonly SOURCE="${PROJECT}/training/cervical_cropped/nnunet_custom_trainers/rootlets_early_stopping"
readonly TRAINER=nnUNetTrainer_2000epochsEarlyStopping

destination=$(
    "$PYTHON" - <<'PY'
from pathlib import Path
import nnunetv2

print(
    Path(nnunetv2.__path__[0])
    / "training"
    / "nnUNetTrainer"
    / "variants"
    / "rootlets_early_stopping"
)
PY
)

mkdir -p "$destination"
install -m 0644 "${SOURCE}/__init__.py" "$destination/__init__.py"
install -m 0644 "${SOURCE}/early_stopping.py" "$destination/early_stopping.py"
install -m 0644 \
    "${SOURCE}/nnUNetTrainer_2000epochsEarlyStopping.py" \
    "$destination/nnUNetTrainer_2000epochsEarlyStopping.py"

"$PYTHON" - "$TRAINER" <<'PY'
import sys
from os.path import join

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

trainer_name = sys.argv[1]
trainer = recursive_find_python_class(
    join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
    trainer_name,
    "nnunetv2.training.nnUNetTrainer",
)
if trainer is None:
    raise SystemExit(f"Could not discover {trainer_name}")
print(f"Installed and discovered {trainer.__module__}.{trainer.__name__}")
PY
