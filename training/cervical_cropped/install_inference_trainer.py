#!/usr/bin/env python3
"""Install the Dataset403 trainer class required to load its checkpoint."""

from __future__ import annotations

import importlib
import shutil
from os.path import join
from pathlib import Path

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


TRAINER = "nnUNetTrainer_2000epochsEarlyStopping"


def find_source() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (
        here / "nnunet_custom_trainers" / "rootlets_early_stopping",
        here.parent / "nnunet_custom_trainers" / "rootlets_early_stopping",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find nnunet_custom_trainers/rootlets_early_stopping")


def main() -> None:
    source = find_source()
    destination = (
        Path(nnunetv2.__path__[0])
        / "training"
        / "nnUNetTrainer"
        / "variants"
        / "rootlets_early_stopping"
    )
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "early_stopping.py", f"{TRAINER}.py"):
        shutil.copy2(source / name, destination / name)

    importlib.invalidate_caches()
    trainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        TRAINER,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer is None:
        raise RuntimeError(f"Could not discover {TRAINER}")
    print(f"Installed: {trainer.__module__}.{trainer.__name__}")


if __name__ == "__main__":
    main()
