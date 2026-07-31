#!/usr/bin/env python3
"""Replay the rootlets early-stopping policy against an nnU-Net checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nnunet_custom_trainers.rootlets_early_stopping.early_stopping import (
    should_stop_early,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--min-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--min-delta", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    scores = np.asarray(checkpoint["logging"]["ema_fg_dice"], dtype=float)

    stop_after_epoch = None
    stop_state = None
    for completed_epochs in range(1, len(scores) + 1):
        stop, state = should_stop_early(
            scores[:completed_epochs],
            min_epochs=args.min_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        if stop:
            stop_after_epoch = completed_epochs
            stop_state = state
            break

    result = {
        "checkpoint": str(args.checkpoint),
        "recorded_epochs": len(scores),
        "absolute_best_epoch": int(np.nanargmax(scores)) + 1,
        "absolute_best_ema_fg_dice": float(np.nanmax(scores)),
        "policy": {
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "would_stop_early": stop_after_epoch is not None,
        "stop_after_epoch": stop_after_epoch,
        "last_significant_improvement_epoch": (
            None
            if stop_state is None
            or stop_state.last_significant_improvement_epoch is None
            else stop_state.last_significant_improvement_epoch + 1
        ),
        "significant_best": (
            None if stop_state is None else stop_state.significant_best
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
