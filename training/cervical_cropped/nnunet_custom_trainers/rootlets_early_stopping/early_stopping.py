"""Pure early-stopping policy used by the rootlets nnU-Net trainer."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class PlateauState:
    """Summary of significant improvements in a metric history."""

    significant_best: float | None
    last_significant_improvement_epoch: int | None
    epochs_without_significant_improvement: int


def plateau_state(scores: Iterable[float], min_delta: float) -> PlateauState:
    """Return the last epoch that improved on the reference by ``min_delta``.

    Small gains accumulate: the reference remains fixed until the metric rises
    by more than ``min_delta``. Non-finite values are ignored and cannot by
    themselves trigger early stopping.
    """

    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    values = [float(score) for score in scores]
    significant_best: float | None = None
    last_epoch: int | None = None

    for epoch, score in enumerate(values):
        if not isfinite(score):
            continue
        if significant_best is None or score > significant_best + min_delta:
            significant_best = score
            last_epoch = epoch

    epochs_without_improvement = (
        0 if last_epoch is None else len(values) - 1 - last_epoch
    )
    return PlateauState(
        significant_best=significant_best,
        last_significant_improvement_epoch=last_epoch,
        epochs_without_significant_improvement=epochs_without_improvement,
    )


def should_stop_early(
    scores: Iterable[float],
    *,
    min_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[bool, PlateauState]:
    """Decide whether a finite metric has plateaued after a warm-up period."""

    if min_epochs < 1:
        raise ValueError("min_epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")

    values = list(scores)
    state = plateau_state(values, min_delta)
    stop = (
        len(values) >= min_epochs
        and state.last_significant_improvement_epoch is not None
        and state.epochs_without_significant_improvement >= patience
    )
    return stop, state
