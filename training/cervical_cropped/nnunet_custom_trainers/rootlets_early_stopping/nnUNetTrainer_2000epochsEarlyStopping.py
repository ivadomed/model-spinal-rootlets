"""A conservative 2,000-epoch nnU-Net trainer with plateau stopping."""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_2000epochs,
)

from .early_stopping import should_stop_early


class nnUNetTrainer_2000epochsEarlyStopping(nnUNetTrainer_2000epochs):
    """Stop only after a long, materially flat EMA pseudo-Dice plateau.

    The thresholds were checked retrospectively against two rootlets histories:
    the rule would preserve all 2,000 epochs of the released uncropped
    ``fold_all`` model, whose metric improved through epoch 1,998, while
    stopping the earlier cropped fold-0 run at approximately epoch 1,445 after
    its last material gain.
    """

    min_epochs_before_early_stopping = 1000
    early_stopping_patience = 400
    early_stopping_min_delta = 0.002

    def run_training(self):
        self.on_train_start()
        self.print_to_log_file(
            "Early stopping enabled:",
            f"metric=ema_fg_dice, min_epochs={self.min_epochs_before_early_stopping},",
            f"patience={self.early_stopping_patience},",
            f"min_delta={self.early_stopping_min_delta},",
            f"max_epochs={self.num_epochs}",
        )
        if self.fold == "all":
            self.print_to_log_file(
                "WARNING: fold_all uses all training cases for both training and "
                "nnU-Net pseudo-validation. Its plateau signal measures optimization "
                "convergence, not held-out generalization."
            )

        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(
                        self.validation_step(next(self.dataloader_val))
                    )
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            stop, state = should_stop_early(
                self.logger.my_fantastic_logging["ema_fg_dice"],
                min_epochs=self.min_epochs_before_early_stopping,
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
            )
            if stop:
                self.print_to_log_file(
                    "Early stopping:",
                    f"completed_epochs={self.current_epoch},",
                    "last_significant_improvement_epoch="
                    f"{state.last_significant_improvement_epoch},",
                    "epochs_without_significant_improvement="
                    f"{state.epochs_without_significant_improvement},",
                    f"significant_best={state.significant_best:.6f}",
                )
                break

        self.on_train_end()
