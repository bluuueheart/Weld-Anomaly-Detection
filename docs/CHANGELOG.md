# CHANGELOG

This file consolidates recent updates and code-change notes (V4 optimization round). It is a curated, human-friendly timeline and summary—original detailed docs are archived under `docs/archive/`.

## Summary (Top-level)

- Date: 2025-10-21
- Release: V4 — Early Stop & Feature-Level MixUp
- Goal: Fix early-overfitting where best validation models appear at Epoch 4-10 and later epochs overfit.
- Key strategies:
  - Feature-level MixUp (config `use_mixup: True`, `mixup_alpha: 0.2`)
  - Faster warmup and learning rate adjustment (`learning_rate: 5e-5`, `warmup_epochs: 5`)
  - Stronger L2 (`weight_decay: 1e-2`) and dropout tuning
  - Aggressive Early Stopping (`early_stopping_patience: 8`)
  - Removed: Label Smoothing (user-requested removal — no longer applied in SupCon)

## Timeline

- 2025-10-21 15:00 — V4 strategy implemented
  - Add Feature-Level MixUp
  - Adjust LR scheduling and warmup
  - Implement aggressive Early Stopping
  - Add validation & verification scripts

- 2025-10-21 16:00 — Label Smoothing removed
  - Per user decision, `label_smoothing` parameter and related logic removed from configs and loss implementation
  - Keep MixUp + Early Stop
  - Updated docs and verification scripts

## Code-level highlights

- configs/train_config.py
  - Added/kept: `use_mixup`, `mixup_alpha`, `early_stopping_patience`, altered lr/warmup/min_lr/weight_decay
  - Removed: `label_smoothing` (no longer present)

- src/train.py
  - Added: `_mixup_features()` implementation (feature-level MixUp)
  - Integrated MixUp into training loop when `use_mixup=True`
  - `_setup_loss()` no longer passes `label_smoothing` to SupConLoss

- src/losses.py
  - Restored to standard SupConLoss (no label smoothing) — behavior identical to original when `label_smoothing` absent

- scripts/verify_v4_config.sh
  - Script added/updated to verify critical config keys and V4 readiness

## How to use

- Keep top-level docs small. Recommended docs to read:
  - `README.md` (root)
  - `docs/QUICKSTART.md`
  - `docs/CHANGELOG.md` (this file)

- All other historical/auxiliary docs are copied into `docs/archive/` for reference.

## Files archived

The following files were copied into `docs/archive/` (originals remain in `docs/` until you choose to delete them):

- ANTI_OVERFITTING_V3.md
- ANTI_OVERFITTING_V4_EARLY_STOP.md
- LOGGING_AND_VISUALIZATION.md
- PROGRESS_QUADMODAL.md
- PROJECT_STRUCTURE.md
- STEP3_FUSION_REPORT.md
- TRAINABLE_MODULES_DROPOUT.md
- TRAINING_LOG_20251020.md
- TRAINING_OPTIMIZATION.md
- V4_CODE_CHANGES.md
- UPDATE_2025-10-21_V4_EARLY_STOP.md

(You can remove the originals from `docs/` after verifying the archive. Recommended: keep `QUICKSTART.md` and `README.md` in their places.)

## Notes

- I did not delete any original files. I only created this consolidated changelog and archived copies. Please confirm which originals you want removed; I list recommended deletions below for your convenience.

---

Generated: 2025-10-30
Branch: main
