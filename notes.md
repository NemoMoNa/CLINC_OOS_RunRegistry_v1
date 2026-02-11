# Experiment Notes — CLINC-OOS Run Registry

## 0) Goal
Decide practical hyperparameters by reading logs and performing failure analysis.

Primary decision targets:
- `max_len` (128 vs 256)
- `epochs` (overfit boundary / best stopping point)
- OOS failure patterns (receiver intents)

## 1) Runs Executed
I ran 3 experiments and compared their logs:

- Run-A: baseline (max_len=128, epochs=3)
- Run-B: longer training (max_len=128, epochs=8)
- Run-C: longer training + longer context (max_len=256, epochs=8)

Actual run folders are under: `runs/run_YYYYMMDD_HHMMSS/`

## 2) Evidence from train_log.tsv
- Training loss decreases as epochs increase (expected).
- Validation metrics improve until a certain point, then plateau or worsen.
- This indicates an overfitting boundary. A practical choice is to stop near the best validation region.

## 3) Final Hyperparameter Decision

### 3.1 max_len (128 vs 256)
**Decision:** choose `max_len = 128`

**Reason:**
- Validation behavior is comparable (or better) than 256 in my runs
- 128 is cheaper (faster training & inference), so it is preferable if performance is similar

### 3.2 epochs
**Decision:** best region is around **epoch 4–5** (based on validation loss trend)

**Reason:**
- Past that point, val_loss begins to fluctuate/worsen while train_loss still decreases
- This is typical overfitting behavior

Practical implementation choices:
- Option A: set `epochs=5` (simple)
- Option B: set `epochs=8` with early stopping (robust)

## 4) OOS Error Analysis

### 4.1 Definition
“OOS analysis” here means:
- Filter `errors.tsv` where `true_label == "oos"`
- Count where the model incorrectly sends those samples (`pred_label` top-N)

This identifies the **receiver intents** absorbing OOS errors.

### 4.2 Observation (from my run)
Top receiver intents (example):
- smart_home
- directions
- jump_start
- travel_suggestion
- ...

Interpretation:
- When the model fails OOS detection, it tends to map to a few frequent/generic intents.
- This is a practical improvement target (e.g., strengthen OOS decision boundary or introduce a 2-stage approach).

## 5) Next Practical Improvements

### Option 1: Confidence logging (recommended)
- Save `max_prob` (confidence score) into `errors.tsv`
- Analyze **high-confidence wrong** samples (most dangerous in production)

### Option 2: Two-stage classification (OOS head)
- 1st model: `in_scope` vs `oos` (binary)
- 2nd model: classify only in-scope into 150 intents

Note: `in_scope` means “not oos” (a concept derived from labels), not a dataset column name.

## 6) “Final Model” rule (proposal)
- Primary: best `val_f1_macro`
- Tie-break: lower `val_loss`
- Tie-break: lower inference cost (smaller `max_len`)
