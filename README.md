# CLINC_OOS_RunRegistry_v1
Mini project: Intent classification on CLINC-OOS with run registry, metrics, and OOS error analysis.

# CLINC-OOS Intent Classification (Run Registry Mini Project)

## Overview
This repository is a hands-on mini project to practice a practical AI engineering workflow:
not only model training, but also logging, run comparison, error analysis, and selecting final hyperparameters.

- Task: Intent classification with OOS (out-of-scope)
- Model: distilbert-base-uncased (Hugging Face Transformers)
- Environment: macOS (Apple Silicon / MPS)
- Style: Run Registry (each run is saved as an isolated folder with inputs/outputs)

## Dataset
- Hugging Face dataset: `clinc_oos` (config: `plus`)
- Labels: 150+ in-scope intents + `oos`

## Run Registry Outputs
Each run is stored under `runs/run_YYYYMMDD_HHMMSS/` and includes:

- `run_cfg.json` : training configuration (inputs)
- `train_log.tsv`: training/validation metrics per epoch
- `metrics.json` : final summary metrics (outputs)
- `errors.tsv`   : misclassified samples (for failure analysis)
- `confusion_matrix_top25.png` : readable confusion matrix (top-K)
- `confusion_matrix_oos_receiver.png` : OOS → predicted intent distribution
- `model/` : saved model artifacts

A summary table is maintained in:
- `runs/registry.csv`

## How to Run (High Level)
1. `conda activate ai-train-2025`
2. Start Jupyter Lab and run the notebook top-to-bottom
3. Check `runs/` for generated artifacts per run
4. Compare runs using `runs/registry.csv`

## What I Did in This Update
I executed three runs, compared training logs and validation trends, and decided:

- `max_len`: 128 is preferable to 256 (similar/better validation behavior with lower cost)
- `epochs`: the best validation region is around epoch ~4–5; longer training shows overfitting tendency
- I performed OOS error analysis and identified top “receiver intents” when the model fails OOS detection

## Notes (AI assistance)
This mini project was developed with support from an AI assistant (ChatGPT) as part of my training (iteration and improvement) to become an AI Engineer.
