# Fold-Rank Project

This repository provides a **machine learning pipeline for predicting pregnancy success outcomes** based on **IVF/DI treatment data**.  
The codebase is a refactored and modularized version of the original notebook implementation, covering **data preprocessing, model training, cross-validation, and weighted rank ensemble logic**, organized as a clean Python package.

---

## Development Environment

- Designed for a **Colab + GitHub workflow**, allowing experiments and model training in Colab while maintaining version-controlled code in GitHub.
- Required Python libraries are listed in `requirements.txt` and can be installed easily using:

    pip install -r requirements.txt

---

## Directory Structure

    ├── src/                     # Core executable code (Python package)
    │   ├── __init__.py          # Package initialization
    │   ├── dataset.py           # Data loading and preprocessing
    │   ├── model.py             # Model construction utilities
    │   ├── trainer.py           # Training loop (K-Fold, per-model training)
    │   ├── losses.py            # Weighted rank ensemble and shared functions
    │   └── utils.py             # Utility functions (e.g., random seed fixing)
    ├── train.py                 # Training entry script (reads config, calls src)
    ├── inference.py             # Inference script (loads models, creates submission)
    ├── configs/                 # Configuration files (no code changes required)
    │   ├── train.yaml           # Training and experiment settings
    │   └── submit.yaml          # Inference and submission settings
    ├── data/                    # Input data (example files provided)
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── assets/                  # Model artifacts (Git LFS recommended)
    │   └── README.md            # Asset usage notes
    ├── outputs/                 # Output files (e.g., submission results)
    ├── requirements.txt         # Python dependency list
    ├── .gitignore               # Files and directories to ignore
    └── .gitattributes           # Git attributes and LFS configuration

---

## Quick Start

### 1. Install Dependencies

    pip install -r requirements.txt

---

### 2. Prepare the Dataset

- Replace the example files in `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv` with the actual dataset.
- The training data must include a **pregnancy success label column**.

---

### 3. Train Models

    python train.py --config configs/train.yaml

- Model hyperparameters, K-Fold settings, and ensemble weights can be adjusted in `configs/train.yaml`.
- After training:
  - AUC scores are printed to the console
  - Trained models are saved to the `assets/` directory
  - Final predictions are written to `outputs/submission.csv`

---

### 4. Inference & Submission Generation

    python inference.py --config configs/submit.yaml

- Specify which trained models and weights to use in `configs/submit.yaml`.
- The script loads saved models, performs predictions on the test data, applies the ensemble strategy, and generates the final submission file.

---

## Customization

- **Hyperparameter Tuning**  
  Modify per-model hyperparameters under the `models` section in `configs/train.yaml`.

- **Ensemble Strategy**  
  Edit or replace the `weighted_rank_ensemble` function in `src/losses.py` to experiment with different ensemble methods.

- **Data Preprocessing**  
  The preprocessing logic in `src/dataset.py` is a functionalized version of the original notebook workflow.  
  Feature engineering and column handling can be customized based on dataset characteristics.

---

## Notes

- The `assets/` directory stores large binary files such as trained models.  
  Use **Git LFS** or ensure these files are excluded via `.gitignore` to avoid accidental commits.
- Example datasets are provided for structural reference only.  
  Replace them with the actual competition data before training or inference.

---

## License

This project is intended for **educational and research purposes only**.  
Any other usage requires explicit permission from the original author.
