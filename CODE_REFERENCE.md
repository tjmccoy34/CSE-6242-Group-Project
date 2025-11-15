# Code Reference – `moe_fantasy_predictions.py`

| Component | Lines | Purpose |
|-----------|-------|---------|
| Module Docstring | 1–4 | High-level description of the script and supported positions. |
| Imports | 6–22 | Loads pandas/numpy, scikit-learn utilities, warnings, and optionally xgboost (`XGBOOST_AVAILABLE`). |
| `MixtureOfExperts` class | 24–167 | Wraps a suite of regression experts and handles training, validation scoring, best-model selection, re-training, and prediction. |
| `load_position_data(file_path, position)` | 170–269 | Reads a CSV, ensures required columns, constructs `fp_per_week`, helps with missing IDs/names, lags numeric features by one season, and marks `feature_source_season`. |
| `prepare_features(df, required_features=None)` | 272–322 | Builds the numeric feature matrix, optionally enforcing a pre-defined feature list, converts to numeric, and fills missing values via median. |
| `temporal_split(df, test_year=2024, val_year=None)` | 325–341 | Splits the data chronologically into train, optional validation, and test sets. |
| `evaluate_model(model, X_test, y_test, position)` | 344–365 | Uses the trained MoE model to compute MAE, RMSE, R², and MAPE on the test set and returns predictions. |
| `train_position_model(position, file_path, test_year=2024)` | 368–452 | End-to-end pipeline for one position: load, split, prep features, train MoE, evaluate, and package predictions. |
| `main()` | 455–627 | Discovers all positional CSVs, iteratively trains/evaluates each position, aggregates predictions, writes CSV outputs, and prints summary stats plus top-10 projections. |
| Script entry (`if __name__ == "__main__":`) | 630–638 | Calls `main()` and wraps up with a completion message. |

## Detailed Notes

### 1. `MixtureOfExperts`
- **Initialization** (24–37): Stores the position label, allocates dicts, and sets flags.  
- **`_initialize_experts`** (39–75): Builds regressors – Random Forest, Gradient Boosting, Ridge, Linear, and optionally XGBoost; also sets up scalers for linear models.  
- **`fit`** (77–148):  
  - Splits train data into internal train/validation if none provided.  
  - Trains each expert and records MAE/RMSE/R² on the validation set.  
  - Selects the lowest-MAE model, retrains it on the full training set, and marks the class as trained.  
- **`predict`** (150–159): Applies the best model (with scaling if required) to new matrices.  
- **`get_best_model_info`** (161–167): Returns status metadata (position, model name).

### 2. Data Loading & Feature Engineering
- **`load_position_data`** (170–269):  
  - Converts relative paths to absolute by searching the script directory, current working directory, and `Code Updates/`.  
  - Reads the positional CSV; for kickers uses `ppg` as the target, otherwise computes `fp_per_week = fp / games`.  
  - Guarantees `fantasy_player_id` and `fantasy_player_name` exist (defenses use `team`).  
  - Sorts by player/team and season, then shifts each numeric feature by one season to enforce previous-year inputs.  
  - Adds `feature_source_season = season - 1` so it’s clear which year those inputs came from.

- **`prepare_features`** (272–322):  
  - Excludes identifiers, targets, and derived columns from the feature matrix.  
  - When `required_features` is supplied (validation/test), it reorders/reshapes to match the training feature list and creates missing columns as NaNs.  
  - Converts all features to numeric, fills NaNs via median, and returns `(X, y, feature_names)`.

### 3. Temporal Management & Evaluation
- **`temporal_split`** (325–341):  
  - Training data includes all seasons prior to `test_year`.  
  - Optional `val_year` can carve out a specific season as validation.  
  - Test data is exactly the `test_year`.  
- **`evaluate_model`** (344–365): Runs the best model on the test set, computing MAE, RMSE, MAPE, and R², and returns the metrics plus predictions.

### 4. Position Pipeline
- **`train_position_model`** (368–452):  
  - Loads processed data, performs the temporal split, and triggers feature preparation.  
  - If the test set is empty (e.g., future season absent), the code falls back to the latest available season.  
  - Uses the MoE class to train & evaluate, then appends predictions to the test dataframe.

### 5. Script Orchestration
- **`main`** (455–627):  
  - Searches for: `QB_features.csv`, `RB_features.csv`, `WR_features.csv`, `TE_features.csv`, `kicker_features.csv`, and `Defense_features.csv`.  
  - Iterates through positions, calling `train_position_model`.  
  - Collects predictions from every position and writes two outputs:  
    - `fantasy_predictions_2024_full.csv` (projected + actual).  
    - `fantasy_predictions_2024.csv` (projected only).  
  - Prints validation/test metrics and top-10 projected players per position for quick inspection.

### 6. Entry Point
- **`if __name__ == "__main__":`** (630–638):  
  - Prints a header, runs `main()`, and displays a completion footer.

---

This reference serves as a quick map across the script so you can locate specific logic, understand its role in the pipeline, and extend or debug modules with confidence.



