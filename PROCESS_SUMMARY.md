# Fantasy MoE Workflow – High-Level Summary

This document explains how the fantasy projection pipeline operates end-to-end, highlighting the temporal split strategy, the way prior-season data is leveraged, and how the system handles missing inputs.

---

## 1. Data Inputs

For each position group we supply a season-level CSV:

| Position | File | Required Targets | Notes |
|----------|------|------------------|-------|
| Quarterbacks | `QB_features.csv` | `fp`, `games` | Additional passing/rushing features |
| Running Backs | `RB_features.csv` | `fp`, `games` | Rushing/receiving mix |
| Wide Receivers | `WR_features.csv` | `fp`, `games` | Target/air yards metrics |
| Tight Ends | `TE_features.csv` | `fp`, `games` | Receiving usage |
| Kickers | `kicker_features.csv` | `ppg` (per-game fantasy points) | Field-goal and PAT stats |
| Defenses / Special Teams | `Defense_features.csv` | `fp`, `games` | Takeaways, points allowed, etc. |

All files contain a `season` column (e.g., 1999–2024) along with numeric features describing that season’s performance.

---

## 2. Temporal Pipeline

1. **Lagging features by one season**  
   - When `load_position_data` ingests a CSV, it sorts each `fantasy_player_id` (player or defense) by season, then shifts every numeric feature down one row.  
   - For season *N*, we therefore feed the models season *N – 1* stats.  
   - `feature_source_season = season - 1` is stored for clarity.

2. **Temporal split definition**  
   - Training set: every row with `season < test_year`.  
   - Validation set: optional season specified by `val_year` (disabled in the current run).  
   - Test set: rows where `season == test_year` (default 2024).

3. **Effect**  
   - When we predict 2024, the features reflect 2023 performance (or imputed medians if no 2023 data exists).  
   - If we targeted a different year, the shift dynamically aligns with that prediction year.

---

## 3. Handling Missing or New Players

1. **Lag-induced gaps**  
   - Shifting features creates `NaN` for the earliest season per player/team.  
   - Rather than dropping those records, we fill gaps later via median imputation.

2. **Rookies/new teams**  
   - A rookie with no prior-season data inherits the positional median for each feature.  
   - They remain in the prediction set with a “typical” statistical profile.

3. **Consistent feature sets**  
   - Training defines the canonical list of feature columns per position.  
   - Validation/test data reuses that list, adding any missing columns as nulls before imputation to avoid shape mismatches.

---

## 4. Model Selection (Mixture of Experts)

1. **Experts**  
   - RandomForestRegressor  
   - GradientBoostingRegressor  
   - Ridge Regression  
   - Linear Regression  
   - XGBoost Regressor (if available in the environment)

2. **Training loop**  
   - Each expert trains on the training subset (with validation fallback).  
   - MAE, RMSE, and R² are logged on the validation set.  
   - The expert with the lowest MAE is selected and re-trained on the full training data.

3. **Evaluation**  
   - Best model predictions are compared against the 2024 test set.  
   - We report MAE, RMSE, MAPE, and R² plus per-player projected vs actual points.

---

## 5. Outputs

1. **Console**  
   - Training metrics per expert (per position).  
   - Test-set metrics (MAE/RMSE/R²/MAPE).  
   - Top-10 projected players by position.

2. **CSV files (saved beside the script)**  
   - `fantasy_predictions_2024_full.csv`: includes projected and actual average fantasy points per week with position/season metadata.  
   - `fantasy_predictions_2024.csv`: simplified version containing only the required columns (`Player Name`, `Player Position`, `Projected Average Fantasy Points Per Week`).

3. **Model objects**  
   - The pipeline keeps a dictionary of trained `MixtureOfExperts` instances keyed by position should we want to inspect or reuse them later in the session.

---

## 6. Key Assumptions & Notes

* **Seasonal availability**: the default run expects data through 2024. If the latest season is earlier, the script automatically adjusts the test year to the most recent season.  
* **Imputation**: median values are computed per column after any new features are added/matched, ensuring a consistent feature vector.  
* **Scalability**: as long as new CSVs respect the `season + fp/games` pattern, the pipeline can assimilate additional positions with minimal code change.  
* **Environment**: XGBoost is optional—if the import fails, a warning is printed and the rest of the experts remain.  
* **Performance**: R² values for some positions (K, DST) can be low because of the randomness inherent in those positions. The system still returns unbiased predictions thanks to lagged inputs and validation-based model selection.

---

With this workflow, fantasy projections are generated in a forward-looking manner: every prediction uses only information that would have been available entering the season, and all results are consolidated in easily consumable CSV outputs alongside human-friendly reports.



