"""
Mixture of Experts (MoE) Model for Fantasy Football Point Predictions
Predicts fantasy points per week for QB, RB, WR, TE, K, and DST positions
"""
# xgboost is imported to allow the code to run in case a peer reviewer doesn't have it installed.
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    
    XGBOOST_AVAILABLE = True
    import xgboost as xgb
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. XGBoost model will be skipped.")

class MixtureOfExperts:
    """
    Mixture of Experts model that sets up multiple models
    to predict fantasy football points per week.
    """
    
    def __init__(self, position='QB'):
        self.position = position
        self.experts = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.expert_weights = {}
        self.trained = False
        
    def _initialize_experts(self):
        """Initialize multiple expert models"""
        self.experts = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.experts['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # Initialize scalers for models that need scaling
        for model_name in ['ridge', 'linear']:
            self.scalers[model_name] = StandardScaler()
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):

        #Training all of our models for each position. We also are completing comparison of the models to allow for model selection.

        self._initialize_experts()
        
        # If no validation set provided, split training data
        if X_val is None or y_val is None:
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        else:
            X_train_fit, y_train_fit = X_train, y_train
        
        # Train each expert and evaluate performance
        model_scores = {}
        model_predictions = {}
        
        for model_name, model in self.experts.items():
            try:
                # Scale features for models
                if model_name in self.scalers:
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train_fit)
                    X_val_scaled = self.scalers[model_name].transform(X_val)
                    model.fit(X_train_scaled, y_train_fit)
                    y_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train_fit, y_train_fit)
                    y_pred = model.predict(X_val)
                
                # Calculate comparisons
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                # Use negative MAE as score
                model_scores[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'score': -mae
                }
                
                model_predictions[model_name] = y_pred
                
                print(f"  {model_name:20s} - MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                model_scores[model_name] = {'score': -np.inf}
        
        # Select best model based on MAE
        if model_scores:
            self.best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['score'])
            self.best_model = self.experts[self.best_model_name]
            print(f"\n  Best model for {self.position}: {self.best_model_name}")
            print(f"  Best model metrics - MAE: {model_scores[self.best_model_name]['mae']:.4f}, "
                  f"RMSE: {np.sqrt(model_scores[self.best_model_name]['mse']):.4f}, "
                  f"R²: {model_scores[self.best_model_name]['r2']:.4f}\n")
        
        # Retrain best model on full training set
        if self.best_model_name:
            if self.best_model_name in self.scalers:
                X_train_scaled = self.scalers[self.best_model_name].fit_transform(X_train)
                self.best_model.fit(X_train_scaled, y_train)
            else:
                self.best_model.fit(X_train, y_train)
        
        self.trained = True
        return model_scores
    
    def predict(self, X):
        """Make predictions using the best model"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.best_model_name in self.scalers:
            X_scaled = self.scalers[self.best_model_name].transform(X)
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X)
    
    def get_best_model_info(self):
        """Return information about the best model"""
        return {
            'position': self.position,
            'best_model': self.best_model_name,
            'trained': self.trained
        }


def load_position_data(file_path, position):
    """
    Load data for a specific position from CSV file and prepare previous-season features.
    """
    # Convert to absolute path if relative
    if not os.path.isabs(file_path):
        # Try script directory first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            os.path.join(script_dir, file_path),
            os.path.join(os.getcwd(), file_path),
            os.path.join(os.getcwd(), 'Code Updates', file_path),
            file_path
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                file_path = path
                break
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Please ensure the file exists in one of these locations:\n"
            f"  - {os.path.dirname(os.path.abspath(__file__))}\n"
            f"  - {os.getcwd()}\n"
            f"  - {os.path.join(os.getcwd(), 'Code Updates')}"
        )
    
    # All inputs are CSV files
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file {file_path}: {exc}")
    
    # Ensure position column is set
    if 'position' not in df.columns:
        df['position'] = position
    
    # Calculate fantasy points per week
    if position == 'K' and 'ppg' in df.columns:
        # Kickers provide ppg directly
        df['fp_per_week'] = df['ppg'].fillna(0)
    elif 'fp' in df.columns and 'games' in df.columns:
        df['fp_per_week'] = df['fp'] / df['games'].replace(0, 1)
        df['fp_per_week'] = df['fp_per_week'].fillna(0)
    else:
        # Provide helpful error message
        available_cols = list(df.columns)
        missing_cols = [col for col in ['fp', 'games'] if col not in df.columns]
        raise ValueError(
            f"Missing required columns in {file_path}:\n"
            f"  Missing: {missing_cols}\n"
            f"  Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}"
        )
    
    # Ensure fantasy_player_id exists
    if 'fantasy_player_id' not in df.columns:
        if 'team' in df.columns:
            df['fantasy_player_id'] = df['team'].astype(str)
        else:
            df['fantasy_player_id'] = df.index.astype(str)
    
    # Ensure name column exists for reporting
    if 'fantasy_player_name' not in df.columns:
        if 'team' in df.columns:
            df['fantasy_player_name'] = df['team']
        else:
            df['fantasy_player_name'] = df['fantasy_player_id']
    
    # Normalize position labels
    df['position'] = df['position'].fillna(position)
    df = df.sort_values(['fantasy_player_id', 'season']).reset_index(drop=True)
    
    no_shift_cols = {
        'season', 'fantasy_player_id', 'fantasy_player_name', 'team', 'position',
        'fp', 'games', 'ppg', 'fp_per_week', 'fp_delta', 'k_fp_calc', 'dst_fp_calc'
    }
    
    feature_columns_to_shift = [
        col for col in df.columns
        if col not in no_shift_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if feature_columns_to_shift:
        df[feature_columns_to_shift] = (
            df.groupby('fantasy_player_id')[feature_columns_to_shift].shift(1)
        )
    
    # Track which season our features came from
    df['feature_source_season'] = df['season'] - 1
    
    return df


def prepare_features(df, required_features=None):

    # Prepare features for modeling by excluding non-feature columns.
    # Columns to exclude from features
    exclude_cols = [
        'season', 'fantasy_player_id', 'fantasy_player_name',
        'position', 'fp', 'games', 'fp_per_week', 'team',
        'ppg', 'feature_source_season', 'fp_delta', 'k_fp_calc', 'dst_fp_calc',
        'team_ppr'
    ]
    
    if required_features is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        valid_features = []
        for col in feature_cols:
            if df[col].isna().sum() / len(df) < 0.5:  
                if df[col].nunique() > 1:
                    valid_features.append(col)
    else:
        valid_features = required_features
    
    # Ensure all required feature columns exist
    missing_cols = [col for col in valid_features if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan
    
    # Select features in the specified order
    X = df[valid_features].copy()
    
    # Ensure all features are numeric; convert non-numeric to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    y = df['fp_per_week'].values
    
    return X, y, valid_features


def temporal_split(df, test_year=2024, val_year=None):
    """
    Perform temporal split of data.
    - Training: All years before test_year (and before val_year if provided)
    - Validation: val_year if provided, or 20% of training data
    - Test: test_year
    """
    train_df = df[df['season'] < test_year].copy()
    test_df = df[df['season'] == test_year].copy()
    
    if val_year and val_year < test_year:
        val_df = train_df[train_df['season'] == val_year].copy()
        train_df = train_df[train_df['season'] < val_year].copy()
    else:
        val_df = None
    
    return train_df, val_df, test_df


def evaluate_model(model, X_test, y_test, position):
    """Evaluate model performance on test set"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    metrics = {
        'position': position,
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'r2': r2,
        'mape': mape
    }
    
    return metrics, y_pred


def train_position_model(position, file_path, test_year=2024):

    print(f"\n{'='*60}")
    print(f"Training MoE model for {position}")
    print(f"{'='*60}")
    
    print(f"Loading data from {file_path}...")
    df = load_position_data(file_path, position)
    print(f"Loaded {len(df)} records spanning {df['season'].min()}-{df['season'].max()}")
    
    # Temporal split
    print(f"\nPerforming temporal split (test year: {test_year})...")
    train_df, val_df, test_df = temporal_split(df, test_year=test_year)
    
    print(f"Training set: {len(train_df)} records (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    if val_df is not None:
        print(f"Validation set: {len(val_df)} records (season {val_df['season'].unique()})")
    print(f"Test set: {len(test_df)} records (season {test_year})")
    
    if len(test_df) == 0:
        print(f"Warning: No test data found for {test_year}.")
        latest_year = df['season'].max()
        if latest_year < test_year:
            print(f"Latest available year is {latest_year}. Using {latest_year} as test set.")
            test_year = latest_year
            train_df, val_df, test_df = temporal_split(df, test_year=test_year)
            print(f"Updated test set: {len(test_df)} records (season {test_year})")
        else:
            raise ValueError(f"No data available for {test_year} or any suitable test year.")
    
    if len(train_df) < 10:
        raise ValueError(f"Insufficient training data: only {len(train_df)} records. Need at least 10.")
    
    print(f"\nPreparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df, required_features=feature_names)
    
    if val_df is not None and len(val_df) > 0:
        X_val, y_val, _ = prepare_features(val_df, required_features=feature_names)
    else:
        X_val, y_val = None, None
    
    if len(feature_names) == 0:
        raise ValueError("No valid features found after preprocessing. Check data quality.")
    
    print(f"Features: {len(feature_names)}")
    if len(feature_names) <= 20:
        print(f"Feature names: {', '.join(feature_names)}")
    else:
        print(f"Feature names (first 10): {', '.join(feature_names[:10])}...")
        print(f"Total features: {len(feature_names)}")
    
    # Train MoE model
    print(f"\nTraining MoE model...")
    moe = MixtureOfExperts(position=position)
    model_scores = moe.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print(f"Evaluating on test set...")
    metrics, y_pred = evaluate_model(moe, X_test, y_test, position)
    
    print(f"\nTest Set Performance for {position}:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Prepare results
    test_df = test_df.copy()
    test_df['predicted_fp_per_week'] = y_pred
    test_df['actual_fp_per_week'] = y_test
    
    return moe, test_df, metrics, feature_names


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths - look in script directory first, then current directory
    base_paths = [
        script_dir,
        os.getcwd(),
        os.path.join(os.getcwd(), 'Code Updates'),
    ]
    
    # Try to find files in various locations
    file_paths = {}
    
    # CSV files for skill positions
    for position in ['QB', 'RB', 'WR', 'TE', 'kicker', 'Defense']:
        filename = f'{position}_features.csv'
        found = False
        for base_path in base_paths:
            potential_path = os.path.join(base_path, filename)
            if os.path.exists(potential_path):
                file_paths[position] = potential_path
                found = True
                print(f"Found {position} file at: {potential_path}")
                break
        
        if not found:
            file_paths[position] = filename
            print(f"Warning: Could not find {filename} in common locations. Will try relative path.")
    

    # Kicker
    k_found = False
    for base_path in base_paths:
        potential_path = os.path.join(base_path, 'kicker_features.csv')
        if os.path.exists(potential_path):
            file_paths['K'] = potential_path
            k_found = True
            print(f"Found K file at: {potential_path}")
            break
    if not k_found:
        file_paths['K'] = 'kicker_features.csv'
        print("Warning: Could not find kicker_features.csv in common locations. Will try relative path.")
    
    # Defense (DST)
    dst_found = False
    for base_path in base_paths:
        potential_path = os.path.join(base_path, 'Defense_features.csv')
        if os.path.exists(potential_path):
            file_paths['DST'] = potential_path
            dst_found = True
            print(f"Found DST file at: {potential_path}")
            break
    if not dst_found:
        file_paths['DST'] = 'Defense_features.csv'
        print("Warning: Could not find Defense_features.csv in common locations. Will try relative path.")
    
    print(f"\nUsing file paths: {file_paths}\n")
    
    # Storage for models and results
    models = {}
    all_test_results = {}
    all_metrics = {}
    
    # Train models for each position
    for position, file_path in file_paths.items():
        try:
            model, test_results, metrics, features = train_position_model(
                position, file_path, test_year=2024
            )
            models[position] = model
            all_test_results[position] = test_results
            all_metrics[position] = metrics
        except Exception as e:
            print(f"Error training {position} model: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate final predictions DataFrame
    print(f"\n{'='*60}")
    print("Generating Final Predictions DataFrame")
    print(f"{'='*60}")
    
    predictions_list = []
    
    for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        if position in all_test_results:
            test_df = all_test_results[position]
            
            for idx, row in test_df.iterrows():
                predictions_list.append({
                    'Player Name': row.get('fantasy_player_name', 'Unknown'),
                    'Player Position': position,
                    'Projected Average Fantasy Points Per Week': row.get('predicted_fp_per_week', 0),
                    'Actual Average Fantasy Points Per Week': row.get('actual_fp_per_week', 0),
                    'Season': row.get('season', 2024)
                })
    
    # Create final DataFrame
    final_predictions = pd.DataFrame(predictions_list)
    
    # Check if we have any predictions
    if len(final_predictions) == 0:
        print("\nWARNING: No predictions were generated!")
        print("This likely means no models were successfully trained.")
        print("Please check that:")
        print("  1. Data files exist in the correct location")
        print("  2. Data files contain 2024 season data")
        print("  3. Data files have the required columns (fp, games, season)")
        return models, pd.DataFrame(), all_metrics
    
    # Sort by position and projected points
    final_predictions = final_predictions.sort_values(
        ['Player Position', 'Projected Average Fantasy Points Per Week'], 
        ascending=[True, False]
    )
    
    # Get output directory (script directory or current directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir if os.path.exists(script_dir) else os.getcwd()
    
    # Save results - full version with actuals for comparison
    output_file_full = os.path.join(output_dir, 'fantasy_predictions_2024_full.csv')
    final_predictions.to_csv(output_file_full, index=False)
    print(f"\nFull predictions (with actuals) saved to {output_file_full}")
    
    # Save required output format (only projected values)
    required_columns = ['Player Name', 'Player Position', 'Projected Average Fantasy Points Per Week']
    final_predictions_required = final_predictions[required_columns].copy()
    output_file = os.path.join(output_dir, 'fantasy_predictions_2024.csv')
    final_predictions_required.to_csv(output_file, index=False)
    print(f"Required predictions saved to {output_file}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    
    if len(all_metrics) == 0:
        print("\nNo models were successfully trained. Cannot display statistics.")
    else:
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            if position in all_metrics:
                metrics = all_metrics[position]
                print(f"\n{position} Model Performance:")
                print(f"  MAE:  {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²:   {metrics['r2']:.4f}")
                print(f"  MAPE: {metrics['mape']:.2f}%")
                if position in models:
                    print(f"  Best Model: {models[position].best_model_name}")
    
    # Print top predictions by position
    print(f"\n{'='*60}")
    print("Top 10 Projected Players by Position")
    print(f"{'='*60}")
    
    for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        pos_preds = final_predictions[final_predictions['Player Position'] == position]
        if len(pos_preds) > 0:
            top_10 = pos_preds.head(10)
            print(f"\n{position}:")
            for idx, row in top_10.iterrows():
                print(f"  {row['Player Name']:30s} - "
                      f"Projected: {row['Projected Average Fantasy Points Per Week']:6.2f} | "
                      f"Actual: {row['Actual Average Fantasy Points Per Week']:6.2f}")
        else:
            print(f"\n{position}: No predictions available")
    
    return models, final_predictions, all_metrics


if __name__ == "__main__":
    print("="*60)
    print("Mixture of Experts (MoE) Fantasy Football Predictions")
    print("="*60)
    
    models, predictions, metrics = main()
    
    print(f"\n{'='*60}")
    print("Model Training Complete!")
    print(f"{'='*60}")

