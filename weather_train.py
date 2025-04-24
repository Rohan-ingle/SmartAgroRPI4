import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import os
import joblib
import xgboost as xgb
import sqlite3
import argparse
import sys

# --- Configuration ---
# Default DB path, can be overridden by command line arg
DEFAULT_DB_PATH = 'weather_data.db'
# Feature columns expected from the weather_data.db
# Adjust these if your DB schema or desired features differ
FEATURE_COLUMNS = ['temp', 'humidity', 'pressure', 'wind_speed'] # Example, add others like clouds, uv etc. if available and needed
TARGET_COLUMN = 'totalprecip_mm'
TEST_SIZE = 0.1
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# XGBoost specific parameters
N_ESTIMATORS = 1000
XGB_LEARNING_RATE = 0.05
MAX_DEPTH = 5
EARLY_STOPPING_ROUNDS = 50

MODEL_SAVE_PATH = 'weather_xgboost_best_model.joblib'
SCALER_SAVE_PATH = 'weather_xgboost_scaler.joblib'
# --- End Configuration ---

def train_weather_model(db_path, from_scratch=True):
    """
    Trains the weather prediction model using data from the specified SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        from_scratch (bool): If True, trains a new model from scratch.
                               If False, loads the existing model and continues training.
    """
    print(f"Loading data from database: {db_path}...")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}.")
        return False

    try:
        conn = sqlite3.connect(db_path)
        # Select only the necessary columns, handle potential missing columns gracefully
        all_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        query = f"SELECT {', '.join(all_columns)} FROM weather_forecast"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            print(f"Error: No data found in table 'weather_forecast' in {db_path}.")
            return False
    except sqlite3.OperationalError as e:
        print(f"Error querying database (check table/column names): {e}")
        return False
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return False

    print("Dataset loaded successfully from database.")

    # --- Data Preprocessing ---
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the database data: {missing_cols}")
        return False

    # Handle missing values (using fillna with mean for simplicity)
    for col in FEATURE_COLUMNS:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Filled NaN in '{col}' with mean: {mean_val:.2f}")

    if df[TARGET_COLUMN].isnull().any():
        mean_target = df[TARGET_COLUMN].mean()
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(mean_target)
        print(f"Filled NaN in target '{TARGET_COLUMN}' with mean: {mean_target:.2f}")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[[TARGET_COLUMN]].values.astype(np.float32)

    if len(df) < 20:
        print(f"Error: Not enough data ({len(df)} rows) for train/validation/test split.")
        return False

    # Split data
    X_train_val, X_test_orig, y_train_val, y_test_orig = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT / (1.0 - TEST_SIZE),
        random_state=RANDOM_STATE
    )

    print(f"Data split: Train={len(X_train_orig)}, Validation={len(X_val_orig)}, Test={len(X_test_orig)}")

    # --- Scaling ---
    if from_scratch or not os.path.exists(SCALER_SAVE_PATH):
        print("Fitting new scaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_orig)
        # Save the newly fitted scaler
        print(f"Saving scaler to {SCALER_SAVE_PATH}...")
        joblib.dump(scaler, SCALER_SAVE_PATH)
    else:
        print(f"Loading existing scaler from {SCALER_SAVE_PATH}...")
        scaler = joblib.load(SCALER_SAVE_PATH)
        X_train_scaled = scaler.transform(X_train_orig) # Use transform, not fit_transform

    X_val_scaled = scaler.transform(X_val_orig)
    X_test_scaled = scaler.transform(X_test_orig)

    y_train_orig_1d = y_train_orig.ravel()
    y_val_orig_1d = y_val_orig.ravel()
    y_test_orig_1d = y_test_orig.ravel()

    # --- Model Training ---
    eval_set = [(X_train_scaled, y_train_orig_1d), (X_val_scaled, y_val_orig_1d)]

    if from_scratch or not os.path.exists(MODEL_SAVE_PATH):
        print("Training model from scratch...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=MAX_DEPTH,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='rmse',
            early_stopping_rounds=EARLY_STOPPING_ROUNDS # Moved here
        )
        # Pass eval_set for early stopping to work
        model.fit(
            X_train_scaled,
            y_train_orig_1d,
            eval_set=eval_set,
            verbose=True
        )
    else:
        print(f"Loading existing model from {MODEL_SAVE_PATH} for incremental training...")
        try:
            model = joblib.load(MODEL_SAVE_PATH)
            print("Continuing training with new data...")
            model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS) # Ensure it's set for this fit call
            model.fit(
                X_train_scaled,
                y_train_orig_1d,
                xgb_model=model.get_booster(), # Pass the existing booster
                eval_set=eval_set,
                verbose=True
            )
        except FileNotFoundError:
             print(f"Error: Model file {MODEL_SAVE_PATH} not found for incremental training. Training from scratch instead.")
             # Fallback to training from scratch
             return train_weather_model(db_path, from_scratch=True)
        except Exception as e:
             print(f"Error loading or continuing training: {e}. Training from scratch instead.")
             # Fallback to training from scratch
             return train_weather_model(db_path, from_scratch=True)

    # --- Evaluation (Optional but recommended) ---
    if len(X_test_scaled) > 0:
        print("\nEvaluating model on the test set...")
        y_pred_np = model.predict(X_test_scaled)
        y_test_np = y_test_orig_1d

        mae = mean_absolute_error(y_test_np, y_pred_np)
        mse = mean_squared_error(y_test_np, y_pred_np)
        rmse = sqrt(mse)
        r2 = r2_score(y_test_np, y_pred_np)

        print(f"\nTest Set Evaluation Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R-squared: {r2:.4f}")
    else:
        print("Test set is empty. Skipping evaluation.")

    # --- Save Model ---
    print(f"Saving trained XGBoost model to {MODEL_SAVE_PATH}...")
    joblib.dump(model, MODEL_SAVE_PATH)
    print("Model training/update finished.")
    return True

# train_weather_model("weather_data.db", from_scratch=True)