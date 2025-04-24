import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration
DATA_FILE = "../../data/normalized_nba_data_with_MVP_rank_simple.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define features (updated from selected_features.txt)
features = [
    'ws',
    'vorp',
    'ws_per_48',
    'bpm',
    'win_loss_pct',
    'pts_per_g',
    'fg_per_g',
    'fg2_per_g',
    'ft_per_g',
    'fga_per_g'
]

print(f"{'='*20} MVP Award Share Prediction {'='*20}")

try:
    # 1. Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data. Shape: {df.shape}")
    
    # Verify essential columns exist
    essential_cols = features + ['season', 'award_share', 'player']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        # If award_share is missing, check for MVP_rank as a fallback check
        if 'award_share' in missing_cols and 'MVP_rank' not in df.columns:
             raise ValueError(f"Error: Missing essential columns including 'award_share' or 'MVP_rank': {missing_cols}")
        elif 'award_share' in missing_cols:
             print("Warning: 'award_share' column not found. Ensure the data file contains it.")
             # Attempt to proceed, assuming it might be handled later or is an error
        else:
             raise ValueError(f"Error: Missing essential columns: {missing_cols}")
    
    # Use all data, handle missing award_share later
    df_processed = df.copy()
    print(f"Processing {len(df_processed)} total rows.")
    
    # Convert data types and handle missing values
    # Convert feature columns to numeric
    for col in features:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Drop rows with NaN in features
    df_processed.dropna(subset=features, inplace=True)
    print(f"After dropping NaNs in features: {len(df_processed)} rows")
    
    # Handle missing award_share - fill with 0
    if 'award_share' in df_processed.columns:
        df_processed['award_share'] = df_processed['award_share'].fillna(0)
    else:
        # Handle case where award_share column doesn't exist after check
        raise ValueError("Error: 'award_share' column is required but not found in the data.")
    
    # Display distribution of award shares (optional, uncomment if needed)
    # print(f"Award Share distribution summary:")
    # print(df_processed['award_share'].describe())
    # print(f"Number of players with non-zero award share: {len(df_processed[df_processed['award_share'] > 0])}")
    
    # 2. Season-wise Z-score normalization of features
    print(f"\nNormalizing statistical features using Z-score within each season...")
    
    # Function to apply scaling within a group
    def scale_group(group):
        scaler = StandardScaler()
        # Check if group is empty or has NaNs before scaling
        if not group.empty and not group[features].isnull().values.any():
             if len(group) > 1:  # Only scale if more than 1 row
                 group[features] = scaler.fit_transform(group[features])
             else:
                 # For single rows, set to 0 or handle appropriately. Avoid fit_transform.
                 # Setting to 0 assumes average performance for that season.
                 group[features] = 0
        else:
            # Handle empty groups or groups with NaNs if necessary
            # For simplicity, we might fill NaNs before grouping or handle here
            pass # Or set to 0, or skip scaling for this group
        return group
    
    # Group by season and apply scaling
    df_with_season = df_processed.copy()
    X = df_with_season[features + ['season']].copy()
    X_normalized = X.groupby('season').apply(scale_group)
    
    # Remove season column after scaling
    X_normalized.drop(columns=['season'], inplace=True)
    
    # Target variable - award_share
    y = df_processed['award_share']
    
    # Reset indices to ensure alignment after splitting
    X_normalized = X_normalized.reset_index(drop=True)
    df_processed = df_processed.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 3. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Save test indices
    test_indices = y_test.index
    
    # Save info about test set for later analysis
    test_data = df_processed.iloc[test_indices].copy()
    
    # 4. Model training and evaluation
    
    # 4.1 Linear Regression
    print(f"\n{'-'*10} Linear Regression {'-'*10}")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    # Clip predictions to valid range [0, 1]
    y_pred_lr = np.clip(y_pred_lr, 0, 1)
    
    # Performance metrics (MSE, MAE, R2)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print(f"Mean Squared Error (MSE): {mse_lr:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_lr:.4f}")
    print(f"R-squared (R2): {r2_lr:.4f}")
    
    # 4.2 Random Forest Regressor
    print(f"\n{'-'*10} Random Forest Regressor {'-'*10}")
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    # Clip predictions to valid range [0, 1]
    y_pred_rf = np.clip(y_pred_rf, 0, 1)
    
    # Performance metrics (MSE, MAE, R2)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_rf:.4f}")
    print(f"R-squared (R2): {r2_rf:.4f}")
    
    # 4.3 Gradient Boosting Regressor
    print(f"\n{'-'*10} Gradient Boosting Regressor {'-'*10}")
    gb_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    # Clip predictions to valid range [0, 1]
    y_pred_gb = np.clip(y_pred_gb, 0, 1)
    
    # Performance metrics (MSE, MAE, R2)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    
    print(f"Mean Squared Error (MSE): {mse_gb:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_gb:.4f}")
    print(f"R-squared (R2): {r2_gb:.4f}")
    
    # 5. Result comparison (using regression metrics)
    print(f"\n{'-'*10} Model Comparison {'-'*10}")
    models = ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"]
    mses = [mse_lr, mse_rf, mse_gb]
    maes = [mae_lr, mae_rf, mae_gb]
    r2s = [r2_lr, r2_rf, r2_gb]
    
    print("Model Performance Metrics:")
    print(f"{'Model':<30} {'MSE':<10} {'MAE':<10} {'R2':<10}")
    for model, mse, mae, r2 in zip(models, mses, maes, r2s):
        print(f"{model:<30} {mse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
    
    # Display some example predictions for analysis
    print(f"\n{'-'*10} Sample Predictions {'-'*10}")
    test_data['LR_pred_share'] = y_pred_lr
    test_data['RF_pred_share'] = y_pred_rf
    test_data['GB_pred_share'] = y_pred_gb
    
    # Display a sample of predictions with actual and predicted award shares
    sample_preds = test_data[['season', 'player', 'award_share', 'LR_pred_share', 'RF_pred_share', 'GB_pred_share']].head(10)
    print(f"\nSample of actual vs predicted award shares:")
    # Format prediction columns for better readability
    for col in ['LR_pred_share', 'RF_pred_share', 'GB_pred_share']:
         sample_preds[col] = sample_preds[col].map('{:.4f}'.format)
    print(sample_preds.to_string(index=False))

    # Save test data with predictions to CSV
    print(f"\nSaving test predictions to 'award_share_model_predictions.csv'...")
    output_cols = ['season', 'player', 'award_share', 'LR_pred_share', 'RF_pred_share', 'GB_pred_share'] + features
    test_data_to_save = test_data[output_cols].copy()
    # Optionally format the prediction columns before saving
    for col in ['LR_pred_share', 'RF_pred_share', 'GB_pred_share']:
         test_data_to_save[col] = test_data_to_save[col].map('{:.6f}'.format) # Use more precision for CSV
    test_data_to_save.sort_values(by=['season', 'award_share'], ascending=[False, False], inplace=True)
    test_data_to_save.to_csv('../../data/ml_predictions.csv', index=False)
    print("Predictions saved.")

except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {DATA_FILE} is empty.")
except ValueError as ve:
    print(f"Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*20} MVP Award Share Prediction Finished {'='*20}") 