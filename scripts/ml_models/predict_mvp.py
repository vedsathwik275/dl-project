import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, mean_squared_error, r2_score, confusion_matrix
import numpy as np
import warnings

# Suppress specific warnings if needed (e.g., convergence warnings)
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define file paths for the era datasets
era_files = {
    "Physical Era (1980-1989)": "physical_era_data.csv",
    "Isolation Era (1995-2010)": "isolation_era_data.csv",
    "Analytics Era (2011-Present)": "analytics_era_data.csv"
}

# --- Feature Selection (Define features globally) ---
# Using the same features as before
features = [
    'g', 'gs', 'mp_per_g',
    'fg_per_g', 'fga_per_g', # 'fg_pct',
    'fg3_per_g', 'fg3a_per_g', # 'fg3_pct',
    'fg2_per_g', 'fg2a_per_g', # 'fg2_pct',
    'ft_per_g', 'fta_per_g', # 'ft_pct',
    'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g',
    'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g',
    'pts_per_g',
    'ws', 'ws_per_48', 'bpm', 'vorp',
    'win_loss_pct' # Team win percentage
]
essential_cols = features + ['season', 'award_share', 'player']

# --- Loop through each era dataset ---
for era_name, filepath in era_files.items():
    print(f"\n{'='*20} Processing Era: {era_name} ({filepath}) {'='*20}")

    # --- 1. Load Data for the Era ---
    try:
        # Reading the first row to get headers correctly
        header = pd.read_csv(filepath, nrows=1).columns.tolist()
        cleaned_header = [col.strip() for col in header if 'Unnamed' not in col and col.strip()]
        df = pd.read_csv(filepath, header=None, skiprows=1, names=cleaned_header)
        print(f"Loaded {era_name} data. Shape: {df.shape}")

        # Check if essential columns exist
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing essential columns in {filepath}: {missing_cols}. Skipping this era.")
            continue

        # Convert feature columns and award_share to numeric, coercing errors
        for col in features + ['award_share', 'season']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN in FEATURES or essential non-feature columns ('season', 'award_share')
        df.dropna(subset=features + ['season', 'award_share'], inplace=True)
        print(f"Shape after dropping NaNs in features/target: {df.shape}")

        if df.empty:
            print(f"Skipping {era_name} due to empty DataFrame after cleaning.")
            continue

        # Ensure 'season' is integer
        df['season'] = df['season'].astype(int)

        # --- 2. Target Variable Creation ---
        # Target 1: award_share (continuous) - Fill NaN just in case, although dropna should handle it
        y_award_share = df['award_share'].fillna(0)

        # Target 2: is_mvp (binary)
        max_award_share_season = df.groupby('season')['award_share'].transform('max')
        df['is_mvp'] = np.where((df['award_share'] == max_award_share_season) & (df['award_share'] > 0), 1, 0)
        y_is_mvp = df['is_mvp']

        print(f"MVP counts for {era_name}:\n{y_is_mvp.value_counts()}")

        # Check if there are any MVPs in this era's data
        if y_is_mvp.sum() == 0:
            print(f"Warning: No MVPs identified in the {era_name} dataset based on 'award_share'. Classification metrics might be skewed or fail.")
            # Decide whether to skip or proceed with caution
            # continue # Option to skip if no MVPs

        # Check if there's enough data to split
        if len(df) < 5: # Need at least a few samples for train/test split
             print(f"Skipping {era_name} due to insufficient data ({len(df)} rows) after cleaning.")
             continue


        # --- 3. Data Preprocessing ---
        X = df[features]

        # Split data - Use y_is_mvp for stratification if possible, otherwise don't stratify
        try:
            # Stratify requires at least 2 members in the smallest class
            should_stratify = y_is_mvp.nunique() > 1 and y_is_mvp.value_counts().min() >= 2
            stratify_param = y_is_mvp if should_stratify else None

            X_train, X_test, y_train_is_mvp, y_test_is_mvp, y_train_award_share, y_test_award_share = train_test_split(
                X, y_is_mvp, y_award_share, test_size=0.2, random_state=42, stratify=stratify_param
            )
        except ValueError as e:
             print(f"Train-test split failed for {era_name} (likely due to insufficient samples per class for stratification): {e}. Skipping this era.")
             continue


        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"Training set shape for {era_name}: {X_train_scaled.shape}")
        print(f"Test set shape for {era_name}: {X_test_scaled.shape}")


        # --- 4. Model Training & Evaluation ---

        # Model 1: Linear Regression (predicting award_share)
        print(f"\n--- Training Linear Regression for {era_name} ---")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train_award_share)

        print(f"--- Evaluating Linear Regression for {era_name} ---")
        y_pred_award_share = lr_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test_award_share, y_pred_award_share)
        r2 = r2_score(y_test_award_share, y_pred_award_share)
        print(f"Linear Regression - Mean Squared Error: {mse:.4f}")
        print(f"Linear Regression - R-squared: {r2:.4f}")


        # Models 2 & 3: Classifiers (predicting is_mvp)
        classifiers = {
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        for name, model in classifiers.items():
            print(f"\n--- Training {name} for {era_name} ---")
            # Handle cases where only one class might be present in y_train after split
            if len(np.unique(y_train_is_mvp)) < 2:
                 print(f"Skipping {name} training for {era_name}: only one class present in training data.")
                 continue
            model.fit(X_train_scaled, y_train_is_mvp)

            print(f"--- Evaluating {name} for {era_name} ---")
             # Handle cases where only one class might be present in y_test after split
            if len(np.unique(y_test_is_mvp)) < 2:
                print(f"Skipping {name} evaluation metrics (like ROC AUC and TP count) for {era_name}: only one class present in test data.")
                y_pred_is_mvp = model.predict(X_test_scaled)
                print(f"{name} - Accuracy: {accuracy_score(y_test_is_mvp, y_pred_is_mvp):.4f}")
                print(classification_report(y_test_is_mvp, y_pred_is_mvp, zero_division=0, labels=[0, 1] if y_is_mvp.nunique() > 1 else [y_is_mvp.unique()[0]]))
            else:
                y_pred_is_mvp = model.predict(X_test_scaled)
                y_pred_proba_is_mvp = model.predict_proba(X_test_scaled)[:, 1] # Probability for ROC AUC

                print(f"{name} - Accuracy: {accuracy_score(y_test_is_mvp, y_pred_is_mvp):.4f}")
                # Use zero_division=0 and specify labels if needed
                print(classification_report(y_test_is_mvp, y_pred_is_mvp, zero_division=0, labels=[0, 1]))
                try:
                    roc_auc = roc_auc_score(y_test_is_mvp, y_pred_proba_is_mvp)
                    print(f"{name} - ROC AUC Score: {roc_auc:.4f}")
                except ValueError as e:
                    print(f"Could not calculate ROC AUC for {name} in {era_name}: {e}")

                # Calculate and print True Positives for MVP class
                cm = confusion_matrix(y_test_is_mvp, y_pred_is_mvp, labels=[0, 1])
                true_positives = cm[1, 1]
                total_actual_mvps = np.sum(y_test_is_mvp == 1)
                if total_actual_mvps > 0:
                    print(f"{name} - Correctly Predicted MVPs: {true_positives} out of {total_actual_mvps}")
                else:
                    print(f"{name} - No actual MVPs in the test set for {era_name} to evaluate TP count.")


    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Skipping this era.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filepath} is empty. Skipping this era.")
    except Exception as e:
        print(f"An unexpected error occurred processing {era_name} ({filepath}): {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

print("\nMVP Prediction Script Finished Processing All Eras.") 