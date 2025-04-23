import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, mean_squared_error, r2_score, confusion_matrix
import numpy as np
import warnings

# Suppress warnings if necessary
# from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# --- Configuration ---
DATA_FILE = "NBA_Dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define features (same as before, adjust if needed)
statistical_features = [
    'g', 'gs', 'mp_per_g',
    'fg_per_g', 'fga_per_g', # 'fg_pct',
    'fg3_per_g', 'fg3a_per_g', # 'fg3_pct',
    'fg2_per_g', 'fg2a_per_g', # 'fg2_pct',
    'ft_per_g', 'fta_per_g', # 'ft_pct',
    'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g',
    'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g',
    'pts_per_g',
    'ws', 'ws_per_48', 'bpm', 'vorp',
    'win_loss_pct' # Team win percentage (consider if this should be normalized)
]
essential_cols = statistical_features + ['season', 'award_share', 'player'] # Columns needed before normalization

print(f"{'='*20} Starting Unified MVP Prediction Script {'='*20}")

# --- 1. Load Data ---
try:
    # Reading the first row to get headers correctly
    header = pd.read_csv(DATA_FILE, nrows=1).columns.tolist()
    # Clean potential leading/trailing spaces and filter unnamed columns
    cleaned_header = [col.strip() for col in header if 'Unnamed' not in col and col.strip()]
    df = pd.read_csv(DATA_FILE, header=None, skiprows=1, names=cleaned_header, low_memory=False) # low_memory=False for mixed types
    print(f"Loaded data from {DATA_FILE}. Shape: {df.shape}")

    # Check if essential columns exist
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing essential columns in {DATA_FILE}: {missing_cols}")

    # Convert relevant columns to numeric, coercing errors
    cols_to_numeric = statistical_features + ['award_share', 'season']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             print(f"Warning: Column '{col}' expected but not found during numeric conversion.")


    # Drop rows with NaN in FEATURES or essential non-feature columns ('season', 'award_share')
    print(f"Shape before dropping NaNs: {df.shape}")
    df.dropna(subset=cols_to_numeric, inplace=True)
    print(f"Shape after dropping NaNs in essential numeric columns: {df.shape}")

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning NaNs. Cannot proceed.")

    # Ensure 'season' is integer
    df['season'] = df['season'].astype(int)

except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found. Exiting.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Error: The file {DATA_FILE} is empty. Exiting.")
    exit()
except ValueError as ve:
    print(ve)
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading/cleaning: {e}")
    import traceback
    traceback.print_exc()
    exit()


# --- 2. Target Variable Creation ---
# Target 1: award_share (continuous)
y_award_share = df['award_share'].fillna(0) # Fill any remaining NaNs just in case

# Target 2: is_mvp (binary)
# Calculate max award share *per season*
max_award_share_season = df.groupby('season')['award_share'].transform('max')
# MVP is the player(s) with the max share in a season, provided share > 0
df['is_mvp'] = np.where((df['award_share'] == max_award_share_season) & (df['award_share'] > 0), 1, 0)
y_is_mvp = df['is_mvp']

print(f"\nTotal MVP counts across all seasons:\n{y_is_mvp.value_counts()}")
if y_is_mvp.sum() == 0:
    print("Warning: No MVPs identified in the dataset based on 'award_share'. Check data and logic.")
    # Decide if script should exit or proceed
    # exit()


# --- 3. Feature Engineering: Season-wise Z-Score Normalization ---
print("\nNormalizing statistical features using Z-score within each season...")
X = df[statistical_features].copy() # Features to be normalized
other_data = df.drop(columns=statistical_features) # Keep other columns like 'player', 'season', 'is_mvp' etc.

# Function to apply scaling within a group
def scale_group(group):
    scaler = StandardScaler()
    # Only scale if there's more than 1 row to avoid std dev = 0 issues
    if len(group) > 1:
        group[statistical_features] = scaler.fit_transform(group[statistical_features])
    else:
        # Handle single-row groups: set to 0 or keep original? Setting to 0 is common for z-score with std=0.
        # Alternatively, could skip scaling for these groups, but might introduce NaNs if fit_transform fails.
        group[statistical_features] = 0 # Or np.nan, or keep original - depends on desired behavior
    return group

# Group by season and apply scaling
# We need 'season' temporarily in X for grouping
X['season'] = df['season']
X_normalized_grouped = X.groupby('season', group_keys=False).apply(scale_group)

# Drop the temporary season column after scaling
X_normalized = X_normalized_grouped.drop(columns=['season'])

# Check for NaNs introduced by scaling (e.g., from single-row groups if not handled)
if X_normalized.isnull().sum().sum() > 0:
    print("Warning: NaNs detected after normalization. Check for seasons with single players or constant feature values.")
    # Option: Fill NaNs, e.g., with 0
    # X_normalized.fillna(0, inplace=True)

print(f"Normalization complete. Shape of normalized features: {X_normalized.shape}")

# Recombine with non-normalized data if needed for context later, but not for model input X
# df_normalized = pd.concat([other_data.reset_index(drop=True), X_normalized.reset_index(drop=True)], axis=1)


# --- 4. Data Splitting ---
print(f"\nSplitting data into training ({1-TEST_SIZE:.0%}) and testing ({TEST_SIZE:.0%}) sets...")

# Check for sufficient data
if len(X_normalized) < 10: # Arbitrary small number
     print(f"Error: Insufficient data ({len(X_normalized)} rows) after processing to perform train/test split. Exiting.")
     exit()

# Stratify based on the binary MVP target to ensure representation in both sets
try:
    # Stratify requires at least 2 members in the smallest class for splitting
    should_stratify = y_is_mvp.nunique() > 1 and y_is_mvp.value_counts().min() >= 2
    stratify_param = y_is_mvp if should_stratify else None
    if not should_stratify:
        print("Warning: Cannot stratify split (likely due to < 2 samples in the minority class). Performing regular split.")

    X_train, X_test, y_train_is_mvp, y_test_is_mvp, y_train_award_share, y_test_award_share = train_test_split(
        X_normalized, y_is_mvp, y_award_share,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_param
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"MVPs in training set: {y_train_is_mvp.sum()} (Total: {len(y_train_is_mvp)})")
    print(f"MVPs in test set: {y_test_is_mvp.sum()} (Total: {len(y_test_is_mvp)})")

except ValueError as e:
     print(f"Train-test split failed (likely due to stratification issues): {e}. Exiting.")
     exit()

# --- 5. Model Training & Evaluation ---

# Model 1: Linear Regression (predicting award_share)
print(f"\n--- Training Linear Regression (Predicting Award Share) ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_award_share)

print(f"--- Evaluating Linear Regression ---")
y_pred_award_share = lr_model.predict(X_test)
mse = mean_squared_error(y_test_award_share, y_pred_award_share)
r2 = r2_score(y_test_award_share, y_pred_award_share)
print(f"Linear Regression - Mean Squared Error: {mse:.4f}")
print(f"Linear Regression - R-squared: {r2:.4f}")


# Models 2 & 3: Classifiers (predicting is_mvp)
classifiers = {
    # Balance classes for RF due to MVP rarity
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

# Check if there are enough classes in training data for classification
if len(np.unique(y_train_is_mvp)) < 2:
     print("\nWarning: Only one class present in the training data for MVP classification. Skipping classifier training and evaluation.")
else:
    for name, model in classifiers.items():
        print(f"\n--- Training {name} (Predicting Is MVP) ---")
        model.fit(X_train, y_train_is_mvp)

        print(f"--- Evaluating {name} ---")
        y_pred_is_mvp = model.predict(X_test)

         # Handle cases where only one class might be present in y_test after split
        if len(np.unique(y_test_is_mvp)) < 2:
            print(f"Warning: Only one class present in the test data for {name}. Reporting accuracy and classification report only.")
            print(f"{name} - Accuracy: {accuracy_score(y_test_is_mvp, y_pred_is_mvp):.4f}")
            # Ensure labels are correctly specified if only one class exists
            report_labels = [0, 1] if y_is_mvp.nunique() > 1 else [y_test_is_mvp.unique()[0]]
            print(classification_report(y_test_is_mvp, y_pred_is_mvp, zero_division=0, labels=report_labels))
            print(f"{name} - ROC AUC Score: Not applicable (only one class in test set)")
            print(f"{name} - Confusion Matrix:\n {confusion_matrix(y_test_is_mvp, y_pred_is_mvp, labels=report_labels)}")

        else:
            y_pred_proba_is_mvp = model.predict_proba(X_test)[:, 1] # Probability for ROC AUC

            print(f"{name} - Accuracy: {accuracy_score(y_test_is_mvp, y_pred_is_mvp):.4f}")
            # Use zero_division=0 and specify labels for robustness
            print(classification_report(y_test_is_mvp, y_pred_is_mvp, zero_division=0, labels=[0, 1]))
            try:
                # ROC AUC requires both classes in y_test
                roc_auc = roc_auc_score(y_test_is_mvp, y_pred_proba_is_mvp)
                print(f"{name} - ROC AUC Score: {roc_auc:.4f}")
            except ValueError as e:
                print(f"Could not calculate ROC AUC for {name}: {e}")

            # Calculate and print True Positives for MVP class (class 1)
            cm = confusion_matrix(y_test_is_mvp, y_pred_is_mvp, labels=[0, 1])
            true_positives = cm[1, 1]
            total_actual_mvps = np.sum(y_test_is_mvp == 1)
            false_positives = cm[0, 1]
            print(f"{name} - Confusion Matrix (Labels: [0, 1]):\n{cm}")
            if total_actual_mvps > 0:
                print(f"{name} - Correctly Predicted MVPs (TP): {true_positives} out of {total_actual_mvps} actual MVPs")
                print(f"{name} - Incorrectly Predicted MVPs (FP): {false_positives}")

            else:
                print(f"{name} - No actual MVPs in the test set to evaluate TP/FP count.")

print(f"\n{'='*20} Unified MVP Prediction Script Finished {'='*20}") 