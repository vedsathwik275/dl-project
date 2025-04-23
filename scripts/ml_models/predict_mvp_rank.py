import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
DATA_FILE = "NBA_Dataset_with_MVP_rank.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define features
features = [
    'g', 'gs', 'mp_per_g',
    'fg_per_g', 'fga_per_g', 
    'fg3_per_g', 'fg3a_per_g',
    'fg2_per_g', 'fg2a_per_g',
    'ft_per_g', 'fta_per_g',
    'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g',
    'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g',
    'pts_per_g',
    'ws', 'ws_per_48', 'bpm', 'vorp',
    'win_loss_pct'
]

# Custom accuracy function for regression models: 
# counts cases where rounded prediction equals actual rank
def exact_rank_accuracy(y_true, y_pred):
    """Calculate accuracy based on exact rank matches after rounding."""
    # Round predictions to nearest integer
    y_pred_rounded = np.round(y_pred).astype(int)
    # Count matches
    matches = np.sum(y_pred_rounded == y_true)
    return matches / len(y_true)

# New flexible accuracy function that gives partial credit for being off by one or two ranks
def flexible_rank_accuracy(y_true, y_pred_rounded):
    """
    Calculate accuracy with partial credit:
    - Exact match: 1.0 (100% correct)
    - Off by 1 rank: 0.8 (80% correct)
    - Off by 2 ranks: 0.2 (20% correct)
    - Off by >2 ranks: 0.0 (incorrect)
    """
    # Calculate absolute difference between predicted and actual ranks
    rank_diff = np.abs(y_pred_rounded - y_true)
    
    # Assign scores based on rank difference
    scores = np.zeros_like(rank_diff, dtype=float)
    scores[rank_diff == 0] = 1.0    # Exact matches
    scores[rank_diff == 1] = 0.8    # Off by one rank
    scores[rank_diff == 2] = 0.2    # Off by two ranks
    
    # Return the average score
    return scores.mean()

print(f"{'='*20} MVP Rank Prediction {'='*20}")

try:
    # 1. Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data. Shape: {df.shape}")
    
    # Verify essential columns exist
    essential_cols = features + ['season', 'MVP_rank', 'player']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing essential columns: {missing_cols}")
    
    # Focus only on rows with MVP_rank (1-5)
    df_mvp = df.dropna(subset=['MVP_rank'])
    print(f"Number of rows with MVP rank: {len(df_mvp)} out of {len(df)} total rows")
    
    # Convert data types and handle missing values
    # Convert feature columns to numeric
    for col in features:
        if col in df_mvp.columns:
            df_mvp[col] = pd.to_numeric(df_mvp[col], errors='coerce')
    
    # Drop rows with NaN in features
    df_mvp.dropna(subset=features, inplace=True)
    print(f"After dropping NaNs in features: {len(df_mvp)} rows")
    
    # Ensure MVP_rank is integer
    df_mvp['MVP_rank'] = df_mvp['MVP_rank'].astype(int)
    
    # Display distribution of MVP ranks
    rank_distribution = df_mvp['MVP_rank'].value_counts().sort_index()
    print(f"\nDistribution of MVP ranks:\n{rank_distribution}")
    
    # 2. Season-wise Z-score normalization of features
    print(f"\nNormalizing statistical features using Z-score within each season...")
    
    # Function to apply scaling within a group
    def scale_group(group):
        scaler = StandardScaler()
        if len(group) > 1:  # Only scale if more than 1 row
            group[features] = scaler.fit_transform(group[features])
        else:
            group[features] = 0  # Set to 0 for single-row groups
        return group
    
    # Group by season and apply scaling
    df_mvp_with_season = df_mvp.copy()
    X = df_mvp_with_season[features + ['season']].copy()
    X_normalized = X.groupby('season').apply(scale_group)
    
    # Remove season column after scaling
    X_normalized.drop(columns=['season'], inplace=True)
    
    # Target variable - MVP rank
    y = df_mvp['MVP_rank']
    
    # Reset indices to ensure alignment after splitting
    X_normalized = X_normalized.reset_index(drop=True)
    df_mvp = df_mvp.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 3. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Ensure balanced distribution of ranks
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"MVP rank distribution in training set:\n{y_train.value_counts().sort_index()}")
    print(f"MVP rank distribution in test set:\n{y_test.value_counts().sort_index()}")
    
    # Save test indices
    test_indices = y_test.index
    
    # Save info about test set for later analysis
    test_data = df_mvp.iloc[test_indices].copy()
    
    # 4. Model training and evaluation
    
    # 4.1 Linear Regression (predicting MVP_rank as a continuous value, then rounding)
    print(f"\n{'-'*10} Linear Regression {'-'*10}")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_lr_rounded = np.round(y_pred_lr).astype(int)
    
    # Performance metrics
    mse = mean_squared_error(y_test, y_pred_lr)
    r2 = r2_score(y_test, y_pred_lr)
    acc_lr_exact = exact_rank_accuracy(y_test, y_pred_lr)
    acc_lr_flex = flexible_rank_accuracy(y_test, y_pred_lr_rounded)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Exact Rank Accuracy: {acc_lr_exact:.4f} ({int(acc_lr_exact * len(y_test))} exact matches out of {len(y_test)})")
    print(f"Flexible Rank Accuracy: {acc_lr_flex:.4f} (includes 80% credit for being off by one or two ranks)")
    
    # Calculate counts for each type of match
    exact_matches = np.sum(y_pred_lr_rounded == y_test)
    off_by_one = np.sum(np.abs(y_pred_lr_rounded - y_test) == 1)
    off_by_two = np.sum(np.abs(y_pred_lr_rounded - y_test) == 2)
    off_by_more = len(y_test) - exact_matches - off_by_one - off_by_two
    
    print(f"Prediction breakdown:")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Off by 1 rank: {off_by_one}")
    print(f"  Off by 2 ranks: {off_by_two}")
    print(f"  Off by >2 ranks: {off_by_more}")
    
    # 4.2 Random Forest Classifier
    print(f"\n{'-'*10} Random Forest Classifier {'-'*10}")
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    
    # Performance metrics
    acc_rf_exact = accuracy_score(y_test, y_pred_rf)
    acc_rf_flex = flexible_rank_accuracy(y_test, y_pred_rf)
    
    print(f"Exact Accuracy: {acc_rf_exact:.4f} ({int(acc_rf_exact * len(y_test))} exact matches out of {len(y_test)})")
    print(f"Flexible Accuracy: {acc_rf_flex:.4f} (includes 80% credit for being off by one or two ranks)")
    
    # Calculate counts for each type of match
    exact_matches = np.sum(y_pred_rf == y_test)
    off_by_one = np.sum(np.abs(y_pred_rf - y_test) == 1)
    off_by_two = np.sum(np.abs(y_pred_rf - y_test) == 2)
    off_by_more = len(y_test) - exact_matches - off_by_one - off_by_two
    
    print(f"Prediction breakdown:")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Off by 1 rank: {off_by_one}")
    print(f"  Off by 2 ranks: {off_by_two}")
    print(f"  Off by >2 ranks: {off_by_more}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    
    # 4.3 Gradient Boosting Classifier
    print(f"\n{'-'*10} Gradient Boosting Classifier {'-'*10}")
    gb_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_model.fit(X_train, y_train)
    
    y_pred_gb = gb_model.predict(X_test)
    
    # Performance metrics
    acc_gb_exact = accuracy_score(y_test, y_pred_gb)
    acc_gb_flex = flexible_rank_accuracy(y_test, y_pred_gb)
    
    print(f"Exact Accuracy: {acc_gb_exact:.4f} ({int(acc_gb_exact * len(y_test))} exact matches out of {len(y_test)})")
    print(f"Flexible Accuracy: {acc_gb_flex:.4f} (includes 80% credit for being off by one or two ranks)")
    
    # Calculate counts for each type of match
    exact_matches = np.sum(y_pred_gb == y_test)
    off_by_one = np.sum(np.abs(y_pred_gb - y_test) == 1)
    off_by_two = np.sum(np.abs(y_pred_gb - y_test) == 2)
    off_by_more = len(y_test) - exact_matches - off_by_one - off_by_two
    
    print(f"Prediction breakdown:")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Off by 1 rank: {off_by_one}")
    print(f"  Off by 2 ranks: {off_by_two}")
    print(f"  Off by >2 ranks: {off_by_more}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_gb))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_gb))
    
    # 5. Result comparison
    print(f"\n{'-'*10} Model Comparison {'-'*10}")
    models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    exact_accuracies = [acc_lr_exact, acc_rf_exact, acc_gb_exact]
    flexible_accuracies = [acc_lr_flex, acc_rf_flex, acc_gb_flex]
    
    print("Exact Accuracy (perfect predictions):")
    for model, acc in zip(models, exact_accuracies):
        print(f"{model}: {acc:.4f} ({int(acc * len(y_test))} / {len(y_test)} correct)")
    
    print("\nFlexible Accuracy (with 80% credit for being off by one or two ranks):")
    for model, acc in zip(models, flexible_accuracies):
        print(f"{model}: {acc:.4f}")
    
    # Display some example predictions for analysis
    print(f"\n{'-'*10} Sample Predictions {'-'*10}")
    test_data['LR_pred'] = np.round(y_pred_lr).astype(int)
    test_data['RF_pred'] = y_pred_rf
    test_data['GB_pred'] = y_pred_gb
    
    # Display a sample of predictions
    sample_preds = test_data[['season', 'player', 'MVP_rank', 'LR_pred', 'RF_pred', 'GB_pred']].head(10)
    print(f"\nSample of actual vs predicted MVP ranks:")
    print(sample_preds.to_string(index=False))
    
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

print(f"\n{'='*20} MVP Rank Prediction Finished {'='*20}") 