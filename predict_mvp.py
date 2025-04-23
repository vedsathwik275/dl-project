import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import numpy as np

# Define file path
input_csv = "NBA_Dataset.csv"

# --- 1. Load Data ---
def load_and_clean_data(filepath):
    """Loads, cleans header, and handles initial NaN conversion."""
    try:
        # Reading the first row to get headers correctly
        header = pd.read_csv(filepath, nrows=1).columns.tolist()
        # Clean potential unnamed columns resulting from trailing commas
        cleaned_header = [col.strip() for col in header if 'Unnamed' not in col and col.strip()]

        # Read the full CSV using the cleaned header, skipping the original header row
        df = pd.read_csv(filepath, header=None, skiprows=1, names=cleaned_header)
        print(f"Original dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None

df = load_and_clean_data(input_csv)

if df is not None:
    # --- 2. Feature Selection ---
    # Select potential features (numerical stats and team performance)
    # Added 'ws_per_48' as potentially indicative
    # Excluded percentage fields initially to avoid multicollinearity with raw counts, can add back if needed
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
    essential_cols = features + ['season', 'award_share', 'player'] # Include player for MVP identification

    # Ensure selected feature columns and essential cols exist
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing essential columns: {missing_cols}")
        exit()

    # Convert feature columns and award_share to numeric, coercing errors
    for col in features + ['award_share', 'season']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in features or award_share (essential for modeling)
    df.dropna(subset=features + ['award_share', 'season'], inplace=True)
    print(f"Shape after dropping NaNs in features/target: {df.shape}")

    # --- 3. Target Variable Creation (is_mvp) ---
    df['award_share'] = df['award_share'].fillna(0) # Assume NaN in award_share means 0 votes
    df['season'] = df['season'].astype(int)

    # Find the max award share per season
    max_award_share = df.groupby('season')['award_share'].transform('max')

    # MVP is the player with the max award share for that season (and share > 0)
    df['is_mvp'] = np.where((df['award_share'] == max_award_share) & (df['award_share'] > 0), 1, 0)

    print(f"MVP counts:\n{df['is_mvp'].value_counts()}")

    if df['is_mvp'].sum() == 0:
        print("Error: No MVPs identified based on the 'award_share' logic. Check data.")
        exit()

    # --- 4. Data Preprocessing ---
    X = df[features]
    y = df['is_mvp']

    # Split data (stratify by y due to imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")

    # --- 5. Model Training & Evaluation ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train)

        print(f"--- Evaluating {name} ---")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability for ROC AUC

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        # Use zero_division=0 for precision/recall/f1 in case of no predicted positives
        print(classification_report(y_test, y_pred, zero_division=0))
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC AUC Score: {roc_auc:.4f}")
        except ValueError as e:
            print(f"Could not calculate ROC AUC: {e}") # Handle cases where only one class present in y_true

    print("\nMVP Prediction Script Finished.") 