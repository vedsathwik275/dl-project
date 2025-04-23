import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Neural network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = "../data/NBA_Dataset_with_MVP_rank.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 20  # For early stopping

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

# Flexible accuracy function that gives partial credit for being off by one or two ranks
def flexible_rank_accuracy(y_true, y_pred_rounded):
    """
    Calculate accuracy with partial credit:
    - Exact match: 1.0 (100% correct)
    - Off by 1 rank: 0.8 (80% correct)
    - Off by 2 ranks: 0.2 (20% correct)
    - Off by >2 ranks: 0.0 (incorrect)
    """
    # Ensure predictions are within 1-5 range
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    
    # Calculate absolute difference between predicted and actual ranks
    rank_diff = np.abs(y_pred_rounded - y_true)
    
    # Assign scores based on rank difference
    scores = np.zeros_like(rank_diff, dtype=float)
    scores[rank_diff == 0] = 1.0    # Exact matches
    scores[rank_diff == 1] = 0.8    # Off by one rank
    scores[rank_diff == 2] = 0.2    # Off by two ranks
    
    # Return the average score
    return scores.mean()

# Function to build the MLP model with the specified architecture
def build_mlp_model(input_shape):
    model = Sequential([
        # Input layer
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layer 1
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layer 2
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layer 3
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Output layer - sigmoid activation scaled to 1-5 range
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Function to get prediction breakdown
def get_prediction_breakdown(y_true, y_pred):
    exact_matches = np.sum(y_pred == y_true)
    off_by_one = np.sum(np.abs(y_pred - y_true) == 1)
    off_by_two = np.sum(np.abs(y_pred - y_true) == 2)
    off_by_more = len(y_true) - exact_matches - off_by_one - off_by_two
    
    return {
        'exact': exact_matches,
        'off_by_one': off_by_one, 
        'off_by_two': off_by_two,
        'off_by_more': off_by_more
    }

print(f"{'='*20} MVP Rank Prediction with Neural Network {'='*20}")

try:
    # Check for TensorFlow GPU support
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus}")
        # Memory growth needs to be set before GPUs have been initialized
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU.")
    
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
    
    # Neural Network (MLP)
    print(f"\n{'-'*10} MLP Neural Network {'-'*10}")
    
    # Prepare data for neural network
    # Scale target for sigmoid output (1-5 to 0-1)
    y_train_scaled = (y_train - 1) / 4  # Scale from [1,5] to [0,1]
    y_test_scaled = (y_test - 1) / 4
    
    # Build model
    mlp_model = build_mlp_model(X_train.shape[1])
    print(mlp_model.summary())
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining neural network...")
    history = mlp_model.fit(
        X_train, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=2
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_training_history.png')
    print("Training history saved to 'mlp_training_history.png'")
    
    # Make predictions
    y_pred_mlp_scaled = mlp_model.predict(X_test)
    
    # Convert predictions back to 1-5 scale
    y_pred_mlp = (y_pred_mlp_scaled * 4) + 1
    y_pred_mlp_rounded = np.round(y_pred_mlp).astype(int)
    y_pred_mlp_rounded = np.clip(y_pred_mlp_rounded, 1, 5)  # Ensure predictions are between 1-5
    
    # Flatten predictions to ensure 1D arrays for metrics calculation
    y_pred_mlp = y_pred_mlp.flatten()
    y_pred_mlp_rounded = y_pred_mlp_rounded.flatten()
    
    # Performance metrics
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)
    acc_mlp_exact = np.sum(y_pred_mlp_rounded == y_test) / len(y_test)
    acc_mlp_flex = flexible_rank_accuracy(y_test, y_pred_mlp_rounded)
    
    print(f"\nMean Squared Error: {mse_mlp:.4f}")
    print(f"R-squared: {r2_mlp:.4f}")
    print(f"Exact Rank Accuracy: {acc_mlp_exact:.4f} ({int(acc_mlp_exact * len(y_test))} exact matches out of {len(y_test)})")
    print(f"Flexible Rank Accuracy: {acc_mlp_flex:.4f} (includes 80% credit for being off by one or two ranks)")
    
    # Calculate counts for each type of match
    mlp_breakdown = get_prediction_breakdown(y_test, y_pred_mlp_rounded)
    
    print(f"Prediction breakdown:")
    print(f"  Exact matches: {mlp_breakdown['exact']}")
    print(f"  Off by 1 rank: {mlp_breakdown['off_by_one']}")
    print(f"  Off by 2 ranks: {mlp_breakdown['off_by_two']}")
    print(f"  Off by >2 ranks: {mlp_breakdown['off_by_more']}")
    
    # Display some example predictions for analysis
    print(f"\n{'-'*10} Sample Predictions {'-'*10}")
    test_data['MLP_pred'] = y_pred_mlp_rounded
    
    # Display a sample of predictions
    sample_preds = test_data[['season', 'player', 'MVP_rank', 'MLP_pred']].head(10)
    print(f"\nSample of actual vs predicted MVP ranks:")
    print(sample_preds.to_string(index=False))
    
    # Save model
    print("\nSaving model...")
    mlp_model.save("mlp_mvp_rank_model")
    print("Model saved to 'mlp_mvp_rank_model'")
    
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