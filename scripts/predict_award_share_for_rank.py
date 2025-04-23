import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import seaborn as sns

# Neural network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Add, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = "../data/filtered_nba_data_with_MVP_rank.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 20  # For early stopping
TOP_N_FEATURES = 10  # Number of top correlated features to select

# Define all possible features
all_features = [
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

# Use these specific features for award_share based on PCA analysis
award_share_features = [
    'vorp', 'bpm', 'ws', 'ws_per_48', 'pts_per_g',
    'ft_per_g', 'fg_per_g', 'fta_per_g', 'fg2_per_g', 'fga_per_g'
]

# Flexible accuracy function that gives partial credit for being off by one or two ranks
def flexible_rank_accuracy(y_true, y_pred_rounded):
    """
    Calculate accuracy with partial credit:
    - Exact match: 1.0 (100% correct)
    - Off by 1 rank: 0.8 (80% credit)
    - Off by 2 ranks: 0.2 (20% credit)
    - Special case: When actual rank is 5 and predicted is >5, or vice versa, treated as off by 1
    - Off by >2 ranks: 0.0 (incorrect)
    """
    # Ensure predictions are within 1-5 range
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    
    # Calculate absolute difference between predicted and actual ranks
    rank_diff = np.abs(y_pred_rounded - y_true)
    
    # Special case handling for boundary between top-5 and outside top-5
    # If true rank is 5 and predicted as 6 or 7, count as off-by-1
    # If true rank is 6 or 7 and predicted as 5, count as off-by-1
    special_case = ((y_true == 5) & (y_pred_rounded > 5)) | ((y_true > 5) & (y_pred_rounded == 5))
    
    # Assign scores based on rank difference
    scores = np.zeros_like(rank_diff, dtype=float)
    scores[rank_diff == 0] = 1.0    # Exact matches
    scores[rank_diff == 1] = 0.8    # Off by one rank
    scores[rank_diff == 2] = 0.2    # Off by two ranks
    scores[special_case] = 0.8      # Special case (treat as off by one)
    
    # Return the average score
    return scores.mean()

# Function to build improved MLP model with residual connections and better architecture
def build_mlp_model(input_shape):
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # First block with residual connection
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    # Residual block 1
    block_input = x
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, block_input])  # Residual connection
    x = Dropout(0.4)(x)
    
    # Split into two branches for feature extraction at different levels
    # Branch 1: Wide branch for general patterns
    wide_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    wide_branch = BatchNormalization()(wide_branch)
    wide_branch = LeakyReLU(alpha=0.1)(wide_branch)
    wide_branch = Dropout(0.3)(wide_branch)
    
    # Branch 2: Deep branch for more complex patterns
    deep_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.3)(deep_branch)
    deep_branch = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(deep_branch)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.3)(deep_branch)
    
    # Merge branches
    merged = Concatenate()([wide_branch, deep_branch])
    
    # Final layers
    x = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(merged)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    
    # Output layer - use linear activation for award_share prediction
    output = Dense(1)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean squared error for continuous variable
        metrics=['mae']
    )
    
    return model

# Function to convert award_share predictions to ranks within each season
def convert_to_ranks(df, pred_col='predicted_award_share'):
    """
    Convert award_share predictions to ranks (1-5) within each season
    """
    # Create a copy to avoid modifying the original
    ranked_df = df.copy()
    
    # Group by season and rank the predictions (higher award_share = better rank)
    ranked_df['predicted_rank'] = ranked_df.groupby('season')[pred_col].rank(ascending=False, method='first')
    
    # Ensure only ranks 1-5 are used
    def limit_ranks(group):
        mask = group['predicted_rank'] > 5
        group.loc[mask, 'predicted_rank'] = 5
        return group
    
    ranked_df = ranked_df.groupby('season').apply(limit_ranks)
    
    # Convert to integer
    ranked_df['predicted_rank'] = ranked_df['predicted_rank'].astype(int)
    
    return ranked_df

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

# Function to save detailed prediction results
def save_detailed_predictions(df, full_results_df):
    """
    Save comprehensive prediction results to CSV with additional stats
    """
    # Calculate error
    results_df = full_results_df.copy()
    results_df['rank_error'] = results_df['predicted_rank'] - results_df['MVP_rank']
    results_df['abs_rank_error'] = np.abs(results_df['rank_error'])
    
    # Add prediction quality labels
    conditions = [
        (results_df['abs_rank_error'] == 0),
        (results_df['abs_rank_error'] == 1),
        (results_df['abs_rank_error'] == 2),
        (results_df['abs_rank_error'] > 2)
    ]
    labels = ['Exact Match', 'Off by 1', 'Off by 2', 'Off by >2']
    results_df['prediction_quality'] = np.select(conditions, labels)
    
    # Sort by season (descending) and MVP rank (ascending)
    results_df = results_df.sort_values(['season', 'MVP_rank'], ascending=[False, True])
    
    # Save all results to CSV
    results_df.to_csv('award_share_rank_prediction_results.csv', index=False)
    
    # Create a separate dataframe with just the important columns for easy viewing
    summary_df = results_df[['season', 'player', 'award_share', 'predicted_award_share', 
                            'MVP_rank', 'predicted_rank', 'rank_error', 'prediction_quality']]
    summary_df.to_csv('award_share_rank_prediction_summary.csv', index=False)
    
    # Generate yearly summary - prediction accuracy by season
    yearly_summary = results_df.groupby('season').apply(
        lambda x: pd.Series({
            'num_players': len(x),
            'exact_matches': sum(x['abs_rank_error'] == 0),
            'off_by_one': sum(x['abs_rank_error'] == 1),
            'off_by_two': sum(x['abs_rank_error'] == 2),
            'off_by_more': sum(x['abs_rank_error'] > 2),
            'accuracy': sum(x['abs_rank_error'] == 0) / len(x),
            'flexible_accuracy': flexible_rank_accuracy(x['MVP_rank'], x['predicted_rank']),
            'award_share_mse': mean_squared_error(x['award_share'], x['predicted_award_share']),
            'rank_mse': mean_squared_error(x['MVP_rank'], x['predicted_rank'])
        })
    ).reset_index()
    
    yearly_summary.to_csv('award_share_rank_yearly_accuracy.csv', index=False)
    
    return results_df

# Function to create additional visualizations
def create_visualizations(results_df):
    """
    Create additional visualizations for model evaluation
    """
    # 1. Confusion matrix for ranks
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(results_df['MVP_rank'], results_df['predicted_rank'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 6), 
                yticklabels=range(1, 6))
    plt.xlabel('Predicted Rank')
    plt.ylabel('True Rank')
    plt.title('Confusion Matrix of MVP Rank Predictions (from Award Share)')
    plt.tight_layout()
    plt.savefig('award_share_rank_confusion_matrix.png')
    
    # 2. Distribution of prediction errors
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rank_error', data=results_df, palette='viridis')
    plt.xlabel('Rank Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.title('Distribution of Rank Prediction Errors')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('award_share_rank_error_distribution.png')
    
    # 3. Prediction accuracy by season
    yearly_accuracy = results_df.groupby('season')['abs_rank_error'].apply(
        lambda x: sum(x == 0) / len(x)
    ).reset_index()
    yearly_accuracy.columns = ['season', 'accuracy']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='season', y='accuracy', data=yearly_accuracy, palette='viridis')
    plt.xlabel('Season')
    plt.ylabel('Accuracy (Exact Matches)')
    plt.title('Rank Prediction Accuracy by Season')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('award_share_rank_accuracy_by_season.png')
    
    # 4. Actual vs Predicted award_share scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results_df['award_share'], 
        results_df['predicted_award_share'],
        c=results_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add perfect prediction line
    max_val = max(results_df['award_share'].max(), results_df['predicted_award_share'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.colorbar(scatter, label='Season')
    plt.xlabel('Actual Award Share')
    plt.ylabel('Predicted Award Share')
    plt.title('Actual vs Predicted Award Share')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('award_share_actual_vs_predicted.png')
    
    # 5. Actual vs Predicted rank scatter
    plt.figure(figsize=(10, 8))
    # Add jitter to prevent points from overlapping exactly
    jitter = np.random.normal(0, 0.1, size=len(results_df))
    scatter = plt.scatter(
        results_df['MVP_rank'] + jitter, 
        results_df['predicted_rank'] + jitter,
        c=results_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add perfect prediction line
    plt.plot([1, 5], [1, 5], 'r--', alpha=0.7)
    
    plt.colorbar(scatter, label='Season')
    plt.xlabel('Actual MVP Rank')
    plt.ylabel('Predicted Rank (from Award Share)')
    plt.title('Actual vs Predicted MVP Ranks')
    plt.grid(alpha=0.3)
    plt.xticks(range(1, 6))
    plt.yticks(range(1, 6))
    plt.tight_layout()
    plt.savefig('award_share_derived_rank_actual_vs_predicted.png')

# Main execution
print(f"{'='*20} MVP Award Share Prediction for Ranking {'='*20}")

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
    essential_cols = all_features + ['season', 'MVP_rank', 'player', 'award_share']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing essential columns: {missing_cols}")
    
    # Focus only on rows with MVP_rank and award_share
    df_mvp = df.dropna(subset=['MVP_rank', 'award_share'])
    print(f"Number of rows with MVP rank and award_share: {len(df_mvp)} out of {len(df)} total rows")
    
    # Convert data types and handle missing values
    # Convert feature columns to numeric
    for col in all_features:
        if col in df_mvp.columns:
            df_mvp[col] = pd.to_numeric(df_mvp[col], errors='coerce')
    
    # Convert award_share to numeric
    df_mvp['award_share'] = pd.to_numeric(df_mvp['award_share'], errors='coerce')
    
    # Drop rows with NaN in features or award_share
    df_mvp.dropna(subset=all_features + ['award_share'], inplace=True)
    print(f"After dropping NaNs in features and award_share: {len(df_mvp)} rows")
    
    # Ensure MVP_rank is integer
    df_mvp['MVP_rank'] = df_mvp['MVP_rank'].astype(int)
    
    # Display distribution of MVP ranks
    rank_distribution = df_mvp['MVP_rank'].value_counts().sort_index()
    print(f"\nDistribution of MVP ranks:\n{rank_distribution}")
    
    # Display award_share statistics
    print(f"\nAward Share statistics:")
    print(f"Mean: {df_mvp['award_share'].mean():.4f}")
    print(f"Min: {df_mvp['award_share'].min():.4f}")
    print(f"Max: {df_mvp['award_share'].max():.4f}")
    print(f"Median: {df_mvp['award_share'].median():.4f}")
    
    # 2. Season-wise Z-score normalization of features
    print(f"\nNormalizing statistical features using Z-score within each season...")
    
    # Function to apply scaling within a group
    def scale_group(group):
        scaler = StandardScaler()
        if len(group) > 1:  # Only scale if more than 1 row
            group[all_features] = scaler.fit_transform(group[all_features])
        else:
            group[all_features] = 0  # Set to 0 for single-row groups
        return group
    
    # Group by season and apply scaling
    df_mvp_with_season = df_mvp.copy()
    X_all = df_mvp_with_season[all_features + ['season']].copy()
    X_normalized_all = X_all.groupby('season').apply(scale_group)
    
    # Remove season column after scaling
    X_normalized_all.drop(columns=['season'], inplace=True)
    
    # 3. Feature Selection - Use directly the award_share features from PCA
    print(f"\n{'-'*10} Feature Selection for Award Share {'-'*10}")
    
    # Use the top features from PCA analysis
    selected_features = award_share_features
    
    print(f"Top {len(selected_features)} features for award_share prediction (from PCA analysis):")
    for i, feature in enumerate(selected_features):
        print(f"{i+1}. {feature}")
    
    # Create feature matrix with only selected features
    X_normalized = X_normalized_all[selected_features]
    
    # Target variable - award_share
    y = df_mvp['award_share']
    
    # Reset indices to ensure alignment after splitting
    X_normalized = X_normalized.reset_index(drop=True)
    df_mvp = df_mvp.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 4. Split data for training and testing - by season to ensure proper evaluation
    # Get unique seasons
    seasons = df_mvp['season'].unique()
    # Use the most recent 20% of seasons for testing
    seasons.sort()
    split_idx = int(len(seasons) * 0.8)
    train_seasons = seasons[:split_idx]
    test_seasons = seasons[split_idx:]
    
    # Split based on seasons
    train_mask = df_mvp['season'].isin(train_seasons)
    X_train = X_normalized[train_mask]
    y_train = y[train_mask]
    X_test = X_normalized[~train_mask]
    y_test = y[~train_mask]
    
    print(f"Training set seasons: {train_seasons}")
    print(f"Test set seasons: {test_seasons}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Save test indices
    test_indices = df_mvp[~train_mask].index
    
    # Save info about test set for later analysis
    test_data = df_mvp.iloc[test_indices].copy()
    
    # 5. Neural Network (MLP) with selected features
    print(f"\n{'-'*10} MLP Neural Network for Award Share Prediction {'-'*10}")
    
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
    
    # Learning rate reduction when plateau is reached
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\nTraining neural network...")
    history = mlp_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
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
    plt.savefig('award_share_training_history.png')
    print("Training history saved to 'award_share_training_history.png'")
    
    # Make predictions
    y_pred_award_share = mlp_model.predict(X_test).flatten()
    
    # Performance metrics for award_share prediction
    mse_award_share = mean_squared_error(y_test, y_pred_award_share)
    r2_award_share = r2_score(y_test, y_pred_award_share)
    
    print(f"\nAward Share Prediction Performance:")
    print(f"Mean Squared Error: {mse_award_share:.4f}")
    print(f"R-squared: {r2_award_share:.4f}")
    
    # 6. Convert predictions to rankings within each season
    print(f"\n{'-'*10} Converting Award Share Predictions to Rankings {'-'*10}")
    
    # Add predictions to test_data
    test_data['predicted_award_share'] = y_pred_award_share
    
    # Convert to ranks within each season
    ranked_test_data = convert_to_ranks(test_data)
    
    # Performance metrics for rank prediction
    y_test_rank = ranked_test_data['MVP_rank']
    y_pred_rank = ranked_test_data['predicted_rank']
    
    mse_rank = mean_squared_error(y_test_rank, y_pred_rank)
    acc_rank_exact = np.sum(y_pred_rank == y_test_rank) / len(y_test_rank)
    acc_rank_flex = flexible_rank_accuracy(y_test_rank, y_pred_rank)
    
    print(f"\nRank Prediction Performance:")
    print(f"Mean Squared Error: {mse_rank:.4f}")
    print(f"Exact Rank Accuracy: {acc_rank_exact:.4f} ({int(acc_rank_exact * len(y_test_rank))} exact matches out of {len(y_test_rank)})")
    print(f"Flexible Rank Accuracy: {acc_rank_flex:.4f} (includes 80% credit for being off by one or 20% for being off by two ranks)")
    
    # Calculate counts for each type of match
    rank_breakdown = get_prediction_breakdown(y_test_rank, y_pred_rank)
    
    print(f"Prediction breakdown:")
    print(f"  Exact matches: {rank_breakdown['exact']}")
    print(f"  Off by 1 rank: {rank_breakdown['off_by_one']}")
    print(f"  Off by 2 ranks: {rank_breakdown['off_by_two']}")
    print(f"  Off by >2 ranks: {rank_breakdown['off_by_more']}")
    
    # Save detailed predictions and create visualizations
    print("\nSaving detailed prediction results...")
    results_df = save_detailed_predictions(df_mvp, ranked_test_data)
    
    # Create additional visualizations
    print("Creating additional visualizations...")
    create_visualizations(results_df)
    
    # Print information about where to find the results
    print("\nDetailed Results Files:")
    print("- Complete prediction results: 'award_share_rank_prediction_results.csv'")
    print("- Prediction summary: 'award_share_rank_prediction_summary.csv'")
    print("- Yearly accuracy summary: 'award_share_rank_yearly_accuracy.csv'")
    print("\nVisualization Files:")
    print("- Confusion matrix: 'award_share_rank_confusion_matrix.png'")
    print("- Error distribution: 'award_share_rank_error_distribution.png'")
    print("- Seasonal accuracy: 'award_share_rank_accuracy_by_season.png'")
    print("- Award Share predictions: 'award_share_actual_vs_predicted.png'")
    print("- Rank predictions: 'award_share_derived_rank_actual_vs_predicted.png'")
    
    # Display all predictions
    print(f"\n{'-'*10} All Test Set Predictions {'-'*10}")
    all_preds = ranked_test_data[['season', 'player', 'award_share', 'predicted_award_share', 
                                 'MVP_rank', 'predicted_rank']]
    print(all_preds.sort_values(['season', 'predicted_rank']).to_string(index=False))
    
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

print(f"\n{'='*20} MVP Award Share Prediction for Ranking Finished {'='*20}") 