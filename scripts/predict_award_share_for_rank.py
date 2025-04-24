import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
DATA_FILE = "../data/normalized_nba_data_with_MVP_rank_simple.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 423
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
PATIENCE = 30  # For early stopping
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

# Function to build improved MLP model with residual connections and better architecture
def build_mlp_model(input_shape):
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # First block with residual connection
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Residual block 1
    block_input = x
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, block_input])  # Residual connection
    x = Dropout(0.3)(x)
    
    # Split into two branches for feature extraction at different levels
    # Branch 1: Wide branch for general patterns
    wide_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    wide_branch = BatchNormalization()(wide_branch)
    wide_branch = LeakyReLU(alpha=0.1)(wide_branch)
    wide_branch = Dropout(0.2)(wide_branch)
    
    # Branch 2: Deep branch for more complex patterns
    deep_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.2)(deep_branch)
    deep_branch = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(deep_branch)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.2)(deep_branch)
    
    # Merge branches
    merged = Concatenate()([wide_branch, deep_branch])
    
    # Final layers
    x = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(merged)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    
    # Output layer - use linear activation and clip later
    output = Dense(1, activation='linear')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean squared error for continuous variable
        metrics=['mae']
    )
    
    return model

# Function to save detailed prediction results
def save_prediction_results(df, test_indices, y_test, y_pred):
    """
    Save comprehensive prediction results to CSV with basic stats
    """
    # Create results dataframe
    results_df = df.iloc[test_indices].copy()
    
    # Add predictions
    results_df['predicted_award_share'] = y_pred
    
    # Calculate error
    results_df['award_share_error'] = results_df['predicted_award_share'] - results_df['award_share']
    results_df['abs_error'] = np.abs(results_df['award_share_error'])
    
    # Create a flag for vote-getters
    results_df['received_votes'] = results_df['award_share'] > 0
    
    # Sort by season (descending) and actual award share (descending)
    results_df = results_df.sort_values(['season', 'award_share'], ascending=[False, False])
    
    # Save all results to CSV
    results_df.to_csv('../data/award_share_prediction_results.csv', index=False)
    
    # Create a separate summary for just vote-getters
    vote_getters_df = results_df[results_df['received_votes']]
    vote_getters_df.to_csv('../data/award_share_vote_getters_prediction.csv', index=False)
    
    # Generate summary stats by season
    yearly_summary = results_df.groupby('season').apply(
        lambda x: pd.Series({
            'num_players': len(x),
            'num_vote_getters': sum(x['received_votes']),
            'mse': mean_squared_error(x['award_share'], x['predicted_award_share']),
            'mae': mean_absolute_error(x['award_share'], x['predicted_award_share']),
            'r2': r2_score(x['award_share'], x['predicted_award_share'])
        })
    ).reset_index()
    
    yearly_summary.to_csv('../data/award_share_yearly_metrics.csv', index=False)
    
    return results_df

# Function to create visualizations
def create_visualizations(results_df):
    """
    Create visualizations for model evaluation
    """
    # Filter to only vote getters for some visualizations
    vote_getters_df = results_df[results_df['received_votes']].copy()
    
    # 1. Error distribution (all players)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='award_share_error', bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Award Share Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Award Share Prediction Errors')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../pics/award_share_error_distribution.png')
    
    # 2. Prediction accuracy by season
    plt.figure(figsize=(12, 6))
    yearly_metrics = results_df.groupby('season').apply(
        lambda x: mean_squared_error(x['award_share'], x['predicted_award_share'])
    ).reset_index()
    yearly_metrics.columns = ['season', 'mse']
    
    sns.barplot(x='season', y='mse', data=yearly_metrics, palette='viridis')
    plt.xlabel('Season')
    plt.ylabel('Mean Squared Error')
    plt.title('Award Share Prediction MSE by Season')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../pics/award_share_mse_by_season.png')
    
    # 3. Actual vs Predicted award_share scatter (all players)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results_df['award_share'], 
        results_df['predicted_award_share'],
        c=results_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=50
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
    plt.savefig('../pics/award_share_actual_vs_predicted.png')
    
    # 4. Actual vs Predicted award_share scatter (only vote getters for clarity)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        vote_getters_df['award_share'], 
        vote_getters_df['predicted_award_share'],
        c=vote_getters_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add perfect prediction line
    max_val = max(vote_getters_df['award_share'].max(), vote_getters_df['predicted_award_share'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.colorbar(scatter, label='Season')
    plt.xlabel('Actual Award Share')
    plt.ylabel('Predicted Award Share')
    plt.title('Actual vs Predicted Award Share (Vote Getters Only)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../pics/award_share_vote_getters_actual_vs_predicted.png')
    
    # 5. Feature importance 
    # Can't directly get feature importance from neural network
    # But we can look at correlations with award_share
    feature_corr = pd.DataFrame(index=award_share_features)
    feature_corr['correlation'] = [results_df[feature].corr(results_df['award_share']) for feature in award_share_features]
    feature_corr = feature_corr.sort_values('correlation', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='correlation', y=feature_corr.index, data=feature_corr, palette='viridis')
    plt.title('Feature Correlation with Award Share')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('../pics/feature_correlation_with_award_share.png')

# Main execution
print(f"{'='*20} MVP Award Share Prediction {'='*20}")

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
    essential_cols = all_features + ['season', 'player']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing essential columns: {missing_cols}")
    
    # Keep all players, don't filter by MVP_rank or award_share
    df_mvp = df.copy()
    
    # Handle missing values in award_share - set to 0 for players who didn't receive votes
    df_mvp['award_share'] = df_mvp['award_share'].fillna(0)
    
    print(f"Total number of player seasons: {len(df_mvp)}")
    
    # Convert data types and handle missing values
    # Convert feature columns to numeric
    for col in all_features:
        if col in df_mvp.columns:
            df_mvp[col] = pd.to_numeric(df_mvp[col], errors='coerce')
    
    # Drop rows with NaN in features (but keep rows even if award_share is 0)
    df_mvp.dropna(subset=all_features, inplace=True)
    print(f"After dropping NaNs in features: {len(df_mvp)} rows")
    
    # Display distribution of award shares
    vote_getters = df_mvp[df_mvp['award_share'] > 0]
    print(f"\nNumber of players who received MVP votes: {len(vote_getters)}")
    print(f"Number of players who didn't receive MVP votes: {len(df_mvp) - len(vote_getters)}")
    
    # Display award_share statistics
    print(f"\nAward Share statistics:")
    print(f"Mean: {df_mvp['award_share'].mean():.4f}")
    print(f"Non-zero Mean: {df_mvp[df_mvp['award_share'] > 0]['award_share'].mean():.4f}")
    print(f"Min: {df_mvp['award_share'].min():.4f}")
    print(f"Max: {df_mvp['award_share'].max():.4f}")
    print(f"Median: {df_mvp['award_share'].median():.4f}")
    print(f"Percentage of players with votes: {len(vote_getters) / len(df_mvp) * 100:.2f}%")
    
    # Create feature matrix with only selected features
    X = df_mvp[award_share_features]
    
    # Target variable - award_share
    y = df_mvp['award_share']
    
    # Reset indices to ensure alignment after splitting
    X = X.reset_index(drop=True)
    df_mvp = df_mvp.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 4. Split data for training and testing - random selection of seasons 
    print(f"\n{'-'*10} Train-Test Split {'-'*10}")

    # Get unique seasons
    seasons = df_mvp['season'].unique()
    print(f"Total number of unique seasons: {len(seasons)}")

    # Set number of training seasons
    TRAIN_SEASONS_COUNT = 33
    TEST_SEASONS_COUNT = len(seasons) - TRAIN_SEASONS_COUNT

    # Randomly select training seasons
    np.random.seed(RANDOM_STATE)  # For reproducibility
    train_seasons = np.random.choice(seasons, size=TRAIN_SEASONS_COUNT, replace=False)
    test_seasons = np.array([s for s in seasons if s not in train_seasons])

    print(f"Training on {TRAIN_SEASONS_COUNT} randomly selected seasons, testing on {TEST_SEASONS_COUNT} seasons")
    print(f"Training seasons: {sorted(train_seasons)}")
    print(f"Test seasons: {sorted(test_seasons)}")

    # Split based on selected seasons
    train_mask = df_mvp['season'].isin(train_seasons)
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    # Save test indices for later use
    test_indices = df_mvp[~train_mask].index

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Training data distribution by era
    train_seasons_df = pd.DataFrame({'season': train_seasons})
    train_seasons_df['era'] = train_seasons_df['season'].apply(lambda s: 
        "Physical Play (1980s)" if 1980 <= s <= 1989 else
        "Isolation (1995-2010)" if 1995 <= s <= 2010 else
        "Analytics/3PT (2011+)" if s >= 2011 else "Other"
    )
    train_era_counts = train_seasons_df['era'].value_counts()
    print("\nTraining seasons distribution by era:")
    for era, count in train_era_counts.items():
        print(f"  {era}: {count} seasons")

    # Test data distribution by era  
    test_seasons_df = pd.DataFrame({'season': test_seasons})
    test_seasons_df['era'] = test_seasons_df['season'].apply(lambda s: 
        "Physical Play (1980s)" if 1980 <= s <= 1989 else
        "Isolation (1995-2010)" if 1995 <= s <= 2010 else
        "Analytics/3PT (2011+)" if s >= 2011 else "Other"
    )
    test_era_counts = test_seasons_df['era'].value_counts()
    print("\nTest seasons distribution by era:")
    for era, count in test_era_counts.items():
        print(f"  {era}: {count} seasons")
    
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
        patience=15,
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
    plt.savefig('../pics/award_share_training_history.png')
    print("Training history saved to 'award_share_training_history.png'")
    
    # Make predictions
    y_pred_award_share = mlp_model.predict(X_test).flatten()
    
    # Ensure predictions are in valid range (0-1)
    y_pred_award_share = np.clip(y_pred_award_share, 0, 1)
    
    # Performance metrics for award_share prediction
    mse = mean_squared_error(y_test, y_pred_award_share)
    mae = mean_absolute_error(y_test, y_pred_award_share)
    r2 = r2_score(y_test, y_pred_award_share)
    
    print(f"\nAward Share Prediction Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    # Performance metrics for vote getters only
    vote_getters_mask = y_test > 0
    if np.sum(vote_getters_mask) > 0:
        vote_getters_mse = mean_squared_error(y_test[vote_getters_mask], y_pred_award_share[vote_getters_mask])
        vote_getters_mae = mean_absolute_error(y_test[vote_getters_mask], y_pred_award_share[vote_getters_mask])
        vote_getters_r2 = r2_score(y_test[vote_getters_mask], y_pred_award_share[vote_getters_mask])
        
        print(f"\nAward Share Prediction Performance (Vote Getters Only):")
        print(f"Mean Squared Error: {vote_getters_mse:.4f}")
        print(f"Mean Absolute Error: {vote_getters_mae:.4f}")
        print(f"R-squared: {vote_getters_r2:.4f}")
    
    # Save detailed predictions and create visualizations
    print("\nSaving detailed prediction results...")
    results_df = save_prediction_results(df_mvp, test_indices, y_test, y_pred_award_share)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results_df)
    
    # Print information about where to find the results
    print("\nDetailed Results Files:")
    print("- Complete prediction results: 'award_share_prediction_results.csv'")
    print("- Vote getters predictions: 'award_share_vote_getters_prediction.csv'")
    print("- Yearly metrics summary: 'award_share_yearly_metrics.csv'")
    print("\nVisualization Files:")
    print("- Error distribution: 'award_share_error_distribution.png'")
    print("- MSE by season: 'award_share_mse_by_season.png'")
    print("- Actual vs Predicted (All players): 'award_share_actual_vs_predicted.png'")
    print("- Actual vs Predicted (Vote getters): 'award_share_vote_getters_actual_vs_predicted.png'")
    print("- Feature correlation: 'feature_correlation_with_award_share.png'")
    
    # Save the model if needed
    mlp_model.save("award_share_prediction_model")
    print("Model saved to 'award_share_prediction_model'")
    
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