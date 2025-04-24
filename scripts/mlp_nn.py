import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Neural network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Add, LeakyReLU, LayerNormalization, Multiply
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
DATA_FILE = "../data/normalized_nba_data_with_MVP_rank_simple.csv"
OUTPUT_DIR = "../data/nn_evaluation_results"
PICS_DIR = "../pics"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICS_DIR, exist_ok=True)

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 423
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
PATIENCE = 30  # For early stopping

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

# Use these specific features for award_share based on feature selection
award_share_features = [
    'vorp', 'bpm', 'ws', 'ws_per_48', 'pts_per_g',
    'ft_per_g', 'fg_per_g', 'fta_per_g', 'fg2_per_g', 'fga_per_g'
]

# =============== MODEL BUILDING FUNCTIONS ===============

def build_mlp_model(input_shape):
    """
    Build advanced MLP neural network model with:
    - Multiple residual blocks
    - Attention mechanism
    - Bottleneck architecture
    - Varying dropout rates
    - Layer normalization
    """
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # First block
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)  # Higher dropout at early layers
    
    # Residual block 1 with bottleneck
    block_input = x
    # Bottleneck down
    x = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    # Bottleneck up
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Residual connection
    x = Add()([x, block_input])
    x = Dropout(0.3)(x)
    
    # Residual block 2
    block_input = x
    # Bottleneck down
    x = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    # Bottleneck up
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    # Residual connection
    x = Add()([x, block_input])
    x = Dropout(0.3)(x)
    
    # Self-attention mechanism for feature importance
    # Create a more straightforward attention mechanism for tabular data
    attention_features = Dense(64, activation='tanh', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    attention_weights = Dense(64, activation='softmax', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(attention_features)
    attention_output = Multiply()([x, attention_weights])
    attention_output = Dropout(0.2)(attention_output)
    
    # Split into three branches for multi-scale feature extraction
    # Wide branch for general patterns
    wide_branch = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    wide_branch = LayerNormalization()(wide_branch)
    wide_branch = LeakyReLU(alpha=0.1)(wide_branch)
    wide_branch = Dropout(0.2)(wide_branch)
    
    # Middle branch
    middle_branch = Dense(96, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    middle_branch = LayerNormalization()(middle_branch)
    middle_branch = LeakyReLU(alpha=0.1)(middle_branch)
    middle_branch = Dropout(0.2)(middle_branch)
    middle_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(middle_branch)
    middle_branch = LayerNormalization()(middle_branch)
    middle_branch = LeakyReLU(alpha=0.1)(middle_branch)
    middle_branch = Dropout(0.15)(middle_branch)
    
    # Deep branch for more complex patterns
    deep_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    deep_branch = LayerNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.2)(deep_branch)
    deep_branch = Dense(48, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(deep_branch)
    deep_branch = LayerNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.15)(deep_branch)
    deep_branch = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(deep_branch)
    deep_branch = LayerNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.1)(deep_branch)
    
    # Combine attention with branches
    merged = Concatenate()([attention_output, wide_branch, middle_branch, deep_branch])
    
    # Final layers with gradual reduction
    x = Dense(96, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(merged)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    
    x = Dense(48, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.05)(x)  # Lower dropout near output
    
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

# =============== PREDICTION EVALUATION FUNCTIONS ===============

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
    results_df.to_csv(f'{OUTPUT_DIR}/award_share_prediction_results.csv', index=False)
    
    # Create a separate summary for just vote-getters
    vote_getters_df = results_df[results_df['received_votes']]
    vote_getters_df.to_csv(f'{OUTPUT_DIR}/award_share_vote_getters_prediction.csv', index=False)
    
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
    
    yearly_summary.to_csv(f'{OUTPUT_DIR}/award_share_yearly_metrics.csv', index=False)
    
    return results_df

def create_prediction_visualizations(results_df):
    """
    Create visualizations for predicted award shares evaluation
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
    plt.savefig(f'{PICS_DIR}/award_share_error_distribution.png')
    
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
    plt.savefig(f'{PICS_DIR}/award_share_mse_by_season.png')
    
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
    plt.savefig(f'{PICS_DIR}/award_share_actual_vs_predicted.png')
    
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
    plt.savefig(f'{PICS_DIR}/award_share_vote_getters_actual_vs_predicted.png')
    
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
    plt.savefig(f'{PICS_DIR}/feature_correlation_with_award_share.png')

# =============== RANKING EVALUATION FUNCTIONS ===============

def convert_to_ranks(df, value_col, rank_col_name):
    """
    Convert values to ranks within each season (higher value = better rank)
    """
    # Create a temporary copy to avoid modifying the original
    temp_df = df.copy()
    
    # Group by season and rank the values (higher value = better rank)
    temp_df[rank_col_name] = temp_df.groupby('season')[value_col].rank(ascending=False, method='min')
    
    return temp_df

def calculate_rank_accuracy(true_ranks, pred_ranks, max_rank=5):
    """
    Calculate accuracy with partial credit:
    - Exact match: 1.0 (100% correct)
    - Off by 1 rank: 0.8 (80% credit)
    - Off by 2 ranks: 0.6 (60% credit)
    - Off by 3 ranks: 0.4 (40% credit)
    - Off by 4 ranks: 0.2 (20% credit)
    - Off by >4 ranks: 0.0 (0% credit)
    
    Only considers players with true_rank <= max_rank
    """
    # Filter to only include top max_rank players by true rank
    mask = true_ranks <= max_rank
    filtered_true = true_ranks[mask]
    filtered_pred = pred_ranks[mask]
    
    # If no players match the criteria, return 0
    if len(filtered_true) == 0:
        return 0
    
    # Calculate absolute difference
    rank_diff = np.abs(filtered_pred - filtered_true)
    
    # Assign scores based on rank difference
    scores = np.zeros_like(rank_diff, dtype=float)
    scores[rank_diff == 0] = 1.0    # Exact matches
    scores[rank_diff == 1] = 0.8    # Off by one rank
    scores[rank_diff == 2] = 0.6    # Off by two ranks
    scores[rank_diff == 3] = 0.4    # Off by three ranks
    scores[rank_diff == 4] = 0.2    # Off by four ranks
    scores[rank_diff > 4] = 0.0     # Off by five or more ranks
    
    # Return the sum of scores (not the average)
    return scores.sum()

def evaluate_ranking_performance(results_df):
    """
    Evaluate ranking performance based on award share predictions
    """
    print(f"\n{'-'*20} Ranking Evaluation {'-'*20}")
    
    # Convert award shares to ranks
    df = convert_to_ranks(results_df, 'award_share', 'actual_rank')
    df = convert_to_ranks(df, 'predicted_award_share', 'predicted_rank')
    
    # Calculate accuracy for each season
    seasons = df['season'].unique()
    seasons.sort()
    
    results = []
    total_score = 0
    total_possible = 0
    
    for season in seasons:
        season_df = df[df['season'] == season].copy()
        
        # Get number of top-5 players in this season
        top5_count = len(season_df[season_df['actual_rank'] <= 5])
        total_possible += top5_count
        
        # Calculate accuracy for this season
        score = calculate_rank_accuracy(
            season_df['actual_rank'].values,
            season_df['predicted_rank'].values
        )
        
        total_score += score
        
        # Get details for top 5 players
        top5_df = season_df[season_df['actual_rank'] <= 5].sort_values('actual_rank')
        player_details = []
        
        for _, row in top5_df.iterrows():
            player_detail = {
                'player': row['player'],
                'actual_rank': int(row['actual_rank']),
                'predicted_rank': int(row['predicted_rank']),
                'actual_award_share': row['award_share'],
                'predicted_award_share': row['predicted_award_share'],
                'rank_diff': int(row['predicted_rank'] - row['actual_rank']),
                'points': 0.0
            }
            
            # Calculate points for this player
            diff = abs(player_detail['rank_diff'])
            if diff == 0:
                player_detail['points'] = 1.0
            elif diff == 1:
                player_detail['points'] = 0.8
            elif diff == 2:
                player_detail['points'] = 0.6
            elif diff == 3:
                player_detail['points'] = 0.4
            elif diff == 4:
                player_detail['points'] = 0.2
            else:
                player_detail['points'] = 0.0
            
            player_details.append(player_detail)
        
        results.append({
            'season': season,
            'score': score,
            'max_score': top5_count,
            'accuracy': score / top5_count if top5_count > 0 else 0,
            'player_details': player_details
        })
    
    # 4. Display results by season
    print("\nRanking accuracy by season:")
    print(f"{'Season':<10} {'Score':<10} {'Accuracy':<10}")
    print("-" * 30)
    
    for r in results:
        print(f"{r['season']:<10} {r['score']:.1f}/{r['max_score']:.1f}{'':<5} {r['accuracy']*100:.1f}%")
    
    print("-" * 30)
    overall_accuracy = total_score / total_possible if total_possible > 0 else 0
    print(f"Overall: {total_score:.1f}/{total_possible:.1f} ({overall_accuracy*100:.1f}%)")
    
    return results, df

def create_ranking_visualizations(results, df):
    """
    Create visualizations for ranking evaluation
    """
    # Prepare data for detailed CSV
    detail_rows = []
    for r in results:
        for p in r['player_details']:
            detail_rows.append({
                'season': r['season'],
                'player': p['player'],
                'actual_rank': p['actual_rank'],
                'predicted_rank': p['predicted_rank'],
                'actual_award_share': p['actual_award_share'],
                'predicted_award_share': p['predicted_award_share'],
                'rank_diff': p['rank_diff'],
                'points': p['points']
            })
    
    detail_df = pd.DataFrame(detail_rows)
    
    # Reorder columns to have award shares side by side before accuracy metrics
    column_order = ['season', 'player', 'actual_rank', 'predicted_rank', 
                   'actual_award_share', 'predicted_award_share', 
                   'rank_diff', 'points']
    detail_df = detail_df[column_order]
    detail_df.to_csv(f"{OUTPUT_DIR}/top5_rank_evaluation.csv", index=False)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame([{
        'season': r['season'],
        'score': r['score'],
        'accuracy': r['accuracy']
    } for r in results])
    summary_df.to_csv(f"{OUTPUT_DIR}/season_accuracy_summary.csv", index=False)
    
    # Bar chart of accuracy by season
    plt.figure(figsize=(12, 6))
    sns.barplot(x='season', y='accuracy', data=summary_df, palette='viridis')
    plt.title('Top-5 Ranking Accuracy by Season')
    plt.xlabel('Season')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PICS_DIR}/rank_accuracy_by_season.png")
    
    # Distribution of rank differences
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rank_diff', data=detail_df, palette='viridis')
    plt.title('Distribution of Rank Prediction Errors (Top-5 Players Only)')
    plt.xlabel('Rank Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PICS_DIR}/rank_error_distribution.png")
    
    # Confusion matrix for top-5 ranks
    plt.figure(figsize=(10, 8))
    
    # Create confusion matrix
    cm = confusion_matrix(
        detail_df['actual_rank'],
        detail_df['predicted_rank'],
        labels=range(1, 6)
    )
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.xlabel('Predicted Rank')
    plt.ylabel('True Rank')
    plt.title('Confusion Matrix of Top-5 MVP Rank Predictions')
    plt.tight_layout()
    plt.savefig(f"{PICS_DIR}/rank_confusion_matrix.png")
    
    return detail_df

# =============== MAIN EXECUTION ===============

def main():
    print(f"{'='*20} MVP Award Share Prediction + Ranking Evaluation {'='*20}")

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
        
        # Split data for training and testing by season (year)
        print(f"\nSplitting data by seasons...")
        
        # Get unique seasons and sort them
        seasons = df_mvp['season'].unique()
        seasons.sort()
        print(f"Available seasons: {seasons}")
        
        # Number of seasons to use for testing (e.g., 20% of seasons)
        num_test_seasons = max(1, int(len(seasons) * TEST_SIZE))
        
        # Use the most recent seasons for testing
        test_seasons = seasons[-num_test_seasons:]
        train_seasons = seasons[:-num_test_seasons]
        
        print(f"Training seasons: {train_seasons}")
        print(f"Testing seasons: {test_seasons}")
        
        # Create train/test masks based on seasons
        train_mask = df_mvp['season'].isin(train_seasons)
        test_mask = df_mvp['season'].isin(test_seasons)
        
        # Split data using the masks
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]  
        y_test = y[test_mask]
        
        # Save test indices
        test_indices = df_mvp[test_mask].index
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Build and train the neural network model
        print(f"\n{'-'*20} Neural Network Training {'-'*20}")
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
        plt.savefig(f'{PICS_DIR}/award_share_training_history.png')
        print(f"Training history saved to '{PICS_DIR}/award_share_training_history.png'")
        
        # Make predictions
        print(f"\n{'-'*20} Prediction and Evaluation {'-'*20}")
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
        
        # Save detailed prediction results
        print("\nSaving detailed prediction results...")
        results_df = save_prediction_results(df_mvp, test_indices, y_test, y_pred_award_share)
        
        # Create prediction visualizations
        print("Creating prediction visualizations...")
        create_prediction_visualizations(results_df)
        
        # Evaluate ranking performance
        print("\nEvaluating ranking performance...")
        results, ranked_df = evaluate_ranking_performance(results_df)
        
        # Create ranking visualizations
        print("Creating ranking visualizations...")
        detail_df = create_ranking_visualizations(results, ranked_df)
        
        # Print detailed breakdown of each season's top 5 players
        print("\nDetailed breakdown of top-5 players by season:")
        
        for r in results:
            print(f"\nSeason: {r['season']} (Score: {r['score']:.1f}/{r['max_score']:.1f})")
            print(f"{'Player':<25} {'True Rank':<10} {'Pred Rank':<10} {'True Award':<12} {'Pred Award':<12} {'Diff':<6} {'Points':<6}")
            print("-" * 85)
            
            for p in r['player_details']:
                print(f"{p['player']:<25} {p['actual_rank']:<10} {p['predicted_rank']:<10} " +
                      f"{p['actual_award_share']:<12.3f} {p['predicted_award_share']:<12.3f} " +
                      f"{p['rank_diff']:<6} {p['points']:.1f}")
        
        # Save the model
        model_path = f"{OUTPUT_DIR}/award_share_prediction_model"
        mlp_model.save(model_path)
        print(f"\nModel saved to '{model_path}'")
        
        print(f"\n{'='*20} MLP Neural Network Evaluation Complete {'='*20}")
        print(f"Results saved to '{OUTPUT_DIR}/' directory")
        print(f"Visualizations saved to '{PICS_DIR}/' directory")
        
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

if __name__ == "__main__":
    main() 