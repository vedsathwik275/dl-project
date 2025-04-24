import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Neural network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_FILE = "../data/filtered_nba_data.csv"
OUTPUT_DIR = "../data/simple_nn_evaluation_results_ndcg"
PICS_DIR = "../pics/simple_ndcg"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICS_DIR, exist_ok=True)

# Training parameters
RANDOM_STATE = 423
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# =============== SIMPLE MODEL BUILDING FUNCTION ===============

def build_simple_model(input_shape):
    """
    Build a simple neural network for award share prediction following Occam's razor
    - Minimal architecture with just enough capacity to learn the patterns
    - Light regularization to prevent overfitting
    - Simple and interpretable
    
    Parameters:
    - input_shape: Shape of input features
    
    Returns:
    - Compiled Keras model
    """
    model = Sequential([
        # Input layer
        Dense(32, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_shape,)),
        Dropout(0.2),
        
        # Just one hidden layer is enough for this task
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # Output layer with sigmoid activation for [0,1] range
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with standard MSE loss
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# =============== PREDICTION EVALUATION FUNCTIONS ===============

def save_prediction_results(df, test_indices, y_test, y_pred):
    """
    Save prediction results to CSV with basic stats
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

def calculate_ndcg(true_values, pred_values, k=5):
    """
    Calculate NDCG (Normalized Discounted Cumulative Gain) for a ranking
    
    Parameters:
    - true_values: array of true relevance scores (award_share)
    - pred_values: array of predicted relevance scores (predicted_award_share)
    - k: number of top items to consider (default: 5)
    
    Returns:
    - NDCG score (0-1, with 1 being perfect ranking)
    """
    # If we don't have enough items, return 0
    if len(true_values) < k:
        return 0
    
    # Convert to numpy arrays
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)
    
    # Get indices that would sort the arrays in descending order
    true_indices = np.argsort(-true_values)
    pred_indices = np.argsort(-pred_values)
    
    # Create binary relevance matrix for scikit-learn ndcg_score
    # For each element, we need a binary vector where 1 indicates the items in the top-k
    true_relevance = np.zeros_like(true_values)
    true_relevance[true_indices[:k]] = 1
    
    # Reshape for ndcg_score format (expects 2D arrays)
    true_relevance = true_relevance.reshape(1, -1)
    y_score = pred_values.reshape(1, -1)
    
    # Calculate NDCG score
    return ndcg_score(true_relevance, y_score, k=k)

def evaluate_ranking_performance_ndcg(results_df, k=5):
    """
    Evaluate ranking performance using NDCG metric
    
    Parameters:
    - results_df: DataFrame with actual and predicted award shares
    - k: top-k players to consider (default: 5)
    
    Returns:
    - NDCG results by season and overall average
    """
    print(f"\n{'-'*20} NDCG Ranking Evaluation {'-'*20}")
    
    # Get unique seasons
    seasons = results_df['season'].unique()
    seasons.sort()
    
    ndcg_results = []
    all_ndcg_scores = []
    
    for season in seasons:
        season_df = results_df[results_df['season'] == season].copy()
        
        # Calculate NDCG for this season
        ndcg = calculate_ndcg(
            season_df['award_share'].values,
            season_df['predicted_award_share'].values,
            k=k
        )
        
        ndcg_results.append({
            'season': season,
            'ndcg': ndcg,
        })
        all_ndcg_scores.append(ndcg)
    
    # Display results by season
    print(f"\nNDCG@{k} by season:")
    print(f"{'Season':<10} {'NDCG@'+str(k):<10}")
    print("-" * 30)
    
    for r in ndcg_results:
        print(f"{r['season']:<10} {r['ndcg']:.4f}")
    
    # Calculate and display average NDCG
    average_ndcg = np.mean(all_ndcg_scores)
    print("-" * 30)
    print(f"Average NDCG@{k}: {average_ndcg:.4f}")
    
    return ndcg_results, average_ndcg

def save_top5_comparison(results_df, k=5):
    """
    Save a comparison of the top 5 actual players vs top 5 predicted players for each season
    """
    print(f"\n{'-'*20} Top {k} Actual vs Predicted Comparison {'-'*20}")
    
    # Get unique seasons
    seasons = results_df['season'].unique()
    seasons.sort()
    
    # Create an empty DataFrame to store all comparisons
    all_top_comparisons = pd.DataFrame()
    
    for season in seasons:
        season_df = results_df[results_df['season'] == season].copy()
        
        # Get top k by actual award share
        top_actual = season_df.nlargest(k, 'award_share')
        
        # Get top k by predicted award share
        top_predicted = season_df.nlargest(k, 'predicted_award_share')
        
        # Create a combined DataFrame for this season
        comparison_df = pd.DataFrame()
        comparison_df['season'] = [season] * k
        comparison_df['rank'] = list(range(1, k+1))
        
        # Add actual top players
        comparison_df['actual_player'] = top_actual['player'].values
        comparison_df['actual_award_share'] = top_actual['award_share'].values
        
        # Add predicted top players
        comparison_df['predicted_player'] = top_predicted['player'].values
        comparison_df['predicted_award_share'] = top_predicted['predicted_award_share'].values
        
        # Add whether prediction was correct (player in same position)
        comparison_df['correct_player'] = comparison_df['actual_player'] == comparison_df['predicted_player']
        
        # Append to the overall results
        all_top_comparisons = pd.concat([all_top_comparisons, comparison_df], ignore_index=True)
    
    # Save to CSV
    all_top_comparisons.to_csv(f"{OUTPUT_DIR}/top{k}_actual_vs_predicted.csv", index=False)
    
    # Print a summary
    for season in seasons:
        season_comp = all_top_comparisons[all_top_comparisons['season'] == season]
        correct_count = season_comp['correct_player'].sum()
        print(f"Season {season}: {correct_count}/{k} correct predictions in top {k}")
    
    # Overall accuracy
    overall_accuracy = all_top_comparisons['correct_player'].mean() * 100
    print(f"\nOverall accuracy for top {k} predictions: {overall_accuracy:.2f}%")
    
    return all_top_comparisons

# =============== MAIN EXECUTION ===============

def main():
    print(f"{'='*20} Simple Neural Network MVP Prediction {'='*20}")
    print("Using Occam's razor approach - simplicity often outperforms complexity")

    try:
        # 1. Load data
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded data. Shape: {df.shape}")
        
        # Handle missing values in award_share - set to 0 for players who didn't receive votes
        df['award_share'] = df['award_share'].fillna(0)
        
        # Identify non-statistical columns to exclude from normalization
        non_stat_columns = ['player', 'pos', 'team_id', 'season', 'award_share', 'MVP_rank', 'MVP_winner']
        
        # Step 1: Perform one-hot encoding for categorical variables
        print("Performing one-hot encoding for categorical variables...")
        
        # One-hot encode position (pos)
        pos_dummies = pd.get_dummies(df['pos'], prefix='pos')
        
        # One-hot encode team_id (limit to most common teams to avoid too many columns)
        # Get top 30 teams by frequency
        top_teams = df['team_id'].value_counts().nlargest(30).index
        # Replace less common teams with 'OTHER'
        df['team_id_grouped'] = df['team_id'].apply(lambda x: x if x in top_teams else 'OTHER')
        # Create dummies
        team_dummies = pd.get_dummies(df['team_id_grouped'], prefix='team')
        
        # Step 2: Get numeric columns excluding the non-statistical ones
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in non_stat_columns]
        print(f"Number of numeric features: {len(feature_columns)}")
        
        # Step 3: Normalize statistical features using StandardScaler
        print(f"Normalizing {len(feature_columns)} statistical features...")
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Step 4: Combine all features: normalized numeric + one-hot encoded categorical
        # Concatenate the original dataframe with the one-hot encoded features
        df_encoded = pd.concat([df[feature_columns], pos_dummies, team_dummies], axis=1)
        
        # Create feature matrix with all features
        X = df_encoded
        
        # Report final feature dimensions
        print(f"Final feature matrix shape after encoding: {X.shape}")
        print(f"Total number of features: {X.shape[1]}")
        
        # Target variable - award_share
        y = df['award_share']
        
        # Reset indices to ensure alignment after splitting
        X = X.reset_index(drop=True)
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Split data for training and testing by season (year)
        print(f"\nSplitting data by seasons...")
        
        # Get unique seasons and sort them
        seasons = df['season'].unique()
        seasons.sort()
        print(f"Available seasons: {seasons}")
        
        # Set the number of training seasons (33) and the rest for testing
        NUM_TRAIN_SEASONS = 33
        NUM_TEST_SEASONS = len(seasons) - NUM_TRAIN_SEASONS
        
        # Always include 2005 in the training set, then randomly select the rest
        np.random.seed(RANDOM_STATE)  # For reproducibility
        available_seasons = np.array([s for s in seasons if s != 2005])  # Remove 2005 from random selection pool
        random_train_seasons = np.random.choice(available_seasons, size=NUM_TRAIN_SEASONS-1, replace=False)  # Select one less
        train_seasons = np.append(random_train_seasons, [2005])  # Add 2005 to training set
        test_seasons = np.array([season for season in seasons if season not in train_seasons])
        
        print(f"Training seasons: {train_seasons}")
        print(f"Testing seasons: {test_seasons}")
        
        # Create train/test masks based on seasons
        train_mask = df['season'].isin(train_seasons)
        test_mask = df['season'].isin(test_seasons)
        
        # Split data using the masks
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]  
        y_test = y[test_mask]
        
        # Save test indices
        test_indices = df[test_mask].index
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Build and train the simple model
        print(f"\n{'-'*20} Simple Model Training {'-'*20}")
        simple_model = build_simple_model(X_train.shape[1])
        
        # Print model summary
        print(simple_model.summary())
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        print("\nTraining simple model...")
        history = simple_model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=2
        )
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PICS_DIR}/training_history.png')
        
        # Plot feature importance
        print("\nCalculating feature importance...")
        # Create a basic feature importance visualization based on weights of the first layer
        weights = simple_model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        
        # Create a DataFrame with feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Features by Importance')
        plt.tight_layout()
        plt.savefig(f'{PICS_DIR}/feature_importance.png')
        
        # Make predictions
        print(f"\n{'-'*20} Prediction and Evaluation {'-'*20}")
        y_pred = simple_model.predict(X_test).flatten()
        
        # Ensure predictions are in valid range (0-1)
        y_pred = np.clip(y_pred, 0, 1)
        
        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nAward Share Prediction Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        # Vote getters only metrics
        vote_getters_mask = y_test > 0
        if np.sum(vote_getters_mask) > 0:
            vote_getters_mse = mean_squared_error(y_test[vote_getters_mask], y_pred[vote_getters_mask])
            vote_getters_mae = mean_absolute_error(y_test[vote_getters_mask], y_pred[vote_getters_mask])
            vote_getters_r2 = r2_score(y_test[vote_getters_mask], y_pred[vote_getters_mask])
            
            print(f"\nAward Share Prediction (Vote Getters Only):")
            print(f"Mean Squared Error: {vote_getters_mse:.4f}")
            print(f"Mean Absolute Error: {vote_getters_mae:.4f}")
            print(f"R-squared: {vote_getters_r2:.4f}")
        
        # Save results
        results_df = save_prediction_results(df, test_indices, y_test, y_pred)
        
        # Evaluate ranking performance with NDCG
        results, average_ndcg = evaluate_ranking_performance_ndcg(results_df)
        
        # Save top5 comparison
        top5_comparison = save_top5_comparison(results_df)
        
        # Save important information to a summary file
        with open(f"{OUTPUT_DIR}/model_summary.txt", "w") as f:
            f.write(f"Simple Neural Network MVP Prediction Summary\n")
            f.write(f"================================================\n\n")
            f.write(f"Dataset: {DATA_FILE}\n")
            f.write(f"Number of features: {len(feature_columns)}\n")
            f.write(f"Training samples: {X_train.shape[0]}\n")
            f.write(f"Test samples: {X_test.shape[0]}\n\n")
            
            f.write(f"Model Architecture:\n")
            f.write(f"- Input layer: Dense(32, relu)\n")
            f.write(f"- Hidden layer: Dense(16, relu)\n")
            f.write(f"- Output layer: Dense(1, sigmoid)\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"- MSE: {mse:.4f}\n")
            f.write(f"- MAE: {mae:.4f}\n")
            f.write(f"- R²: {r2:.4f}\n")
            f.write(f"- Average NDCG@5: {average_ndcg:.4f}\n\n")
            
            f.write(f"Top 10 Most Important Features:\n")
            for i, row in importance_df.head(10).iterrows():
                f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
            
        print(f"\n{'='*20} Simple Model Evaluation Complete {'='*20}")
        print("Sometimes, less is more. The simple model often generalizes better.")
        print(f"Results saved to '{OUTPUT_DIR}/' directory")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 