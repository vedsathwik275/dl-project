import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_FILE = "../../data/normalized_nba_data_with_MVP_rank_simple.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = "../../data/ml_ndcg_evaluation_results"  # Directory to save evaluation results
PICS_DIR = "../../pics/ndcg"  # Directory to save visualizations

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICS_DIR, exist_ok=True)

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

# Ranking and evaluation functions
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

def evaluate_model_rankings(test_data, model_name, pred_col):
    """
    Evaluate the ranking performance of a model
    """
    # Convert award share values to ranks
    df = convert_to_ranks(test_data, 'award_share', 'actual_rank')
    df = convert_to_ranks(df, pred_col, 'predicted_rank')
    
    # Calculate accuracy for each season
    seasons = df['season'].unique()
    seasons.sort()
    
    results = []
    total_score = 0
    total_possible = 0
    
    for season in seasons:
        season_df = df[df['season'] == season].copy()
        
        # Skip seasons with no top 5 players in the test set
        top5_count = len(season_df[season_df['actual_rank'] <= 5])
        if top5_count == 0:
            continue
            
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
                'predicted_award_share': row[pred_col],
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
    
    # Calculate overall accuracy
    overall_accuracy = total_score / total_possible if total_possible > 0 else 0
    
    # Display results by season
    print(f"\n{'-'*10} {model_name} Ranking Evaluation {'-'*10}")
    print(f"{'Season':<10} {'Score':<10} {'Accuracy':<10}")
    print("-" * 30)
    
    for r in results:
        print(f"{r['season']:<10} {r['score']:.1f}/{r['max_score']:.1f}{'':<5} {r['accuracy']*100:.1f}%")
    
    print("-" * 30)
    print(f"Overall: {total_score:.1f}/{total_possible:.1f} ({overall_accuracy*100:.1f}%)")
    
    return {
        'model_name': model_name,
        'results': results,
        'total_score': total_score,
        'total_possible': total_possible,
        'overall_accuracy': overall_accuracy
    }

def generate_evaluation_visualizations(eval_results, model_name):
    """Generate visualization charts for ranking evaluation"""
    # Prepare data for plots
    detail_rows = []
    for r in eval_results['results']:
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
    
    if not detail_rows:  # Check if there are any results
        print(f"No evaluation data available for {model_name}")
        return None
    
    detail_df = pd.DataFrame(detail_rows)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'season': r['season'],
        'score': r['score'],
        'accuracy': r['accuracy']
    } for r in eval_results['results']])
    
    # Bar chart of accuracy by season
    plt.figure(figsize=(12, 6))
    sns.barplot(x='season', y='accuracy', data=summary_df, palette='viridis')
    plt.title(f'{model_name}: Top-5 Ranking Accuracy by Season')
    plt.xlabel('Season')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_accuracy_by_season.png")
    
    # Distribution of rank differences
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rank_diff', data=detail_df, palette='viridis')
    plt.title(f'{model_name}: Distribution of Rank Prediction Errors (Top-5 Players Only)')
    plt.xlabel('Rank Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_error_distribution.png")
    
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
    plt.title(f'{model_name}: Confusion Matrix of Top-5 MVP Rank Predictions')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    
    return detail_df

def save_detailed_evaluation(model_name, detail_df):
    """Save detailed evaluation results to CSV files"""
    if detail_df is None or detail_df.empty:
        return
    
    # Reorder columns to match the requested format
    column_order = ['season', 'player', 'actual_rank', 'predicted_rank', 
                    'actual_award_share', 'predicted_award_share', 
                    'rank_diff', 'points']
    detail_df = detail_df[column_order]
    
    # Save to CSV
    detail_df.to_csv(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_top5_evaluation.csv", index=False)

# New NDCG evaluation functions
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

def evaluate_ranking_performance_ndcg(results_df, pred_col, model_name, k=5):
    """
    Evaluate ranking performance using NDCG metric
    
    Parameters:
    - results_df: DataFrame with actual and predicted award shares
    - pred_col: column name with predicted award shares
    - model_name: name of the model for display
    - k: top-k players to consider (default: 5)
    
    Returns:
    - NDCG results by season and overall average
    """
    print(f"\n{'-'*20} {model_name} NDCG Ranking Evaluation {'-'*20}")
    
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
            season_df[pred_col].values,
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

def create_ranking_visualizations_ndcg(ndcg_results, model_name, k=5):
    """
    Create visualizations for NDCG evaluation
    """
    # Create a DataFrame for easier plotting
    ndcg_df = pd.DataFrame(ndcg_results)
    
    # Save NDCG results to CSV
    ndcg_df.to_csv(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_ndcg_results.csv", index=False)
    
    # Bar chart of NDCG by season
    plt.figure(figsize=(12, 6))
    sns.barplot(x='season', y='ndcg', data=ndcg_df, palette='viridis')
    plt.title(f'{model_name}: NDCG@{k} by Season')
    plt.xlabel('Season')
    plt.ylabel(f'NDCG@{k}')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PICS_DIR}/{model_name.lower().replace(' ', '_')}_ndcg_by_season.png")
    
    # Calculate average NDCG
    average_ndcg = ndcg_df['ndcg'].mean()
    
    # Create a DataFrame with season-wise and average NDCG
    summary_df = ndcg_df.copy()
    summary_df['average_ndcg'] = average_ndcg
    summary_df.to_csv(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_ndcg_summary.csv", index=False)
    
    return summary_df

def save_top5_comparison(results_df, pred_col, model_name, k=5):
    """
    Save a comparison of the top 5 actual players vs top 5 predicted players for each season
    
    Parameters:
    - results_df: DataFrame with actual and predicted award shares
    - pred_col: column name with predicted award shares
    - model_name: name of the model for display
    - k: Number of top players to include (default: 5)
    
    Returns:
    - DataFrame with top-k comparison for each season
    """
    print(f"\n{'-'*20} {model_name} Top {k} Actual vs Predicted Comparison {'-'*20}")
    
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
        top_predicted = season_df.nlargest(k, pred_col)
        
        # Create a combined DataFrame for this season
        comparison_df = pd.DataFrame()
        comparison_df['season'] = [season] * k
        comparison_df['rank'] = list(range(1, k+1))
        
        # Add actual top players
        comparison_df['actual_player'] = top_actual['player'].values
        comparison_df['actual_award_share'] = top_actual['award_share'].values
        
        # Add predicted top players
        comparison_df['predicted_player'] = top_predicted['player'].values
        comparison_df['predicted_award_share'] = top_predicted[pred_col].values
        
        # Add whether prediction was correct (player in same position)
        comparison_df['correct_player'] = comparison_df['actual_player'] == comparison_df['predicted_player']
        
        # Append to the overall results
        all_top_comparisons = pd.concat([all_top_comparisons, comparison_df], ignore_index=True)
    
    # Save to CSV
    all_top_comparisons.to_csv(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_top{k}_actual_vs_predicted.csv", index=False)
    
    # Print a summary
    for season in seasons:
        season_comp = all_top_comparisons[all_top_comparisons['season'] == season]
        correct_count = season_comp['correct_player'].sum()
        print(f"Season {season}: {correct_count}/{k} correct predictions in top {k}")
    
    # Overall accuracy
    overall_accuracy = all_top_comparisons['correct_player'].mean() * 100
    print(f"\nOverall accuracy for top {k} predictions: {overall_accuracy:.2f}%")
    
    return all_top_comparisons

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
    
    # 3. Split data for training and testing by season (year)
    print(f"\nSplitting data by seasons...")
    
    # Get unique seasons and sort them
    seasons = df_processed['season'].unique()
    seasons.sort()
    print(f"Available seasons: {seasons}")
    
    # Set the number of training seasons (33) and testing seasons (8)
    NUM_TRAIN_SEASONS = 33
    NUM_TEST_SEASONS = len(seasons) - NUM_TRAIN_SEASONS
    
    # Randomly select seasons for training and testing
    np.random.seed(RANDOM_STATE)  # For reproducibility
    train_seasons = np.random.choice(seasons, size=NUM_TRAIN_SEASONS, replace=False)
    test_seasons = np.array([season for season in seasons if season not in train_seasons])
    
    print(f"Training seasons: {train_seasons}")
    print(f"Testing seasons: {test_seasons}")
    
    # Create train/test masks based on seasons
    train_mask = df_processed['season'].isin(train_seasons)
    test_mask = df_processed['season'].isin(test_seasons)
    
    # Split data using the masks
    X_train = X_normalized[train_mask]
    y_train = y[train_mask]
    X_test = X_normalized[test_mask]  
    y_test = y[test_mask]
    
    # Save test indices
    test_indices = df_processed[test_mask].index
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of training seasons: {len(train_seasons)}")
    print(f"Number of testing seasons: {len(test_seasons)}")
    
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
    print(f"\nSaving test predictions to '../data/ml_predictions.csv'...")
    output_cols = ['season', 'player', 'award_share', 'LR_pred_share', 'RF_pred_share', 'GB_pred_share'] + features
    test_data_to_save = test_data[output_cols].copy()
    # Optionally format the prediction columns before saving
    for col in ['LR_pred_share', 'RF_pred_share', 'GB_pred_share']:
         test_data_to_save[col] = test_data_to_save[col].map('{:.6f}'.format) # Use more precision for CSV
    test_data_to_save.sort_values(by=['season', 'award_share'], ascending=[False, False], inplace=True)
    test_data_to_save.to_csv('../../data/ml_predictions.csv', index=False)
    print("Predictions saved.")
    
    # 6. Evaluate ranking performance for each model
    print(f"\n{'-'*10} MVP Rank Evaluation {'-'*10}")
    print("Evaluating model performance based on ranking accuracy...")
    
    # Evaluate Linear Regression model
    lr_results = evaluate_model_rankings(test_data, "Linear Regression", 'LR_pred_share')
    lr_detail_df = generate_evaluation_visualizations(lr_results, "Linear Regression")
    save_detailed_evaluation("Linear_Regression", lr_detail_df)
    
    # Evaluate Random Forest model
    rf_results = evaluate_model_rankings(test_data, "Random Forest", 'RF_pred_share')
    rf_detail_df = generate_evaluation_visualizations(rf_results, "Random Forest")
    save_detailed_evaluation("Random_Forest", rf_detail_df)
    
    # Evaluate Gradient Boosting model
    gb_results = evaluate_model_rankings(test_data, "Gradient Boosting", 'GB_pred_share')
    gb_detail_df = generate_evaluation_visualizations(gb_results, "Gradient Boosting")
    save_detailed_evaluation("Gradient_Boosting", gb_detail_df)
    
    # 7. NDCG Evaluation for each model
    print(f"\n{'-'*10} NDCG Evaluation {'-'*10}")
    print("Evaluating model performance using NDCG metric...")
    
    # Evaluate Linear Regression model using NDCG
    lr_ndcg_results, lr_avg_ndcg = evaluate_ranking_performance_ndcg(test_data, 'LR_pred_share', "Linear Regression")
    lr_ndcg_summary = create_ranking_visualizations_ndcg(lr_ndcg_results, "Linear Regression")
    lr_top5 = save_top5_comparison(test_data, 'LR_pred_share', "Linear Regression")
    
    # Evaluate Random Forest model using NDCG
    rf_ndcg_results, rf_avg_ndcg = evaluate_ranking_performance_ndcg(test_data, 'RF_pred_share', "Random Forest")
    rf_ndcg_summary = create_ranking_visualizations_ndcg(rf_ndcg_results, "Random Forest")
    rf_top5 = save_top5_comparison(test_data, 'RF_pred_share', "Random Forest")
    
    # Evaluate Gradient Boosting model using NDCG
    gb_ndcg_results, gb_avg_ndcg = evaluate_ranking_performance_ndcg(test_data, 'GB_pred_share', "Gradient Boosting")
    gb_ndcg_summary = create_ranking_visualizations_ndcg(gb_ndcg_results, "Gradient Boosting")
    gb_top5 = save_top5_comparison(test_data, 'GB_pred_share', "Gradient Boosting")
    
    # Create a model comparison summary with NDCG
    print("\nCreating model comparison summary with NDCG...")
    model_comparison = pd.DataFrame([
        {'Model': 'Linear Regression', 
         'Accuracy Score': lr_results['total_score'], 
         'Possible Score': lr_results['total_possible'],
         'Accuracy': lr_results['overall_accuracy'],
         'NDCG@5': lr_avg_ndcg},
        {'Model': 'Random Forest', 
         'Accuracy Score': rf_results['total_score'], 
         'Possible Score': rf_results['total_possible'],
         'Accuracy': rf_results['overall_accuracy'],
         'NDCG@5': rf_avg_ndcg},
        {'Model': 'Gradient Boosting', 
         'Accuracy Score': gb_results['total_score'], 
         'Possible Score': gb_results['total_possible'],
         'Accuracy': gb_results['overall_accuracy'],
         'NDCG@5': gb_avg_ndcg}
    ])
    model_comparison.to_csv(f"{OUTPUT_DIR}/model_comparison_with_ndcg.csv", index=False)
    
    print(f"\nEvaluation results saved to '{OUTPUT_DIR}/' directory")
    print(f"Visualization results saved to '{PICS_DIR}/' directory")

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