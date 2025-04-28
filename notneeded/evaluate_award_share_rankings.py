import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Configuration
PREDICTION_FILE = "../data/award_share_prediction_results.csv"
OUTPUT_DIR = "../data/evaluation_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    - Off by 2 ranks: 0.2 (20% credit)
    - Off by >2 ranks: 0.0 (incorrect)
    
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
    scores[rank_diff > 4] = 0.0    # Off by five or more ranks
    
    # Return the sum of scores (not the average)
    return scores.sum()

def main():
    print(f"{'='*20} Award Share Ranking Evaluation {'='*20}")
    
    try:
        # 1. Load the prediction results file
        print(f"Loading prediction results from {PREDICTION_FILE}...")
        df = pd.read_csv(PREDICTION_FILE)
        print(f"Loaded data. Shape: {df.shape}")
        
        # 2. Convert award shares to ranks within each season
        print("\nConverting award shares to ranks within each season...")
        df = convert_to_ranks(df, 'award_share', 'actual_rank')
        df = convert_to_ranks(df, 'predicted_award_share', 'predicted_rank')
        
        # 3. Calculate accuracy for each season
        print("\nCalculating ranking accuracy for each season...")
        seasons = df['season'].unique()
        seasons.sort()
        
        results = []
        total_score = 0
        total_possible = len(seasons) * 5  # 5 ranks per season
        
        for season in seasons:
            season_df = df[df['season'] == season].copy()
            
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
                'max_score': 5,
                'accuracy': score / 5,
                'player_details': player_details
            })
        
        # 4. Display results by season
        print("\nRanking accuracy by season:")
        print(f"{'Season':<10} {'Score':<10} {'Accuracy':<10}")
        print("-" * 30)
        
        for r in results:
            print(f"{r['season']:<10} {r['score']:.1f}/5.0{'':<5} {r['accuracy']*100:.1f}%")
        
        print("-" * 30)
        overall_accuracy = total_score / total_possible
        print(f"Overall: {total_score:.1f}/{total_possible:.1f} ({overall_accuracy*100:.1f}%)")
        
        # 5. Save detailed results to CSV
        print("\nSaving detailed results...")
        
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
        
        # 6. Create a summary DataFrame
        summary_df = pd.DataFrame([{
            'season': r['season'],
            'score': r['score'],
            'accuracy': r['accuracy']
        } for r in results])
        summary_df.to_csv(f"{OUTPUT_DIR}/season_accuracy_summary.csv", index=False)
        
        # 7. Create visualizations
        print("\nGenerating visualizations...")
        
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
        plt.savefig(f"{OUTPUT_DIR}/rank_accuracy_by_season.png")
        
        # Distribution of rank differences
        plt.figure(figsize=(10, 6))
        sns.countplot(x='rank_diff', data=detail_df, palette='viridis')
        plt.title('Distribution of Rank Prediction Errors (Top-5 Players Only)')
        plt.xlabel('Rank Error (Predicted - Actual)')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/rank_error_distribution.png")
        
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
        plt.savefig(f"{OUTPUT_DIR}/rank_confusion_matrix.png")
        
        # 8. Print detailed breakdown of each season's top 5 players
        print("\nDetailed breakdown of top-5 players by season:")
        
        for r in results:
            print(f"\nSeason: {r['season']} (Score: {r['score']:.1f}/5.0)")
            print(f"{'Player':<25} {'True Rank':<10} {'Pred Rank':<10} {'True Award':<12} {'Pred Award':<12} {'Diff':<6} {'Points':<6}")
            print("-" * 85)
            
            for p in r['player_details']:
                print(f"{p['player']:<25} {p['actual_rank']:<10} {p['predicted_rank']:<10} " +
                      f"{p['actual_award_share']:<12.3f} {p['predicted_award_share']:<12.3f} " +
                      f"{p['rank_diff']:<6} {p['points']:.1f}")
        
        print(f"\n{'='*20} Evaluation Complete {'='*20}")
        print(f"Results saved to {OUTPUT_DIR}/ directory")
        
    except FileNotFoundError:
        print(f"Error: The file {PREDICTION_FILE} was not found.")
        print("Make sure to run the award share prediction script first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 