import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
INPUT_FILE = "../data/filtered_nba_data_with_MVP_rank.csv"
OUTPUT_FILE = "../data/normalized_nba_data_with_MVP_rank.csv"

# Define eras
ERAS = {
    "Physical Play": (1980, 1989),
    "Isolation": (1995, 2010),
    "Analytics/3PT": (2011, 2023)  # Adjust end year as needed
}

# Statistical features to normalize
STAT_FEATURES = [
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

# List of features we want to keep unnormalized
KEEP_ORIGINAL = ['player', 'season', 'team', 'MVP_rank', 'award_share']

def determine_era(season):
    """
    Determine which era a season belongs to.
    Returns era name or 'Other' if it doesn't fit any defined era.
    """
    for era_name, (start, end) in ERAS.items():
        if start <= season <= end:
            return era_name
    return "Other"

def normalize_stats():
    """
    Load NBA data, normalize statistical features within each season, and save results.
    """
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check if all required features exist
        missing_features = [col for col in STAT_FEATURES if col not in df.columns]
        if missing_features:
            print(f"Warning: The following features are missing: {missing_features}")
            # Remove missing features from the list to normalize
            for feat in missing_features:
                STAT_FEATURES.remove(feat)
        
        # Ensure the season column exists
        if 'season' not in df.columns:
            raise ValueError("The dataset must have a 'season' column to perform era-based normalization.")
        
        # Convert numeric columns properly
        for col in STAT_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add era column
        df['era'] = df['season'].apply(determine_era)
        
        # Count players in each era
        era_counts = df['era'].value_counts()
        print("\nPlayers per era:")
        for era, count in era_counts.items():
            print(f"  {era}: {count} players")
        
        # Create empty DataFrame to store normalized data
        df_normalized = pd.DataFrame()
        
        # Add non-statistical columns directly
        for col in KEEP_ORIGINAL:
            if col in df.columns:
                df_normalized[col] = df[col]
        
        # Add era column
        df_normalized['era'] = df['era']
        
        # Store original values in a separate DataFrame for validation
        df_original = df[STAT_FEATURES].copy()
        
        # Normalize within each season
        print("\nNormalizing statistics within each season...")
        
        # Group by season and normalize each group separately
        for season, season_df in df.groupby('season'):
            era = determine_era(season)
            print(f"  Normalizing season {season} (Era: {era})")
            
            # Create a scaler for this season's data
            scaler = StandardScaler()
            
            # Extract only the statistical features for this season
            season_stats = season_df[STAT_FEATURES].copy()
            
            # Replace any NaN values with season mean before scaling
            for col in season_stats.columns:
                season_stats[col] = season_stats[col].fillna(season_stats[col].mean())
            
            # Fit and transform the data
            normalized_stats = pd.DataFrame(
                scaler.fit_transform(season_stats),
                columns=STAT_FEATURES,
                index=season_stats.index
            )
            
            # Add the normalized stats to the main DataFrame
            for col in STAT_FEATURES:
                df_normalized.loc[season_stats.index, f"norm_{col}"] = normalized_stats[col]
        
        # Validate the normalization (check that each season's features have mean ≈ 0 and std ≈ 1)
        print("\nValidating normalization...")
        validation_results = []
        
        for season, season_df in df_normalized.groupby('season'):
            for col in STAT_FEATURES:
                norm_col = f"norm_{col}"
                if norm_col in season_df.columns:
                    mean_val = season_df[norm_col].mean()
                    std_val = season_df[norm_col].std()
                    validation_results.append({
                        'season': season,
                        'feature': col,
                        'mean': mean_val,
                        'std': std_val
                    })
        
        validation_df = pd.DataFrame(validation_results)
        mean_deviation = np.abs(validation_df['mean']).mean()
        std_deviation = np.abs(validation_df['std'] - 1).mean()
        
        print(f"Average deviation of means from zero: {mean_deviation:.4f}")
        print(f"Average deviation of standard deviations from one: {std_deviation:.4f}")
        
        if mean_deviation < 0.1 and std_deviation < 0.1:
            print("Normalization validation successful! ✓")
        else:
            print("Warning: Normalization may not be optimal. Check for seasons with few players.")
        
        # Generate visualizations to compare original vs. normalized distributions
        print("\nGenerating visualization of original vs. normalized distributions...")
        
        # Select a sample of important features to visualize
        viz_features = ['pts_per_g', 'fg3_per_g', 'ast_per_g', 'ws', 'vorp']
        viz_features = [f for f in viz_features if f in STAT_FEATURES]
        
        for feature in viz_features:
            norm_feature = f"norm_{feature}"
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(data=df, x=feature, hue='era', kde=True, alpha=0.4)
            plt.title(f'Original {feature} Distribution by Era')
            plt.xlabel(feature)
            
            plt.subplot(1, 2, 2)
            sns.histplot(data=df_normalized, x=norm_feature, hue='era', kde=True, alpha=0.4)
            plt.title(f'Normalized {feature} Distribution by Era')
            plt.xlabel(f'Normalized {feature}')
            
            plt.tight_layout()
            plt.savefig(f'era_normalized_{feature}_distribution.png')
            print(f"  Saved visualization: era_normalized_{feature}_distribution.png")
        
        # Save the normalized dataset
        print(f"\nSaving normalized data to {OUTPUT_FILE}...")
        df_normalized.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved normalized data with shape: {df_normalized.shape}")
        
        # Also save a version with only normalized features replacing the originals (same structure as input)
        df_normalized_simple = df.copy()
        for col in STAT_FEATURES:
            norm_col = f"norm_{col}"
            if norm_col in df_normalized.columns:
                df_normalized_simple[col] = df_normalized[norm_col]
        
        simple_output = OUTPUT_FILE.replace('.csv', '_simple.csv')
        df_normalized_simple.to_csv(simple_output, index=False)
        print(f"Saved simplified normalized data to {simple_output}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: The file {INPUT_FILE} was not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("NBA Data Normalization by Season and Era")
    print("=" * 80)
    
    success = normalize_stats()
    
    if success:
        print("\nNormalization complete! The data has been normalized within each season.")
        print("Each normalized feature has prefix 'norm_' in the output file.")
        print("\nTwo output files were created:")
        print(f"1. {OUTPUT_FILE} - Contains original features plus normalized versions")
        print(f"2. {OUTPUT_FILE.replace('.csv', '_simple.csv')} - Original features replaced with normalized versions")
    else:
        print("\nNormalization failed. Please check the error messages above.")
    
    print("=" * 80) 