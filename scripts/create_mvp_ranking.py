import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = "../data/NBA_Dataset.csv"
OUTPUT_FILE = "../data/NBA_Dataset_with_MVP_rank.csv"
TOP_RANKS = 7  # Number of top MVP candidates to rank (1-7)

print(f"{'='*20} Creating MVP Rankings {'='*20}")

try:
    # Read the dataset
    print(f"Loading data from {INPUT_FILE}...")
    # Reading the first row to get headers correctly
    header = pd.read_csv(INPUT_FILE, nrows=1).columns.tolist()
    # Clean potential leading/trailing spaces and filter unnamed columns
    cleaned_header = [col.strip() for col in header if 'Unnamed' not in col and col.strip()]
    df = pd.read_csv(INPUT_FILE, header=None, skiprows=1, names=cleaned_header, low_memory=False)
    print(f"Loaded data. Shape: {df.shape}")
    
    # Verify essential columns exist
    essential_cols = ['season', 'award_share', 'player']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing essential columns in {INPUT_FILE}: {missing_cols}")

    # Convert relevant columns to numeric
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['award_share'] = pd.to_numeric(df['award_share'], errors='coerce')
    
    # Drop rows with NaN in essential columns
    orig_rows = len(df)
    df.dropna(subset=['season', 'award_share'], inplace=True)
    print(f"Dropped {orig_rows - len(df)} rows with missing season or award_share values")
    
    # Ensure 'season' is integer
    df['season'] = df['season'].astype(int)
    
    # For each season, rank players by award_share
    print(f"Ranking players by MVP award share within each season...")
    
    # Add a rank column (smaller rank = higher award_share)
    # Players with 0 award_share will get a high rank, which we'll handle next
    df['temp_rank'] = df.groupby('season')['award_share'].rank(method='min', ascending=False)
    
    # Create the MVP_rank column:
    # - Players ranked 1-7 get their actual rank
    # - Players ranked >7 or with 0 award_share get NaN
    df['MVP_rank'] = np.where(
        (df['temp_rank'] <= TOP_RANKS) & (df['award_share'] > 0),
        df['temp_rank'],
        np.nan
    )
    
    # Convert MVP_rank to integer where not NaN
    # Note: Int64 pandas type allows for NaN values unlike regular int
    df['MVP_rank'] = df['MVP_rank'].astype('Int64')
    
    # Remove the temporary rank column
    df.drop(columns=['temp_rank'], inplace=True)
    
    # Count MVP ranked players by season
    mvp_count = df.groupby('season')['MVP_rank'].count()
    print(f"\nSummary of MVP ranked players (should be up to {TOP_RANKS} per season):")
    print(mvp_count.describe())
    
    # Get seasons with fewer than expected ranks
    incomplete_seasons = mvp_count[mvp_count < TOP_RANKS]
    if not incomplete_seasons.empty:
        print(f"\nSeasons with fewer than {TOP_RANKS} ranked players (likely due to ties or missing data):")
        print(incomplete_seasons)
    
    # Show example of the rankings for a sample season
    sample_season = df['season'].max()  # Use the most recent season as sample
    print(f"\nExample MVP rankings for season {sample_season}:")
    sample = df[df['season'] == sample_season].sort_values('award_share', ascending=False).head(TOP_RANKS+2)
    print(sample[['season', 'player', 'award_share', 'MVP_rank']].to_string(index=False))
    
    # Save the updated dataset
    print(f"\nSaving updated dataset to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved dataset with MVP rankings")
    
    # Additional statistics
    ranked_count = df['MVP_rank'].count()
    total_count = len(df)
    print(f"\nTotal players with MVP rank (1-{TOP_RANKS}): {ranked_count} ({ranked_count/total_count:.1%} of all players)")
    
except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found. Exiting.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {INPUT_FILE} is empty. Exiting.")
except ValueError as ve:
    print(ve)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    
print(f"\n{'='*20} MVP Ranking Script Finished {'='*20}")