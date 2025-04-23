import pandas as pd

# Define file paths
input_csv = "../data/NBA_Dataset.csv"
# Output file for filtered data
filtered_output = "../data/filtered_nba_data.csv"

# Define filtering thresholds
min_games = 58
min_avg_minutes = 28
min_avg_points = 10

try:
    # Read the CSV file into a pandas DataFrame
    # Need to handle potential trailing commas in the header causing unnamed columns
    # Reading the first row to get headers correctly
    header = pd.read_csv(input_csv, nrows=1).columns.tolist()
    # Clean potential unnamed columns resulting from trailing commas
    cleaned_header = [col.strip() for col in header if 'Unnamed' not in col and col.strip()]

    # Read the full CSV using the cleaned header, skipping the original header row
    df = pd.read_csv(input_csv, header=None, skiprows=1, names=cleaned_header)

    print(f"Original dataset shape: {df.shape}")

    # Ensure necessary columns are numeric, coercing errors to NaN
    numeric_cols = ['g', 'mp_per_g', 'pts_per_g', 'season']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")

    # Drop rows where numeric conversion failed in essential columns
    df.dropna(subset=numeric_cols, inplace=True)

    # Convert season to integer after handling potential NaNs
    df['season'] = df['season'].astype(int)

    # Apply the filters
    filtered_df = df[
        (df['g'] >= min_games) &
        (df['mp_per_g'] >= min_avg_minutes) &
        (df['pts_per_g'] >= min_avg_points)
    ].copy()

    print(f"Filtered dataset shape: {filtered_df.shape}")

    # Save filtered DataFrame to a new CSV file
    filtered_df.to_csv(filtered_output, index=False)
    print(f"Filtered data saved to {filtered_output} ({filtered_df.shape[0]} rows)")

except FileNotFoundError:
    print(f"Error: The file {input_csv} was not found.")
except KeyError as e:
    # Check if the error is related to the 'season' column during filtering
    if 'df' in locals() and e.args[0] not in df.columns:
         print(f"Error: Column {e} not found in the original CSV. Please check the column names.")
         print(f"Available columns: {df.columns.tolist()}")
    else:
         print(f"Error: A KeyError occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 