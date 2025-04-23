import pandas as pd

# Define file paths
input_csv = "NBA_Dataset.csv"
# Output files for eras
physical_era_output = "physical_era_data.csv"
isolation_era_output = "isolation_era_data.csv"
analytics_era_output = "analytics_era_data.csv"

# Define filtering thresholds
min_games = 58
min_avg_minutes = 28
min_avg_points = 10

# Define era year ranges
physical_era_start, physical_era_end = 1980, 1989
isolation_era_start, isolation_era_end = 1995, 2010
analytics_era_start = 2011

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
    numeric_cols = ['g', 'mp_per_g', 'pts_per_g', 'season'] # Added 'season'
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in the dataset.")
            # Handle missing column if necessary, e.g., raise error or skip filtering on it

    # Drop rows where numeric conversion failed in essential columns
    df.dropna(subset=numeric_cols, inplace=True)

    # Convert season to integer after handling potential NaNs
    df['season'] = df['season'].astype(int)

    # Apply the initial filters
    filtered_df = df[
        (df['g'] >= min_games) &
        (df['mp_per_g'] >= min_avg_minutes) &
        (df['pts_per_g'] >= min_avg_points)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    print(f"Filtered dataset shape (before era split): {filtered_df.shape}")

    # Split into eras based on 'season'
    physical_era_df = filtered_df[
        (filtered_df['season'] >= physical_era_start) &
        (filtered_df['season'] <= physical_era_end)
    ]
    isolation_era_df = filtered_df[
        (filtered_df['season'] >= isolation_era_start) &
        (filtered_df['season'] <= isolation_era_end)
    ]
    analytics_era_df = filtered_df[
        filtered_df['season'] >= analytics_era_start
    ]

    # Save each era DataFrame to a new CSV file
    physical_era_df.to_csv(physical_era_output, index=False)
    print(f"Physical Play Era ({physical_era_start}-{physical_era_end}) data saved to {physical_era_output} ({physical_era_df.shape[0]} rows)")

    isolation_era_df.to_csv(isolation_era_output, index=False)
    print(f"Isolation Era ({isolation_era_start}-{isolation_era_end}) data saved to {isolation_era_output} ({isolation_era_df.shape[0]} rows)")

    analytics_era_df.to_csv(analytics_era_output, index=False)
    print(f"Analytics/3PT Era ({analytics_era_start}-Present) data saved to {analytics_era_output} ({analytics_era_df.shape[0]} rows)")

except FileNotFoundError:
    print(f"Error: The file {input_csv} was not found.")
except KeyError as e:
    # Check if the error is related to the 'season' column during filtering or splitting
    if 'filtered_df' in locals() and e.args[0] not in filtered_df.columns:
         print(f"Error: Column {e} required for era splitting not found after initial filtering. Check CSV header.")
         print(f"Filtered columns: {filtered_df.columns.tolist()}")
    elif 'df' in locals() and e.args[0] not in df.columns:
         print(f"Error: Column {e} not found in the original CSV. Please check the column names.")
         print(f"Available columns: {df.columns.tolist()}")
    else:
         print(f"Error: A KeyError occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 