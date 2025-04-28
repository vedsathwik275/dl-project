# Data Acquisition and Preprocessing

## 1. Data Source

The primary dataset used is `data/normalized_nba_data_with_MVP_rank_simple.csv`. The origin of this pre-normalized dataset is not explicitly detailed in the scripts, but it appears to contain aggregated player statistics per season, team win percentage, and MVP award share information.

## 2. Initial Filtering

A preliminary filtering step might have been performed using `scripts/filter_nba_data.py`. This script appears to filter a raw CSV (`nba_data_with_MVP_rank.csv`) based on minimum games played and minutes played per game, saving the result to `filtered_nba_data.csv`. However, the main modeling scripts directly use the `normalized_nba_data_with_MVP_rank_simple.csv` file.

## 3. Normalization

Two primary normalization strategies seem to be employed or available:

*   **Season-wise Z-score Normalization:** Most training scripts (`predict_mvp_rank.py`, `predict_mvp_rank_ndcg.py`, etc.) implement Z-score normalization *within each season*. This scales features based on the mean and standard deviation of that specific season, accounting for statistical variations across different eras.
*   **Era-based Normalization:** The script `scripts/normalize_nba_data_by_era.py` suggests an alternative approach where normalization might be performed based on defined NBA eras (e.g., "Physical Play (1980s)", "Isolation (1995-2010)", "Analytics/3PT (2011+)"). This aims to group seasons with similar playstyles before normalizing.

It appears the `normalized_nba_data_with_MVP_rank_simple.csv` file likely contains data pre-processed using one of these methods (most likely season-wise Z-score, given its direct use in multiple scripts). The scripts using this file often re-apply season-wise Z-score normalization as a safety measure or for flexibility.

## 4. Target Variable Handling

*   **`award_share`:** This is the primary target variable for regression. It represents the proportion of the total possible MVP votes a player received.
*   **Missing `award_share`:** Missing values in `award_share` are consistently filled with `0`. This assumes that players not listed in the voting results received no votes.
*   **`MVP_rank`:** While present in filenames, the `MVP_rank` column itself isn't typically used as a direct target variable in the provided NN or ML regression scripts. Instead, ranks are derived *after* predicting `award_share` using the `convert_to_ranks` function.
*   The script `scripts/create_mvp_ranking.py` might be involved in the initial calculation or formatting of `award_share` or `MVP_rank` in the dataset creation pipeline.

## 5. Final Dataset Schema

The dataset `normalized_nba_data_with_MVP_rank_simple.csv` used in the main training scripts generally contains columns like:

*   `player`: Player name
*   `season`: NBA season year (e.g., 2023 for 2022-23 season)
*   `award_share`: Target variable (proportion of MVP votes, 0-1 scale)
*   `MVP_rank`: The player's official MVP rank for the season (if available)
*   `win_loss_pct`: Team's regular season win percentage
*   Various player statistics (potentially normalized): `g`, `gs`, `mp_per_g`, `fg_per_g`, `fga_per_g`, `ws`, `vorp`, `bpm`, etc.
*   Features selected for modeling (e.g., `ws`, `vorp`, `bpm`, `pts_per_g`, etc.)

## 6. Handling Missing Data (Features)

Rows containing `NaN` values in any of the selected *feature* columns are typically dropped before training (`df.dropna(subset=features, inplace=True)`). 