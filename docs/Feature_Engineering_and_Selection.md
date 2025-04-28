# Feature Engineering and Selection

## 1. Initial Feature Set

Several neural network scripts (e.g., `mlp_nn.py`, `transformer_mvp_ndcg.py`) define a comprehensive list of potential features (`all_features`):

```python
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
```

This includes basic counting stats (games, games started), per-game averages (minutes, points, rebounds, assists, etc.), shooting efficiency stats, advanced metrics (Win Shares, WS/48, Box Plus/Minus, VORP), and team win percentage.

## 2. Feature Analysis

The script `scripts/pca_mvp_analysis.py` suggests that Principal Component Analysis (PCA) was performed to understand the relationships between features and their importance concerning MVP rankings or `award_share`. The specific findings of this analysis would ideally be summarized here, but the script itself would contain the details.

The script `predict_award_share_for_rank.py` also includes a step to calculate and visualize the correlation between selected features and the `award_share` target variable.

## 3. Selected Feature Set

Based on analysis (likely including PCA and correlation analysis), a smaller, refined set of features was selected for most models. This set is often defined as `features` or `award_share_features` in the scripts and is also listed in `scripts/selected_features.txt`:

```python
# From selected_features.txt or script definitions
features = [
    'ws',           # Win Shares
    'vorp',         # Value Over Replacement Player
    'ws_per_48',    # Win Shares per 48 minutes
    'bpm',          # Box Plus/Minus
    'win_loss_pct', # Team Win Percentage
    'pts_per_g',    # Points Per Game
    'fg_per_g',     # Field Goals Per Game
    'fg2_per_g',    # 2-Point Field Goals Per Game
    'ft_per_g',     # Free Throws Per Game
    'fga_per_g'     # Field Goal Attempts Per Game
]
```

**Rationale for Selection (Inferred):**

This set likely represents a balance between:

*   **Advanced Metrics:** `ws`, `vorp`, `ws_per_48`, `bpm` often capture overall player impact effectively.
*   **Team Success:** `win_loss_pct` is crucial as MVP voting heavily considers team performance.
*   **Scoring Volume/Efficiency:** `pts_per_g` and various field goal/free throw metrics reflect scoring, a key aspect of perceived value.

The selection aims to use highly informative features while potentially reducing multicollinearity and model complexity compared to using `all_features`. 