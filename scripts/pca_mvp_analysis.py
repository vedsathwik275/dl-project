import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Configuration
DATA_FILE = "../data/NBA_Dataset_with_MVP_rank.csv"

# Define features (same as in prediction scripts)
features = [
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

print(f"{'='*20} PCA Analysis of MVP Rankings {'='*20}")

try:
    # 1. Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data. Shape: {df.shape}")
    
    # Focus on rows with MVP_rank data
    df_mvp = df.dropna(subset=['MVP_rank'])
    print(f"Number of rows with MVP rank: {len(df_mvp)} out of {len(df)} total rows")
    
    # Convert data types and handle missing values
    for col in features:
        if col in df_mvp.columns:
            df_mvp[col] = pd.to_numeric(df_mvp[col], errors='coerce')
    
    # Drop rows with NaN in features
    df_mvp.dropna(subset=features, inplace=True)
    print(f"After dropping NaNs in features: {len(df_mvp)} rows")
    
    # 2. Season-wise Z-score normalization of features
    print("\nNormalizing statistical features using Z-score within each season...")
    
    # Function to apply scaling within a group
    def scale_group(group):
        scaler = StandardScaler()
        if len(group) > 1:  # Only scale if more than 1 row
            group[features] = scaler.fit_transform(group[features])
        else:
            group[features] = 0  # Set to 0 for single-row groups
        return group
    
    # Group by season and apply scaling
    df_mvp_with_season = df_mvp.copy()
    X = df_mvp_with_season[features + ['season']].copy()
    X_normalized = X.groupby('season').apply(scale_group)
    
    # Remove season column after scaling
    X_normalized.drop(columns=['season'], inplace=True)
    
    # 3. Apply PCA
    print("\nApplying PCA to the features...")
    pca = PCA()
    principal_components = pca.fit_transform(X_normalized)
    
    # Explained variance by each component
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Print PCA results
    print(f"\nExplained variance by top 5 components:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
    
    # 4. Create DataFrame with principal components and MVP rank
    pc_df = pd.DataFrame(
        data=principal_components[:, :5],  # First 5 components
        columns=[f'PC{i+1}' for i in range(5)]
    )
    pc_df['MVP_rank'] = df_mvp['MVP_rank'].values
    pc_df['player'] = df_mvp['player'].values
    pc_df['season'] = df_mvp['season'].values
    
    # 5. Feature importance - how much each original feature contributes to PCs
    print("\nTop 5 features contributing to first principal component:")
    feature_importance = pd.DataFrame(
        pca.components_[:2, :].T,  # First 2 principal components
        index=features,
        columns=['PC1', 'PC2']
    ).abs()
    
    # Sort by PC1 importance
    top_features_pc1 = feature_importance.sort_values('PC1', ascending=False)
    print(top_features_pc1.head(5))
    
    # 6. Visualizations
    
    # 6.1 Scree plot (explained variance)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative', color='red')
    plt.axhline(y=0.8, color='k', linestyle='--', alpha=0.3, label='80% Threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot - Explained Variance by Principal Components')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_scree_plot.png')
    print("\nScree plot saved to 'pca_scree_plot.png'")
    
    # 6.2 Feature contribution heatmap
    plt.figure(figsize=(14, 8))
    abs_components = np.abs(pca.components_[:5, :])  # First 5 PCs
    sns.heatmap(
        abs_components, 
        cmap='viridis',
        yticklabels=[f'PC{i+1}' for i in range(5)],
        xticklabels=features,
        annot=False
    )
    plt.title('Feature Contributions to Principal Components')
    plt.tight_layout()
    plt.savefig('pca_feature_contributions.png')
    print("Feature contributions heatmap saved to 'pca_feature_contributions.png'")
    
    # 6.3 Biplot of first two principal components
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = plt.scatter(
        principal_components[:, 0], 
        principal_components[:, 1],
        c=df_mvp['MVP_rank'],  # Color by MVP rank
        cmap='viridis',
        alpha=0.7,
        s=100  # Size of points
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('MVP Rank')
    
    # Label top players
    top_players = df_mvp[df_mvp['MVP_rank'] <= 2]
    for idx, row in top_players.iterrows():
        i = df_mvp.index.get_loc(idx)
        plt.annotate(
            f"{row['player']} ({row['season']})",
            (principal_components[i, 0], principal_components[i, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    # Plot feature vectors
    feature_vectors = pca.components_.T[:, :2]
    for i, feature in enumerate(features):
        plt.arrow(
            0, 0,  # Start at origin
            feature_vectors[i, 0] * 3,  # Scaling for visibility
            feature_vectors[i, 1] * 3,
            head_width=0.1,
            head_length=0.1,
            fc='red', 
            ec='red', 
            alpha=0.3
        )
        plt.text(
            feature_vectors[i, 0] * 3.2,
            feature_vectors[i, 1] * 3.2,
            feature,
            color='red',
            ha='center',
            va='center',
            fontsize=8
        )
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.title('Biplot: MVP Ranks in Principal Component Space')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig('pca_biplot.png')
    print("PCA biplot saved to 'pca_biplot.png'")
    
    # 6.4 MVP rank vs PC1
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MVP_rank', y='PC1', data=pc_df)
    plt.title('MVP Rank vs First Principal Component')
    plt.tight_layout()
    plt.savefig('mvp_rank_vs_pc1.png')
    print("MVP rank vs PC1 plot saved to 'mvp_rank_vs_pc1.png'")
    
    # 7. Calculate correlation between PCs and MVP rank
    print("\nCorrelation between principal components and MVP rank:")
    # Use only numeric columns for correlation
    numeric_cols = pc_df.select_dtypes(include=['number']).columns
    correlations = pc_df[numeric_cols].corr()['MVP_rank'].drop('MVP_rank')  # Exclude self-correlation
    print(correlations)
    
    # Spearman rank correlation (better for ordinal data like ranks)
    spearman_corr = pc_df[numeric_cols].corr(method='spearman')['MVP_rank'].drop('MVP_rank')
    print("\nSpearman rank correlation (more appropriate for rank data):")
    print(spearman_corr)
    
    # 8. Direct feature importance - correlation between original features and MVP rank
    print("\n" + "="*50)
    print("DIRECT FEATURE IMPORTANCE FOR MVP RANKING")
    print("="*50)
    
    # Create a combined dataset with normalized features and MVP rank
    feature_importance_df = X_normalized.copy()
    feature_importance_df['MVP_rank'] = df_mvp['MVP_rank'].values
    
    # Calculate correlations between each feature and MVP rank
    feature_correlations = pd.DataFrame({
        'Pearson': [feature_importance_df[feature].corr(feature_importance_df['MVP_rank']) for feature in features],
        'Spearman': [feature_importance_df[feature].corr(feature_importance_df['MVP_rank'], method='spearman') for feature in features]
    }, index=features)
    
    # Take absolute values for ranking importance
    feature_correlations['Pearson_abs'] = feature_correlations['Pearson'].abs()
    feature_correlations['Spearman_abs'] = feature_correlations['Spearman'].abs()
    
    # Sort by Spearman correlation (more appropriate for ranks)
    feature_correlations_sorted = feature_correlations.sort_values('Spearman_abs', ascending=False)
    
    # Display top correlations
    print("\nTop features correlated with MVP rank (by absolute Spearman correlation):")
    print(feature_correlations_sorted[['Pearson', 'Spearman']].head(10))
    
    # Visualize correlations
    plt.figure(figsize=(12, 10))
    
    # Plot absolute correlations for top 15 features
    top_features = feature_correlations_sorted.head(15).index
    
    # Create a bar chart of correlations
    plt.figure(figsize=(12, 8))
    
    # Get the top features and their correlations
    corr_data = feature_correlations_sorted.loc[top_features, ['Pearson', 'Spearman']]
    
    # Create a colorful bar chart
    ax = corr_data.plot(kind='barh', figsize=(12, 10), 
                         color=['#3498db', '#e74c3c'], 
                         alpha=0.7, 
                         title='Top 15 Features Correlated with MVP Rank')
    
    # Add a vertical line at zero
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add grid lines
    plt.grid(axis='x', alpha=0.3)
    
    # Add labels
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('NBA Statistical Features')
    plt.title('Top 15 Features Correlated with MVP Rank', fontsize=14)
    
    # Add a legend with better positioning
    plt.legend(loc='lower right', title='Correlation Type')
    
    # Improve layout
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('feature_correlations_with_mvp.png')
    print("\nFeature correlations chart saved to 'feature_correlations_with_mvp.png'")
    
    print("\nPCA Analysis Complete!")
    
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
    
print(f"\n{'='*20} PCA Analysis Finished {'='*20}") 