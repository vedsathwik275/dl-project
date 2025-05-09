import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import seaborn as sns

# Neural network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Add, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = "../data/filtered_nba_data_with_MVP_rank.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 20  # For early stopping
TOP_N_FEATURES = 10  # Number of top correlated features to select

# Define all possible features
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

# Flexible accuracy function that gives partial credit for being off by one or two ranks
def flexible_rank_accuracy(y_true, y_pred_rounded):
    """
    Calculate accuracy with partial credit:
    - Exact match: 1.0 (100% correct)
    - Off by 1 rank: 0.8 (80% credit)
    - Off by 2 ranks: 0.2 (20% credit)
    - Special case: When actual rank is 5 and predicted is >5, or vice versa, treated as off by 1
    - Off by >2 ranks: 0.0 (incorrect)
    """
    # Ensure predictions are within 1-5 range
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    
    # Calculate absolute difference between predicted and actual ranks
    rank_diff = np.abs(y_pred_rounded - y_true)
    
    # Special case handling for boundary between top-5 and outside top-5
    # If true rank is 5 and predicted as 6 or 7, count as off-by-1
    # If true rank is 6 or 7 and predicted as 5, count as off-by-1
    special_case = ((y_true == 5) & (y_pred_rounded > 5)) | ((y_true > 5) & (y_pred_rounded == 5))
    
    # Assign scores based on rank difference
    scores = np.zeros_like(rank_diff, dtype=float)
    scores[rank_diff == 0] = 1.0    # Exact matches
    scores[rank_diff == 1] = 0.8    # Off by one rank
    scores[rank_diff == 2] = 0.2    # Off by two ranks
    scores[special_case] = 0.8      # Special case (treat as off by one)
    
    # Return the average score
    return scores.mean()

# Custom ordinal loss function that penalizes more for larger ranking errors
def ordinal_rank_loss(y_true, y_pred):
    """
    Custom loss function that penalizes larger ranking errors more severely
    """
    # Rescale predictions and true values back to 1-5 range
    y_pred_rescaled = y_pred * 4 + 1
    y_true_rescaled = y_true * 4 + 1
    
    # Calculate absolute error
    abs_error = tf.abs(y_pred_rescaled - y_true_rescaled)
    
    # Square the error to increase penalty for larger differences
    squared_error = tf.square(abs_error)
    
    # The exponent makes the loss increase more rapidly for larger errors
    return tf.reduce_mean(squared_error)

# Function to build improved MLP model with residual connections and better architecture
def build_mlp_model(input_shape):
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # First block with residual connection
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    # Residual block 1
    block_input = x
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Add()([x, block_input])  # Residual connection
    x = Dropout(0.4)(x)
    
    # Split into two branches for feature extraction at different levels
    # Branch 1: Wide branch for general patterns
    wide_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    wide_branch = BatchNormalization()(wide_branch)
    wide_branch = LeakyReLU(alpha=0.1)(wide_branch)
    wide_branch = Dropout(0.3)(wide_branch)
    
    # Branch 2: Deep branch for more complex patterns
    deep_branch = Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.3)(deep_branch)
    deep_branch = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(deep_branch)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = LeakyReLU(alpha=0.1)(deep_branch)
    deep_branch = Dropout(0.3)(deep_branch)
    
    # Merge branches
    merged = Concatenate()([wide_branch, deep_branch])
    
    # Final layers
    x = Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(merged)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    
    # Output layer - sigmoid activation scaled to 1-5 range
    output = Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=ordinal_rank_loss,  # Custom loss function
        metrics=['mae']
    )
    
    return model

# Function to get prediction breakdown
def get_prediction_breakdown(y_true, y_pred):
    exact_matches = np.sum(y_pred == y_true)
    off_by_one = np.sum(np.abs(y_pred - y_true) == 1)
    off_by_two = np.sum(np.abs(y_pred - y_true) == 2)
    off_by_more = len(y_true) - exact_matches - off_by_one - off_by_two
    
    return {
        'exact': exact_matches,
        'off_by_one': off_by_one, 
        'off_by_two': off_by_two,
        'off_by_more': off_by_more
    }

# Function to save detailed prediction results
def save_detailed_predictions(df, test_indices, y_pred, y_pred_rounded, selected_features, feature_correlations):
    """
    Save comprehensive prediction results to CSV with additional stats
    """
    # Create results dataframe
    results_df = df.iloc[test_indices].copy()
    
    # Add predictions
    results_df['predicted_rank_raw'] = y_pred
    results_df['predicted_rank'] = y_pred_rounded
    
    # Calculate error
    results_df['rank_error'] = results_df['predicted_rank'] - results_df['MVP_rank']
    results_df['abs_rank_error'] = np.abs(results_df['rank_error'])
    
    # Add prediction quality labels
    conditions = [
        (results_df['abs_rank_error'] == 0),
        (results_df['abs_rank_error'] == 1),
        (results_df['abs_rank_error'] == 2),
        (results_df['abs_rank_error'] > 2)
    ]
    labels = ['Exact Match', 'Off by 1', 'Off by 2', 'Off by >2']
    results_df['prediction_quality'] = np.select(conditions, labels)
    
    # Sort by season (descending) and MVP rank (ascending)
    results_df = results_df.sort_values(['season', 'MVP_rank'], ascending=[False, True])
    
    # Save all results to CSV
    results_df.to_csv('mvp_rank_prediction_results.csv', index=False)
    
    # Create a separate dataframe with just the important columns for easy viewing
    summary_df = results_df[['season', 'player', 'MVP_rank', 'predicted_rank', 'rank_error', 'prediction_quality']]
    summary_df.to_csv('mvp_rank_prediction_summary.csv', index=False)
    
    # Generate yearly summary - prediction accuracy by season
    yearly_summary = results_df.groupby('season').apply(
        lambda x: pd.Series({
            'num_players': len(x),
            'exact_matches': sum(x['abs_rank_error'] == 0),
            'off_by_one': sum(x['abs_rank_error'] == 1),
            'off_by_two': sum(x['abs_rank_error'] == 2),
            'off_by_more': sum(x['abs_rank_error'] > 2),
            'accuracy': sum(x['abs_rank_error'] == 0) / len(x),
            'flexible_accuracy': flexible_rank_accuracy(x['MVP_rank'], x['predicted_rank']),
            'mse': mean_squared_error(x['MVP_rank'], x['predicted_rank'])
        })
    ).reset_index()
    
    yearly_summary.to_csv('mvp_rank_yearly_accuracy.csv', index=False)
    
    return results_df

# Function to create additional visualizations
def create_visualizations(results_df, test_data, y_test, y_pred_rounded, feature_correlations, selected_features):
    """
    Create additional visualizations for model evaluation
    """
    # 1. Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_rounded)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 6), 
                yticklabels=range(1, 6))
    plt.xlabel('Predicted Rank')
    plt.ylabel('True Rank')
    plt.title('Confusion Matrix of MVP Rank Predictions')
    plt.tight_layout()
    plt.savefig('mvp_rank_confusion_matrix.png')
    
    # 2. Distribution of prediction errors
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rank_error', data=results_df, palette='viridis')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.title('Distribution of MVP Rank Prediction Errors')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('mvp_rank_error_distribution.png')
    
    # 3. Prediction accuracy by season
    yearly_accuracy = results_df.groupby('season')['abs_rank_error'].apply(
        lambda x: sum(x == 0) / len(x)
    ).reset_index()
    yearly_accuracy.columns = ['season', 'accuracy']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='season', y='accuracy', data=yearly_accuracy, palette='viridis')
    plt.xlabel('Season')
    plt.ylabel('Accuracy (Exact Matches)')
    plt.title('MVP Rank Prediction Accuracy by Season')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('mvp_rank_accuracy_by_season.png')
    
    # 4. Actual vs Predicted scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results_df['MVP_rank'], 
        results_df['predicted_rank'],
        c=results_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add perfect prediction line
    plt.plot([1, 5], [1, 5], 'r--', alpha=0.7)
    
    plt.colorbar(scatter, label='Season')
    plt.xlabel('Actual MVP Rank')
    plt.ylabel('Predicted MVP Rank')
    plt.title('Actual vs Predicted MVP Ranks')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('mvp_rank_actual_vs_predicted.png')

print(f"{'='*20} MVP Rank Prediction with Feature Selection {'='*20}")

try:
    # Check for TensorFlow GPU support
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus}")
        # Memory growth needs to be set before GPUs have been initialized
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU.")
    
    # 1. Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded data. Shape: {df.shape}")
    
    # Verify essential columns exist
    essential_cols = all_features + ['season', 'MVP_rank', 'player']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing essential columns: {missing_cols}")
    
    # Focus only on rows with MVP_rank
    df_mvp = df.dropna(subset=['MVP_rank'])
    print(f"Number of rows with MVP rank: {len(df_mvp)} out of {len(df)} total rows")
    
    # Convert data types and handle missing values
    # Convert feature columns to numeric
    for col in all_features:
        if col in df_mvp.columns:
            df_mvp[col] = pd.to_numeric(df_mvp[col], errors='coerce')
    
    # Drop rows with NaN in features
    df_mvp.dropna(subset=all_features, inplace=True)
    print(f"After dropping NaNs in features: {len(df_mvp)} rows")
    
    # Ensure MVP_rank is integer
    df_mvp['MVP_rank'] = df_mvp['MVP_rank'].astype(int)
    
    # Display distribution of MVP ranks
    rank_distribution = df_mvp['MVP_rank'].value_counts().sort_index()
    print(f"\nDistribution of MVP ranks:\n{rank_distribution}")
    
    # 2. Season-wise Z-score normalization of features
    print(f"\nNormalizing statistical features using Z-score within each season...")
    
    # Function to apply scaling within a group
    def scale_group(group):
        scaler = StandardScaler()
        if len(group) > 1:  # Only scale if more than 1 row
            group[all_features] = scaler.fit_transform(group[all_features])
        else:
            group[all_features] = 0  # Set to 0 for single-row groups
        return group
    
    # Group by season and apply scaling
    df_mvp_with_season = df_mvp.copy()
    X_all = df_mvp_with_season[all_features + ['season']].copy()
    X_normalized_all = X_all.groupby('season').apply(scale_group)
    
    # Remove season column after scaling
    X_normalized_all.drop(columns=['season'], inplace=True)
    
    # 3. Feature Selection - Identify top features by correlation with MVP rank
    print(f"\n{'-'*10} Feature Selection {'-'*10}")
    
    # Create a dataframe for correlation calculation
    feature_selection_df = X_normalized_all.copy()
    feature_selection_df['MVP_rank'] = df_mvp['MVP_rank'].values
    
    # Calculate correlations between each feature and MVP rank
    feature_correlations = pd.DataFrame({
        'Pearson': [feature_selection_df[feature].corr(feature_selection_df['MVP_rank']) for feature in all_features],
        'Spearman': [feature_selection_df[feature].corr(feature_selection_df['MVP_rank'], method='spearman') for feature in all_features]
    }, index=all_features)
    
    # Take absolute values for ranking importance
    feature_correlations['Pearson_abs'] = feature_correlations['Pearson'].abs()
    feature_correlations['Spearman_abs'] = feature_correlations['Spearman'].abs()
    
    # Sort by Spearman correlation (more appropriate for ranks)
    feature_correlations_sorted = feature_correlations.sort_values('Spearman_abs', ascending=False)
    
    # Select top N features
    selected_features = feature_correlations_sorted.index[:TOP_N_FEATURES].tolist()
    
    print(f"Top {TOP_N_FEATURES} features selected by correlation with MVP rank:")
    for i, feature in enumerate(selected_features):
        corr = feature_correlations_sorted.loc[feature, 'Spearman']
        print(f"{i+1}. {feature}: {corr:.4f}")
    
    # Create feature matrix with only selected features
    X_normalized = X_normalized_all[selected_features]
    
    # Target variable - MVP rank
    y = df_mvp['MVP_rank']
    
    # Reset indices to ensure alignment after splitting
    X_normalized = X_normalized.reset_index(drop=True)
    df_mvp = df_mvp.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 4. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Ensure balanced distribution of ranks
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"MVP rank distribution in training set:\n{y_train.value_counts().sort_index()}")
    print(f"MVP rank distribution in test set:\n{y_test.value_counts().sort_index()}")
    
    # Save test indices
    test_indices = y_test.index
    
    # Save info about test set for later analysis
    test_data = df_mvp.iloc[test_indices].copy()
    
    # 5. Neural Network (MLP) with selected features
    print(f"\n{'-'*10} MLP Neural Network with Selected Features {'-'*10}")
    
    # Prepare data for neural network
    # Scale target for sigmoid output (1-5 to 0-1)
    y_train_scaled = (y_train - 1) / 4  # Scale from [1,5] to [0,1]
    y_test_scaled = (y_test - 1) / 4
    
    # Build model
    mlp_model = build_mlp_model(X_train.shape[1])
    print(mlp_model.summary())
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction when plateau is reached
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\nTraining neural network...")
    history = mlp_model.fit(
        X_train, y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],  # Added learning rate schedule
        verbose=2
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_selected_training_history.png')
    print("Training history saved to 'mlp_selected_training_history.png'")
    
    # Make predictions
    y_pred_mlp_scaled = mlp_model.predict(X_test)
    
    # Convert predictions back to 1-5 scale
    y_pred_mlp = (y_pred_mlp_scaled * 4) + 1
    y_pred_mlp_rounded = np.round(y_pred_mlp).astype(int)
    y_pred_mlp_rounded = np.clip(y_pred_mlp_rounded, 1, 5)  # Ensure predictions are between 1-5
    
    # Flatten predictions to ensure 1D arrays for metrics calculation
    y_pred_mlp = y_pred_mlp.flatten()
    y_pred_mlp_rounded = y_pred_mlp_rounded.flatten()
    
    # Performance metrics
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)
    acc_mlp_exact = np.sum(y_pred_mlp_rounded == y_test) / len(y_test)
    acc_mlp_flex = flexible_rank_accuracy(y_test, y_pred_mlp_rounded)
    
    print(f"\nMean Squared Error: {mse_mlp:.4f}")
    print(f"R-squared: {r2_mlp:.4f}")
    print(f"Exact Rank Accuracy: {acc_mlp_exact:.4f} ({int(acc_mlp_exact * len(y_test))} exact matches out of {len(y_test)})")
    print(f"Flexible Rank Accuracy: {acc_mlp_flex:.4f} (includes 80% credit for being off by one or 20% for being off by two ranks)")
    
    # Calculate counts for each type of match
    mlp_breakdown = get_prediction_breakdown(y_test, y_pred_mlp_rounded)
    
    print(f"Prediction breakdown:")
    print(f"  Exact matches: {mlp_breakdown['exact']}")
    print(f"  Off by 1 rank: {mlp_breakdown['off_by_one']}")
    print(f"  Off by 2 ranks: {mlp_breakdown['off_by_two']}")
    print(f"  Off by >2 ranks: {mlp_breakdown['off_by_more']}")
    
    # Visualize feature importance 
    plt.figure(figsize=(10, 6))
    plt.barh(selected_features, feature_correlations.loc[selected_features, 'Spearman'])
    plt.xlabel('Spearman Correlation with MVP Rank')
    plt.ylabel('Selected Features')
    plt.title('Selected Feature Importance for MVP Rank Prediction')
    plt.tight_layout()
    plt.savefig('selected_feature_importance.png')
    print("\nFeature importance chart saved to 'selected_feature_importance.png'")
    
    # Display some example predictions for analysis
    print(f"\n{'-'*10} Sample Predictions {'-'*10}")
    test_data['MLP_pred'] = y_pred_mlp_rounded
    
    # Display a sample of predictions
    sample_preds = test_data[['season', 'player', 'MVP_rank', 'MLP_pred']].head(10)
    print(f"\nSample of actual vs predicted MVP ranks:")
    print(sample_preds.to_string(index=False))
    
    # Save model
    # print("\nSaving model...")
    # mlp_model.save("mlp_mvp_rank_selected_model")
    # print("Model saved to 'mlp_mvp_rank_selected_model'")
    
    # Save selected features for later use
    with open('selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    print(f"Selected features saved to 'selected_features.txt'")
    
    # Save detailed predictions and create additional visualizations
    print("\nSaving detailed prediction results...")
    results_df = save_detailed_predictions(df_mvp, test_indices, y_pred_mlp, y_pred_mlp_rounded, 
                                          selected_features, feature_correlations)
    
    # Create additional visualizations
    print("Creating additional visualizations...")
    create_visualizations(results_df, test_data, y_test, y_pred_mlp_rounded, 
                         feature_correlations, selected_features)
    
    # Print information about where to find the results
    print("\nDetailed Results Files:")
    print("- Complete prediction results: 'mvp_rank_prediction_results.csv'")
    print("- Prediction summary: 'mvp_rank_prediction_summary.csv'")
    print("- Yearly accuracy summary: 'mvp_rank_yearly_accuracy.csv'")
    print("\nVisualization Files:")
    print("- Confusion matrix: 'mvp_rank_confusion_matrix.png'")
    print("- Error distribution: 'mvp_rank_error_distribution.png'")
    print("- Seasonal accuracy: 'mvp_rank_accuracy_by_season.png'")
    print("- Actual vs Predicted: 'mvp_rank_actual_vs_predicted.png'")
    
    # Display all predictions
    print(f"\n{'-'*10} All Test Set Predictions {'-'*10}")
    all_preds = results_df[['season', 'player', 'MVP_rank', 'predicted_rank', 'rank_error', 'prediction_quality']]
    print(all_preds.to_string(index=False))
    
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

print(f"\n{'='*20} MVP Rank Prediction with Feature Selection Finished {'='*20}") 