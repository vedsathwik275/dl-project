import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, ndcg_score
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Neural network imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Layer, LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import Concatenate, Average, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Configuration
DATA_FILE = "../data/normalized_nba_data_with_MVP_rank_simple.csv"
OUTPUT_DIR = "../data/ensemble_evaluation_results_ndcg"
PICS_DIR = "../pics/ensemble_ndcg"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICS_DIR, exist_ok=True)

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 423
EPOCHS = 300
BATCH_SIZE = 48
LEARNING_RATE = 0.001
PATIENCE = 25

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

# Use these specific features for award_share based on feature selection
award_share_features = [
    'vorp', 'bpm', 'ws', 'ws_per_48', 'pts_per_g',
    'ft_per_g', 'fg_per_g', 'fta_per_g', 'fg2_per_g', 'fga_per_g'
]

# =============== HELPER LAYER DEFINITIONS ===============

class ExpandDimsLayer(Layer):
    """Wrapper for tf.expand_dims as a Keras layer"""
    def __init__(self, axis):
        super(ExpandDimsLayer, self).__init__()
        self.axis = axis
        
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

class SqueezeLayer(Layer):
    """Wrapper for tf.squeeze as a Keras layer"""
    def __init__(self, axis):
        super(SqueezeLayer, self).__init__()
        self.axis = axis
        
    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

class StackLayer(Layer):
    """Wrapper for tf.stack as a Keras layer"""
    def __init__(self, axis):
        super(StackLayer, self).__init__()
        self.axis = axis
        
    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(StackLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

# =============== TRANSFORMER COMPONENTS ===============

class TransformerBlock(Layer):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3)),
            Dropout(rate/2),
            Dense(embed_dim, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3)),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# =============== MLP COMPONENTS ===============

class ResidualBlock(Layer):
    """
    Residual block with skip connections for better gradient flow
    """
    def __init__(self, units, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.dense1 = Dense(units, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.dense2 = Dense(units, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.layernorm(inputs + x)

# =============== ADVANCED HELPER LAYERS ===============

class FeatureInteractionLayer(Layer):
    """
    Layer that explicitly models pairwise feature interactions
    """
    def __init__(self, output_dim):
        super(FeatureInteractionLayer, self).__init__()
        self.output_dim = output_dim
        self.interaction_weight = None  # Will be created in build
        
    def build(self, input_shape):
        # Create weights for feature interactions
        self.interaction_weight = self.add_weight(
            name='interaction_weights',
            shape=(input_shape[-1], input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(FeatureInteractionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Compute pairwise feature interactions
        batch_size = tf.shape(inputs)[0]
        # Reshape for broadcasting
        x_reshaped = tf.reshape(inputs, [batch_size, inputs.shape[1], 1])
        # First interaction term (batch_size, input_dim, 1) * (batch_size, 1, input_dim)
        interactions = tf.matmul(x_reshaped, tf.expand_dims(inputs, 1))
        # Apply weights to interactions and reduce
        weighted = tf.tensordot(interactions, self.interaction_weight, axes=[[1, 2], [0, 1]])
        return weighted
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class SelfNormalizedAttention(Layer):
    """
    Self-normalized attention layer specifically designed for tabular data
    """
    def __init__(self, attention_dim=32):
        super(SelfNormalizedAttention, self).__init__()
        self.attention_dim = attention_dim
        self.W = None
        self.b = None
        self.u = None
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='context_vector',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SelfNormalizedAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Compute attention weights
        # (batch_size, time_steps, attention_dim)
        attention = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        # (batch_size, time_steps)
        attention_weights = tf.nn.softmax(tf.reduce_sum(attention * self.u, axis=-1), axis=1)
        # Apply attention weights
        # (batch_size, dim)
        weighted_sum = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        return weighted_sum, attention_weights

# =============== ENSEMBLE MODEL BUILDING FUNCTION ===============

def build_ensemble_model(input_shape, transformer_dim=96, mlp_dim=128, num_heads=4, num_transformer_blocks=2, dropout_rate=0.3):
    """
    Build an enhanced ensemble model combining Transformer and MLP architectures
    with feature interactions and self-normalized attention
    
    Parameters:
    - input_shape: Shape of input features
    - transformer_dim: Dimension for transformer embedding space
    - mlp_dim: Dimension for MLP hidden layers
    - num_heads: Number of attention heads
    - num_transformer_blocks: Number of transformer blocks
    - dropout_rate: Dropout rate
    
    Returns:
    - Compiled Keras model
    """
    # Input layer (shared)
    inputs = Input(shape=(input_shape,))
    
    # Feature interactions - explicitly model pairwise feature interactions
    feature_interactions = FeatureInteractionLayer(output_dim=32)(inputs)
    
    # ---- TRANSFORMER BRANCH ----
    # Embed input features for transformer
    t_embed = Dense(transformer_dim, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(inputs)
    t_embed = LayerNormalization()(t_embed)
    t_embed = Dropout(dropout_rate)(t_embed)
    
    # Reshape for transformer: [batch_size, 1, embed_dim]
    t_embed = ExpandDimsLayer(axis=1)(t_embed)
    
    # Apply transformer blocks
    t_output = t_embed
    for _ in range(num_transformer_blocks):
        t_output = TransformerBlock(
            transformer_dim, 
            num_heads, 
            transformer_dim * 2, 
            dropout_rate
        )(t_output)
    
    # Flatten transformer output
    t_output = SqueezeLayer(axis=1)(t_output)
    
    # ---- MLP BRANCH ----
    # First MLP layer with BatchNormalization
    m_output = Dense(mlp_dim, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(inputs)
    m_output = BatchNormalization()(m_output)
    m_output = LeakyReLU(alpha=0.2)(m_output)
    m_output = Dropout(dropout_rate)(m_output)
    
    # Residual blocks for MLP branch
    for _ in range(2):  # Use two residual blocks
        m_output = ResidualBlock(mlp_dim, dropout_rate)(m_output)
    
    # Additional dense layers for MLP branch
    m_output = Dense(mlp_dim//2, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(m_output)
    m_output = BatchNormalization()(m_output)
    m_output = LeakyReLU(alpha=0.2)(m_output)
    m_output = Dropout(dropout_rate)(m_output)
    
    # ---- SELF-NORMALIZED ATTENTION BRANCH ----
    # Create transformations for attention
    attn_relu = Dense(32, activation='relu')(inputs)
    attn_tanh = Dense(32, activation='tanh')(inputs)
    attn_sigmoid = Dense(32, activation='sigmoid')(inputs)

    # Use the wrapper layer instead of direct tf.stack
    attention_input = StackLayer(axis=1)([attn_relu, attn_tanh, attn_sigmoid])
    
    # Apply self-normalized attention
    attention_output, attention_weights = SelfNormalizedAttention()(attention_input)
    
    # ---- COMBINE ALL BRANCHES ----
    # Process each branch before combining
    t_processed = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(t_output)
    t_processed = LayerNormalization()(t_processed)
    t_processed = Dropout(dropout_rate/2)(t_processed)
    
    m_processed = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(m_output)
    m_processed = BatchNormalization()(m_processed)
    m_processed = Dropout(dropout_rate/2)(m_processed)
    
    # Process feature interactions
    interactions_processed = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(feature_interactions)
    interactions_processed = BatchNormalization()(interactions_processed)
    interactions_processed = Dropout(dropout_rate/2)(interactions_processed)
    
    # Process attention output
    attention_processed = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(attention_output)
    attention_processed = LayerNormalization()(attention_processed)
    attention_processed = Dropout(dropout_rate/2)(attention_processed)
    
    # Concatenate all processed outputs
    combined = Concatenate()([
        t_processed, 
        m_processed, 
        interactions_processed,
        attention_processed
    ])
    
    # Integration layers
    combined = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(combined)
    combined = LayerNormalization()(combined)
    combined = Dropout(dropout_rate/2)(combined)
    
    combined = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(combined)
    combined = LayerNormalization()(combined)
    combined = Dropout(dropout_rate/2)(combined)
    
    combined = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(combined)
    combined = LayerNormalization()(combined)
    combined = Dropout(dropout_rate/4)(combined)
    
    # Output layer - with sigmoid for award share prediction (0-1 range)
    output = Dense(1, activation='sigmoid')(combined)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=output)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean squared error for continuous variable
        metrics=['mae']
    )
    
    return model

# =============== PREDICTION EVALUATION FUNCTIONS ===============

def save_prediction_results(df, test_indices, y_test, y_pred):
    """
    Save comprehensive prediction results to CSV with basic stats
    """
    # Create results dataframe
    results_df = df.iloc[test_indices].copy()
    
    # Add predictions
    results_df['predicted_award_share'] = y_pred
    
    # Calculate error
    results_df['award_share_error'] = results_df['predicted_award_share'] - results_df['award_share']
    results_df['abs_error'] = np.abs(results_df['award_share_error'])
    
    # Create a flag for vote-getters
    results_df['received_votes'] = results_df['award_share'] > 0
    
    # Sort by season (descending) and actual award share (descending)
    results_df = results_df.sort_values(['season', 'award_share'], ascending=[False, False])
    
    # Save all results to CSV
    results_df.to_csv(f'{OUTPUT_DIR}/award_share_prediction_results.csv', index=False)
    
    # Create a separate summary for just vote-getters
    vote_getters_df = results_df[results_df['received_votes']]
    vote_getters_df.to_csv(f'{OUTPUT_DIR}/award_share_vote_getters_prediction.csv', index=False)
    
    # Generate summary stats by season
    yearly_summary = results_df.groupby('season').apply(
        lambda x: pd.Series({
            'num_players': len(x),
            'num_vote_getters': sum(x['received_votes']),
            'mse': mean_squared_error(x['award_share'], x['predicted_award_share']),
            'mae': mean_absolute_error(x['award_share'], x['predicted_award_share']),
            'r2': r2_score(x['award_share'], x['predicted_award_share'])
        })
    ).reset_index()
    
    yearly_summary.to_csv(f'{OUTPUT_DIR}/award_share_yearly_metrics.csv', index=False)
    
    return results_df

def create_prediction_visualizations(results_df):
    """
    Create visualizations for predicted award shares evaluation
    """
    # Filter to only vote getters for some visualizations
    vote_getters_df = results_df[results_df['received_votes']].copy()
    
    # 1. Error distribution (all players)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='award_share_error', bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Award Share Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Award Share Prediction Errors')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PICS_DIR}/award_share_error_distribution.png')
    
    # 2. Prediction accuracy by season
    plt.figure(figsize=(12, 6))
    yearly_metrics = results_df.groupby('season').apply(
        lambda x: mean_squared_error(x['award_share'], x['predicted_award_share'])
    ).reset_index()
    yearly_metrics.columns = ['season', 'mse']
    
    sns.barplot(x='season', y='mse', data=yearly_metrics, palette='viridis')
    plt.xlabel('Season')
    plt.ylabel('Mean Squared Error')
    plt.title('Award Share Prediction MSE by Season')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PICS_DIR}/award_share_mse_by_season.png')
    
    # 3. Actual vs Predicted award_share scatter (all players)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results_df['award_share'], 
        results_df['predicted_award_share'],
        c=results_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add perfect prediction line
    max_val = max(results_df['award_share'].max(), results_df['predicted_award_share'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.colorbar(scatter, label='Season')
    plt.xlabel('Actual Award Share')
    plt.ylabel('Predicted Award Share')
    plt.title('Actual vs Predicted Award Share')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PICS_DIR}/award_share_actual_vs_predicted.png')
    
    # 4. Actual vs Predicted award_share scatter (only vote getters for clarity)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        vote_getters_df['award_share'], 
        vote_getters_df['predicted_award_share'],
        c=vote_getters_df['season'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add perfect prediction line
    max_val = max(vote_getters_df['award_share'].max(), vote_getters_df['predicted_award_share'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.colorbar(scatter, label='Season')
    plt.xlabel('Actual Award Share')
    plt.ylabel('Predicted Award Share')
    plt.title('Actual vs Predicted Award Share (Vote Getters Only)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PICS_DIR}/award_share_vote_getters_actual_vs_predicted.png')

# =============== RANKING EVALUATION FUNCTIONS ===============

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

def evaluate_ranking_performance_ndcg(results_df, k=5):
    """
    Evaluate ranking performance using NDCG metric
    
    Parameters:
    - results_df: DataFrame with actual and predicted award shares
    - k: top-k players to consider (default: 5)
    
    Returns:
    - NDCG results by season and overall average
    """
    print(f"\n{'-'*20} NDCG Ranking Evaluation {'-'*20}")
    
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
            season_df['predicted_award_share'].values,
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

def create_ranking_visualizations_ndcg(ndcg_results, k=5):
    """
    Create visualizations for NDCG evaluation
    """
    # Create a DataFrame for easier plotting
    ndcg_df = pd.DataFrame(ndcg_results)
    
    # Save NDCG results to CSV
    ndcg_df.to_csv(f"{OUTPUT_DIR}/ndcg_results.csv", index=False)
    
    # Bar chart of NDCG by season
    plt.figure(figsize=(12, 6))
    sns.barplot(x='season', y='ndcg', data=ndcg_df, palette='viridis')
    plt.title(f'NDCG@{k} by Season (Ensemble Model)')
    plt.xlabel('Season')
    plt.ylabel(f'NDCG@{k}')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PICS_DIR}/ndcg_by_season.png")
    
    # Calculate average NDCG
    average_ndcg = ndcg_df['ndcg'].mean()
    
    # Create a DataFrame with season-wise and average NDCG
    summary_df = ndcg_df.copy()
    summary_df['average_ndcg'] = average_ndcg
    summary_df.to_csv(f"{OUTPUT_DIR}/ndcg_summary.csv", index=False)
    
    return summary_df

def save_top5_comparison(results_df, k=5):
    """
    Save a comparison of the top 5 actual players vs top 5 predicted players for each season
    
    Parameters:
    - results_df: DataFrame with actual and predicted award shares
    - k: Number of top players to include (default: 5)
    
    Returns:
    - DataFrame with top-k comparison for each season
    """
    print(f"\n{'-'*20} Top {k} Actual vs Predicted Comparison {'-'*20}")
    
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
        top_predicted = season_df.nlargest(k, 'predicted_award_share')
        
        # Create a combined DataFrame for this season
        comparison_df = pd.DataFrame()
        comparison_df['season'] = [season] * k
        comparison_df['rank'] = list(range(1, k+1))
        
        # Add actual top players
        comparison_df['actual_player'] = top_actual['player'].values
        comparison_df['actual_award_share'] = top_actual['award_share'].values
        
        # Add predicted top players
        comparison_df['predicted_player'] = top_predicted['player'].values
        comparison_df['predicted_award_share'] = top_predicted['predicted_award_share'].values
        
        # Add whether prediction was correct (player in same position)
        comparison_df['correct_player'] = comparison_df['actual_player'] == comparison_df['predicted_player']
        
        # Append to the overall results
        all_top_comparisons = pd.concat([all_top_comparisons, comparison_df], ignore_index=True)
    
    # Save to CSV
    all_top_comparisons.to_csv(f"{OUTPUT_DIR}/top{k}_actual_vs_predicted.csv", index=False)
    
    # Print a summary
    for season in seasons:
        season_comp = all_top_comparisons[all_top_comparisons['season'] == season]
        correct_count = season_comp['correct_player'].sum()
        print(f"Season {season}: {correct_count}/{k} correct predictions in top {k}")
    
    # Overall accuracy
    overall_accuracy = all_top_comparisons['correct_player'].mean() * 100
    print(f"\nOverall accuracy for top {k} predictions: {overall_accuracy:.2f}%")
    
    return all_top_comparisons

# =============== CUSTOM TRAINING FUNCTION WITH CLASS WEIGHTING ===============

def train_model_with_weighting(model, X_train, y_train, epochs, batch_size, validation_split=0.2, callbacks=None):
    """
    Train model with sample weighting to handle class imbalance
    - Gives higher weight to actual vote-getters (non-zero award shares)
    """
    # Create sample weights - higher for actual vote getters
    sample_weights = np.ones(len(y_train))
    
    # Identify vote getters (non-zero award shares)
    vote_getters = (y_train > 0).values  # Convert to numpy array
    
    # Calculate class ratio for balancing
    n_non_vote = np.sum(~vote_getters)
    n_vote = np.sum(vote_getters)
    vote_weight = n_non_vote / n_vote if n_vote > 0 else 1.0
    
    # Set higher weights for vote getters - the higher their award share, the higher the weight
    y_train_values = y_train.values  # Convert to numpy array
    for i in range(len(y_train)):
        if vote_getters[i]:
            # Scale weight by award share value (so higher shares get higher weights)
            # Minimum weight is vote_weight, max is vote_weight * 10 for award_share=1.0
            sample_weights[i] = vote_weight * (1.0 + 9.0 * y_train_values[i])
    
    print(f"Training with weighted samples - Vote getters weight range: {vote_weight:.2f} to {vote_weight * 10:.2f}")
    
    # Train the model with sample weights
    history = model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2
    )
    
    return history

# =============== MAIN EXECUTION ===============

def main():
    print(f"{'='*20} MVP Award Share Prediction - Enhanced Ensemble Model {'='*20}")

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
        essential_cols = all_features + ['season', 'player']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Error: Missing essential columns: {missing_cols}")
        
        # Keep all players, don't filter by MVP_rank or award_share
        df_mvp = df.copy()
        
        # Handle missing values in award_share - set to 0 for players who didn't receive votes
        df_mvp['award_share'] = df_mvp['award_share'].fillna(0)
        
        print(f"Total number of player seasons: {len(df_mvp)}")
        
        # Convert data types and handle missing values
        # Convert feature columns to numeric
        for col in all_features:
            if col in df_mvp.columns:
                df_mvp[col] = pd.to_numeric(df_mvp[col], errors='coerce')
        
        # Drop rows with NaN in features (but keep rows even if award_share is 0)
        df_mvp.dropna(subset=all_features, inplace=True)
        print(f"After dropping NaNs in features: {len(df_mvp)} rows")
        
        # Display distribution of award shares
        vote_getters = df_mvp[df_mvp['award_share'] > 0]
        print(f"\nNumber of players who received MVP votes: {len(vote_getters)}")
        print(f"Number of players who didn't receive MVP votes: {len(df_mvp) - len(vote_getters)}")
        
        # Display award_share statistics
        print(f"\nAward Share statistics:")
        print(f"Mean: {df_mvp['award_share'].mean():.4f}")
        print(f"Non-zero Mean: {df_mvp[df_mvp['award_share'] > 0]['award_share'].mean():.4f}")
        print(f"Min: {df_mvp['award_share'].min():.4f}")
        print(f"Max: {df_mvp['award_share'].max():.4f}")
        print(f"Median: {df_mvp['award_share'].median():.4f}")
        print(f"Percentage of players with votes: {len(vote_getters) / len(df_mvp) * 100:.2f}%")
        
        # Create feature matrix with only selected features
        X = df_mvp[award_share_features]
        
        # Target variable - award_share
        y = df_mvp['award_share']
        
        # Reset indices to ensure alignment after splitting
        X = X.reset_index(drop=True)
        df_mvp = df_mvp.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Split data for training and testing by season (year)
        print(f"\nSplitting data by seasons...")
        
        # Get unique seasons and sort them
        seasons = df_mvp['season'].unique()
        seasons.sort()
        print(f"Available seasons: {seasons}")
        
        # Set the number of training seasons (33) and the rest for testing
        NUM_TRAIN_SEASONS = 33
        NUM_TEST_SEASONS = len(seasons) - NUM_TRAIN_SEASONS
        
        # Always include 2005 in the training set, then randomly select the rest
        np.random.seed(RANDOM_STATE)  # For reproducibility
        available_seasons = np.array([s for s in seasons if s != 2005])  # Remove 2005 from random selection pool
        random_train_seasons = np.random.choice(available_seasons, size=NUM_TRAIN_SEASONS-1, replace=False)  # Select one less
        train_seasons = np.append(random_train_seasons, [2005])  # Add 2005 to training set
        test_seasons = np.array([season for season in seasons if season not in train_seasons])
        
        print(f"Training seasons: {train_seasons}")
        print(f"Testing seasons: {test_seasons}")
        print(f"Confirmed 2005 in training set: {2005 in train_seasons}")
        
        # Create train/test masks based on seasons
        train_mask = df_mvp['season'].isin(train_seasons)
        test_mask = df_mvp['season'].isin(test_seasons)
        
        # Split data using the masks
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]  
        y_test = y[test_mask]
        
        # Save test indices
        test_indices = df_mvp[test_mask].index
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of training seasons: {len(train_seasons)}")
        print(f"Number of testing seasons: {len(test_seasons)}")
        
        # Build and train the ensemble model
        print(f"\n{'-'*20} Enhanced Ensemble Model Training {'-'*20}")
        ensemble_model = build_ensemble_model(X_train.shape[1])
        
        # Calculate model size
        trainable_count = np.sum([K.count_params(w) for w in ensemble_model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in ensemble_model.non_trainable_weights])
        
        # Convert to MB
        trainable_mb = trainable_count * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
        non_trainable_mb = non_trainable_count * 4 / (1024 * 1024)
        
        print(f"\nModel Size Summary:")
        print(f"Total parameters: {trainable_count + non_trainable_count:,}")
        print(f"Trainable parameters: {trainable_count:,} ({trainable_mb:.2f} MB)")
        print(f"Non-trainable parameters: {non_trainable_count:,} ({non_trainable_mb:.2f} MB)")
        
        print(ensemble_model.summary())
        
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
            min_lr=1e-5,
            verbose=1
        )
        
        # Train model with sample weighting
        print("\nTraining enhanced ensemble model with class weighting...")
        history = train_model_with_weighting(
            ensemble_model,
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[
                early_stopping,
                reduce_lr,
            ]
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
        plt.savefig(f'{PICS_DIR}/award_share_training_history.png')
        print(f"Training history saved to '{PICS_DIR}/award_share_training_history.png'")
        
        # Make predictions
        print(f"\n{'-'*20} Prediction and Evaluation {'-'*20}")
        y_pred_award_share = ensemble_model.predict(X_test).flatten()
        
        # Ensure predictions are in valid range (0-1)
        y_pred_award_share = np.clip(y_pred_award_share, 0, 1)
        
        # Performance metrics for award_share prediction
        mse = mean_squared_error(y_test, y_pred_award_share)
        mae = mean_absolute_error(y_test, y_pred_award_share)
        r2 = r2_score(y_test, y_pred_award_share)
        
        print(f"\nAward Share Prediction Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        # Performance metrics for vote getters only
        vote_getters_mask = y_test > 0
        if np.sum(vote_getters_mask) > 0:
            vote_getters_mse = mean_squared_error(y_test[vote_getters_mask], y_pred_award_share[vote_getters_mask])
            vote_getters_mae = mean_absolute_error(y_test[vote_getters_mask], y_pred_award_share[vote_getters_mask])
            vote_getters_r2 = r2_score(y_test[vote_getters_mask], y_pred_award_share[vote_getters_mask])
            
            print(f"\nAward Share Prediction Performance (Vote Getters Only):")
            print(f"Mean Squared Error: {vote_getters_mse:.4f}")
            print(f"Mean Absolute Error: {vote_getters_mae:.4f}")
            print(f"R-squared: {vote_getters_r2:.4f}")
        
        # Save detailed prediction results
        print("\nSaving detailed prediction results...")
        results_df = save_prediction_results(df_mvp, test_indices, y_test, y_pred_award_share)
        
        # Create prediction visualizations
        print("Creating prediction visualizations...")
        create_prediction_visualizations(results_df)
        
        # Evaluate ranking performance
        print("\nEvaluating ranking performance...")
        results, average_ndcg = evaluate_ranking_performance_ndcg(results_df)
        
        # Create ranking visualizations
        print("Creating ranking visualizations...")
        summary_df = create_ranking_visualizations_ndcg(results)
        
        # Save comparison of top 5 actual vs predicted players
        print("\nSaving top 5 actual vs predicted players comparison...")
        top5_comparison = save_top5_comparison(results_df, k=5)
        
        print(f"\n{'='*20} Enhanced Ensemble Model Evaluation Complete {'='*20}")
        print(f"Results saved to '{OUTPUT_DIR}/' directory")
        print(f"Visualizations saved to '{PICS_DIR}/' directory")
        
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

if __name__ == "__main__":
    main() 