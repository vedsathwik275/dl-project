# Model Architecture Documentation

This document details the architectures of the various models developed for NBA MVP award share prediction and ranking.

## 1. Baseline Machine Learning Models

Source: `scripts/predict_mvp_rank_ndcg.py`

These models serve as traditional benchmarks.

*   **Linear Regression:** Standard `sklearn.linear_model.LinearRegression` with default parameters.
*   **Random Forest Regressor:** Standard `sklearn.ensemble.RandomForestRegressor` with default parameters (`random_state` set for reproducibility).
*   **Gradient Boosting Regressor:** Standard `sklearn.ensemble.GradientBoostingRegressor` with default parameters (`random_state` set for reproducibility).

**Input:** Selected features (typically 10) after season-wise Z-score normalization.
**Output:** Predicted `award_share` (clipped between 0 and 1).

## 2. Simple MLP (Occam's Razor Model)

Source: `scripts/simple_nn_mvp_ndcg.py`

Designed for simplicity and efficiency.

*   **Input Layer:** Dense(32 units, ReLU activation, L2 regularization)
*   Dropout(0.2)
*   **Hidden Layer 1:** Dense(16 units, ReLU activation, L2 regularization)
*   Dropout(0.2)
*   **Output Layer:** Dense(1 unit, Sigmoid activation)

**Input:** 10 selected features.
**Output:** Predicted `award_share` (0-1 scale due to Sigmoid).
**Key Idea:** Minimalist architecture to test if a simple model suffices.

## 3. Advanced MLP

Source: `scripts/mlp_nn_ndcg.py` (Represents a refined version compared to `mlp_nn.py`)

This model incorporates several modern deep learning techniques for potentially improved performance.

*   **Input Layer:** Input(shape=(num_features,))
*   **Initial Block:** Dense(256) -> LayerNormalization -> LeakyReLU(0.2) -> Dropout(0.3)
*   **Deep Path (with Residual Blocks):**
    *   Dense(128) -> LayerNorm -> LeakyReLU(0.2) -> Dropout(0.3)
    *   *Loop 3 times:*
        *   Residual Input = current layer
        *   Dense(64) -> LayerNorm -> LeakyReLU(0.2) -> Dropout(0.2) # Down-projection
        *   Dense(128) -> LayerNorm -> LeakyReLU(0.2) # Up-projection
        *   Add()([Residual Input, Up-projection]) # Residual Connection
        *   Dropout(0.2)
*   **Wide Path:**
    *   Dense(256) -> LayerNorm -> LeakyReLU(0.2) -> Dropout(0.3)
    *   Dense(196) -> LayerNorm -> LeakyReLU(0.2) -> Dropout(0.25)
*   **Attention Mechanism (on Deep Path):**
    *   Query = Dense(64)(Deep Path)
    *   Key = Dense(64)(Deep Path)
    *   Value = Dense(64)(Deep Path)
    *   Attention Scores = Dense(64, activation='softmax')(Query)
    *   Attention Output = Multiply()([Attention Scores, Value])
*   **Concatenation:** Concatenate([Deep Path, Wide Path, Attention Output])
*   **Final Dense Layers:**
    *   Dense(128) -> LayerNorm -> LeakyReLU(0.2) -> Dropout(0.2)
    *   Dense(64) -> LayerNorm -> LeakyReLU(0.2) -> Dropout(0.1)
    *   Dense(32) -> LayerNorm -> LeakyReLU(0.2)
*   **Output Layer:** Dense(1, activation='linear')

**Input:** 10 selected features.
**Output:** Predicted `award_share` (linear activation, clipped between 0 and 1 post-prediction).
**Key Ideas:** Multi-path learning (deep & wide), residual connections for gradient flow, layer normalization instead of batch normalization, self-attention on features.

## 4. Transformer-Based Model

Source: `scripts/transformer_mvp_ndcg.py`

Adapts the core self-attention mechanism from Transformers for the tabular data.

*   **Input Layer:** Input(shape=(num_features,))
*   **Initial Embedding:** Dense(embed_dim=128) -> LayerNormalization -> Dropout
*   **Initial Residual Block:** ResidualBlock(embed_dim, dropout_rate)
*   **Reshape:** ExpandDimsLayer(axis=1) # Add sequence dimension: (batch, 1, embed_dim)
*   **Transformer Blocks:**
    *   *Loop `num_transformer_blocks` (e.g., 4) times:*
        *   TransformerBlock(embed_dim, num_heads, ff_dim=embed_dim*2, rate=dropout_rate)
            *   MultiHeadAttention(num_heads, key_dim=embed_dim/num_heads)
            *   Dropout -> Add -> LayerNormalization
            *   FeedForward(ff_dim -> embed_dim)
            *   Dropout -> Add -> LayerNormalization
*   **Reshape:** SqueezeLayer(axis=1) # Remove sequence dimension: (batch, embed_dim)
*   **Head:**
    *   ResidualBlock(embed_dim, dropout_rate)
    *   Dense(64) -> LayerNorm -> ReLU -> Dropout
    *   Dense(32) -> LayerNorm -> ReLU -> Dropout
*   **Output Layer:** Dense(1, activation='sigmoid')

**Custom Layers:**
*   `ExpandDimsLayer`, `SqueezeLayer`: Wrappers for TensorFlow dimension manipulation.
*   `TransformerBlock`: Encapsulates multi-head attention and feed-forward network.
*   `ResidualBlock`: Standard residual connection block.

**Input:** 10 selected features.
**Output:** Predicted `award_share` (0-1 scale due to Sigmoid).
**Key Ideas:** Applying self-attention to input features, multiple transformer layers for context aggregation, residual connections throughout.

## 5. Ensemble Model

Source: `scripts/ensemble_mvp_ndcg.py`

Combines elements from MLP and Transformer approaches, plus explicit feature interaction.

*   **Input Layer:** Input(shape=(num_features,))

*   **Branch 1: Transformer Path**
    *   Initial Embedding: Dense(transformer_dim=96) -> LayerNorm -> Dropout
    *   Reshape: ExpandDimsLayer(axis=1)
    *   *Loop `num_transformer_blocks` (e.g., 2) times:*
        *   TransformerBlock(transformer_dim, num_heads, ...)
    *   Reshape: SqueezeLayer(axis=1)
    *   Transformer Output: Dense(transformer_dim / 2) -> ReLU

*   **Branch 2: MLP Path**
    *   Initial Embedding: Dense(mlp_dim=128) -> LayerNorm -> LeakyReLU -> Dropout
    *   Residual Block 1: ResidualBlock(mlp_dim)
    *   Residual Block 2: ResidualBlock(mlp_dim)
    *   MLP Output: Dense(mlp_dim / 2) -> ReLU

*   **Branch 3: Feature Interaction Path**
    *   Interaction: FeatureInteractionLayer(output_dim=num_features * (num_features - 1) // 2)
    *   Interaction Output: Dense(32) -> ReLU

*   **Concatenation:** Concatenate([Transformer Output, MLP Output, Interaction Output])

*   **Combined Path:**
    *   Dense(128) -> LayerNorm -> LeakyReLU -> Dropout
    *   Attention: SelfNormalizedAttention(attention_dim=32)
    *   Dense(64) -> LayerNorm -> LeakyReLU -> Dropout
    *   Dense(32) -> LayerNorm -> LeakyReLU

*   **Output Layer:** Dense(1, activation='linear')

**Custom Layers:**
*   `ExpandDimsLayer`, `SqueezeLayer`, `TransformerBlock`, `ResidualBlock`: Similar to Transformer model.
*   `FeatureInteractionLayer`: Computes explicit pairwise feature interactions.
*   `SelfNormalizedAttention`: Custom attention mechanism applied after branch merging.

**Input:** 10 selected features.
**Output:** Predicted `award_share` (linear activation, clipped 0-1 post-prediction).
**Key Ideas:** Combining strengths of different architectures, explicitly modeling feature interactions, using custom attention on combined features. 