# Training Procedures

This document outlines the common procedures used for training the various MVP prediction models.

## 1. Train/Test Split Strategy

Most scripts employ a **season-based splitting** strategy rather than a simple random row split. This is crucial to prevent data leakage and evaluate how well models generalize to unseen seasons.

*   **Method:** The data is split such that entire seasons are allocated to either the training set or the test set.
*   **Selection:**
    *   Some scripts use a fixed number of the *most recent* seasons for testing (e.g., `mlp_nn.py`).
    *   Many scripts (especially the `_ndcg` variants like `mlp_nn_ndcg.py`, `transformer_mvp_ndcg.py`, `simple_nn_mvp_ndcg.py`, `ensemble_mvp_ndcg.py`) use a fixed number of training seasons (e.g., 33) and randomly select these seasons, with the remaining seasons used for testing.
    *   A common pattern in the random selection is to *ensure a specific season (e.g., 2005) is always included in the training set* before randomly selecting the remaining training seasons. The reason for fixing 2005 is not explicitly stated but might relate to ensuring a specific data distribution or era is represented in training.
*   **Implementation:** This is typically achieved by getting unique season values, selecting train/test seasons, and then creating boolean masks on the main DataFrame based on whether a row's `season` is in the train or test set.

## 2. Hyperparameters

Hyperparameters vary slightly across models, but common ranges/values include:

*   **Epochs:** Typically set high (e.g., 200, 400, 1000, 1200), relying on Early Stopping to determine the actual number of training epochs.
*   **Batch Size:** Values like 32, 48, or 64 are common.
*   **Learning Rate:** Often around 0.001 or 0.0005, sometimes used with a learning rate scheduler.
*   **Optimizer:** Adam is consistently used.
*   **Loss Function:** Mean Squared Error (`mse`) is used for the `award_share` regression task.
*   **Metrics:** Mean Absolute Error (`mae`) is typically monitored during training alongside the loss.
*   **Regularization:** L1/L2 regularization is applied to Dense layers in NN models, with varying strengths (e.g., `l1=1e-5`, `l2=1e-4`). Dropout layers are also used extensively with rates from 0.1 to 0.4.
*   **Random State:** A fixed `RANDOM_STATE` (e.g., 42, 423) is often used for reproducibility in data splitting and model initialization (where applicable, e.g., RandomForest, GradientBoosting, NN weight initialization potentially influenced by TF/NumPy seeds).

## 3. Callbacks (Neural Networks)

TensorFlow/Keras callbacks are frequently used to manage the training process:

*   **`EarlyStopping`:** Monitors the validation loss (`val_loss`) and stops training if it doesn't improve for a specified number of epochs (`patience`, e.g., 15, 25, 30, 50). `restore_best_weights=True` ensures the model reverts to the state from the epoch with the best validation loss.
*   **`ReduceLROnPlateau`:** Monitors the validation loss and reduces the learning rate (e.g., by a factor of 0.5) if it plateaus for a set number of epochs (`patience`, e.g., 5, 15). This helps fine-tune the model when learning slows down.

## 4. Data Scaling

As mentioned in the Data Preprocessing document, input features (`X_train`, `X_test`) are typically scaled using `StandardScaler` applied *within each season* before being fed into the models. This is often done *after* the train/test split to avoid data leakage from the test set into the training set scaling factors. 