# Repository Structure and Usage Guide

## 1. Directory Structure

The project follows this general structure:

```
dl-project/
├── data/                 # Input data, model outputs (CSVs)
│   ├── normalized_nba_data_with_MVP_rank_simple.csv  # Main input dataset
│   ├── ml_evaluation_results/  # Outputs from predict_mvp_rank.py
│   ├── ml_ndcg_evaluation_results/ # Outputs from predict_mvp_rank_ndcg.py
│   ├── nn_evaluation_results/    # Outputs from mlp_nn.py
│   ├── nn_evaluation_results_ndcg/ # Outputs from mlp_nn_ndcg.py
│   ├── simple_nn_evaluation_results_ndcg/ # Outputs from simple_nn_mvp_ndcg.py
│   ├── transformer_evaluation_results_ndcg/ # Outputs from transformer_mvp_ndcg.py
│   ├── ensemble_evaluation_results_ndcg/ # Outputs from ensemble_mvp_ndcg.py
│   └── ...                 # Other intermediate or output CSVs
├── scripts/              # All Python scripts
│   ├── ml_models/          # Subdirectory for ML/NN training scripts
│   │   ├── predict_mvp_rank_ndcg.py
│   │   ├── predict_mvp_rank.py
│   │   ├── mlp_nn.py
│   │   ├── mlp_nn_ndcg.py
│   │   ├── simple_nn_mvp_ndcg.py
│   │   ├── transformer_mvp_ndcg.py
│   │   ├── ensemble_mvp_ndcg.py
│   │   └── ...
│   ├── filter_nba_data.py
│   ├── normalize_nba_data_by_era.py
│   ├── pca_mvp_analysis.py
│   ├── create_mvp_ranking.py
│   ├── evaluate_award_share_rankings.py
│   ├── selected_features.txt
│   └── ...
├── pics/                 # Output visualizations (PNGs)
│   ├── ndcg/               # Subdir for NDCG related plots
│   ├── transformer_ndcg/   # Subdir for Transformer NDCG plots
│   ├── simple_ndcg/        # Subdir for Simple NN NDCG plots
│   ├── ensemble_ndcg/      # Subdir for Ensemble NDCG plots
│   └── ...               # Other plots from various scripts
├── docs/                 # Technical documentation (Markdown files)
│   ├── Project_Overview_and_Goals.md
│   ├── Data_Acquisition_and_Preprocessing.md
│   ├── Feature_Engineering_and_Selection.md
│   ├── Model_Architecture_Documentation.md
│   ├── Training_Procedures.md
│   ├── Evaluation_Strategy_and_Metrics.md
│   ├── Results_and_Analysis.md
│   └── Repository_Structure_and_Usage_Guide.md
├── requirements.txt      # (Recommended) Python dependencies
└── README.md             # Main project README
```

## 2. Setup

1.  **Clone the repository.**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:** (Assuming a `requirements.txt` file exists or is created)
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`.

## 3. Running Scripts

*   Navigate to the `scripts` or `scripts/ml_models` directory in your terminal.
*   Execute the desired Python script.

**Example: Running the Advanced MLP with NDCG evaluation:**

```bash
cd scripts/ml_models
python mlp_nn_ndcg.py
```

**Example: Running the baseline ML models with NDCG evaluation:**

```bash
cd scripts/ml_models
python predict_mvp_rank_ndcg.py
```

## 4. Outputs

*   **Data:** Evaluation results, detailed predictions, and model comparisons are saved as CSV files in the corresponding subdirectories within `data/` (e.g., `data/nn_evaluation_results_ndcg/`).
*   **Visualizations:** Plots illustrating model performance, training history, and ranking analysis are saved as PNG files in the corresponding subdirectories within `pics/` (e.g., `pics/ndcg/`).
*   **Trained Models:** Some scripts might save the trained model artifacts (e.g., TensorFlow SavedModel format in directories like `award_share_prediction_model`). 