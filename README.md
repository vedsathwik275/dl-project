# NBA MVP Prediction using Deep Learning

This repository contains code and documentation for predicting NBA Most Valuable Player (MVP) award rankings using various machine learning and deep learning approaches.

## Project Overview

The NBA MVP award represents the highest individual honor in the NBA regular season. In this project, we develop models that:
- Predict a player's "award_share" (proportion of maximum possible MVP votes)
- Rank players within each season based on these predictions
- Compare the performance of different model architectures
- Analyze the most influential statistical features for MVP prediction

Our approaches range from traditional machine learning baselines to sophisticated neural network architectures including MLPs with residual connections, transformer-based models, and custom ensemble architectures.

## Repository Structure

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
│   └── ensemble_evaluation_results_ndcg/ # Outputs from ensemble_mvp_ndcg.py
├── scripts/              # All Python scripts
│   ├── ml_models/          # ML training scripts
│   │   ├── predict_mvp_rank_ndcg.py      # Baseline ML models with NDCG evaluation
│   │   ├── predict_mvp_rank.py           # Baseline ML models with custom evaluation
│   │── mlp_nn.py                     # Advanced MLP with custom evaluation
│   │── mlp_nn_ndcg.py                # Advanced MLP with NDCG evaluation
│   │── simple_nn_mvp_ndcg.py         # Simple MLP with NDCG evaluation
│   │── transformer_mvp_ndcg.py       # Transformer model with NDCG evaluation
│   │── ensemble_mvp_ndcg.py          # Ensemble model with NDCG evaluation
│   ├── filter_nba_data.py                # Data filtering script
│   ├── normalize_nba_data_by_era.py      # Data normalization script
│   ├── pca_mvp_analysis.py               # Feature analysis using PCA
│   ├── create_mvp_ranking.py             # Create MVP rankings
│   ├── evaluate_award_share_rankings.py  # Standalone evaluation script
│   └── selected_features.txt             # List of selected features
├── pics/                 # Output visualizations (PNGs)
│   ├── ndcg/               # NDCG related plots
│   ├── transformer_ndcg/   # Transformer NDCG plots
│   ├── simple_ndcg/        # Simple NN NDCG plots
│   └── ensemble_ndcg/      # Ensemble NDCG plots
├── docs/                 # Technical documentation
│   ├── Project_Overview_and_Goals.md
│   ├── Data_Acquisition_and_Preprocessing.md
│   ├── Feature_Engineering_and_Selection.md
│   ├── Model_Architecture_Documentation.md
│   ├── Training_Procedures.md
│   ├── Evaluation_Strategy_and_Metrics.md
│   ├── Results_and_Analysis.md
│   └── Repository_Structure_and_Usage_Guide.md
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

The main dataset `normalized_nba_data_with_MVP_rank_simple.csv` should be placed in the `data/` directory. It should contain the following key columns:
- `player`: Player name
- `season`: NBA season year (e.g., 2023 for 2022-23 season)
- `award_share`: Target variable (proportion of MVP votes, 0-1 scale)
- Various player statistics including those listed in `selected_features.txt`

## Usage

### Running Baseline ML Models
```bash
cd scripts/ml_models
python predict_mvp_rank_ndcg.py
```

### Running Simple MLP
```bash
cd scripts
python simple_nn_mvp_ndcg.py
```

### Running Advanced MLP
```bash
cd scripts
python mlp_nn_ndcg.py
```

### Running Transformer Model
```bash
cd scripts
python transformer_mvp_ndcg.py
```

### Running Ensemble Model
```bash
cd scripts
python ensemble_mvp_ndcg.py
```

## Key Models

1. **Baseline ML Models**: Linear Regression, Random Forest, and Gradient Boosting models
2. **Simple MLP**: Minimalist neural network with just enough capacity to learn the mapping
3. **Advanced MLP**: Multi-path neural network with residual connections and attention mechanisms
4. **Transformer-Based Model**: Adapts the transformer architecture for tabular player statistics
5. **Ensemble Model**: Complex architecture combining MLP, transformer, and feature interaction components

## Results Summary

Model performance evaluated using Normalized Discounted Cumulative Gain at 5 (NDCG@5):

| Model              | Average NDCG@5 |
|:-------------------|:--------------:|
| Linear Regression  | 0.580          |
| Random Forest      | 0.663          |
| Gradient Boosting  | 0.678          |
| Simple MLP         | 0.799          |
| Advanced MLP       | 0.802          |
| Transformer        | 0.772          |
| Ensemble           | 0.759          |

The Advanced MLP model achieved the best performance in ranking players according to MVP voting patterns.

## Team Members

- Naman Tellakula (ntellakula3@gatech.com)
- Vedanth Sathwik Toduru Madabushi (vmadabushi6@gatech.edu)