# fantasy-football-assistant
A data-driven fantasy football assistant that helps managers make smarter start-sit decisions via the week's fantasy point projections.
This repository is for U-M SIADS 496 — Milestone II. 

## 👥 Team Members
The team for this project consists of [Cedric Lambert](https://github.com/cedlamb-122), [Austin Miller](https://github.com/milleau98), and [Ryan Pierce](https://github.com/ryanapierce).

## 🎯 Project Goals
Ingest multi-source NFL data (historical stats, depth charts, ADP).
Model player projections and uncertainty.
Explain recommendations (feature importances, scenario tips).

## 🔭 Milestone II Scope
Project problem statement & success metrics
Data audit + EDA notebooks
Clean data schema + processing pipeline
Baseline projection models (e.g., regressor)

## 🗂️ Data Source
Source: [nfl_data_py](https://github.com/nflverse/nfl_data_py)

## 🧠 Modeling (Milestone 2 Baselines)
Targets: Fantasy points under league scoring
Features (initial): prior-year stats, usage (attempts, targets), team pace/efficiency proxies, depth chart role
Models: Regularized linear models / Gradient boosting baseline
Evaluation: MAE/RMSE on holdout; position-wise error analysis
Uncertainty: Residual-based intervals (Milestone 3: quantile/NGBoost)

## 📓 Notebooks
#### Dataset/Feature Engineering Generator Notebooks
- [Main Dataset](dataset_generators/init_1_lag_dataset_generator.ipynb)
- [MinMaxScaler 5 Fold Dataset Generator](dataset_generators/kfolds_dataset_generator.ipynb)
- [PCA 5 Fold Dataset Generator](dataset_generators/pca_dataset_generator.ipynb)
#### Model Generator Notebooks
- [MLP Regressor Model](model_generators/mlp/mlp_model_generator_pytorch.ipynb)
- [Linear Regression Model](model_generators/lr/lr_model_generator.ipynb)
#### Model Analysis Notebooks
- [MLP Data Analysis](model_analysis/mlp_data_analysis.ipynb)
- [MLP Ablation Analysis](model_analysis/mlp_model_ablation_analysis_features.ipynb)
- [MLP Model Analysis](model_analysis/mlp_model_analysis.ipynb)
- [MLP Feature Analysis](model_analysis/mlp_model_permutation_importance.ipynb)
- [MLP SHAP Analysis (Unused in report)](model_analysis/mlp_model_shap.ipynb)

## 📜 License
MIT — see LICENSE.

## 🙏 Acknowledgments
Thanks to open-source NFL analytics communities and academic resources used in this project. Please see notebook citations for specific datasets and papers.

