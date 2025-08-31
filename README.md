# fantasy-football-assistant
A data-driven draft assistant that helps fantasy football managers make smarter picks via projections, value-over-replacement (VORP), and roster construction optimization during the drafting process.
This repository is for U-M SIADS 496 — Milestone II. 

## 👥 Team Members
The team for this project consists of [Cedric Lambert](https://github.com/cedlamb-122), [Austin Miller](https://github.com/milleau98), and [Ryan Pierce](https://github.com/ryanapierce).

## 🎯 Project Goals
Ingest multi-source NFL data (historical stats, depth charts, ADP).
Model player projections and uncertainty.
Rank players using VORP and positional scarcity.
Optimize draft decisions under league settings (scoring, roster slots).
Explain recommendations (feature importances, scenario tips).

## 🔭 Milestone II Scope
Project problem statement & success metrics
Data audit + EDA notebooks
Clean data schema + processing pipeline
Baseline projection models (e.g., regressor + heuristics)
Initial ranking logic (VORP) & sanity checks
Draft-time UI (alpha) with position filters and replacement-level logic
Unit tests for core utilities


## 🗂️ Datasets
**TBP**


## 🧠 Modeling (Milestone 2 Baselines)
Targets: Fantasy points under league scoring
Features (initial): prior-year stats, usage (attempts, targets), team pace/efficiency proxies, depth chart role
Models: Regularized linear models / Gradient boosting baseline
Evaluation: MAE/RMSE on holdout; position-wise error analysis
Uncertainty: Residual-based intervals (Milestone 3: quantile/NGBoost)


## 📊 Ranking Methodology
**TBP**

## 📓 Notebooks
**TBP**

## 🤝 Contributing
PRs welcome for:
Data loaders for additional sources
Feature ideas (air yards, route rates, target share stability)
Modeling improvements and calibration
Open an issue with:
What you changed
Why it helps
How to reproduce


## 📜 License
MIT — see LICENSE.


## 🙏 Acknowledgments
Thanks to open-source NFL analytics communities and academic resources used in this project. Please see notebook citations for specific datasets and papers.

