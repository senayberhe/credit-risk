# Credit Risk Modeling

A Python project for learning credit risk modeling — aligned with the Udemy course **"Credit Risk Modeling in Python"**.

## Project Structure

```
credit_risk_modeling/
├── data/
│   ├── raw/          # Original, unmodified data (put Udemy datasets here)
│   └── processed/    # Cleaned and feature-engineered data
├── notebooks/        # Jupyter notebooks for exploration & learning
├── src/              # Reusable Python modules
├── models/           # Saved trained models
└── reports/          # Outputs: charts, metrics, summaries
```

## Setup with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it once if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
cd ~/Desktop/credit_risk_modeling

# Create virtual environment and install all dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook
```

That's it — no need to activate the environment manually.

## Running a single script

```bash
uv run python src/utils.py
```

## Udemy Course Alignment

| Course Section | Notebook |
|---|---|
| EDA & Data Understanding | `01_eda.ipynb` |
| Preprocessing & WoE/IV | `02_preprocessing.ipynb` |
| PD Model (Logistic Regression) | `03_modeling.ipynb` |
| Model Evaluation (AUC, KS, Gini) | `04_model_evaluation.ipynb` |

> **Tip:** Put the datasets from the Udemy course into `data/raw/` and update the file paths in each notebook.

## Key Concepts

- **PD** – Probability of Default
- **LGD** – Loss Given Default
- **EAD** – Exposure at Default
- **WoE** – Weight of Evidence (used in scorecard modeling)
- **IV** – Information Value (feature selection for scorecards)
- **Expected Loss** = PD × LGD × EAD
# credit-risk
# credit-risk
