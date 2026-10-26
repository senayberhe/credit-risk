# Credit Risk Modeling

**By Senay Berhe** · [github.com/senayberhe/credit-risk](https://github.com/senayberhe/credit-risk)

> How much is this loan portfolio expected to lose? A full credit risk model on 466,285 LendingClub loans (2007-2014): a PD scorecard, a two-stage LGD model, an EAD model, the portfolio's Expected Loss, and a stability check against 2015 applicants.

Built alongside the Udemy course **"Credit Risk Modeling in Python"**.

---

## Key results

| Model | What it predicts | Result (held-out data) |
|---|---|---|
| PD scorecard | Probability a borrower defaults | AUROC 0.69 · Gini 0.39 · KS 0.28 |
| LGD stage 1 | Is anything recovered after default? | AUROC 0.69 |
| LGD stage 2 | Share recovered, when something is | Correlation 0.27 · MAE 0.044 |
| EAD | Share of the loan still owed at default (CCF) | Correlation 0.52 · MAE 0.131 |

**Expected Loss = PD × LGD × EAD = $515M on $6.66B funded (7.73%).**

| Grade | Loans | Mean PD | Mean LGD | Expected loss rate |
|---|---|---|---|---|
| A | 74,867 | 3.8% | 96.7% | 2.12% |
| B | 136,929 | 7.9% | 95.9% | 4.80% |
| C | 125,293 | 11.5% | 95.2% | 7.73% |
| D | 76,888 | 15.5% | 94.1% | 10.69% |
| E | 35,757 | 19.3% | 93.0% | 14.25% |
| F | 13,229 | 24.3% | 89.9% | 18.22% |
| G | 3,322 | 27.9% | 91.5% | 19.19% |

Expected loss rises steadily from grade A to G, as it should. LGD is high across grades: 43.6% of written-off loans recovered nothing, and the mean recovery rate is only 6%.

**Monitoring (PSI, 2015 vs 2007-2014):** every borrower attribute is stable (PSI < 0.10). The score shifts strongly (PSI 1.12), but that comes from `mths_since_issue_d`, a loan-age variable that structurally favours newer loans, and from a real switch to whole-loan listings (`initial_list_status`, PSI 0.34). See `04_monitoring_psi.ipynb` for the recommendation.

---

## Notebooks

| Notebook | What it does |
|---|---|
| `eda.ipynb` | Exploration, general preprocessing, WoE / IV fine and coarse classing |
| `01_load_data_creditrisk_preprocessing_pd.ipynb` | Preprocessing pipeline: dummies, WoE bins, train/test split |
| `credit_risk_model.ipynb` | PD model (logistic regression with p-values), validation (AUROC, Gini, KS), scorecard, cut-offs |
| `02_loaddata_creditrisk_pd_scorecard_cutoff.ipynb` | Alternative PD specification with fewer features (AUROC 0.68) |
| `03_lgd_ead_expected_loss.ipynb` | Two-stage LGD model, EAD model, Expected Loss by loan and grade |
| `04_monitoring_psi.ipynb` | Population Stability Index of the scorecard on 2015 loans |

Shared code lives in `src/`:

- `utils.py`: WoE / IV, coarse classing, logistic and linear regression with p-values
- `preprocessing.py`: raw-to-model preprocessing, so new data is prepared exactly like the training data
- `expected_loss.py`: recovery rate, CCF, LGD/EAD features, PSI

Saved models are in `models/` (`pd_model.sav`, `lgd_model_stage_1.sav`, `lgd_model_stage_2.sav`, `ead_model.sav`); the PSI table is in `reports/`.

---

## Data

LendingClub loan data, 2007-2014 (training) and 2015 (monitoring), plus `LCDataDictionary.xlsx`. The files are 190-240MB each and are not in the repo: put them in `data/raw/` as `loan_data_2007_2014.csv` and `loan_data_2015.csv`, then run the notebooks in the order above. `filter_loan.py` trims a full Kaggle `loan.csv` to the 74 columns used here.

## Setup with uv

```bash
uv sync
uv run jupyter notebook
```

## Key concepts

- **PD** – Probability of Default
- **LGD** – Loss Given Default = 1 − recovery rate
- **EAD** – Exposure at Default = credit conversion factor × funded amount
- **WoE / IV** – Weight of Evidence and Information Value, used to bin and select scorecard variables
- **PSI** – Population Stability Index: < 0.10 stable, 0.10–0.25 some shift, > 0.25 significant shift
- **Expected Loss** = PD × LGD × EAD
