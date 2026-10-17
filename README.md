# Student Dropout Prediction

Predicts whether a university student will **drop out, remain enrolled, or graduate**, and identifies the academic and financial factors that drive dropout risk.

## Dataset

[UCI: Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (Realinho et al.), 4,424 students from a Portuguese higher-education institution with 36 features:

- Demographics and socio-economic factors
- 1st and 2nd semester academic performance (units enrolled, approved, grades)
- Financial status (tuition payment, debtor, scholarship)
- Macroeconomic indicators (unemployment, inflation, GDP)

The target has three classes and is imbalanced: about 50% Graduate, 32% Dropout, 18% Enrolled. Because of this, models are compared on **macro F1** rather than accuracy. See the dataset page for license terms.

## Approach

1. **EDA:** univariate, bivariate and multivariate analysis, with a written takeaway under each plot
2. **Feature engineering:** one-hot encoding of nominal columns and an engineered `1st_sem_approval_rate` (approved ÷ enrolled units)
3. **Modeling:** Logistic Regression (scaled, inside a pipeline), Random Forest and XGBoost on a stratified 80/20 split
4. **Evaluation:** stratified 5-fold cross-validation on macro F1, plus per-class precision, recall and F1
5. **Tuning:** `RandomizedSearchCV` on XGBoost (25 candidates, 5-fold CV)
6. **Interpretation:** feature importance checked against the EDA findings

## Results

**Cross-validated comparison (5-fold, macro F1)**

| Model | Macro F1 (mean ± std) |
|---|---|
| Logistic Regression | 0.675 ± 0.008 |
| Random Forest | 0.692 ± 0.015 |
| **XGBoost** | **0.703 ± 0.008** |

**Tuned XGBoost on the held-out test set (885 students):** 77% accuracy, 0.70 macro F1

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Dropout | 0.82 | 0.74 | 0.78 |
| Enrolled | 0.49 | 0.45 | 0.47 |
| Graduate | 0.82 | 0.90 | 0.86 |

Best parameters found: `n_estimators=100`, `max_depth=6`, `learning_rate=0.2`, `subsample=0.8`, `colsample_bytree=1.0`. Tuning gave little improvement over the default model, and the gap between the ensemble models and Logistic Regression is modest, which suggests much of the signal in this data is linear.

## Key findings

- **Strongest predictors** (XGBoost feature importance, in order): 2nd-semester approved units, tuition fees up to date, the engineered 1st-semester approval rate, 1st-semester enrolled units, 2nd-semester enrolled units, debtor status, scholarship holder, 1st-semester approved units.
- **Financial status matters:** students with tuition not up to date show a far higher dropout rate, and scholarship holders drop out less.
- **Admission grade is a weak predictor:** outcomes overlap heavily on admission grade (ranks 30 of 57 features).
- **Course ranks highly despite near-zero linear correlation:** it is a categorical ID, so tree models can split on it even though a correlation coefficient cannot capture it.
- **Gender:** male students drop out at a higher rate (~46% vs ~26% for female students), but gender ranks only 22nd in importance. Female students approve more 1st-semester units on average (5.1 vs 3.9), which suggests the gap is largely carried through academic performance rather than adding independent signal.

## Limitations

- **Enrolled is hard to predict (F1 0.47).** These students' outcomes are still undetermined, so their features overlap with both other groups.
- Feature importance shows what the model uses, not what causes dropout, and it has no direction. The EDA plots show the direction of each effect.
- Correlated features (such as 1st and 2nd semester units) share importance, so their relative ordering should not be over-interpreted.
- The strongest predictor is only known after the 2nd semester, which limits use for early intervention.

## Next steps

- Train an early-warning variant using only enrollment-time and 1st-semester features, and measure how much performance drops
- Test class weighting or resampling to improve Enrolled recall
- Build a small app that returns dropout-risk probabilities for a student profile

## Repository structure

```
student-dropout-prediction/
├── data/
│   └── data.csv
├── notebooks/
│   └── Untitled-checkpoint.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

```bash
git clone https://github.com/<ashishg2502>/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt
jupyter notebook notebooks/Untitled-checkpoint.ipynb
```

Run the notebook from top to bottom (**Kernel → Restart & Run All**). It reads the data from `../data/data.csv`, so launch Jupyter from the project root as shown.

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost
