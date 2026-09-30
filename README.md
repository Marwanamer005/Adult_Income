# Adult Income Classification: Decision Tree vs Random Forest

NTI Assignment 6. Predicts whether a person earns **more than $50K a year** from 1994 US census data (the "Adult" dataset), using a Decision Tree and a Random Forest.

## Project structure

| File | What it is |
|---|---|
| `adult.csv` | The dataset: 48,842 rows, 15 columns, target column `income` (`<=50K` / `>50K`) |
| `Assigment_6.ipynb` | The full analysis: cleaning, EDA, visualization, Decision Tree, Random Forest, comparison |
| `requirements.txt` | Python libraries needed to run the notebook |

## How to run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then open `Assigment_6.ipynb` in VS Code or Jupyter, pick the `.venv` kernel and run all cells. A full run takes about a minute.

## What the notebook does

### 1. Data cleaning
- **Missing values** are stored as `'?'`, not as empty cells: `workclass` (2,799), `occupation` (2,809), `native-country` (857). They are replaced with an `Unknown` category rather than dropped, because people with missing occupation earn >50K far less often (9% vs 24%), so the missingness itself is informative.
- **52 duplicate rows** removed (48,790 rows left).
- **`education` dropped**: it maps one-to-one onto `educational-num`.
- **Outliers kept**: `capital-gain`/`capital-loss` are zero for over 90% of people and the 99999 cap value is real signal (all 244 people with it earn >50K). Tree models are not affected by outliers or scale.
- **Target** mapped to `0` (<=50K) and `1` (>50K).

### 2. EDA and visualization
Class balance, histograms and box plots of numeric features by income, a correlation heatmap, the share earning >50K for each categorical feature, and income rate by education and age.

Main findings:
- The classes are imbalanced: **76% <=50K, 24% >50K**, so accuracy alone is misleading.
- Being **married** is the strongest signal (~45% earn >50K vs ~5% for never-married).
- Income rises steadily with **education**, and with **age** up to about 45–55.
- More **hours per week** and any **capital gain** are linked to higher income.
- About 30% of men earn >50K versus about 11% of women.
- `fnlwgt` (census sampling weight) has no relationship with income and is dropped before modelling.

### 3. Preprocessing
One-hot encoding of the 7 categorical columns (91 features in total), no scaling, and a **stratified 70/30 train/test split**.

### 4. Models
Following the structure of the *Random Forest Tutorial* notebook:
1. **Decision Tree, unlimited depth**: overfits heavily (11,883 nodes, depth 49).
2. **Decision Tree, tuned** with `GridSearchCV` (5-fold, ROC AUC): best `max_depth=15`, `min_samples_leaf=50`.
3. **Random Forest, default** (100 trees, `max_features='sqrt'`).
4. **Random Forest, tuned** with `RandomizedSearchCV` (20 candidates, 3-fold, ROC AUC): best `n_estimators=200`, `max_depth=25`, `max_features=0.3`, `min_samples_split=20`, `min_samples_leaf=2`.

## Results (test set, 14,637 people)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline (always <=50K) | 0.761 | 0.000 | 0.000 | 0.000 | 0.500 |
| Decision Tree (unlimited) | 0.820 | 0.627 | 0.613 | 0.620 | 0.774 |
| Decision Tree (tuned) | 0.858 | 0.756 | 0.601 | 0.670 | 0.907 |
| Random Forest (default) | 0.847 | 0.706 | 0.617 | 0.658 | 0.892 |
| **Random Forest (tuned)** | **0.866** | **0.770** | **0.630** | **0.693** | **0.918** |

Precision, recall and F1 are for the >50K class.

## Conclusions
- The unlimited Decision Tree memorises the training data (train ROC AUC 0.998 vs test 0.774). Limiting its depth and leaf size fixes most of the overfitting.
- The **tuned Random Forest is the best model** on every metric, confirming that averaging many randomised trees reduces variance.
- The most important features are capital gain, being married, education, age and hours per week, which matches the EDA.
- Recall for >50K (~0.63) is the weak point because of the class imbalance. Lowering the decision threshold or using `class_weight='balanced'` would catch more high earners at the cost of precision.

## Dataset
UCI Machine Learning Repository, [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) (Becker & Kohavi, 1996).
