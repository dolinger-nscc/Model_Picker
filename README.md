# Model Pickers

PyCaret-style model selection built on **scikit-learn**, in three separate, self-contained modules:

| Module | Use it when your target is... | Class |
|---|---|---|
| `regression_picker_v2.py` | a **number** (price, sales, temperature) | `RegressionPicker` |
| `classification_picker_v2.py` | a **category** (Yes/No, species, grade) | `ClassificationPicker` |
| `cluster_picker_v2.py` | **none**: you want to find groups (segmentation) | `ClusterPicker` |

Each picker does three things:

1. **Compares** every suitable model with cross-validation and ranks them.
2. **Tunes** the best few and recommends hyperparameters.
3. **Charts** the results (v2 files only).

> The pickers *select* models and hyperparameters. They expect a **clean DataFrame**, so do your cleaning, feature engineering and EDA first.

---

## Contents

- [Installation](#installation)
- [Folder layout and importing](#folder-layout-and-importing)
- [The workflow (same for all three)](#the-workflow-same-for-all-three)
- [How starting hyperparameters are chosen](#how-starting-hyperparameters-are-chosen)
- [RegressionPicker](#regressionpicker)
- [ClassificationPicker](#classificationpicker)
- [ClusterPicker](#clusterpicker)
- [Common options](#common-options)
- [Tips and gotchas](#tips-and-gotchas)
- [v1 vs v2](#v1-vs-v2)

---

## Installation

```bash
pip install scikit-learn pandas numpy scipy matplotlib
# optional extra models (picked up automatically if installed)
pip install xgboost lightgbm catboost
```

- **Python:** 3.10 or newer.
- **scikit-learn:** 1.4 or newer (tested on 1.9).
- **matplotlib:** only needed for the `plot_` methods, and loaded only when one is called.
- **XGBoost / LightGBM / CatBoost:** optional. If they're installed they join the comparison, and if not they're skipped quietly.

---

## Folder layout and importing

```
ML/
├── pickers/
│   ├── __init__.py
│   ├── regression_picker_v2.py
│   ├── classification_picker_v2.py
│   └── cluster_picker_v2.py
├── tests/
│   └── test_regression_picker_v2.ipynb
└── week3/
    └── lab.ipynb
```

**`pickers/__init__.py`**, so you can write `from pickers import ...`:

```python
from .regression_picker_v2 import RegressionPicker
from .classification_picker_v2 import ClassificationPicker
from .cluster_picker_v2 import ClusterPicker

__all__ = ["RegressionPicker", "ClassificationPicker", "ClusterPicker"]
```

**Notebook directly in `ML/`:**

```python
from pickers import RegressionPicker, ClassificationPicker, ClusterPicker
```

**Notebook in any subfolder of `ML/`:** this walks up the folders until it finds `pickers`:

```python
import sys
from pathlib import Path

ml_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pickers").is_dir())
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from pickers import RegressionPicker, ClassificationPicker, ClusterPicker
```

**Editing the `.py` files while you work?** Add this so changes load without restarting the kernel:

```python
%load_ext autoreload
%autoreload 2
```

---

## The workflow (same for all three)

```python
picker = SomePicker(df, ...)     # 1. set up (validates the data, works out starting values)
picker.compare_models()          # 2. ranked table of every model
picker.tune_top(n=3)             # 3. tune the best 3
picker.recommended_params_       # 4. the hyperparameters to use
picker.summary()                 # 5. plain-text recap
picker.plot_leaderboard()        # 6. charts (v2)
```

| Step | Method | Returns |
|---|---|---|
| See what will run | `models()` | Table of every model, its family, and why it would be skipped |
| See starting values | `starting_params("rf")` | dict of hyperparameters that differ from scikit-learn's defaults |
| Compare | `compare_models()` | Leaderboard DataFrame (also stored in `leaderboard_`) |
| Tune | `tune_top(n=3, n_iter=25)` | Start-vs-tuned table (also stored in `tuned_`) |
| Recommended settings | `recommended_params_` | `{model_id: {param: value}}` |
| Use a model | `get_model("rf")` | Fitted pipeline (tuned version if tuning helped) |
| Recap | `print(summary())` | Data facts, winner, its hyperparameters |

What happens behind the scenes:

- **Leakage-safe preprocessing:** numeric columns are standardized where a model needs it, and text/category/bool columns are one-hot encoded. All of it happens inside a scikit-learn `Pipeline`, so nothing leaks between cross-validation folds.
- **Tuned only if it's better:** tuning keeps the tuned version **only if it beats the starting version** in cross-validation. The `Kept` column tells you which version was kept.
- **One bad model doesn't stop the run:** a model that's too slow for your data size, or that fails, is skipped and listed in `skipped_`.

---

## How starting hyperparameters are chosen

Starting values are **worked out from your data**, not copied from the library defaults. Some examples:

| Model | Starting value | Why |
|---|---|---|
| Lasso / Elastic Net | `alpha` = 1% of the data's own `alpha_max` | A fixed alpha is far too strong or too weak depending on the target's scale |
| K Neighbors | `k` grows with the row count (5 to 25), distance-weighted | Small data needs a small k |
| Decision Tree | smallest leaf = about 1% of rows | The default memorises the training data |
| SVR / MLP / SGD | trained on a standardized target | Then `C` and `epsilon` mean the same thing on any dataset |
| Boosting models | learning rate 0.05, early stopping | Stable without hand-tuning |
| Classifiers, uneven classes | `class_weight="balanced"` | Switched on automatically when the smallest class is under 50% of the largest |
| LDA | automatic (Ledoit-Wolf) shrinkage | Picks its own regularisation |
| Clustering: number of clusters | best silhouette from a K-Means scan of 2 to 10 | |
| DBSCAN `eps` | the knee of the nearest-neighbour distance curve | |

Then `tune_top()` runs a random search over a curated range **centred on those starting values**.

---

## RegressionPicker

**For a numeric target.**

```python
from pickers import RegressionPicker

rp = RegressionPicker(df, target="median_house_value", sort="RMSE")
rp.compare_models()
rp.tune_top(n=3, n_iter=20)
print(rp.summary())

model = rp.finalize()                   # refit the winner on ALL rows
predictions = model.predict(new_df)     # new_df has the same columns, minus the target
```

### Key options

| Option | Default | Meaning |
|---|---|---|
| `target` | required | Name of the numeric column to predict |
| `sort` | `"R2"` | `"R2"`, `"RMSE"`, `"MAE"`, `"MAPE"` |
| `test_size` | `0.2` | Share held back as an untouched holdout set |
| `cv` | `5` | Cross-validation folds |

### Metrics

| Metric | Better | Notes |
|---|---|---|
| R2 | higher | 1 = perfect, 0 = no better than predicting the mean |
| RMSE | lower | In target units; punishes big misses |
| MAE | lower | In target units; average miss |
| MAPE | lower | Average % miss; unreliable when the target is near 0 |

### Models (ids)

| Family | Model ids |
|---|---|
| Baseline | `dummy` |
| Linear | `lr`, `ridge`, `lasso`, `en`, `lar`, `llar`, `omp`, `br`, `ard`, `sgd` |
| Robust (resist outliers) | `huber`, `ransac`, `tr`\*, `quantile` |
| GLMs (offered only if the target allows) | `poisson` (target ≥ 0), `gamma` (target > 0), `tweedie` (target ≥ 0) |
| Kernel / distance | `knn`, `svr`, `kr`\* |
| Trees / ensembles | `dt`, `rf`, `et`, `ada`, `gbr`, `hgb` |
| Neural net | `mlp`\* |
| Optional libraries | `xgboost`, `lightgbm`, `catboost` |

\* skipped when `turbo=True` (the default), because they're slow.

### Charts (v2)

| Method | Shows |
|---|---|
| `plot_leaderboard(metric=None, top=None)` | Ranked bars with ±1 SD error bars, dummy baseline in grey |
| `plot_tuning()` | Starting vs tuned CV score, plus the holdout score |
| `plot_prediction_error(model_id=None)` | Predicted vs actual on the holdout set |
| `plot_residuals(model_id=None)` | Residuals vs predicted, and a residual histogram |
| `plot_feature_importance(model_id=None, top=15)` | Permutation importance per **original** column |

---

## ClassificationPicker

**For a category target.** The target can be text (`"Yes"`/`"No"`), bool or integer labels.

```python
from pickers import ClassificationPicker

cp = ClassificationPicker(df, target="Churn", positive_class="Yes", sort="F1")
cp.compare_models()
cp.tune_top(n=3)
print(cp.summary())

cp.confusion_matrix("hgb")              # holdout confusion matrix (DataFrame)
print(cp.class_report("hgb"))           # precision / recall / F1 per class

model = cp.finalize()
model.predict(new_df)                   # returns the original labels, e.g. "Yes"/"No"
model.predict_proba(new_df)             # columns in the order of model.classes_
```

### Key options

| Option | Default | Meaning |
|---|---|---|
| `target` | required | Name of the category column |
| `positive_class` | last label when sorted | **Binary only:** the class Recall/Precision/F1/AUC are measured on (e.g. `"Yes"`) |
| `sort` | `"Accuracy"` | `"Accuracy"`, `"AUC"`, `"Recall"`, `"Precision"`, `"F1"`, `"Kappa"`, `"MCC"` |
| `balance` | `"auto"` | `class_weight="balanced"` as the starting value. `"auto"` = on when classes are uneven; or `True` / `False` |
| `average` | `"weighted"` | Multiclass averaging: `"weighted"` (PyCaret-style) or `"macro"` (every class counts equally) |
| `test_size`, `cv` | `0.2`, `5` | Splits and folds are **stratified** (keep class proportions) |

### Metrics (all: higher is better)

| Metric | Use when |
|---|---|
| Accuracy | Classes are roughly balanced |
| AUC | You care about ranking/probabilities; threshold-free |
| Recall | Missing a positive is costly (fraud, disease, churn) |
| Precision | A false alarm is costly |
| F1 | You want a balance of precision and recall, especially with uneven classes |
| Kappa / MCC | Single-number summaries that account for chance; good with uneven classes |

> **Accuracy trap:** with 27% churners, a model that always says "No" is 73% accurate but has F1 = 0. With uneven classes, sort by F1, AUC or MCC.

### Models (ids)

| Family | Model ids |
|---|---|
| Baseline | `dummy` |
| Linear | `lr`, `ridge`, `svm` (linear), `sgd` |
| Probabilistic | `nb`, `lda`, `qda` |
| Kernel / distance | `knn`, `rbfsvm`, `gpc`\* |
| Trees / ensembles | `dt`, `rf`, `et`, `ada`, `gbc`, `hgb` |
| Neural net | `mlp`\* |
| Optional libraries | `xgboost`, `lightgbm`, `catboost` |

\* skipped when `turbo=True`. With `sort="AUC"`, `ridge` and `svm` are skipped too, because they give no probability scores.

### Charts (v2)

| Method | Shows |
|---|---|
| `plot_leaderboard(metric=None, top=None)` | Ranked bars with ±1 SD error bars |
| `plot_tuning()` | Starting vs tuned CV score, plus the holdout score |
| `plot_confusion_matrix(model_id=None, normalize=False)` | Heatmap; `normalize=True` shows row % |
| `plot_roc(models=None)` | Binary: one curve per model (the tuned ones by default). Multiclass: one curve per class for one model |
| `plot_precision_recall(models=None)` | Same rules as ROC; more honest when the positive class is rare |
| `plot_threshold(model_id=None)` | **Binary only.** Precision, recall and F1 at every threshold; marks the best-F1 threshold |
| `plot_feature_importance(model_id=None, top=15)` | Permutation importance per **original** column |

---

## ClusterPicker

**For data with no target: finding groups.**

```python
from pickers import ClusterPicker

cl = ClusterPicker(df, ignore=["CustomerID"])   # leave ID/name columns out of the clustering
cl.k_scan()                                     # how many clusters? elbow / silhouette / BIC table
cl.compare_models()
cl.tune_top(n=3)
print(cl.summary())

cl.profile()                                    # size + average of each column per cluster
segments = cl.assign_clusters()                 # your df + a 'Cluster' column
```

### Key options

| Option | Default | Meaning |
|---|---|---|
| `n_clusters` | `None` | Starting number of clusters. `None` = pick by K-Means silhouette scan; set it if you know the answer |
| `max_clusters` | `10` | Largest k scanned and tuned |
| `ignore` | `None` | Columns to leave out of the clustering (IDs, names, dates). They still appear in `assign_clusters()` |
| `ground_truth` | `None` | **Teaching only:** a known-label column. It's *not* used to cluster, only to score the result (ARI, NMI, Homogeneity) |
| `sort` | `"Silhouette"` | `"Silhouette"`, `"Calinski-Harabasz"`, `"Davies-Bouldin"`, `"Mean rank"`; plus `"ARI"`, `"NMI"`, `"Homogeneity"` with `ground_truth` |
| `max_noise` | `0.25` | Most rows a density model may label as noise before it's flagged/rejected |

### Metrics

| Metric | Better | Meaning |
|---|---|---|
| Silhouette | higher (−1 to 1) | How well each point fits its own cluster vs the next one. Noise points count as 0 |
| Calinski-Harabasz | higher | Spread between clusters vs within them |
| Davies-Bouldin | lower | How similar each cluster is to its nearest neighbour |
| Mean rank | lower | Average rank across the three above |
| ARI / NMI / Homogeneity | higher (up to 1) | Agreement with `ground_truth` |

> **Caution:** the three shape metrics all favour compact, round clusters, and they often prefer **fewer** clusters than really exist. Check `k_scan()`, look at `profile()`, and use domain knowledge. Don't trust the top row blindly.

### Models (ids)

| Family | Model ids | Chooses its own k? |
|---|---|---|
| Centroid | `kmeans`, `mbkmeans`, `bkmeans` | no |
| Distribution | `gmm` | no |
| Hierarchical | `hclust`, `birch` | no |
| Graph | `spectral`, `ap` | `ap` yes |
| Density | `dbscan`, `hdbscan`, `optics`\*, `meanshift` | yes |

\* skipped when `turbo=True`. Slow models are skipped automatically on large data.

**Labelling new rows:** `get_model(id).predict(new_df)` works for `kmeans`, `mbkmeans`, `bkmeans`, `gmm`, `birch`, `ap`, `meanshift`. The others (`hclust`, `spectral`, `dbscan`, `hdbscan`, `optics`) can only label the data they were fitted on; use `labels(id)` or `assign_clusters(id)`.

### Charts (v2)

| Method | Shows |
|---|---|
| `plot_k_scan()` | Five panels (inertia elbow, silhouette, CH, DB, GMM BIC) with the best k marked, plus a summary |
| `plot_leaderboard(metric=None)` | Ranked bars; flagged models (too much noise, < 2 clusters) in grey |
| `plot_tuning()` | Starting vs tuned score, with the number of clusters in each label |
| `plot_clusters(model_id=None)` | Scatter coloured and shaped by cluster; original axes if there are 2 numeric columns, otherwise a 2-D PCA view |
| `plot_silhouette(model_id=None)` | Silhouette value of every point, grouped by cluster |
| `plot_profile(model_id=None)` | Heatmap: colour = above/below average, numbers = actual averages |
| `plot_crosstab(model_id=None)` | **Needs `ground_truth`.** Clusters vs true labels |

---

## Common options

These work on all three pickers:

| Option | Default | Meaning |
|---|---|---|
| `sort` | see each picker | Metric used to rank **and** to tune |
| `turbo` | `True` | Skip slow models (like PyCaret's turbo mode) |
| `include` | `None` | Only run these model ids, e.g. `include=["rf", "hgb", "lr"]` |
| `exclude` | `None` | Leave these model ids out, e.g. `exclude=["svr"]` |
| `random_state` | `42` | Seed for reproducible results |
| `n_jobs` | `-1` | CPU cores for cross-validation (`-1` = all). Not on `ClusterPicker` |
| `quiet` | `True` | Hide convergence warnings |

And these methods:

```python
picker.tune_top(n=3, n_iter=25)                 # tune the top 3 with 25 random tries each
picker.tune_top(models=["rf", "lr"])            # tune specific models instead
picker.leaderboard_.sort_values("MAE")          # re-view by another metric (no re-run needed)
picker.skipped_                                 # {model_id: reason}
```

**All charts** take an optional `ax=` and return the matplotlib `Axes` (or `Figure` for multi-panel charts):

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cp.plot_roc(ax=axes[0])
cp.plot_confusion_matrix(ax=axes[1])

ax = rp.plot_leaderboard(top=10)
ax.set_title("Top 10 models")
ax.figure.savefig("leaderboard.png", dpi=150, bbox_inches="tight")
```

> In Jupyter, end a plot line with `;` (e.g. `rp.plot_tuning();`) to hide the `<Axes: ...>` text under the chart.

---

## Tips and gotchas

- **Clean first.** Missing values raise an error that names the columns. Also check **dtypes**: a number column read as text (e.g. blanks stored as `" "`) will be one-hot encoded as categories. Fix it with `pd.to_numeric(df[col], errors="coerce")`.
- **Drop ID columns** before regression/classification (e.g. `df.drop(columns="customerID")`). In `ClusterPicker`, use `ignore=`.
- **Wrong module?** A text target in `RegressionPicker`, or a float target with many values in `ClassificationPicker`, raises an error telling you which module to use.
- **Speed.** Start with a sample (`df.sample(5000, random_state=42)`) and `n_iter=10`. The defaults (5-fold CV, `n_iter=25`) on 20,000 rows take a few minutes.
- **Leaderboard vs holdout.** The leaderboard is cross-validation on the training split. `tune_top()` also scores the untouched holdout set, which is your best estimate of real-world performance.
- **Tuning gains are often small.** Good starting values mean tuning often adds only +0.001 to 0.01. That's a useful lesson: data and features usually matter more than hyperparameters.
- **Results vary slightly by version.** Different scikit-learn / XGBoost / LightGBM versions, and whether the optional libraries are installed, can change the exact numbers and the winner.
- **Using the recommended hyperparameters elsewhere:**

  ```python
  from sklearn.ensemble import RandomForestRegressor
  params = rp.recommended_params_["rf"]
  model = RandomForestRegressor(**params, random_state=42)
  ```

  Remember that the picker also scaled and one-hot encoded the data. `get_model()` / `finalize()` give you the full pipeline with that preprocessing built in.

---

## v1 vs v2

| | v1 (`*_picker.py`) | v2 (`*_picker_v2.py`) |
|---|---|---|
| Model comparison, tuning, recommendations | ✓ | ✓ (identical) |
| `plot_` chart methods | n/a | ✓ |
| Needs matplotlib | no | only when a chart is drawn |

v2 is v1 plus the charts, so any code written for v1 works unchanged with v2.

---

## Test notebooks

Each module has a test notebook in `tests/` that runs it on a well-known public dataset and ends with automatic **PASS/FAIL** checks:

| Notebook | Datasets |
|---|---|
| `test_regression_picker_v2.ipynb` | California Housing |
| `test_classification_picker_v2.ipynb` | IBM Telco Customer Churn (binary), Palmer Penguins (3 classes) |
| `test_cluster_picker_v2.ipynb` | Mall Customers, Palmer Penguins |

Run all cells. Each section should end with **All checks passed**.
