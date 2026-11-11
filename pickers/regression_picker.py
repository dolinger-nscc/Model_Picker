"""
regression_picker.py
====================

PyCaret-style model selection for REGRESSION (numeric target) using scikit-learn,
with optional XGBoost / LightGBM / CatBoost if they are installed.

The module does two jobs:
  1. compare_models()  -> cross-validates every candidate model and ranks them
  2. tune_top()        -> tunes the top N models and recommends hyperparameters

Hyperparameters are handled in two layers:
  * Starting values are *data-aware*, not raw library defaults. They are computed
    from the training data (rows, columns, target spread). For example:
    Lasso's alpha is set from the data's own alpha_max, KNN's k scales with the
    row count, and SVR/MLP/SGD train on a standardized target so C and epsilon
    mean the same thing on any dataset.
  * Tuning uses RandomizedSearchCV over a curated search space centred on those
    starting values. The tuned model is kept only if it actually beats the
    starting one in cross-validation.

Input: a CLEAN pandas DataFrame (no missing values). Numeric columns are scaled
where a model needs it; object/category/bool columns are one-hot encoded.
Everything happens inside a Pipeline, so there is no leakage across CV folds.

Example
-------
>>> from regression_picker import RegressionPicker
>>> rp = RegressionPicker(df, target="price")
>>> rp.compare_models()            # leaderboard (DataFrame)
>>> rp.tune_top(n=3)               # tuned results for the top 3
>>> rp.recommended_params_         # {model_id: {param: value}}
>>> model = rp.finalize("hgb")     # refit best version on ALL the data
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet,
                                  GammaRegressor, HuberRegressor, Lars, Lasso,
                                  LassoLars, LinearRegression,
                                  OrthogonalMatchingPursuit, PoissonRegressor,
                                  QuantileRegressor, RANSACRegressor, Ridge,
                                  SGDRegressor, TheilSenRegressor,
                                  TweedieRegressor)
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Optional third-party boosters -------------------------------------------------
try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


# ------------------------------------------------------------------------------
# Data context: facts about the training data that drive the starting values
# ------------------------------------------------------------------------------
@dataclass
class DataContext:
    n: int                 # training rows
    p: int                 # features AFTER one-hot encoding
    y_min: float
    y_std: float
    alpha_max: float       # smallest Lasso alpha that zeroes every coefficient
    seed: int


# ------------------------------------------------------------------------------
# Model specification
# ------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    id: str
    name: str
    family: str
    build: Callable[[DataContext], object]          # returns estimator w/ start params
    space: Callable[[DataContext], dict] = lambda c: {}
    scale_X: bool = True        # standardize numeric features?
    scale_y: bool = False       # train on a standardized target?
    requires: Callable[[DataContext], str | None] = lambda c: None  # reason to skip
    turbo_skip: bool = False    # skipped when turbo=True (slow models)
    tags: list = field(default_factory=list)


def _limit(max_n=None, max_p=None, y_pos=False, y_nonneg=False):
    """Build a 'requires' check. Returns a skip reason string or None."""
    def check(c: DataContext):
        if max_n and c.n > max_n:
            return f"too slow for {c.n:,} rows (limit {max_n:,})"
        if max_p and c.p > max_p:
            return f"too slow for {c.p} features (limit {max_p})"
        if y_pos and c.y_min <= 0:
            return "target must be strictly positive"
        if y_nonneg and c.y_min < 0:
            return "target must be non-negative"
        return None
    return check


def _knn_k(c):  # k grows with data size, kept in a sensible band
    return int(np.clip(round(np.sqrt(c.n) / 2), 5, 25))


def _min_leaf(c):  # leaves of ~1% of rows stop a single tree memorising noise
    return int(np.clip(round(c.n * 0.01), 5, 50))


def _registry() -> list[ModelSpec]:
    S = []

    # ---- Baseline -----------------------------------------------------------
    S.append(ModelSpec("dummy", "Dummy (mean)", "baseline",
                       lambda c: DummyRegressor(strategy="mean"), scale_X=False))

    # ---- Linear -------------------------------------------------------------
    S.append(ModelSpec("lr", "Linear Regression", "linear",
                       lambda c: LinearRegression()))
    S.append(ModelSpec("ridge", "Ridge", "linear",
                       lambda c: Ridge(alpha=1.0),
                       lambda c: {"alpha": loguniform(1e-3, 1e3)}))
    S.append(ModelSpec("lasso", "Lasso", "linear",
                       lambda c: Lasso(alpha=c.alpha_max * 0.01, max_iter=5000),
                       lambda c: {"alpha": loguniform(c.alpha_max * 1e-4, c.alpha_max * 0.5)}))
    S.append(ModelSpec("en", "Elastic Net", "linear",
                       lambda c: ElasticNet(alpha=c.alpha_max * 0.02, l1_ratio=0.5, max_iter=5000),
                       lambda c: {"alpha": loguniform(c.alpha_max * 1e-4, c.alpha_max),
                                  "l1_ratio": uniform(0.05, 0.9)}))
    S.append(ModelSpec("lar", "Least Angle Regression", "linear",
                       lambda c: Lars(n_nonzero_coefs=c.p),
                       lambda c: {"n_nonzero_coefs": randint(1, c.p + 1)}))
    S.append(ModelSpec("llar", "Lasso Least Angle", "linear",
                       lambda c: LassoLars(alpha=c.alpha_max * 0.01),
                       lambda c: {"alpha": loguniform(c.alpha_max * 1e-4, c.alpha_max * 0.5)}))
    S.append(ModelSpec("omp", "Orthogonal Matching Pursuit", "linear",
                       lambda c: OrthogonalMatchingPursuit(n_nonzero_coefs=max(1, c.p // 2)),
                       lambda c: {"n_nonzero_coefs": randint(1, c.p + 1)}))
    S.append(ModelSpec("br", "Bayesian Ridge", "linear",
                       lambda c: BayesianRidge(),
                       lambda c: {"alpha_1": loguniform(1e-7, 1e-3), "alpha_2": loguniform(1e-7, 1e-3),
                                  "lambda_1": loguniform(1e-7, 1e-3), "lambda_2": loguniform(1e-7, 1e-3)}))
    S.append(ModelSpec("ard", "ARD Regression", "linear",
                       lambda c: ARDRegression(),
                       lambda c: {"alpha_1": loguniform(1e-7, 1e-3), "lambda_1": loguniform(1e-7, 1e-3),
                                  "threshold_lambda": loguniform(1e3, 1e5)},
                       requires=_limit(max_p=300)))
    S.append(ModelSpec("sgd", "SGD Regressor", "linear",
                       lambda c: SGDRegressor(penalty="elasticnet", alpha=1e-4, l1_ratio=0.15,
                                              max_iter=3000, early_stopping=True, random_state=c.seed),
                       lambda c: {"alpha": loguniform(1e-6, 1e-1), "l1_ratio": uniform(0, 1),
                                  "loss": ["squared_error", "huber"]},
                       scale_y=True))

    # ---- Robust linear (resist outliers) ------------------------------------
    S.append(ModelSpec("huber", "Huber Regressor", "robust",
                       lambda c: HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000),
                       lambda c: {"epsilon": uniform(1.1, 1.0), "alpha": loguniform(1e-6, 1e-1)}))
    S.append(ModelSpec("ransac", "RANSAC", "robust",
                       lambda c: RANSACRegressor(random_state=c.seed),
                       lambda c: {"min_samples": uniform(0.3, 0.6)}))
    S.append(ModelSpec("tr", "Theil-Sen", "robust",
                       lambda c: TheilSenRegressor(max_subpopulation=2000, random_state=c.seed),
                       requires=_limit(max_n=10_000, max_p=40), turbo_skip=True))
    S.append(ModelSpec("quantile", "Quantile (median)", "robust",
                       lambda c: QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
                       lambda c: {"alpha": loguniform(1e-5, 1e-1)},
                       requires=_limit(max_n=5_000)))

    # ---- GLMs (only offered when the target allows) -------------------------
    S.append(ModelSpec("poisson", "Poisson GLM", "glm",
                       lambda c: PoissonRegressor(alpha=1e-3, max_iter=1000),
                       lambda c: {"alpha": loguniform(1e-6, 1)},
                       requires=_limit(y_nonneg=True)))
    S.append(ModelSpec("gamma", "Gamma GLM", "glm",
                       lambda c: GammaRegressor(alpha=1e-3, max_iter=1000),
                       lambda c: {"alpha": loguniform(1e-6, 1)},
                       requires=_limit(y_pos=True)))
    S.append(ModelSpec("tweedie", "Tweedie GLM", "glm",
                       lambda c: TweedieRegressor(power=1.5, link="log", alpha=1e-3, max_iter=1000),
                       lambda c: {"alpha": loguniform(1e-6, 1), "power": uniform(1.1, 0.8)},
                       requires=_limit(y_nonneg=True)))

    # ---- Distance / kernel --------------------------------------------------
    S.append(ModelSpec("knn", "K Neighbors", "kernel",
                       lambda c: KNeighborsRegressor(n_neighbors=_knn_k(c), weights="distance"),
                       lambda c: {"n_neighbors": randint(2, 51), "weights": ["uniform", "distance"],
                                  "p": [1, 2]}))
    # On a standardized target, C ~ 3 follows the Cherkassky & Ma (2004) rule
    # C = max(|mean +/- 3 sd|); gamma centred on 1/p (= sklearn's 'scale').
    S.append(ModelSpec("svr", "Support Vector (RBF)", "kernel",
                       lambda c: SVR(kernel="rbf", C=3.0, epsilon=0.1, gamma="scale"),
                       lambda c: {"C": loguniform(0.1, 100), "epsilon": loguniform(0.01, 0.5),
                                  "gamma": loguniform(0.1 / c.p, 10 / c.p)},
                       scale_y=True, requires=_limit(max_n=10_000)))
    S.append(ModelSpec("kr", "Kernel Ridge (RBF)", "kernel",
                       lambda c: KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0 / c.p),
                       lambda c: {"alpha": loguniform(1e-3, 10), "gamma": loguniform(0.1 / c.p, 10 / c.p)},
                       scale_y=True, requires=_limit(max_n=5_000), turbo_skip=True))

    # ---- Trees & ensembles --------------------------------------------------
    S.append(ModelSpec("dt", "Decision Tree", "tree",
                       lambda c: DecisionTreeRegressor(min_samples_leaf=_min_leaf(c), random_state=c.seed),
                       lambda c: {"max_depth": [None, 3, 4, 5, 6, 8, 10, 15, 20],
                                  "min_samples_leaf": randint(1, 51),
                                  "max_features": [1.0, 0.7, 0.5, "sqrt"]},
                       scale_X=False))
    S.append(ModelSpec("rf", "Random Forest", "ensemble",
                       lambda c: RandomForestRegressor(n_estimators=300, random_state=c.seed, n_jobs=1),
                       lambda c: {"max_features": [1.0, 0.7, 0.5, 0.33, "sqrt"],
                                  "min_samples_leaf": randint(1, 21),
                                  "max_depth": [None, 8, 12, 16, 24],
                                  "max_samples": [None, 0.5, 0.7, 0.9]},
                       scale_X=False))
    S.append(ModelSpec("et", "Extra Trees", "ensemble",
                       lambda c: ExtraTreesRegressor(n_estimators=300, random_state=c.seed, n_jobs=1),
                       lambda c: {"max_features": [1.0, 0.7, 0.5, 0.33, "sqrt"],
                                  "min_samples_leaf": randint(1, 21),
                                  "max_depth": [None, 8, 12, 16, 24]},
                       scale_X=False))
    S.append(ModelSpec("ada", "AdaBoost", "ensemble",
                       lambda c: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),
                                                   n_estimators=200, learning_rate=0.5,
                                                   random_state=c.seed),
                       lambda c: {"n_estimators": randint(50, 501), "learning_rate": loguniform(0.01, 1),
                                  "loss": ["linear", "square", "exponential"],
                                  "estimator__max_depth": randint(2, 9)},
                       scale_X=False))
    S.append(ModelSpec("gbr", "Gradient Boosting", "ensemble",
                       lambda c: GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                                           max_depth=3, subsample=0.8,
                                                           n_iter_no_change=20, random_state=c.seed),
                       lambda c: {"learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 7),
                                  "subsample": uniform(0.5, 0.5), "min_samples_leaf": randint(1, 31),
                                  "max_features": [1.0, 0.7, 0.5, "sqrt"]},
                       scale_X=False))
    S.append(ModelSpec("hgb", "Hist Gradient Boosting", "ensemble",
                       lambda c: HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05,
                                                               min_samples_leaf=min(20, max(5, c.n // 100)),
                                                               early_stopping=True, n_iter_no_change=20,
                                                               random_state=c.seed),
                       lambda c: {"learning_rate": loguniform(0.01, 0.3),
                                  "max_leaf_nodes": randint(8, 128),
                                  "min_samples_leaf": randint(5, 101),
                                  "l2_regularization": loguniform(1e-4, 10),
                                  "max_features": uniform(0.5, 0.5)},
                       scale_X=False))

    # ---- Neural net ---------------------------------------------------------
    S.append(ModelSpec("mlp", "MLP Neural Net", "neural",
                       lambda c: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                              learning_rate_init=1e-3, max_iter=2000,
                                              early_stopping=True, n_iter_no_change=20,
                                              random_state=c.seed),
                       lambda c: {"hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
                                  "alpha": loguniform(1e-5, 1e-1),
                                  "learning_rate_init": loguniform(1e-4, 1e-2)},
                       scale_y=True, turbo_skip=True))

    # ---- Optional third-party boosters --------------------------------------
    if XGBRegressor is not None:
        S.append(ModelSpec("xgboost", "XGBoost", "ensemble",
                           lambda c: XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  tree_method="hist", n_jobs=1,
                                                  random_state=c.seed, verbosity=0),
                           lambda c: {"learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 11),
                                      "min_child_weight": loguniform(1, 20), "subsample": uniform(0.5, 0.5),
                                      "colsample_bytree": uniform(0.5, 0.5),
                                      "reg_lambda": loguniform(1e-3, 10), "reg_alpha": loguniform(1e-4, 1)},
                           scale_X=False, tags=["optional"]))
    if LGBMRegressor is not None:
        S.append(ModelSpec("lightgbm", "LightGBM", "ensemble",
                           lambda c: LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                                   min_child_samples=int(np.clip(c.n // 100, 5, 20)),
                                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                   n_jobs=1, random_state=c.seed, verbose=-1),
                           lambda c: {"learning_rate": loguniform(0.01, 0.3), "num_leaves": randint(8, 128),
                                      "min_child_samples": randint(5, 101), "subsample": uniform(0.5, 0.5),
                                      "colsample_bytree": uniform(0.5, 0.5),
                                      "reg_lambda": loguniform(1e-3, 10), "reg_alpha": loguniform(1e-4, 1)},
                           scale_X=False, tags=["optional"]))
    if CatBoostRegressor is not None:
        S.append(ModelSpec("catboost", "CatBoost", "ensemble",
                           lambda c: CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                                       random_seed=c.seed, verbose=0, thread_count=1,
                                                       allow_writing_files=False),
                           lambda c: {"learning_rate": loguniform(0.01, 0.3), "depth": randint(4, 11),
                                      "l2_leaf_reg": loguniform(1, 10)},
                           scale_X=False, tags=["optional"]))
    return S


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
_SCORERS = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
}
_HIGHER_IS_BETTER = {"MAE": False, "RMSE": False, "R2": True, "MAPE": False}


def _holdout_scores(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred)}


# ------------------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------------------
class RegressionPicker:
    """Rank regression models on a clean DataFrame and recommend hyperparameters.

    Parameters
    ----------
    df : pandas.DataFrame   Clean data (no missing values), features + target.
    target : str            Name of the numeric target column.
    test_size : float       Fraction held out for a final, untouched check.
    cv : int                Number of cross-validation folds on the training split.
    sort : str              Metric to rank by: 'R2', 'RMSE', 'MAE' or 'MAPE'.
    turbo : bool            Skip slow models (Theil-Sen, Kernel Ridge, MLP), like PyCaret.
    include / exclude : list of model ids to force in / leave out.
    random_state : int      Seed for splits and models (reproducible results).
    n_jobs : int            Parallel CV folds (-1 = all cores).
    quiet : bool            Hide convergence warnings.
    """

    def __init__(self, df: pd.DataFrame, target: str, test_size: float = 0.2, cv: int = 5,
                 sort: str = "R2", turbo: bool = True, include=None, exclude=None,
                 random_state: int = 42, n_jobs: int = -1, quiet: bool = True):
        self._validate(df, target, sort)
        self.target, self.sort, self.turbo = target, sort, turbo
        self.include, self.exclude = set(include or []), set(exclude or [])
        self.seed, self.n_jobs, self.quiet = random_state, n_jobs, quiet

        X, y = df.drop(columns=[target]), df[target].astype(float)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        self.cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        self.cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        self.ctx = self._build_context()
        self.specs = {s.id: s for s in _registry()}
        self.leaderboard_ = None
        self.tuned_ = None
        self.recommended_params_ = {}
        self.skipped_ = {}
        self._best_pipes = {}

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _validate(df, target, sort):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if target not in df.columns:
            raise ValueError(f"target '{target}' not found in df columns")
        if not pd.api.types.is_numeric_dtype(df[target]):
            raise ValueError(f"target '{target}' is not numeric - use the classification module instead")
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"df has missing values in {bad}; clean it first")
        if df[target].nunique() <= 10:
            warnings.warn(f"target '{target}' has only {df[target].nunique()} unique values - "
                          "is this really a regression problem?")
        if sort not in _SCORERS:
            raise ValueError(f"sort must be one of {list(_SCORERS)}")

    def _preprocessor(self, scale_X: bool):
        num = StandardScaler() if scale_X else "passthrough"
        return ColumnTransformer(
            [("num", num, self.num_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)],
            verbose_feature_names_out=False)

    def _build_context(self) -> DataContext:
        Xs = self._preprocessor(True).fit_transform(self.X_train)
        yc = self.y_train.to_numpy() - self.y_train.mean()
        alpha_max = float(np.max(np.abs(Xs.T @ yc)) / len(yc)) or 1.0
        return DataContext(n=len(self.X_train), p=Xs.shape[1], y_min=float(self.y_train.min()),
                           y_std=float(self.y_train.std()), alpha_max=alpha_max, seed=self.seed)

    def _pipeline(self, spec: ModelSpec, estimator=None):
        est = estimator if estimator is not None else spec.build(self.ctx)
        if spec.scale_y:
            est = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
        return Pipeline([("prep", self._preprocessor(spec.scale_X)), ("model", est)])

    def _prefix(self, spec):
        return "model__regressor__" if spec.scale_y else "model__"

    def _candidates(self):
        out = []
        for mid, spec in self.specs.items():
            if self.include and mid not in self.include:
                continue
            if mid in self.exclude:
                continue
            if self.turbo and spec.turbo_skip and mid not in self.include:
                self.skipped_[mid] = "turbo mode (set turbo=False to include)"
                continue
            reason = spec.requires(self.ctx)
            if reason:
                self.skipped_[mid] = reason
                continue
            out.append(spec)
        return out

    def _quiet(self):
        if self.quiet:
            warnings.filterwarnings("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"   # silences parallel worker processes too

    # ------------------------------------------------------------- public API
    def models(self) -> pd.DataFrame:
        """List every available model, its family and whether it will run on this data."""
        rows = []
        for mid, s in self.specs.items():
            reason = s.requires(self.ctx) or ("turbo mode" if self.turbo and s.turbo_skip else "")
            rows.append({"ID": mid, "Model": s.name, "Family": s.family,
                         "Optional lib": "optional" in s.tags, "Skip reason": reason})
        return pd.DataFrame(rows).set_index("ID")

    def starting_params(self, model_id: str) -> dict:
        """The data-aware starting hyperparameters for one model (before tuning)."""
        spec = self.specs[model_id]
        est = spec.build(self.ctx)
        defaults = type(est)().get_params(deep=False)
        housekeeping = {"n_jobs", "random_state", "random_seed", "verbose", "verbosity",
                        "thread_count", "allow_writing_files"}   # not real hyperparameters
        return {k: v for k, v in est.get_params(deep=False).items()
                if k not in housekeeping and (k not in defaults or repr(defaults[k]) != repr(v))}

    def compare_models(self, verbose: bool = True) -> pd.DataFrame:
        """Cross-validate every candidate with its starting hyperparameters and rank them."""
        self._quiet()
        rows = []
        for spec in self._candidates():
            t0 = time.perf_counter()
            try:
                res = cross_validate(self._pipeline(spec), self.X_train, self.y_train, cv=self.cv,
                                     scoring=_SCORERS, n_jobs=self.n_jobs, error_score="raise")
            except Exception as e:  # keep going - one bad model shouldn't stop the run
                self.skipped_[spec.id] = f"failed: {type(e).__name__}: {str(e)[:80]}"
                continue
            row = {"ID": spec.id, "Model": spec.name}
            for m in _SCORERS:
                vals = res[f"test_{m}"] * (1 if _HIGHER_IS_BETTER[m] else -1)
                row[m] = vals.mean()
                if m == self.sort:
                    row[f"{m} SD"] = vals.std()
            row["Fit (s)"] = res["fit_time"].mean()
            rows.append(row)
            if verbose:
                print(f"  {spec.name:<28} {self.sort}={row[self.sort]:.4f}  "
                      f"({time.perf_counter() - t0:.1f}s)")
        lb = pd.DataFrame(rows).set_index("ID")
        lb = lb.sort_values(self.sort, ascending=not _HIGHER_IS_BETTER[self.sort])
        self.leaderboard_ = lb
        if verbose and self.skipped_:
            print("\nSkipped:", ", ".join(f"{k} ({v})" for k, v in self.skipped_.items()))
        return lb.round(4)

    def tune_top(self, n: int = 3, n_iter: int = 25, models=None, verbose: bool = True) -> pd.DataFrame:
        """Tune the top-n models (or the listed `models`) with RandomizedSearchCV.

        Keeps the tuned version only if it beats the starting version in CV,
        then scores both on the untouched holdout set.
        """
        if models is None:
            if self.leaderboard_ is None:
                self.compare_models(verbose=False)
            models = [m for m in self.leaderboard_.index if m != "dummy"][:n]
        self._quiet()
        scorer, sign = _SCORERS[self.sort], (1 if _HIGHER_IS_BETTER[self.sort] else -1)
        rows = []
        for mid in models:
            spec = self.specs[mid]
            base = self._pipeline(spec)
            base_cv = sign * cross_validate(base, self.X_train, self.y_train, cv=self.cv,
                                            scoring=scorer, n_jobs=self.n_jobs)["test_score"].mean()
            space = {self._prefix(spec) + k: v for k, v in spec.space(self.ctx).items()}
            t0 = time.perf_counter()
            if space:
                search = RandomizedSearchCV(base, space, n_iter=n_iter, scoring=scorer, cv=self.cv,
                                            n_jobs=self.n_jobs, random_state=self.seed,
                                            error_score=np.nan)
                search.fit(self.X_train, self.y_train)
                tuned_cv, tuned_pipe = sign * search.best_score_, search.best_estimator_
                tuned_params = {k.replace(self._prefix(spec), ""): v
                                for k, v in search.best_params_.items()}
            else:
                tuned_cv, tuned_pipe, tuned_params = base_cv, None, {}

            better = ((tuned_cv > base_cv) if _HIGHER_IS_BETTER[self.sort] else (tuned_cv < base_cv))
            if tuned_pipe is not None and better:
                final, kept = tuned_pipe, "tuned"
                params = {**self.starting_params(mid), **tuned_params}
            else:
                final, kept = clone(base).fit(self.X_train, self.y_train), "starting"
                params = self.starting_params(mid)
            params = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                      for k, v in params.items()}
            self._best_pipes[mid] = final
            self.recommended_params_[mid] = params
            hold = _holdout_scores(self.y_test, final.predict(self.X_test))
            rows.append({"ID": mid, "Model": spec.name,
                         f"CV {self.sort} (start)": base_cv, f"CV {self.sort} (tuned)": tuned_cv,
                         "Kept": kept, **{f"Holdout {k}": v for k, v in hold.items()},
                         "Tune (s)": time.perf_counter() - t0})
            if verbose:
                print(f"  {spec.name:<28} start={base_cv:.4f}  tuned={tuned_cv:.4f}  -> keep {kept}")
        self.tuned_ = pd.DataFrame(rows).set_index("ID").sort_values(
            f"Holdout {self.sort}", ascending=not _HIGHER_IS_BETTER[self.sort])
        return self.tuned_.round(4)

    def get_model(self, model_id: str):
        """Return the best fitted pipeline for a model (tuned if tuning helped)."""
        if model_id in self._best_pipes:
            return self._best_pipes[model_id]
        return clone(self._pipeline(self.specs[model_id])).fit(self.X_train, self.y_train)

    def finalize(self, model_id: str | None = None):
        """Refit the chosen model (default: best holdout result) on ALL rows for deployment."""
        if model_id is None:
            if self.tuned_ is None:
                raise RuntimeError("run tune_top() first or pass a model_id")
            model_id = self.tuned_.index[0]
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        return clone(self.get_model(model_id)).fit(X, y)

    def summary(self) -> str:
        """Plain-text recap: data facts, winner and its recommended hyperparameters."""
        c = self.ctx
        lines = [f"Rows (train): {c.n:,}   Features (encoded): {c.p}   Target sd: {c.y_std:.4g}",
                 f"Categorical columns one-hot encoded: {self.cat_cols or 'none'}"]
        if self.tuned_ is not None:
            best = self.tuned_.index[0]
            lines += [f"\nBest model: {self.specs[best].name} ({best})",
                      f"Holdout {self.sort}: {self.tuned_.loc[best, f'Holdout {self.sort}']:.4f}",
                      "Recommended hyperparameters:"]
            lines += [f"  {k} = {v}" for k, v in self.recommended_params_[best].items()]
        return "\n".join(lines)


if __name__ == "__main__":  # quick demo:  python regression_picker.py
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True).frame
    rp = RegressionPicker(data, target="target")
    print(rp.compare_models())
    print(rp.tune_top(n=3, n_iter=15))
    print(rp.summary())
