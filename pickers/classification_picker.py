"""
classification_picker.py
========================

PyCaret-style model selection for CLASSIFICATION (category target) using
scikit-learn, with optional XGBoost / LightGBM / CatBoost if they are installed.
It mirrors regression_picker.py, so students learn one interface for both.

The module does two jobs:
  1. compare_models()  -> cross-validates every candidate model and ranks them
  2. tune_top()        -> tunes the top N models and recommends hyperparameters

Hyperparameters are handled in two layers:
  * Starting values are *data-aware*, not raw library defaults. They come from
    the training data (rows, columns, number of classes, class balance). For
    example: KNN's k scales with the row count, LDA uses automatic Ledoit-Wolf
    shrinkage, QDA gets a little regularisation so one-hot columns don't break
    it, and imbalanced targets get class_weight='balanced' automatically.
  * Tuning uses RandomizedSearchCV over a curated search space centred on those
    starting values. The tuned model is kept only if it actually beats the
    starting one in cross-validation.

Input: a CLEAN pandas DataFrame (no missing values). The target can be text,
bool or integer labels. Numeric features are scaled where a model needs it;
object/category/bool features are one-hot encoded. Everything happens inside a
Pipeline, so there is no leakage across CV folds.

Metrics (like PyCaret): Accuracy, AUC, Recall, Precision, F1, Kappa, MCC.
Binary problems score the positive class; multiclass problems use a weighted
average across classes (change with average="macro").

Example
-------
>>> from classification_picker import ClassificationPicker
>>> cp = ClassificationPicker(df, target="churn", positive_class="Yes")
>>> cp.compare_models()            # leaderboard (DataFrame)
>>> cp.tune_top(n=3)               # tuned results for the top 3
>>> cp.recommended_params_         # {model_id: {param: value}}
>>> cp.confusion_matrix("rf")      # holdout confusion matrix
>>> model = cp.finalize()          # refit best model on ALL rows; predicts real labels
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
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  SGDClassifier)
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, get_scorer,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Optional third-party boosters -------------------------------------------------
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


# ------------------------------------------------------------------------------
# Data context: facts about the training data that drive the starting values
# ------------------------------------------------------------------------------
@dataclass
class DataContext:
    n: int                  # training rows
    p: int                  # features AFTER one-hot encoding
    k: int                  # number of classes
    min_class: int          # rows in the smallest class (training split)
    imbalance: float        # smallest class count / largest class count (1.0 = balanced)
    pos_weight: float       # negatives / positives (binary only, for XGBoost)
    balanced: bool          # use class_weight='balanced'?
    seed: int

    @property
    def binary(self):
        return self.k == 2

    @property
    def cw(self):
        return "balanced" if self.balanced else None


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
    scale_X: bool = True
    requires: Callable[[DataContext], str | None] = lambda c: None  # reason to skip
    turbo_skip: bool = False
    tags: list = field(default_factory=list)


def _limit(max_n=None, max_p=None):
    def check(c: DataContext):
        if max_n and c.n > max_n:
            return f"too slow for {c.n:,} rows (limit {max_n:,})"
        if max_p and c.p > max_p:
            return f"too slow for {c.p} features (limit {max_p})"
        return None
    return check


def _qda_ok(c: DataContext):
    if c.min_class <= c.p:
        return f"smallest class ({c.min_class} rows) must exceed feature count ({c.p})"
    return None


def _knn_k(c):  # odd k (avoids ties), grows with data size, kept in a sensible band
    k = int(np.clip(round(np.sqrt(c.n) / 2), 5, 25))
    return k if k % 2 else k + 1


def _min_leaf(c):  # leaves of ~1% of rows stop a single tree memorising noise
    return int(np.clip(round(c.n * 0.01), 5, 50))


_CW = [None, "balanced"]   # class_weight choices searched during tuning


def _registry() -> list[ModelSpec]:
    S = []

    # ---- Baseline -----------------------------------------------------------
    S.append(ModelSpec("dummy", "Dummy (majority class)", "baseline",
                       lambda c: DummyClassifier(strategy="prior"), scale_X=False))

    # ---- Linear -------------------------------------------------------------
    S.append(ModelSpec("lr", "Logistic Regression", "linear",
                       lambda c: LogisticRegression(C=1.0, max_iter=3000, class_weight=c.cw),
                       lambda c: {"C": loguniform(1e-3, 1e2), "class_weight": _CW}))
    S.append(ModelSpec("ridge", "Ridge Classifier", "linear",
                       lambda c: RidgeClassifier(alpha=1.0, class_weight=c.cw),
                       lambda c: {"alpha": loguniform(1e-3, 1e3), "class_weight": _CW}))
    S.append(ModelSpec("svm", "Linear SVM", "linear",
                       lambda c: LinearSVC(C=1.0, max_iter=5000, class_weight=c.cw, random_state=c.seed),
                       lambda c: {"C": loguniform(1e-3, 1e2), "class_weight": _CW}))
    S.append(ModelSpec("sgd", "SGD Classifier", "linear",
                       lambda c: SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-4,
                                               l1_ratio=0.15, max_iter=3000, early_stopping=True,
                                               class_weight=c.cw, random_state=c.seed),
                       lambda c: {"alpha": loguniform(1e-6, 1e-1), "l1_ratio": uniform(0, 1),
                                  "loss": ["log_loss", "modified_huber"], "class_weight": _CW}))

    # ---- Probabilistic / discriminant ---------------------------------------
    S.append(ModelSpec("nb", "Naive Bayes (Gaussian)", "probabilistic",
                       lambda c: GaussianNB(),
                       lambda c: {"var_smoothing": loguniform(1e-11, 1e-1)}, scale_X=False))
    # Ledoit-Wolf shrinkage: picks its own regularisation from the data.
    S.append(ModelSpec("lda", "Linear Discriminant Analysis", "probabilistic",
                       lambda c: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                       lambda c: {"shrinkage": [None, "auto", 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]}))
    S.append(ModelSpec("qda", "Quadratic Discriminant Analysis", "probabilistic",
                       lambda c: QuadraticDiscriminantAnalysis(reg_param=0.1),
                       lambda c: {"reg_param": uniform(0, 1)}, requires=_qda_ok))

    # ---- Distance / kernel --------------------------------------------------
    S.append(ModelSpec("knn", "K Neighbors", "kernel",
                       lambda c: KNeighborsClassifier(n_neighbors=_knn_k(c), weights="distance"),
                       lambda c: {"n_neighbors": randint(2, 51), "weights": ["uniform", "distance"],
                                  "p": [1, 2]}))
    # Probabilities are only switched on for multiclass (needed for AUC there);
    # binary AUC uses the decision function, which is ~5x faster to fit.
    S.append(ModelSpec("rbfsvm", "SVM (RBF kernel)", "kernel",
                       lambda c: SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=c.cw,
                                     probability=not c.binary, random_state=c.seed),
                       lambda c: {"C": loguniform(0.1, 100), "gamma": loguniform(0.1 / c.p, 10 / c.p),
                                  "class_weight": _CW},
                       requires=_limit(max_n=10_000)))
    S.append(ModelSpec("gpc", "Gaussian Process", "kernel",
                       lambda c: GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=c.seed),
                       requires=_limit(max_n=2_000), turbo_skip=True))

    # ---- Trees & ensembles --------------------------------------------------
    S.append(ModelSpec("dt", "Decision Tree", "tree",
                       lambda c: DecisionTreeClassifier(min_samples_leaf=_min_leaf(c), class_weight=c.cw,
                                                        random_state=c.seed),
                       lambda c: {"max_depth": [None, 3, 4, 5, 6, 8, 10, 15, 20],
                                  "min_samples_leaf": randint(1, 51), "criterion": ["gini", "entropy"],
                                  "max_features": [1.0, 0.7, 0.5, "sqrt"], "class_weight": _CW},
                       scale_X=False))
    S.append(ModelSpec("rf", "Random Forest", "ensemble",
                       lambda c: RandomForestClassifier(n_estimators=300, class_weight=c.cw,
                                                        random_state=c.seed, n_jobs=1),
                       lambda c: {"max_features": ["sqrt", "log2", 0.33, 0.5, 0.7],
                                  "min_samples_leaf": randint(1, 21), "max_depth": [None, 8, 12, 16, 24],
                                  "max_samples": [None, 0.5, 0.7, 0.9],
                                  "class_weight": [None, "balanced", "balanced_subsample"]},
                       scale_X=False))
    S.append(ModelSpec("et", "Extra Trees", "ensemble",
                       lambda c: ExtraTreesClassifier(n_estimators=300, class_weight=c.cw,
                                                      random_state=c.seed, n_jobs=1),
                       lambda c: {"max_features": ["sqrt", "log2", 0.33, 0.5, 0.7],
                                  "min_samples_leaf": randint(1, 21), "max_depth": [None, 8, 12, 16, 24],
                                  "class_weight": _CW},
                       scale_X=False))
    S.append(ModelSpec("ada", "AdaBoost", "ensemble",
                       lambda c: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                                                    n_estimators=200, learning_rate=0.5,
                                                    random_state=c.seed),
                       lambda c: {"n_estimators": randint(50, 501), "learning_rate": loguniform(0.01, 1),
                                  "estimator__max_depth": randint(1, 6)},
                       scale_X=False))
    S.append(ModelSpec("gbc", "Gradient Boosting", "ensemble",
                       lambda c: GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05,
                                                            max_depth=3, subsample=0.8,
                                                            n_iter_no_change=20, random_state=c.seed),
                       lambda c: {"learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 7),
                                  "subsample": uniform(0.5, 0.5), "min_samples_leaf": randint(1, 31),
                                  "max_features": [1.0, 0.7, 0.5, "sqrt"]},
                       scale_X=False))
    S.append(ModelSpec("hgb", "Hist Gradient Boosting", "ensemble",
                       lambda c: HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05,
                                                                min_samples_leaf=min(20, max(5, c.n // 100)),
                                                                early_stopping=True, n_iter_no_change=20,
                                                                class_weight=c.cw, random_state=c.seed),
                       lambda c: {"learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": randint(8, 128),
                                  "min_samples_leaf": randint(5, 101), "l2_regularization": loguniform(1e-4, 10),
                                  "max_features": uniform(0.5, 0.5), "class_weight": _CW},
                       scale_X=False))

    # ---- Neural net ---------------------------------------------------------
    S.append(ModelSpec("mlp", "MLP Neural Net", "neural",
                       lambda c: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                               learning_rate_init=1e-3, max_iter=2000,
                                               early_stopping=True, n_iter_no_change=20,
                                               random_state=c.seed),
                       lambda c: {"hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
                                  "alpha": loguniform(1e-5, 1e-1),
                                  "learning_rate_init": loguniform(1e-4, 1e-2)},
                       turbo_skip=True))

    # ---- Optional third-party boosters --------------------------------------
    if XGBClassifier is not None:
        def _xgb(c):
            extra = {"scale_pos_weight": c.pos_weight} if (c.binary and c.balanced) else {}
            return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                                 random_state=c.seed, verbosity=0, **extra)
        S.append(ModelSpec("xgboost", "XGBoost", "ensemble", _xgb,
                           lambda c: {"learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 11),
                                      "min_child_weight": loguniform(1, 20), "subsample": uniform(0.5, 0.5),
                                      "colsample_bytree": uniform(0.5, 0.5),
                                      "reg_lambda": loguniform(1e-3, 10), "reg_alpha": loguniform(1e-4, 1)},
                           scale_X=False, tags=["optional"]))
    if LGBMClassifier is not None:
        S.append(ModelSpec("lightgbm", "LightGBM", "ensemble",
                           lambda c: LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                                    min_child_samples=int(np.clip(c.n // 100, 5, 20)),
                                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                    class_weight=c.cw, n_jobs=1, random_state=c.seed,
                                                    verbose=-1),
                           lambda c: {"learning_rate": loguniform(0.01, 0.3), "num_leaves": randint(8, 128),
                                      "min_child_samples": randint(5, 101), "subsample": uniform(0.5, 0.5),
                                      "colsample_bytree": uniform(0.5, 0.5),
                                      "reg_lambda": loguniform(1e-3, 10), "reg_alpha": loguniform(1e-4, 1),
                                      "class_weight": _CW},
                           scale_X=False, tags=["optional"]))
    if CatBoostClassifier is not None:
        def _cat(c):
            extra = {"auto_class_weights": "Balanced"} if c.balanced else {}
            return CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=c.seed,
                                      verbose=0, thread_count=1, allow_writing_files=False, **extra)
        S.append(ModelSpec("catboost", "CatBoost", "ensemble", _cat,
                           lambda c: {"learning_rate": loguniform(0.01, 0.3), "depth": randint(4, 11),
                                      "l2_leaf_reg": loguniform(1, 10)},
                           scale_X=False, tags=["optional"]))
    return S


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def _scorers(binary: bool, average: str, with_auc: bool) -> dict:
    avg = "binary" if binary else average
    s = {"Accuracy": get_scorer("accuracy")}
    if with_auc:
        s["AUC"] = get_scorer("roc_auc" if binary else f"roc_auc_ovr_{average}")
    s["Recall"] = make_scorer(recall_score, average=avg, zero_division=0)
    s["Precision"] = make_scorer(precision_score, average=avg, zero_division=0)
    s["F1"] = make_scorer(f1_score, average=avg, zero_division=0)
    s["Kappa"] = make_scorer(cohen_kappa_score)
    s["MCC"] = get_scorer("matthews_corrcoef")
    return s


_METRICS = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]   # all: higher is better


# ------------------------------------------------------------------------------
# Wrapper so finalized models predict the ORIGINAL labels (e.g. "Yes"/"No")
# ------------------------------------------------------------------------------
class LabeledModel:
    """A fitted pipeline that returns the original class labels.

    Internally every model trains on integer labels 0..k-1 (XGBoost requires it).
    This wrapper maps predictions back, so students see "Yes"/"No", not 1/0.
    The underlying scikit-learn Pipeline is available as `.pipeline`.
    """

    def __init__(self, pipeline, classes):
        self.pipeline, self.classes_ = pipeline, np.asarray(classes)

    def predict(self, X):
        return self.classes_[self.pipeline.predict(X)]

    def predict_proba(self, X):
        """Columns are in the order of `.classes_`."""
        return self.pipeline.predict_proba(X)

    def __repr__(self):
        return f"LabeledModel(classes={list(self.classes_)}, pipeline={self.pipeline!r})"


# ------------------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------------------
class ClassificationPicker:
    """Rank classification models on a clean DataFrame and recommend hyperparameters.

    Parameters
    ----------
    df : pandas.DataFrame   Clean data (no missing values), features + target.
    target : str            Name of the category target column.
    positive_class          Binary only: which label counts as "positive" for
                            Recall/Precision/F1/AUC (default: the last label when sorted).
    test_size : float       Fraction held out (stratified) for a final, untouched check.
    cv : int                Number of stratified cross-validation folds.
    sort : str              Metric to rank by: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC.
    balance : 'auto'|bool   Use class_weight='balanced' as the starting value.
                            'auto' turns it on when the smallest class is < 50% the size
                            of the largest.
    average : str           Multiclass averaging for Recall/Precision/F1/AUC ('weighted' or 'macro').
    turbo : bool            Skip slow models (Gaussian Process, MLP), like PyCaret.
    include / exclude : list of model ids to force in / leave out.
    random_state : int      Seed for splits and models (reproducible results).
    n_jobs : int            Parallel CV folds (-1 = all cores).
    quiet : bool            Hide convergence warnings.
    """

    def __init__(self, df: pd.DataFrame, target: str, positive_class=None, test_size: float = 0.2,
                 cv: int = 5, sort: str = "Accuracy", balance="auto", average: str = "weighted",
                 turbo: bool = True, include=None, exclude=None, random_state: int = 42,
                 n_jobs: int = -1, quiet: bool = True):
        self._validate(df, target, sort, average, cv)
        self.target, self.sort, self.turbo, self.average = target, sort, turbo, average
        self.include, self.exclude = set(include or []), set(exclude or [])
        self.seed, self.n_jobs, self.quiet = random_state, n_jobs, quiet

        # --- encode labels 0..k-1 (positive class last for binary problems) ----
        classes = sorted(df[target].unique(), key=lambda v: str(v))
        if positive_class is not None:
            if positive_class not in classes:
                raise ValueError(f"positive_class {positive_class!r} not in target labels {classes}")
            if len(classes) != 2:
                raise ValueError("positive_class only applies to binary targets")
            classes = [c for c in classes if c != positive_class] + [positive_class]
        self.classes_ = np.asarray(classes, dtype=object)
        mapping = {c: i for i, c in enumerate(classes)}

        X, y = df.drop(columns=[target]), df[target].map(mapping).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)
        self.cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        if self.y_train.value_counts().min() < cv:
            raise ValueError(f"smallest class has fewer than {cv} training rows; lower cv or merge rare classes")

        self.ctx = self._build_context(balance)
        self.specs = {s.id: s for s in _registry()}
        self.leaderboard_ = None
        self.tuned_ = None
        self.recommended_params_ = {}
        self.skipped_ = {}
        self._best_pipes = {}

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _validate(df, target, sort, average, cv):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if target not in df.columns:
            raise ValueError(f"target '{target}' not found in df columns")
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"df has missing values in {bad}; clean it first")
        k = df[target].nunique()
        if k < 2:
            raise ValueError("target needs at least 2 classes")
        if pd.api.types.is_float_dtype(df[target]) and k > 20:
            raise ValueError(f"target '{target}' is a float with {k} unique values - "
                             "use the regression module instead")
        if k > 20:
            warnings.warn(f"target has {k} classes - is this really a classification problem?")
        if sort not in _METRICS:
            raise ValueError(f"sort must be one of {_METRICS}")
        if average not in ("weighted", "macro"):
            raise ValueError("average must be 'weighted' or 'macro'")

    def _preprocessor(self, scale_X: bool):
        num = StandardScaler() if scale_X else "passthrough"
        return ColumnTransformer(
            [("num", num, self.num_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)],
            verbose_feature_names_out=False)

    def _build_context(self, balance) -> DataContext:
        p = self._preprocessor(True).fit_transform(self.X_train).shape[1]
        counts = self.y_train.value_counts()
        imbalance = float(counts.min() / counts.max())
        balanced = (imbalance < 0.5) if balance == "auto" else bool(balance)
        pos_weight = float(counts.get(0, 1) / counts.get(1, 1)) if len(counts) == 2 else 1.0
        return DataContext(n=len(self.X_train), p=p, k=len(counts), min_class=int(counts.min()),
                           imbalance=imbalance, pos_weight=pos_weight, balanced=balanced, seed=self.seed)

    def _pipeline(self, spec: ModelSpec, estimator=None):
        est = estimator if estimator is not None else spec.build(self.ctx)
        return Pipeline([("prep", self._preprocessor(spec.scale_X)), ("model", est)])

    def _has_auc(self, pipe) -> bool:
        return hasattr(pipe, "predict_proba") or (self.ctx.binary and hasattr(pipe, "decision_function"))

    def _scorers_for(self, pipe) -> dict:
        return _scorers(self.ctx.binary, self.average, self._has_auc(pipe))

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
            if self.sort == "AUC" and not self._has_auc(self._pipeline(spec)):
                self.skipped_[mid] = "no probability scores, so AUC can't be ranked"
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
        est = self.specs[model_id].build(self.ctx)
        defaults = type(est)().get_params(deep=False)
        housekeeping = {"n_jobs", "random_state", "random_seed", "verbose", "verbosity",
                        "thread_count", "allow_writing_files"}   # not real hyperparameters
        return {k: v for k, v in est.get_params(deep=False).items()
                if k not in housekeeping and (k not in defaults or repr(defaults[k]) != repr(v))}

    def compare_models(self, verbose: bool = True) -> pd.DataFrame:
        """Cross-validate every candidate with its starting hyperparameters and rank them."""
        self._quiet()
        if verbose:
            c = self.ctx
            print(f"{c.k} classes, smallest/largest = {c.imbalance:.2f} -> "
                  f"class_weight={'balanced' if c.balanced else 'None'} as starting value")
        rows = []
        for spec in self._candidates():
            pipe = self._pipeline(spec)
            t0 = time.perf_counter()
            try:
                res = cross_validate(pipe, self.X_train, self.y_train, cv=self.cv,
                                     scoring=self._scorers_for(pipe), n_jobs=self.n_jobs,
                                     error_score="raise")
            except Exception as e:  # keep going - one bad model shouldn't stop the run
                self.skipped_[spec.id] = f"failed: {type(e).__name__}: {str(e)[:80]}"
                continue
            row = {"ID": spec.id, "Model": spec.name}
            for m in _METRICS:
                vals = res.get(f"test_{m}")
                row[m] = np.nan if vals is None else vals.mean()
                if m == self.sort:
                    row[f"{m} SD"] = np.nan if vals is None else vals.std()
            row["Fit (s)"] = res["fit_time"].mean()
            rows.append(row)
            if verbose:
                print(f"  {spec.name:<32} {self.sort}={row[self.sort]:.4f}  "
                      f"({time.perf_counter() - t0:.1f}s)")
        lb = pd.DataFrame(rows).set_index("ID").sort_values(self.sort, ascending=False,
                                                            na_position="last")
        self.leaderboard_ = lb
        if verbose and self.skipped_:
            print("\nSkipped:", ", ".join(f"{k} ({v})" for k, v in self.skipped_.items()))
        return lb.round(4)

    def tune_top(self, n: int = 3, n_iter: int = 25, models=None, verbose: bool = True) -> pd.DataFrame:
        """Tune the top-n models (or the listed `models`) with RandomizedSearchCV.

        Keeps the tuned version only if it beats the starting version in CV,
        then scores the kept version on the untouched holdout set.
        """
        if models is None:
            if self.leaderboard_ is None:
                self.compare_models(verbose=False)
            lb = self.leaderboard_.dropna(subset=[self.sort])
            models = [m for m in lb.index if m != "dummy"][:n]
        self._quiet()
        rows = []
        for mid in models:
            spec = self.specs[mid]
            base = self._pipeline(spec)
            scorers = self._scorers_for(base)
            if self.sort not in scorers:
                print(f"  {spec.name}: skipped (no {self.sort})")
                continue
            scorer = scorers[self.sort]
            base_cv = cross_validate(base, self.X_train, self.y_train, cv=self.cv, scoring=scorer,
                                     n_jobs=self.n_jobs)["test_score"].mean()
            space = {"model__" + k: v for k, v in spec.space(self.ctx).items()}
            t0 = time.perf_counter()
            if space:
                search = RandomizedSearchCV(base, space, n_iter=n_iter, scoring=scorer, cv=self.cv,
                                            n_jobs=self.n_jobs, random_state=self.seed,
                                            error_score=np.nan)
                search.fit(self.X_train, self.y_train)
                tuned_cv, tuned_pipe = search.best_score_, search.best_estimator_
                tuned_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            else:
                tuned_cv, tuned_pipe, tuned_params = base_cv, None, {}

            if tuned_pipe is not None and tuned_cv > base_cv:
                final, kept = tuned_pipe, "tuned"
                params = {**self.starting_params(mid), **tuned_params}
            else:
                final, kept = clone(base).fit(self.X_train, self.y_train), "starting"
                params = self.starting_params(mid)
            params = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                      for k, v in params.items()}
            self._best_pipes[mid] = final
            self.recommended_params_[mid] = params
            hold = {m: s(final, self.X_test, self.y_test) for m, s in self._scorers_for(final).items()}
            rows.append({"ID": mid, "Model": spec.name,
                         f"CV {self.sort} (start)": base_cv, f"CV {self.sort} (tuned)": tuned_cv,
                         "Kept": kept, **{f"Holdout {m}": hold.get(m, np.nan) for m in _METRICS},
                         "Tune (s)": time.perf_counter() - t0})
            if verbose:
                print(f"  {spec.name:<32} start={base_cv:.4f}  tuned={tuned_cv:.4f}  -> keep {kept}")
        self.tuned_ = pd.DataFrame(rows).set_index("ID").sort_values(f"Holdout {self.sort}",
                                                                     ascending=False)
        return self.tuned_.round(4)

    def get_model(self, model_id: str) -> LabeledModel:
        """Best fitted version of a model (tuned if tuning helped), trained on the training split."""
        if model_id not in self._best_pipes:
            self._best_pipes[model_id] = clone(self._pipeline(self.specs[model_id])).fit(
                self.X_train, self.y_train)
        return LabeledModel(self._best_pipes[model_id], self.classes_)

    def confusion_matrix(self, model_id: str) -> pd.DataFrame:
        """Holdout confusion matrix (rows = actual, columns = predicted)."""
        pred = self.get_model(model_id).pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, pred, labels=range(self.ctx.k))
        return pd.DataFrame(cm, index=[f"actual {c}" for c in self.classes_],
                            columns=[f"pred {c}" for c in self.classes_])

    def class_report(self, model_id: str) -> str:
        """Holdout precision / recall / F1 for every class."""
        pred = self.get_model(model_id).pipeline.predict(self.X_test)
        return classification_report(self.y_test, pred, labels=range(self.ctx.k),
                                     target_names=[str(c) for c in self.classes_], zero_division=0)

    def finalize(self, model_id: str | None = None) -> LabeledModel:
        """Refit the chosen model (default: best holdout result) on ALL rows for deployment."""
        if model_id is None:
            if self.tuned_ is None:
                raise RuntimeError("run tune_top() first or pass a model_id")
            model_id = self.tuned_.index[0]
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        pipe = self._best_pipes[model_id] if model_id in self._best_pipes else self._pipeline(self.specs[model_id])
        return LabeledModel(clone(pipe).fit(X, y), self.classes_)

    def summary(self) -> str:
        """Plain-text recap: data facts, winner and its recommended hyperparameters."""
        c = self.ctx
        lines = [f"Rows (train): {c.n:,}   Features (encoded): {c.p}   Classes: {list(self.classes_)}",
                 f"Class balance (smallest/largest): {c.imbalance:.2f}   "
                 f"class_weight start: {'balanced' if c.balanced else 'None'}",
                 f"Categorical columns one-hot encoded: {self.cat_cols or 'none'}"]
        if c.binary:
            lines.append(f"Positive class (for Recall/Precision/F1): {self.classes_[1]!r}")
        if self.tuned_ is not None:
            best = self.tuned_.index[0]
            lines += [f"\nBest model: {self.specs[best].name} ({best})",
                      f"Holdout {self.sort}: {self.tuned_.loc[best, f'Holdout {self.sort}']:.4f}",
                      "Recommended hyperparameters:"]
            lines += [f"  {k} = {v}" for k, v in self.recommended_params_[best].items()]
        return "\n".join(lines)


if __name__ == "__main__":  # quick demo:  python classification_picker.py
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True).frame
    data["target"] = data["target"].map({0: "malignant", 1: "benign"})
    cp = ClassificationPicker(data, target="target", positive_class="malignant")
    print(cp.compare_models())
    print(cp.tune_top(n=3, n_iter=15))
    print(cp.summary())
    print(cp.confusion_matrix(cp.tuned_.index[0]))
