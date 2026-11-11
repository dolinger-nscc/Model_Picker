"""
cluster_picker_1a.py
====================

PyCaret-style model selection for CLUSTERING (unsupervised, no target) using
scikit-learn. It follows the same pattern as regression_picker_1a.py and
classification_picker_1a.py, so students learn one interface for all three.

Charts (matplotlib, loaded only when a plot_ method is called):
  plot_k_scan()  plot_leaderboard()  plot_tuning()  plot_clusters()
  plot_silhouette()  plot_profile()  plot_crosstab()
Each returns the matplotlib Axes (or Figure), so it can be restyled or saved.

The module does four jobs:
  1. k_scan()          -> elbow / silhouette / BIC table for choosing the number of clusters
  2. compare_models()  -> fits every candidate model and ranks them
  3. tune_top()        -> tunes the top N models and recommends hyperparameters
  4. profile() / assign_clusters() -> explain and use the clusters found

Without a target there is no "right answer" to score against, so models are
ranked with INTERNAL metrics that judge cluster shape:
  * Silhouette          (-1..1, higher is better)  - how well each point fits its cluster
                                                     (noise points count as 0, so density
                                                     models can't score higher by discarding rows)
  * Calinski-Harabasz   (higher is better)         - between- vs within-cluster spread
  * Davies-Bouldin      (0+,   lower is better)    - average similarity of each cluster to its closest neighbour
  * Mean rank           (lower is better)          - average of the three ranks above
All three favour compact, round clusters, so density models (DBSCAN, HDBSCAN,
OPTICS) can find sensible odd-shaped clusters and still score lower. Look at
the profiles, not just the numbers.

If you have real labels (e.g. to teach with the iris species), pass
ground_truth="species": that column is NOT used for clustering, but external
metrics (ARI, NMI, Homogeneity) are added so students can see how well the
internal metrics agreed with reality.

Hyperparameters are handled in two layers:
  * Starting values are *data-aware*:
      - n_clusters: best silhouette from a KMeans scan of k = 2..max_clusters
      - DBSCAN eps: the "knee" of the k-nearest-neighbour distance curve, with
        min_samples = 2 x features (Sander et al., 1998), kept between 4 and 50
      - HDBSCAN min_cluster_size: about 2% of rows
      - MeanShift bandwidth: estimated from the data with estimate_bandwidth()
      - KMeans uses n_init=10 restarts for stable results
  * Tuning is a random search over curated spaces, scored by the sort metric.
    Configurations that label more than max_noise of the rows as noise are
    rejected, so a density model can't "win" by throwing most points away.
    The tuned model is kept only if it beats the starting one.

Input: a CLEAN pandas DataFrame (no missing values). Numeric columns are
standardized (clustering is distance-based, so scale matters); object/category/
bool columns are one-hot encoded.

Example
-------
>>> from cluster_picker_1a import ClusterPicker
>>> cl = ClusterPicker(df)
>>> cl.k_scan()                    # how many clusters?
>>> cl.compare_models()            # leaderboard
>>> cl.tune_top(n=3)               # tuned results for the top 3
>>> cl.recommended_params_         # {model_id: {param: value}}
>>> cl.profile("kmeans")           # average feature values per cluster
>>> out = cl.assign_clusters("kmeans")   # df with a new 'Cluster' column
>>> cl.plot_k_scan()               # charts: see the plot_ methods
>>> cl.plot_clusters()
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
from sklearn.cluster import (DBSCAN, HDBSCAN, OPTICS, AffinityPropagation,
                             AgglomerativeClustering, Birch, BisectingKMeans,
                             KMeans, MeanShift, MiniBatchKMeans,
                             SpectralClustering, estimate_bandwidth)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, homogeneity_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------------------------
# Data context: facts about the data that drive the starting values
# ------------------------------------------------------------------------------
@dataclass
class DataContext:
    n: int                 # rows
    p: int                 # features AFTER one-hot encoding
    k: int                 # starting number of clusters
    k_max: int             # largest k searched
    eps: float             # DBSCAN starting eps (k-distance knee)
    min_samples: int       # DBSCAN / OPTICS starting min_samples
    bandwidth: float       # MeanShift starting bandwidth
    seed: int


@dataclass
class ModelSpec:
    id: str
    name: str
    family: str
    build: Callable[[DataContext], object]
    space: Callable[[DataContext], dict] = lambda c: {}
    requires: Callable[[DataContext], str | None] = lambda c: None
    turbo_skip: bool = False
    finds_k: bool = False       # chooses its own number of clusters?
    tags: list = field(default_factory=list)


def _limit(max_n):
    def check(c: DataContext):
        return f"too slow for {c.n:,} rows (limit {max_n:,})" if c.n > max_n else None
    return check


def _registry() -> list[ModelSpec]:
    S = []
    k_range = lambda c: randint(2, c.k_max + 1)

    # ---- Centroid -----------------------------------------------------------
    S.append(ModelSpec("kmeans", "K-Means", "centroid",
                       lambda c: KMeans(n_clusters=c.k, n_init=10, random_state=c.seed),
                       lambda c: {"n_clusters": k_range(c), "init": ["k-means++", "random"]}))
    S.append(ModelSpec("mbkmeans", "Mini-Batch K-Means", "centroid",
                       lambda c: MiniBatchKMeans(n_clusters=c.k, n_init=10, batch_size=1024,
                                                 random_state=c.seed),
                       lambda c: {"n_clusters": k_range(c), "batch_size": [256, 1024, 4096]}))
    S.append(ModelSpec("bkmeans", "Bisecting K-Means", "centroid",
                       lambda c: BisectingKMeans(n_clusters=c.k, n_init=5, random_state=c.seed),
                       lambda c: {"n_clusters": k_range(c),
                                  "bisecting_strategy": ["biggest_inertia", "largest_cluster"]}))

    # ---- Distribution -------------------------------------------------------
    S.append(ModelSpec("gmm", "Gaussian Mixture", "distribution",
                       lambda c: GaussianMixture(n_components=c.k, covariance_type="full", n_init=3,
                                                 random_state=c.seed),
                       lambda c: {"n_components": k_range(c),
                                  "covariance_type": ["full", "tied", "diag", "spherical"],
                                  "reg_covar": loguniform(1e-6, 1e-2)}))

    # ---- Hierarchical -------------------------------------------------------
    S.append(ModelSpec("hclust", "Agglomerative (Ward)", "hierarchical",
                       lambda c: AgglomerativeClustering(n_clusters=c.k, linkage="ward"),
                       lambda c: {"n_clusters": k_range(c),
                                  "linkage": ["ward", "complete", "average", "single"]},
                       requires=_limit(15_000)))
    S.append(ModelSpec("birch", "BIRCH", "hierarchical",
                       lambda c: Birch(n_clusters=c.k, threshold=0.5),
                       lambda c: {"n_clusters": k_range(c), "threshold": uniform(0.1, 1.4),
                                  "branching_factor": randint(20, 101)}))

    # ---- Graph --------------------------------------------------------------
    S.append(ModelSpec("spectral", "Spectral Clustering", "graph",
                       lambda c: SpectralClustering(n_clusters=c.k, affinity="nearest_neighbors",
                                                    n_neighbors=int(np.clip(np.log2(c.n) * 2, 10, 50)),
                                                    assign_labels="cluster_qr", random_state=c.seed),
                       lambda c: {"n_clusters": k_range(c), "n_neighbors": randint(5, 51),
                                  "assign_labels": ["cluster_qr", "kmeans", "discretize"]},
                       requires=_limit(5_000)))
    S.append(ModelSpec("ap", "Affinity Propagation", "graph",
                       lambda c: AffinityPropagation(damping=0.9, max_iter=500, random_state=c.seed),
                       lambda c: {"damping": uniform(0.5, 0.49)},
                       requires=_limit(3_000), finds_k=True))

    # ---- Density ------------------------------------------------------------
    S.append(ModelSpec("dbscan", "DBSCAN", "density",
                       lambda c: DBSCAN(eps=c.eps, min_samples=c.min_samples),
                       lambda c: {"eps": loguniform(c.eps / 2, c.eps * 2),
                                  "min_samples": randint(max(3, c.min_samples // 2), c.min_samples * 2 + 1)},
                       finds_k=True))
    S.append(ModelSpec("hdbscan", "HDBSCAN", "density",
                       lambda c: HDBSCAN(min_cluster_size=int(np.clip(c.n * 0.02, 5, 200)), copy=True),
                       lambda c: {"min_cluster_size": randint(5, int(np.clip(c.n // 10, 10, 500)) + 1),
                                  "min_samples": [None, 3, 5, 10, 20],
                                  "cluster_selection_method": ["eom", "leaf"]},
                       finds_k=True))
    S.append(ModelSpec("optics", "OPTICS", "density",
                       lambda c: OPTICS(min_samples=c.min_samples, xi=0.05),
                       lambda c: {"min_samples": randint(max(3, c.min_samples // 2), c.min_samples * 2 + 1),
                                  "xi": uniform(0.01, 0.2)},
                       requires=_limit(20_000), turbo_skip=True, finds_k=True))
    S.append(ModelSpec("meanshift", "Mean Shift", "density",
                       lambda c: MeanShift(bandwidth=c.bandwidth, bin_seeding=True),
                       lambda c: {"bandwidth": uniform(c.bandwidth * 0.5, c.bandwidth * 1.5)},
                       requires=_limit(10_000), finds_k=True))
    return S


# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------
_INTERNAL = {"Silhouette": True, "Calinski-Harabasz": True, "Davies-Bouldin": False}   # True = higher better
_EXTERNAL = {"ARI": True, "NMI": True, "Homogeneity": True}


def _knee(values: np.ndarray) -> float:
    """Point on a sorted curve farthest from the straight line joining its ends."""
    y = np.sort(values)
    x = np.linspace(0, 1, len(y))
    yn = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    return float(y[np.argmax(x - yn)])     # distance below the diagonal



# ------------------------------------------------------------------------------
# Plotting helpers
# matplotlib is imported only when a plot_ method is called, so the module still
# works without it. Every plot_ method takes an optional `ax` (or returns a whole
# figure for multi-panel charts) and returns it, so students can restyle it with
# ordinary matplotlib: ax.set_title(...), ax.figure.savefig(...), etc.
# ------------------------------------------------------------------------------
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",       # fixed categorical order,
           "#e87ba4", "#008300", "#6250d6", "#e34948"]       # colourblind-checked
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]         # second cue besides colour
_INK, _INK2, _MUTED = "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS, _SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
_BASELINE = "#c3c2b7"                                        # dummy / noise / "other"
_SEQ = ["#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_DIV = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2b0ac", "#e34948", "#a62b2b"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting needs matplotlib:  pip install matplotlib") from e
    return plt


def _fmt(v):
    """Readable number for labels: 53,350 / 55.3 / 0.639 / 3.40e-05."""
    if v is None or not np.isfinite(v):
        return ""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.001 or a == 0:
        return f"{v:.3f}"
    return f"{v:.2e}"


def _thousands(ax, axis="both"):
    """Comma thousands separators on large-valued axes (53,000 not 53000)."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:g}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def _style(ax, grid="x"):
    ax.set_facecolor(_SURFACE)
    ax.figure.set_facecolor(_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS)
    ax.tick_params(colors=_INK2, labelsize=9, length=3, color=_AXIS)
    ax.xaxis.label.set_color(_INK2)
    ax.yaxis.label.set_color(_INK2)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return ax


def _axes(ax, figsize, grid="x"):
    if ax is None:
        _, ax = _plt().subplots(figsize=figsize)
    return _style(ax, grid)


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, color=_INK, fontweight="bold",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=_INK2, va="bottom")


def _legend(ax, ncol=3, **kw):
    leg = ax.legend(frameon=False, fontsize=9, ncol=ncol, loc="upper center",
                    bbox_to_anchor=(0.5, -0.16), labelcolor=_INK2, **kw)
    return leg


def _ranked_bars(ax, labels, values, errors=None, muted=None):
    """Horizontal bars, best on top. Direct value labels on the best and muted rows only."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    y = np.arange(n)[::-1]
    muted = np.zeros(n, bool) if muted is None else np.asarray(muted, bool)
    err = None if errors is None else np.nan_to_num(np.asarray(errors, dtype=float))
    ax.barh(y, np.nan_to_num(values), height=0.62, edgecolor=_SURFACE, linewidth=1,
            color=[_BASELINE if m else _SERIES[0] for m in muted])
    if err is not None:
        ax.errorbar(np.nan_to_num(values), y, xerr=err, fmt="none", ecolor=_INK2,
                    elinewidth=1, capsize=2)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=_AXIS, linewidth=1)
    for i in sorted({0, *np.flatnonzero(muted)}):
        v = values[i]
        if np.isnan(v):
            continue
        end = v + (err[i] if err is not None else 0) * (1 if v >= 0 else -1)
        ax.annotate(_fmt(v), (end, y[i]), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=_INK)
    ax.margins(x=0.1)
    ax.grid(axis="y", visible=False)
    _thousands(ax, "x")


def _dumbbell(ax, labels, start, tuned, extra=None, extra_label=None):
    """Start -> tuned dots joined by a line (small differences stay visible)."""
    y = np.arange(len(labels))[::-1]
    for yi, a, b in zip(y, start, tuned):
        ax.plot([a, b], [yi, yi], color=_AXIS, linewidth=2, zorder=1)
    kw = dict(s=60, zorder=3, edgecolor=_SURFACE, linewidth=1.5)
    ax.scatter(start, y, color=_SERIES[0], marker="o", label="Starting values (CV)", **kw)
    ax.scatter(tuned, y, facecolor="none", edgecolor=_SERIES[1], marker="s", s=150, linewidth=2,
               zorder=4, label="Tuned (CV)")      # hollow, so an unchanged start dot shows through
    if extra is not None:
        ax.scatter(extra, y, color=_SERIES[2], marker="D", label=extra_label, **kw)
    ax.set_yticks(y, labels)
    ax.margins(y=0.25, x=0.15)
    ax.grid(axis="y", visible=False)
    _thousands(ax, "x")
    _legend(ax)

# ------------------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------------------
class ClusterPicker:
    """Rank clustering models on a clean DataFrame and recommend hyperparameters.

    Parameters
    ----------
    df : pandas.DataFrame   Clean data (no missing values).
    n_clusters : int        Starting number of clusters. None = pick by KMeans silhouette scan.
    max_clusters : int      Largest k considered in the scan and in tuning.
    ground_truth : str      Optional label column, used ONLY for ARI/NMI/Homogeneity (not for clustering).
    ignore : list           Columns to leave out (IDs, names, dates...).
    sort : str              Silhouette, Calinski-Harabasz, Davies-Bouldin, Mean rank,
                            or (with ground_truth) ARI, NMI, Homogeneity.
    max_noise : float       Largest share of rows a density model may label as noise (-1).
    turbo : bool            Skip slow models (OPTICS), like PyCaret.
    include / exclude : list of model ids to force in / leave out.
    random_state : int      Seed for reproducible results.
    quiet : bool            Hide convergence warnings.
    """

    def __init__(self, df: pd.DataFrame, n_clusters: int | None = None, max_clusters: int = 10,
                 ground_truth: str | None = None, ignore=None, sort: str = "Silhouette",
                 max_noise: float = 0.25, turbo: bool = True, include=None, exclude=None,
                 random_state: int = 42, quiet: bool = True):
        self._validate(df, ground_truth, sort)
        self.df = df
        self.sort, self.max_noise, self.turbo = sort, max_noise, turbo
        self.include, self.exclude = set(include or []), set(exclude or [])
        self.seed, self.quiet = random_state, quiet
        self.rng = np.random.default_rng(random_state)

        drop = list(ignore or []) + ([ground_truth] if ground_truth else [])
        self.X = df.drop(columns=drop)
        self.y_true = df[ground_truth].to_numpy() if ground_truth else None
        self.cat_cols = self.X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.num_cols = [c for c in self.X.columns if c not in self.cat_cols]
        self.prep = self._preprocessor().fit(self.X)
        self.Xt = self.prep.transform(self.X)      # one shared, scaled matrix for every model

        self._quiet()
        self.k_scan_ = None
        self.ctx = self._build_context(n_clusters, max_clusters)
        self.specs = {s.id: s for s in _registry()}
        self.leaderboard_ = None
        self.tuned_ = None
        self.recommended_params_ = {}
        self.skipped_ = {}
        self._models, self._labels = {}, {}

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _validate(df, ground_truth, sort):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"df has missing values in {bad}; clean it first")
        if ground_truth and ground_truth not in df.columns:
            raise ValueError(f"ground_truth '{ground_truth}' not found in df columns")
        allowed = list(_INTERNAL) + ["Mean rank"] + (list(_EXTERNAL) if ground_truth else [])
        if sort not in allowed:
            raise ValueError(f"sort must be one of {allowed}")

    def _preprocessor(self):
        return ColumnTransformer(
            [("num", StandardScaler(), self.num_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)],
            verbose_feature_names_out=False)

    def _sample(self, size=5000):
        n = len(self.Xt)
        return self.Xt if n <= size else self.Xt[self.rng.choice(n, size, replace=False)]

    def _build_context(self, n_clusters, max_clusters) -> DataContext:
        n, p = self.Xt.shape
        k_max = int(max(2, min(max_clusters, n - 1)))
        if n_clusters is None:
            scan = self.k_scan(k_max)
            k = int(scan["Silhouette"].idxmax())
        else:
            k = int(n_clusters)
        min_samples = int(np.clip(2 * p, 4, 50))
        Xs = self._sample()
        dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(Xs).kneighbors(Xs)
        eps = _knee(dist[:, -1])
        bandwidth = float(estimate_bandwidth(Xs, quantile=0.2, random_state=self.seed)) or 1.0
        return DataContext(n=n, p=p, k=k, k_max=k_max, eps=eps, min_samples=min_samples,
                           bandwidth=bandwidth, seed=self.seed)

    def _quiet(self):
        if self.quiet:
            warnings.filterwarnings("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

    def _score(self, labels) -> dict:
        labels = np.asarray(labels)
        mask = labels != -1
        noise = 1 - mask.mean()
        k = len(np.unique(labels[mask]))
        out = {"Clusters": k, "Noise %": 100 * noise}
        if k < 2 or k >= mask.sum():
            out.update({m: np.nan for m in _INTERNAL})
        else:
            Xm, lm = self.Xt[mask], labels[mask]
            # Noise points count as silhouette 0, so dropping points can't inflate the score.
            out["Silhouette"] = (1 - noise) * silhouette_score(Xm, lm, sample_size=min(5000, len(lm)),
                                                               random_state=self.seed)
            out["Calinski-Harabasz"] = calinski_harabasz_score(Xm, lm)
            out["Davies-Bouldin"] = davies_bouldin_score(Xm, lm)
        if self.y_true is not None:     # noise counts as its own group here (honest penalty)
            out["ARI"] = adjusted_rand_score(self.y_true, labels)
            out["NMI"] = normalized_mutual_info_score(self.y_true, labels)
            out["Homogeneity"] = homogeneity_score(self.y_true, labels)
        return out

    def _valid(self, s: dict) -> bool:
        return s["Noise %"] <= 100 * self.max_noise and not np.isnan(s.get("Silhouette", np.nan))

    def _higher_better(self, metric):
        return {**_INTERNAL, **_EXTERNAL, "Mean rank": False}[metric]

    def _better(self, a, b, metric):
        if np.isnan(b):
            return not np.isnan(a)
        if np.isnan(a):
            return False
        return a > b if self._higher_better(metric) else a < b

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

    # ------------------------------------------------------------- public API
    def k_scan(self, k_max: int | None = None) -> pd.DataFrame:
        """Scan k = 2..k_max with KMeans (inertia, silhouette, CH, DB) and GaussianMixture (BIC).

        Elbow in Inertia, peak in Silhouette / CH, dip in Davies-Bouldin / BIC all point
        to a good k. They often disagree - a useful discussion point.
        """
        if k_max is None:
            if self.k_scan_ is not None:
                return self.k_scan_.round(4)
            k_max = self.ctx.k_max
        rows = []
        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(self.Xt)
            gm = GaussianMixture(n_components=k, random_state=self.seed).fit(self.Xt)
            rows.append({"k": k, "Inertia": km.inertia_,
                         "Silhouette": silhouette_score(self.Xt, km.labels_,
                                                        sample_size=min(5000, len(self.Xt)),
                                                        random_state=self.seed),
                         "Calinski-Harabasz": calinski_harabasz_score(self.Xt, km.labels_),
                         "Davies-Bouldin": davies_bouldin_score(self.Xt, km.labels_),
                         "GMM BIC": gm.bic(self.Xt)})
        self.k_scan_ = pd.DataFrame(rows).set_index("k")
        return self.k_scan_.round(4)

    def models(self) -> pd.DataFrame:
        """List every available model, its family and whether it will run on this data."""
        rows = []
        for mid, s in self.specs.items():
            reason = s.requires(self.ctx) or ("turbo mode" if self.turbo and s.turbo_skip else "")
            rows.append({"ID": mid, "Model": s.name, "Family": s.family,
                         "Chooses own k": s.finds_k, "Skip reason": reason})
        return pd.DataFrame(rows).set_index("ID")

    def starting_params(self, model_id: str) -> dict:
        """The data-aware starting hyperparameters for one model (before tuning)."""
        est = self.specs[model_id].build(self.ctx)
        defaults = type(est)().get_params(deep=False)
        housekeeping = {"n_jobs", "random_state", "verbose", "copy"}
        return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                for k, v in est.get_params(deep=False).items()
                if k not in housekeeping and (k not in defaults or repr(defaults[k]) != repr(v))}

    def compare_models(self, verbose: bool = True) -> pd.DataFrame:
        """Fit every candidate with its starting hyperparameters and rank them."""
        self._quiet()
        if verbose:
            c = self.ctx
            print(f"Starting values: n_clusters={c.k}, DBSCAN eps={c.eps:.3f}, "
                  f"min_samples={c.min_samples}, MeanShift bandwidth={c.bandwidth:.3f}")
        rows = []
        for spec in self._candidates():
            t0 = time.perf_counter()
            try:
                est = spec.build(self.ctx)
                labels = est.fit_predict(self.Xt)
            except Exception as e:
                self.skipped_[spec.id] = f"failed: {type(e).__name__}: {str(e)[:80]}"
                continue
            fit_s = time.perf_counter() - t0
            s = self._score(labels)
            note = "" if s["Noise %"] <= 100 * self.max_noise else "too much noise"
            if s["Clusters"] < 2:
                note = "found < 2 clusters"
            self._models[spec.id], self._labels[spec.id] = est, labels
            rows.append({"ID": spec.id, "Model": spec.name, **s, "Fit (s)": fit_s, "Note": note})
            if verbose:
                print(f"  {spec.name:<24} k={s['Clusters']:<3} {self.sort}="
                      f"{s.get(self.sort, np.nan):.4f}  ({fit_s:.1f}s)")
        lb = pd.DataFrame(rows).set_index("ID")
        ranks = [lb[m].rank(ascending=not hb) for m, hb in _INTERNAL.items()]
        lb["Mean rank"] = pd.concat(ranks, axis=1).mean(axis=1)
        lb["_ok"] = lb["Note"] == ""                     # valid models first, then by metric
        lb = lb.sort_values(["_ok", self.sort], ascending=[False, not self._higher_better(self.sort)],
                            na_position="last").drop(columns="_ok")
        cols = ["Model", "Clusters", "Noise %", *_INTERNAL, "Mean rank"]
        cols += [m for m in _EXTERNAL if m in lb] + ["Fit (s)", "Note"]
        self.leaderboard_ = lb[cols]
        if verbose and self.skipped_:
            print("\nSkipped:", ", ".join(f"{k} ({v})" for k, v in self.skipped_.items()))
        return self.leaderboard_.round(4)

    def tune_top(self, n: int = 3, n_iter: int = 25, models=None, verbose: bool = True) -> pd.DataFrame:
        """Random-search the top-n models (or the listed `models`) on the sort metric.

        'Mean rank' can't be computed for a single model, so tuning uses Silhouette
        when sort='Mean rank'. Configurations with too much noise are rejected.
        """
        if models is None:
            if self.leaderboard_ is None:
                self.compare_models(verbose=False)
            ok = self.leaderboard_[self.leaderboard_["Note"] == ""]
            models = list(ok.index[:n])
        metric = "Silhouette" if self.sort == "Mean rank" else self.sort
        self._quiet()
        rows = []
        for mid in models:
            spec = self.specs[mid]
            if mid not in self._labels:
                est = spec.build(self.ctx)
                self._models[mid], self._labels[mid] = est, est.fit_predict(self.Xt)
            base_est, base_s = self._models[mid], self._score(self._labels[mid])
            base_val = base_s.get(metric, np.nan) if self._valid(base_s) else np.nan
            best = (base_val, base_est, self._labels[mid], {})
            t0, tried = time.perf_counter(), 0
            for params in ParameterSampler(spec.space(self.ctx), n_iter=n_iter, random_state=self.seed):
                est = clone(spec.build(self.ctx)).set_params(**params)
                try:
                    labels = est.fit_predict(self.Xt)
                except Exception:
                    continue
                tried += 1
                s = self._score(labels)
                if self._valid(s) and self._better(s[metric], best[0], metric):
                    best = (s[metric], est, labels, params)
            tuned_val, est, labels, params = best
            kept = "tuned" if params else "starting"
            self._models[mid], self._labels[mid] = est, labels
            rec = {**self.starting_params(mid),
                   **{k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                      for k, v in params.items()}}
            self.recommended_params_[mid] = rec
            final = self._score(labels)
            rows.append({"ID": mid, "Model": spec.name, f"{metric} (start)": base_val,
                         f"{metric} (tuned)": tuned_val, "Kept": kept,
                         **{k: v for k, v in final.items() if k != metric},
                         "Configs tried": tried, "Tune (s)": time.perf_counter() - t0})
            if verbose:
                print(f"  {spec.name:<24} start={base_val:.4f}  tuned={tuned_val:.4f}  "
                      f"k={final['Clusters']}  -> keep {kept}")
        self.tuned_ = pd.DataFrame(rows).set_index("ID").sort_values(
            f"{metric} (tuned)", ascending=not self._higher_better(metric), na_position="last")
        return self.tuned_.round(4)

    def get_model(self, model_id: str) -> Pipeline:
        """Best version of a model (tuned if tuning helped) as a fitted Pipeline (prep + model).

        Only some models can label NEW rows with .predict(): K-Means family, Gaussian
        Mixture, BIRCH, Affinity Propagation and Mean Shift. The others (Agglomerative,
        Spectral, DBSCAN, HDBSCAN, OPTICS) only label the data they were fitted on.
        """
        if model_id not in self._models:
            est = self.specs[model_id].build(self.ctx)
            self._labels[model_id] = est.fit_predict(self.Xt)
            self._models[model_id] = est
        return Pipeline([("prep", self.prep), ("model", self._models[model_id])])

    def labels(self, model_id: str) -> np.ndarray:
        """Cluster label for every row (-1 = noise for density models)."""
        self.get_model(model_id)
        return self._labels[model_id]

    def assign_clusters(self, model_id: str | None = None) -> pd.DataFrame:
        """Copy of the original df with a 'Cluster' column added (like PyCaret's assign_model)."""
        model_id = model_id or self.best_model_id()
        out = self.df.copy()
        out["Cluster"] = self.labels(model_id)
        return out

    def profile(self, model_id: str | None = None) -> pd.DataFrame:
        """Cluster sizes plus the average of each numeric feature (and share of each
        category level) per cluster, in ORIGINAL units - the table you'd show a manager."""
        model_id = model_id or self.best_model_id()
        labels = self.labels(model_id)
        X = self.X.copy()
        X["Cluster"] = labels
        prof = X.groupby("Cluster")[self.num_cols].mean() if self.num_cols else pd.DataFrame(
            index=np.unique(labels))
        for col in self.cat_cols:
            shares = pd.crosstab(X["Cluster"], X[col], normalize="index").add_prefix(f"{col}=")
            prof = prof.join(shares)
        sizes = pd.Series(labels).value_counts().sort_index()
        prof.insert(0, "Size", sizes)
        prof.insert(1, "Share %", 100 * sizes / len(labels))
        if self.y_true is not None:
            prof["Most common label"] = pd.Series(self.y_true).groupby(labels).agg(
                lambda s: s.value_counts().index[0])
        return prof.round(3)

    def best_model_id(self) -> str:
        if self.tuned_ is not None:
            return self.tuned_.index[0]
        if self.leaderboard_ is not None:
            return self.leaderboard_.index[0]
        raise RuntimeError("run compare_models() or tune_top() first, or pass a model_id")

    # ------------------------------------------------------------- plots
    def _cluster_style(self, labels):
        """Fixed colour + marker per cluster (same in every chart). Clusters past the
        8th share one grey 'other' style rather than recycling colours."""
        ids = [c for c in np.unique(labels) if c != -1]
        style = {}
        for i, c in enumerate(ids):
            style[c] = ((_SERIES[i], _MARKERS[i], f"Cluster {c}") if i < 8
                        else (_BASELINE, "o", "Other clusters"))
        if -1 in labels:
            style[-1] = ("#d9d8d2", "x", "Noise")
        return style

    def plot_k_scan(self):
        """Six panels: the k-scan metrics (best k marked on each) plus a summary of which
        k each metric prefers. For inertia, the 'best' k is the elbow of the curve."""
        scan = self.k_scan_ if self.k_scan_ is not None else self.k_scan()
        ks = scan.index.to_numpy()
        panels = [("Inertia", "elbow"), ("Silhouette", "max"), ("Calinski-Harabasz", "max"),
                  ("Davies-Bouldin", "min"), ("GMM BIC", "min")]
        fig, axes = _plt().subplots(2, 3, figsize=(13, 7))
        best = {}
        for ax, (col, rule) in zip(axes.flat, panels):
            _style(ax, "y")
            v = scan[col].to_numpy()
            if rule == "elbow":
                xn = (ks - ks[0]) / (ks[-1] - ks[0] or 1)
                yn = (v - v.min()) / (np.ptp(v) or 1)
                i = int(np.argmax((1 - xn) - yn)) if len(ks) > 2 else 0
            else:
                i = int(np.argmax(v) if rule == "max" else np.argmin(v))
            best[col] = int(ks[i])
            ax.plot(ks, v, color=_SERIES[0], linewidth=2, marker="o", markersize=5)
            ax.scatter([ks[i]], [v[i]], s=110, color=_SERIES[1], marker="s", zorder=3,
                       edgecolor=_SURFACE, linewidth=1.5)
            ax.annotate(f"k={ks[i]}", (ks[i], v[i]), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, color=_INK)
            ax.axvline(self.ctx.k, color=_MUTED, linestyle=":", linewidth=1)
            ax.set_xticks(ks)
            ax.set_xlabel("k")
            if np.nanmax(np.abs(v)) >= 1000:
                _thousands(ax, "y")
            ax.set_title(f"{col}  ({'elbow' if rule == 'elbow' else 'higher' if rule == 'max' else 'lower'}"
                         f"{'' if rule == 'elbow' else ' is better'})", loc="left", fontsize=10.5, color=_INK)
        last = axes.flat[-1]
        last.axis("off")
        last.set_facecolor(_SURFACE)
        lines = ["Best k by metric", ""] + [f"{m:<20}{k}" for m, k in best.items()]
        lines += ["", f"Starting k used:    {self.ctx.k}", "(dotted line on each panel)"]
        last.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10, color=_INK)
        fig.suptitle("How many clusters? KMeans scan (GMM for BIC)", x=0.01, ha="left",
                     fontsize=13, fontweight="bold", color=_INK)
        fig.set_facecolor(_SURFACE)
        fig.tight_layout()
        return fig

    def plot_leaderboard(self, metric: str | None = None, ax=None):
        """Bar chart of the compare_models() results, best on top. Models flagged in the
        Note column (too much noise, < 2 clusters) are grey."""
        if self.leaderboard_ is None:
            raise RuntimeError("run compare_models() first")
        metric = metric or self.sort
        if metric not in self.leaderboard_:
            raise ValueError(f"metric must be a leaderboard column, e.g. {list(_INTERNAL)}")
        hb = self._higher_better(metric)
        lb = self.leaderboard_.assign(_ok=self.leaderboard_["Note"] == "").sort_values(
            ["_ok", metric], ascending=[False, not hb], na_position="last")
        labels = [f"{m} (k={k})" + (f" - {n}" if n else "") for m, k, n in
                  zip(lb["Model"], lb["Clusters"], lb["Note"])]
        ax = _axes(ax, (8, 0.3 * len(lb) + 1.6))
        _ranked_bars(ax, labels, lb[metric], muted=~lb["_ok"].to_numpy())
        ax.set_xlabel(metric)
        _title(ax, f"Model comparison: {metric}",
               f"{'higher' if hb else 'lower'} is better - grey = flagged in the Note column")
        return ax

    def plot_tuning(self, ax=None):
        """Starting vs tuned score for each tuned model (number of clusters in the label)."""
        if self.tuned_ is None:
            raise RuntimeError("run tune_top() first")
        t = self.tuned_
        metric = [c for c in t.columns if c.endswith("(start)")][0].replace(" (start)", "")
        labels = [f"{m}  [k={k}, kept: {kept}]" for m, k, kept in zip(t["Model"], t["Clusters"], t["Kept"])]
        ax = _axes(ax, (8, 0.55 * len(t) + 1.8))
        _dumbbell(ax, labels, t[f"{metric} (start)"], t[f"{metric} (tuned)"])
        ax.set_xlabel(metric)
        _title(ax, f"Tuning results: {metric}",
               f"{'higher' if self._higher_better(metric) else 'lower'} is better - axis zoomed in to show small changes")
        return ax

    def _projection(self):
        """2-D view of the data: the original columns if there are exactly two numeric
        features, otherwise the first two principal components of the scaled data."""
        if self.Xt.shape[1] == 2 and len(self.num_cols) == 2:
            return self.X[self.num_cols].to_numpy(), self.num_cols[0], self.num_cols[1], "Original columns"
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=self.seed).fit(self.Xt)
        xy = pca.transform(self.Xt)
        ev = pca.explained_variance_ratio_ * 100
        return (xy, f"PC1 ({ev[0]:.0f}% of variance)", f"PC2 ({ev[1]:.0f}% of variance)",
                f"PCA view - {ev.sum():.0f}% of the variance shown")

    def plot_clusters(self, model_id: str | None = None, ax=None):
        """Scatter of every row coloured (and shaped) by cluster. Uses the two original
        columns when there are only two, otherwise a 2-D PCA projection."""
        mid = model_id or self.best_model_id()
        labels = self.labels(mid)
        xy, xl, yl, how = self._projection()
        ax = _axes(ax, (7.5, 6), grid="both")
        idx = np.random.default_rng(self.seed).permutation(len(labels))[:5000]
        style = self._cluster_style(labels)
        seen = set()
        for c in sorted(style, key=lambda c: (c == -1, c)):
            col, mk, name = style[c]
            m = idx[labels[idx] == c]
            ax.scatter(xy[m, 0], xy[m, 1], s=26 if c != -1 else 18, color=col, marker=mk,
                       edgecolor=_SURFACE if mk != "x" else None, linewidth=0.6 if mk != "x" else 1,
                       alpha=0.85, label=None if name in seen else f"{name} ({(labels == c).sum():,})")
            seen.add(name)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        _thousands(ax)
        k = len([c for c in style if c != -1])
        _title(ax, f"{self.specs[mid].name}: {k} clusters", how)
        _legend(ax, ncol=min(4, len(seen)))
        return ax

    def plot_silhouette(self, model_id: str | None = None, ax=None):
        """Silhouette value of every point, grouped by cluster. Long bars = well placed;
        values below 0 = the point sits closer to another cluster."""
        from sklearn.metrics import silhouette_samples
        mid = model_id or self.best_model_id()
        labels = self.labels(mid)
        mask = np.flatnonzero(labels != -1)
        if len(np.unique(labels[mask])) < 2:
            raise ValueError("need at least 2 clusters")
        if len(mask) > 5000:
            mask = np.random.default_rng(self.seed).choice(mask, 5000, replace=False)
        sil = silhouette_samples(self.Xt[mask], labels[mask])
        lab = labels[mask]
        style = self._cluster_style(labels)
        ax = _axes(ax, (7.5, 6), grid="x")
        y0, ticks, names = 0, [], []
        for c in sorted(c for c in style if c != -1):
            v = np.sort(sil[lab == c])
            ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=style[c][0], linewidth=0)
            ticks.append(y0 + len(v) / 2)
            names.append(f"Cluster {c}")
            y0 += len(v) + max(4, len(mask) // 60)
        avg = sil.mean()
        ax.axvline(avg, color=_INK2, linestyle="--", linewidth=1)
        ax.annotate(f"average {avg:.3f}", (avg, 1), xycoords=ax.get_xaxis_transform(), xytext=(4, -12),
                    textcoords="offset points", fontsize=8.5, color=_INK)
        ax.axvline(0, color=_AXIS, linewidth=1)
        if len(ticks) <= 20:
            ax.set_yticks(ticks, names)
        ax.set_ylim(0, y0)
        ax.invert_yaxis()
        ax.set_xlabel("Silhouette value")
        noise = (labels == -1).mean() * 100
        _title(ax, f"{self.specs[mid].name}: silhouette by cluster",
               "Points with values below 0 may be in the wrong cluster"
               + (f" - {noise:.1f}% noise not shown" if noise else ""))
        return ax

    def plot_profile(self, model_id: str | None = None, max_features: int = 25, ax=None):
        """Heatmap of the cluster profile: colour = how far each cluster's average sits
        from the overall average (in standard deviations, red above / blue below);
        the numbers are the actual averages in original units."""
        if not self.num_cols:
            raise ValueError("plot_profile() needs numeric columns - use profile() for categories")
        mid = model_id or self.best_model_id()
        labels = self.labels(mid)
        X = self.X[self.num_cols]
        means = X.groupby(labels).mean()
        z = (means - X.mean()) / X.std(ddof=0).replace(0, 1)
        if len(z.columns) > max_features:
            keep = z.abs().max().sort_values(ascending=False).index[:max_features]
            z, means = z[keep], means[keep]
        sizes = pd.Series(labels).value_counts()
        rows = [("Noise" if c == -1 else f"Cluster {c}") + f"  (n={sizes[c]:,})" for c in z.index]
        lim = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))
        nr, nc = z.shape
        ax = _axes(ax, (max(6, 1.25 * nc + 3), 0.55 * nr + 2), grid=None)
        im = ax.imshow(z.to_numpy(), cmap=_cmap(_DIV, "div"), vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(means.to_numpy()):
            zz = z.iat[i, j]
            ax.text(j, i, _fmt(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(zz) > 0.6 * lim else _INK)
        ax.set_xticks(range(nc), z.columns, rotation=30 if nc > 4 else 0, ha="right" if nc > 4 else "center")
        ax.set_yticks(range(nr), rows)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.set_label("SDs from the overall average", color=_INK2, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=_INK2, labelsize=8)
        _title(ax, f"{self.specs[mid].name}: cluster profiles",
               "Numbers = cluster averages in original units")
        return ax

    def plot_crosstab(self, model_id: str | None = None, ax=None):
        """Only with ground_truth: how the true labels spread across the clusters."""
        if self.y_true is None:
            raise ValueError("plot_crosstab() needs ground_truth= when creating the picker")
        mid = model_id or self.best_model_id()
        labels = self.labels(mid)
        ct = pd.crosstab(pd.Series(self.y_true, name="truth"), pd.Series(labels, name="cluster"))
        nr, nc = ct.shape
        ax = _axes(ax, (max(5, 0.9 * nc + 3), 0.6 * nr + 2), grid=None)
        ax.imshow(ct.to_numpy(), cmap=_cmap(_SEQ, "seq"), vmin=0, aspect="auto")
        mx = ct.to_numpy().max()
        for (i, j), v in np.ndenumerate(ct.to_numpy()):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v > 0.55 * mx else _INK)
        ax.set_xticks(range(nc), ["Noise" if c == -1 else str(c) for c in ct.columns])
        ax.set_yticks(range(nr), [str(i) for i in ct.index])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("True label")
        for s in ax.spines.values():
            s.set_visible(False)
        s = self._score(labels)
        _title(ax, f"{self.specs[mid].name}: clusters vs true labels",
               f"ARI {s['ARI']:.3f} - NMI {s['NMI']:.3f} (1 = perfect match)")
        return ax

    def summary(self) -> str:
        """Plain-text recap: data facts, starting values, winner and its hyperparameters."""
        c = self.ctx
        lines = [f"Rows: {c.n:,}   Features (encoded): {c.p}",
                 f"Categorical columns one-hot encoded: {self.cat_cols or 'none'}",
                 f"Starting n_clusters: {c.k} (searched 2..{c.k_max})   DBSCAN eps: {c.eps:.3f}   "
                 f"min_samples: {c.min_samples}"]
        if self.tuned_ is not None:
            best = self.tuned_.index[0]
            s = self._score(self._labels[best])
            lines += [f"\nBest model: {self.specs[best].name} ({best})  -  {s['Clusters']} clusters, "
                      f"{s['Noise %']:.1f}% noise",
                      "  " + "   ".join(f"{m}: {s[m]:.4f}" for m in [*_INTERNAL, *_EXTERNAL] if m in s),
                      "Recommended hyperparameters:"]
            lines += [f"  {k} = {v}" for k, v in self.recommended_params_[best].items()]
        return "\n".join(lines)


if __name__ == "__main__":  # quick demo:  python cluster_picker.py
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True).frame
    iris["species"] = iris.pop("target").map(dict(enumerate(load_iris().target_names)))
    cl = ClusterPicker(iris, ground_truth="species")
    print(cl.k_scan())
    print(cl.compare_models())
    print(cl.tune_top(n=3, n_iter=20))
    print(cl.summary())
    print(cl.profile())
    import matplotlib.pyplot as plt
    cl.plot_k_scan(); cl.plot_leaderboard(); cl.plot_tuning(); cl.plot_clusters()
    cl.plot_silhouette(); cl.plot_profile(); cl.plot_crosstab()
    plt.show()
