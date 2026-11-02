# %% [markdown]
# # E-nose classification of areca-nut-blended coffee — revised, leakage-free analysis
#
# Revised analysis pipeline for the manuscript *"Electronic nose for classifying
# arabica, robusta, and liberica coffee blended with areca nut"*.
#
# Main changes compared with the original notebook:
# 1. **Sample-wise validation.** The 20 cycles of one brewed sample are never split
#    between training and testing (leave-one-sample-per-class-out, all 5^3 = 125
#    combinations). The original cycle-wise split is kept only as a comparison.
# 2. **Everything that learns from data is fitted inside the training fold**
#    (min-max scaling, ANOVA feature selection, drift/environment compensation,
#    hyper-parameter tuning via nested CV).
# 3. **DWT is applied per cycle** (300-s window), not on the 90,000-point
#    concatenated signal, so cycles/classes do not bleed into each other.
# 4. **New analyses:** variance decomposition, within-session drift, drift and
#    environment compensation (DEC), log-ratio pattern features, significance
#    tests, sample-level majority voting, sensor ablation and learning curve.
#
# **How to run (Colab / Jupyter):** upload `dataset_enose_kopi_pinang.csv` next to this notebook,
# run `pip install PyWavelets statsmodels` if needed, then *Run all*. Outputs (CSV tables and
# 300-dpi figures) are written to `results/`. A full run takes about 1.5–2 h on 2 CPU cores
# (nested CV over 125 splits + 1,000-permutation tests); set `ENOSE_QUICK=1` for a 3-minute
# smoke test (numbers from quick mode must not be reported). `ENOSE_JOBS` sets the number of cores.
#
# Dataset structure (`dataset_enose_kopi_pinang.csv`):
# 3 classes x 5 brewed samples x 20 cycles x 300 s (t = 201-500 s after sample
# insertion in each 1,000-s cycle) = 90,000 rows; 5 MQ sensors (Rs/Ro) plus
# DHT22 temperature (DHT_S, degC) and relative humidity (DHT_K, %).

# %% 1. Imports and configuration
import itertools, json, os, time, warnings
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"   # also silence joblib worker processes
SEED = 42
DATA = os.environ.get("ENOSE_DATA", "dataset_enose_kopi_pinang.csv")
OUT = os.environ.get("ENOSE_OUT", "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
N_JOBS = int(os.environ.get("ENOSE_JOBS", "2"))
N_PERM = int(os.environ.get("ENOSE_NPERM", "1000"))
QUICK = os.environ.get("ENOSE_QUICK") == "1"   # smoke-test mode (few splits/wavelets); never use for reported numbers

SENSORS = ["MQ2", "MQ4", "MQ7", "MQ135", "MQ137"]
CLASSES = ["Arabica", "Liberica", "Robusta"]
CLASS_MAP = {"Arabika": "Arabica", "Liberika": "Liberica", "Robusta": "Robusta"}
COLORS = {"Arabica": "#2a78d6", "Liberica": "#eb6834", "Robusta": "#1baf7a"}
MARKERS = {"Arabica": "o", "Liberica": "s", "Robusta": "^"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# %% 2. Load data and build the cycle-level table
df = pd.read_csv(DATA)
df["Class"] = df["Class"].map(CLASS_MAP)
df["cycle_global"] = df["Siklus"].str.split("_").str[1].astype(int)
df["cycle"] = (df["cycle_global"] - 1) % 20 + 1          # cycle index within a sample session (1-20)
df["sample"] = df["Sampel"].str.split("_").str[1].astype(int)
df = df.sort_values(["Class", "sample", "cycle", "Time"]).reset_index(drop=True)

sizes = df.groupby(["Class", "sample", "cycle"]).size()
assert sizes.nunique() == 1 and sizes.iloc[0] == 300, "every cycle must have 300 points"
assert len(sizes) == 300
log(f"rows={len(df)}, cycles={len(sizes)}, classes={df.Class.unique().tolist()}")

cycles = [(k, g) for k, g in df.groupby(["Class", "sample", "cycle"], sort=True)]
meta = pd.DataFrame([k for k, _ in cycles], columns=["Class", "sample", "cycle"])
meta["T"] = [g["DHT_S"].mean() for _, g in cycles]
meta["RH"] = [g["DHT_K"].mean() for _, g in cycles]
SIG = {s: np.vstack([g[s].to_numpy() for _, g in cycles]) for s in SENSORS}   # 300 cycles x 300 s
y = meta["Class"].to_numpy()
sample_id = meta["sample"].to_numpy()

dataset_summary = {
    "rows": int(len(df)), "cycles": 300, "points_per_cycle": 300,
    "T_mean": float(df.DHT_S.mean()), "T_sd": float(df.DHT_S.std()),
    "T_min": float(df.DHT_S.min()), "T_max": float(df.DHT_S.max()),
    "RH_mean": float(df.DHT_K.mean()), "RH_sd": float(df.DHT_K.std()),
    "RH_min": float(df.DHT_K.min()), "RH_max": float(df.DHT_K.max()),
    "RH_pct_saturated": float((df.DHT_K >= 100).mean() * 100),
}

# %% 3. DWT denoising (same IQR criterion and VisuShrink soft threshold as the original code, applied per cycle)
def calculate_iqr(x, y_rec, bins=100):
    """Information Quality Ratio (Wijaya et al., 2019): IQR = I(X;Y)/H(X,Y)."""
    joint, _, _ = np.histogram2d(x, y_rec, bins=bins)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    h_xy = -np.sum(p[nz] * np.log2(p[nz]))
    pxpy = (px * py)[nz]
    i_xy = np.sum(p[nz] * np.log2(p[nz] / pxpy))
    return 0.0 if np.isclose(h_xy, 0) else i_xy / h_xy

def dwt_denoise(x, wavelet, level):
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    # 12-bit quantisation can make the finest-scale MAD zero (thr = 0); pywt then returns 0/0 = NaN,
    # so a zero threshold is treated as "no shrinkage".
    den = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") if thr > 0 else c for c in coeffs[1:]]
    return pywt.waverec(den, wavelet, mode="symmetric")[: len(x)]

def search_dwt(sig2d):
    """Unsupervised grid search: wavelet/level maximising mean IQR over all cycles of one sensor."""
    best = (-np.inf, None, None)
    for w in (["db2", "sym4", "bior1.1"] if QUICK else pywt.wavelist(kind="discrete")):
        maxlev = pywt.dwt_max_level(sig2d.shape[1], pywt.Wavelet(w).dec_len)
        for lv in range(1, min(5, maxlev) + 1):
            score = np.mean([calculate_iqr(x, dwt_denoise(x, w, lv)) for x in sig2d])
            if score > best[0]:
                best = (score, w, lv)
    return best

dwt_params, SIG_DWT, noise_rows = {}, {}, []
for s in SENSORS:
    score, w, lv = search_dwt(SIG[s])
    dwt_params[s] = {"wavelet": w, "level": lv, "IQR": score}
    SIG_DWT[s] = np.vstack([dwt_denoise(x, w, lv) for x in SIG[s]])
    removed = SIG[s] - SIG_DWT[s]
    class_means = pd.Series(SIG[s].mean(1)).groupby(y).mean()
    noise_rows.append({
        "Sensor": s, "Wavelet": w, "Level": lv, "IQR": score,
        "removed_component_sd": removed.std(1).mean(),
        "within_cycle_sd": SIG[s].std(1).mean(),
        "max_between_class_mean_diff": class_means.max() - class_means.min(),
    })
    log(f"DWT {s}: {w} level {lv} IQR={score:.4f}")
dwt_table = pd.DataFrame(noise_rows)
dwt_table["noise_to_class_gap_pct"] = 100 * dwt_table.removed_component_sd / dwt_table.max_between_class_mean_diff
dwt_table.to_csv(os.path.join(OUT, "table_dwt_parameters.csv"), index=False)

# %% 4. Feature extraction (per cycle)
def stat_features(sig):
    cols, out = [], []
    for s in SENSORS:
        X = sig[s]
        out += [X.mean(1), X.std(1, ddof=1), X.min(1), X.max(1)]
        cols += [f"{s}_mean", f"{s}_std", f"{s}_min", f"{s}_max"]
    return pd.DataFrame(np.column_stack(out), columns=cols)

F_STAT = stat_features(SIG)
F_STAT_DWT = stat_features(SIG_DWT)
M = np.column_stack([SIG[s].mean(1) for s in SENSORS])
logM = np.log(M)
F_LR = pd.DataFrame(logM - logM.mean(1, keepdims=True), columns=[f"{s}_clr" for s in SENSORS])  # centred log-ratio
COV = meta[["T", "RH", "cycle"]].to_numpy(dtype=float)                                          # covariates for DEC
F_STAT.assign(Class=y, sample=sample_id, cycle=meta.cycle, T=meta["T"], RH=meta.RH).to_csv(
    os.path.join(OUT, "features_cycle_level.csv"), index=False)

# %% 5. Exploratory analysis: variance decomposition and within-session drift
vd_rows = []
for s in SENSORS:
    v = pd.Series(M[:, SENSORS.index(s)])
    cm = v.groupby(y).transform("mean")
    sm = v.groupby([y, sample_id]).transform("mean")
    tot = ((v - v.mean()) ** 2).sum()
    within = v - sm
    rh_dev = meta.RH - meta.RH.groupby([y, sample_id]).transform("mean")
    vd_rows.append({
        "Sensor": s,
        "Between-class (%)": 100 * ((cm - v.mean()) ** 2).sum() / tot,
        "Between-sample within class (%)": 100 * ((sm - cm) ** 2).sum() / tot,
        "Within-sample, across cycles (%)": 100 * (within ** 2).sum() / tot,
        "r(within, cycle)": np.corrcoef(within, meta.cycle)[0, 1],
        "r(within, RH)": np.corrcoef(within, rh_dev)[0, 1],
    })
var_table = pd.DataFrame(vd_rows)
var_table.to_csv(os.path.join(OUT, "table_variance_decomposition.csv"), index=False)

drift = meta.assign(**{s: M[:, i] for i, s in enumerate(SENSORS)})
drift_table = drift.groupby("cycle")[SENSORS + ["T", "RH"]].mean()
drift_table.to_csv(os.path.join(OUT, "table_drift_by_cycle.csv"))
first_last = drift.groupby(["Class", "sample"]).apply(
    lambda g: pd.Series({s: g.loc[g.cycle == 20, s].values[0] / g.loc[g.cycle == 1, s].values[0] for s in SENSORS}))
drift_increase_pct = (100 * (first_last.mean() - 1)).round(1).to_dict()
# class separability vs cycle: one-way ANOVA F of MQ4 mean per cycle index
f_by_cycle = {c: float(np.mean([stats.f_oneway(*[drift.loc[(drift.cycle == c) & (drift.Class == k), s] for k in CLASSES]).statistic
                                for s in SENSORS])) for c in range(1, 21)}
log("variance decomposition:\n" + var_table.round(2).to_string(index=False))

# %% 6. Drift/environment compensation transformer and pipelines
class DEC(BaseEstimator, TransformerMixin):
    """Drift & Environment Compensation.

    The last 3 columns of X are covariates (T, RH, cycle index). In fit(), each
    feature is centred on its *training* class mean and regressed on the
    covariates (within-class OLS), so that class information is not removed.
    transform() subtracts the covariate-explained part and drops the covariates.
    Only training data is used to estimate the coefficients.
    """
    def __init__(self, enabled=True, n_cov=3):
        self.enabled, self.n_cov = enabled, n_cov
    def fit(self, X, y=None):
        X = np.asarray(X, float); F, Z = X[:, :-self.n_cov], X[:, -self.n_cov:]
        if self.enabled:
            centred = F - pd.DataFrame(F).groupby(np.asarray(y)).transform("mean").to_numpy()
            self.reg_ = LinearRegression().fit(Z, centred)
        return self
    def transform(self, X):
        X = np.asarray(X, float); F, Z = X[:, :-self.n_cov], X[:, -self.n_cov:]
        return F - self.reg_.predict(Z) if self.enabled else F

MODELS = {
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7, 9, 15]}),
    "SVM": (SVC(kernel="rbf"), {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.1, 1]}),
    "ANN": (MLPClassifier(hidden_layer_sizes=(24,), activation="relu", solver="lbfgs", max_iter=500, random_state=SEED),
            {"clf__alpha": [0.01, 0.1, 1.0]}),
    "LDA": (LinearDiscriminantAnalysis(solver="lsqr"), {"clf__shrinkage": [None, "auto"]}),
}
if QUICK:
    MODELS.pop("ANN")
STRATEGIES = {   # name: (feature matrix, apply DEC, use ANOVA selection)
    "Stat":          (F_STAT,     False, True),
    "Stat+DWT":      (F_STAT_DWT, False, True),
    "Stat+DEC":      (F_STAT,     True,  True),
    "Stat+DWT+DEC":  (F_STAT_DWT, True,  True),
    "LogRatio":      (F_LR,       False, False),
    "LogRatio+DEC":  (F_LR,       True,  False),
}

def make_pipe(model, dec, anova):
    steps = [("dec", DEC(enabled=dec)), ("scale", MinMaxScaler())]
    if anova:
        steps.append(("anova", SelectKBest(f_classif)))
    steps.append(("clf", clone(model)))
    return Pipeline(steps)

def make_grid(model_name, anova):
    g = dict(MODELS[model_name][1])
    if anova:
        g["anova__k"] = [5, 10, 20] if model_name != "ANN" else [10, 20]
    return g

# %% 7. Validation schemes
def sample_wise_splits():
    """Leave-one-sample-per-class-out: every combination of one held-out sample per class (5^3 = 125)."""
    splits = []
    for combo in itertools.product(range(1, 6), repeat=3):
        test = np.zeros(len(y), bool)
        for cls, s in zip(CLASSES, combo):
            test |= (y == cls) & (sample_id == s)
        splits.append((np.where(~test)[0], np.where(test)[0], combo))
    return splits

def inner_sample_cv(train_idx):
    """Inner CV for tuning: hold out one remaining sample per class (4 folds)."""
    rank = np.zeros(len(train_idx), int)
    for cls in CLASSES:
        m = y[train_idx] == cls
        uniq = np.unique(sample_id[train_idx][m])
        rank[m] = np.searchsorted(uniq, sample_id[train_idx][m])
    return [(np.where(rank != r)[0], np.where(rank == r)[0]) for r in range(4)]

SW_SPLITS = sample_wise_splits()
if QUICK:
    SW_SPLITS = SW_SPLITS[::31]
CW_SPLITS = [(tr, te, None) for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(F_STAT, y)]

def run_fold(strategy, model_name, tr, te, scheme):
    F, dec, anova = STRATEGIES[strategy]
    X = np.hstack([F.to_numpy(), COV])
    inner = inner_sample_cv(tr) if scheme == "sample-wise" else StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipe(MODELS[model_name][0], dec, anova), make_grid(model_name, anova),
                      cv=inner, scoring="accuracy", n_jobs=1, refit=True)
    gs.fit(X[tr], y[tr])
    pred = gs.predict(X[te])
    return {"pred": pred, "test": te, "acc": float(np.mean(pred == y[te])), "params": gs.best_params_}

# %% 8. Run all evaluations (this is the slow part)
RESULTS = {}
for scheme, splits in [("cycle-wise", CW_SPLITS), ("sample-wise", SW_SPLITS)]:
    for strategy in STRATEGIES:
        for model_name in MODELS:
            t = time.time()
            folds = Parallel(n_jobs=N_JOBS)(delayed(run_fold)(strategy, model_name, tr, te, scheme) for tr, te, _ in splits)
            RESULTS[(scheme, strategy, model_name)] = folds
            accs = [f["acc"] for f in folds]
            log(f"{scheme:11s} {strategy:13s} {model_name}: {np.mean(accs):.4f} ± {np.std(accs):.4f} ({time.time()-t:.0f}s)")

# %% 9. Summary tables
def per_class_metrics(folds):
    yt = np.concatenate([y[f["test"]] for f in folds]); yp = np.concatenate([f["pred"] for f in folds])
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
    return yt, yp, p, r, f1

rows = []
for (scheme, strategy, model_name), folds in RESULTS.items():
    accs = np.array([f["acc"] for f in folds])
    yt, yp, p, r, f1 = per_class_metrics(folds)
    row = {"Scheme": scheme, "Strategy": strategy, "Model": model_name,
           "Acc_mean": accs.mean(), "Acc_sd": accs.std(ddof=1),
           "Acc_CI95_low": np.percentile(accs, 2.5), "Acc_CI95_high": np.percentile(accs, 97.5),
           "Macro_F1": f1.mean()}
    for c, pp, rr, ff in zip(CLASSES, p, r, f1):
        row.update({f"{c}_P": pp, f"{c}_R": rr, f"{c}_F1": ff})
    if scheme == "sample-wise":   # sample-level decision by majority vote over the 20 cycles
        votes = []
        for f in folds:
            for cls in CLASSES:
                m = y[f["test"]] == cls
                vals, cnt = np.unique(f["pred"][m], return_counts=True)
                votes.append(vals[np.argmax(cnt)] == cls)
        row["Vote_acc"] = np.mean(votes)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, "table_all_results.csv"), index=False)
log("\n" + summary.pivot_table(index=["Strategy"], columns=["Scheme", "Model"], values="Acc_mean").round(3).to_string())

# %% 10. Statistical tests
tests = []
def paired(scheme, a, b, model_name):
    x = np.array([f["acc"] for f in RESULTS[(scheme, a, model_name)]])
    z = np.array([f["acc"] for f in RESULTS[(scheme, b, model_name)]])
    d = z - x
    p = stats.wilcoxon(z, x, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    return {"Scheme": scheme, "Comparison": f"{b} vs {a}", "Model": model_name,
            "Mean_diff_pp": 100 * d.mean(), "Wilcoxon_p": p}
for m in MODELS:
    for a, b in [("Stat", "Stat+DWT"), ("Stat", "Stat+DEC"), ("LogRatio", "LogRatio+DEC"), ("Stat+DEC", "Stat+DWT+DEC")]:
        tests.append(paired("sample-wise", a, b, m))
    # McNemar on cycle-wise out-of-fold predictions (each cycle predicted exactly once)
    def oof(strategy):
        out = np.empty(len(y), object)
        for f in RESULTS[("cycle-wise", strategy, m)]:
            out[f["test"]] = f["pred"]
        return out == y
    a_ok, b_ok = oof("Stat"), oof("Stat+DWT")
    tbl = [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)], [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]]
    tests.append({"Scheme": "cycle-wise", "Comparison": "Stat+DWT vs Stat (McNemar)", "Model": m,
                  "Mean_diff_pp": 100 * (b_ok.mean() - a_ok.mean()), "McNemar_p": mcnemar(tbl, exact=True).pvalue,
                  "discordant": f"{tbl[0][1]}/{tbl[1][0]}"})
tests = pd.DataFrame(tests)
tests.to_csv(os.path.join(OUT, "table_statistical_tests.csv"), index=False)
log("\n" + tests.round(4).to_string(index=False))

def modal_params(key):
    ps = [json.dumps(f["params"], sort_keys=True, default=str) for f in RESULTS[key]]
    return json.loads(pd.Series(ps).mode()[0])

# %% 11. Permutation test (labels permuted at sample level, i.e. whole samples swap classes)
def fixed_pipe(strategy, model_name):
    F, dec, anova = STRATEGIES[strategy]
    pipe = make_pipe(MODELS[model_name][0], dec, anova)
    params = modal_params(("sample-wise", strategy, model_name))
    pipe.set_params(**params)
    return pipe, np.hstack([F.to_numpy(), COV]), params

def sw_accuracy(pipe, X, labels, splits=SW_SPLITS):
    return np.mean([np.mean(clone(pipe).fit(X[tr], labels[tr]).predict(X[te]) == labels[te]) for tr, te, _ in splits])

best_sw = summary[summary.Scheme == "sample-wise"].sort_values("Acc_mean", ascending=False)
PERM_CONFIGS = [("Stat", "SVM")] + [tuple(r) for r in best_sw[["Strategy", "Model"]].head(2).to_numpy()]
PERM_CONFIGS = list(dict.fromkeys(PERM_CONFIGS))
rng = np.random.default_rng(SEED)
sample_keys = [(c, s) for c in CLASSES for s in range(1, 6)]
key_idx = {k: np.where((y == k[0]) & (sample_id == k[1]))[0] for k in sample_keys}
perm_rows = []
for strategy, model_name in PERM_CONFIGS:
    pipe, X, params = fixed_pipe(strategy, model_name)
    obs = sw_accuracy(pipe, X, y)
    def one(seed):
        r = np.random.default_rng(seed)
        perm_labels = r.permutation([k[0] for k in sample_keys])
        yp = y.copy()
        # permuted labels, then re-index samples 1..5 within each new class so the 125-split design is preserved
        new_sid = sample_id.copy(); counter = {c: 0 for c in CLASSES}
        for k, lab in zip(sample_keys, perm_labels):
            counter[lab] += 1; yp[key_idx[k]] = lab; new_sid[key_idx[k]] = counter[lab]
        splits = []
        for combo in itertools.product(range(1, 6), repeat=3):
            test = np.zeros(len(y), bool)
            for cls, s in zip(CLASSES, combo):
                test |= (yp == cls) & (new_sid == s)
            splits.append((np.where(~test)[0], np.where(test)[0], combo))
        return sw_accuracy(pipe, X, yp, splits)
    null = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(s) for s in rng.integers(0, 1e9, N_PERM)))
    p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
    perm_rows.append({"Strategy": strategy, "Model": model_name, "Params": json.dumps(params, default=str),
                      "Observed_acc": obs, "Null_mean": null.mean(), "Null_95pct": np.percentile(null, 95), "p_value": p})
    log(f"permutation {strategy}/{model_name}: obs={obs:.4f} null={null.mean():.4f} p={p:.4f}")
perm_table = pd.DataFrame(perm_rows)
perm_table.to_csv(os.path.join(OUT, "table_permutation_test.csv"), index=False)

# %% 12. Sensor ablation and learning curve (fixed hyper-parameters, sample-wise validation)
BEST_STRAT, BEST_MODEL = best_sw.iloc[0]["Strategy"], best_sw.iloc[0]["Model"]
def subset_features(strategy, sensors):
    F, dec, anova = STRATEGIES[strategy]
    if strategy.startswith("LogRatio"):
        lm = np.column_stack([logM[:, SENSORS.index(s)] for s in sensors])
        return lm - lm.mean(1, keepdims=True), dec, False
    cols = [c for c in F.columns if c.split("_")[0] in sensors]
    return F[cols].to_numpy(), dec, anova
ablation = []
for strategy, model_name in dict.fromkeys([(BEST_STRAT, BEST_MODEL), ("Stat+DEC", "SVM")]):
    params = modal_params(("sample-wise", strategy, model_name))
    for k in range(1, 6):
        for sub in itertools.combinations(SENSORS, k):
            if strategy.startswith("LogRatio") and k < 2:
                continue
            Fs, dec, anova = subset_features(strategy, list(sub))
            pipe = make_pipe(MODELS[model_name][0], dec, anova)
            p = {kk: v for kk, v in params.items() if kk != "anova__k"}
            if anova:
                p["anova__k"] = min(params.get("anova__k", 10), Fs.shape[1])
            pipe.set_params(**p)
            acc = sw_accuracy(pipe, np.hstack([Fs, COV]), y)
            ablation.append({"Strategy": strategy, "Model": model_name, "n_sensors": k, "Sensors": "+".join(sub), "Acc": acc})
ablation = pd.DataFrame(ablation)
ablation.to_csv(os.path.join(OUT, "table_sensor_ablation.csv"), index=False)
log("ablation top:\n" + ablation.sort_values("Acc", ascending=False).groupby(["Strategy", "n_sensors"]).head(1).to_string(index=False))

lc_rows = []
for strategy, model_name in dict.fromkeys([(BEST_STRAT, BEST_MODEL), ("Stat+DEC", "SVM"), ("Stat", "SVM")]):
    pipe, X, _ = fixed_pipe(strategy, model_name)
    for n in range(1, 5):
        accs = []
        for tr_all, te, combo in SW_SPLITS:
            r = np.random.default_rng(hash(combo) % (2**32))
            keep = np.zeros(len(y), bool)
            for cls, held in zip(CLASSES, combo):
                pool = [s for s in range(1, 6) if s != held]
                for s in r.choice(pool, n, replace=False):
                    keep |= (y == cls) & (sample_id == s)
            tr = np.where(keep)[0]
            if n == 1 and model_name == "KNN":
                continue
            accs.append(np.mean(clone(pipe).fit(X[tr], y[tr]).predict(X[te]) == y[te]))
        lc_rows.append({"Strategy": strategy, "Model": model_name, "n_train_samples_per_class": n,
                        "Acc_mean": np.mean(accs), "Acc_sd": np.std(accs, ddof=1)})
lc = pd.DataFrame(lc_rows)
lc.to_csv(os.path.join(OUT, "table_learning_curve.csv"), index=False)
log("\n" + lc.round(4).to_string(index=False))

# %% 13. Figures
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)

# Fig. 3 — example raw vs DWT signal (one cycle per class, MQ4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
t = np.arange(201, 501)
for cls in CLASSES:
    i = np.where((y == cls) & (sample_id == 1) & (meta.cycle.to_numpy() == 10))[0][0]
    ax1.plot(t, SIG["MQ4"][i], color=COLORS[cls], lw=0.8, alpha=0.35)
    ax1.plot(t, SIG_DWT["MQ4"][i], color=COLORS[cls], lw=1.6, label=cls)
ax1.set_xlabel("Time after sample insertion (s)"); ax1.set_ylabel("MQ-4 response (Rs/Ro)")
ax1.set_title("(a) One cycle per class (sample 1, cycle 10)", fontsize=9); ax1.legend(fontsize=7, loc="center right")
i = np.where((y == "Arabica") & (sample_id == 1) & (meta.cycle.to_numpy() == 10))[0][0]
zoom = (t >= 260) & (t <= 320)
ax2.plot(t[zoom], SIG["MQ4"][i][zoom], color="#9a9890", lw=1.0, marker=".", ms=3, label="Raw")
ax2.plot(t[zoom], SIG_DWT["MQ4"][i][zoom], color=COLORS["Arabica"], lw=1.8,
         label=f"DWT ({dwt_params['MQ4']['wavelet']}, level {dwt_params['MQ4']['level']})")
ax2.set_xlabel("Time after sample insertion (s)"); ax2.set_title("(b) Zoom, Arabica, 260–320 s", fontsize=9)
ax2.legend(fontsize=7, loc="best")
fig.tight_layout(); savefig(fig, "fig3_raw_vs_dwt_MQ4.png")

# Fig. 4 — within-session drift: sensor response and RH vs cycle (two panels, no dual axis)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, s in zip(axes[:2], ["MQ2", "MQ4"]):
    for cls in CLASSES:
        d = drift[drift.Class == cls].groupby("cycle")[s]
        mu, sd = d.mean(), d.std()
        ax.fill_between(mu.index, mu - sd, mu + sd, color=COLORS[cls], alpha=0.12, lw=0)
        ax.plot(mu.index, mu, color=COLORS[cls], lw=2, marker=MARKERS[cls], ms=4, label=cls)
    ax.set_title(f"{s[:2]}-{s[2:]} mean response", fontsize=9); ax.set_xlabel("Cycle index within session"); ax.set_ylabel("Rs/Ro")
axes[0].legend(fontsize=7)
rh = drift.groupby("cycle")[["RH"]].agg(["mean", "std"])["RH"]
axes[2].fill_between(rh.index, rh["mean"] - rh["std"], rh["mean"] + rh["std"], color="#9ec5f4", alpha=0.4, lw=0)
axes[2].plot(rh.index, rh["mean"], color="#2a78d6", lw=2)
axes[2].set_title("Chamber relative humidity", fontsize=9); axes[2].set_xlabel("Cycle index within session"); axes[2].set_ylabel("RH (%)")
fig.tight_layout(); savefig(fig, "fig4_drift_by_cycle.png")

# Fig. 5 — variance decomposition (stacked horizontal bars)
fig, ax = plt.subplots(figsize=(5.2, 2.4))
comp = ["Between-class (%)", "Between-sample within class (%)", "Within-sample, across cycles (%)"]
ccol = ["#2a78d6", "#eda100", "#9a9890"]
left = np.zeros(len(SENSORS))
for c, col in zip(comp, ccol):
    vals = var_table[c].to_numpy()
    ax.barh([s.replace("MQ", "MQ-") for s in SENSORS], vals, left=left, color=col, edgecolor="white", linewidth=1.5,
            label=c.replace(" (%)", ""))
    for yy, (l, v) in enumerate(zip(left, vals)):
        if v > 9:
            ax.text(l + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
    left += vals
ax.set_xlim(0, 100); ax.set_xlabel("Share of total variance of the mean response (%)"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
savefig(fig, "fig5_variance_decomposition.png")

# Fig. 6 — PCA score plots before/after DEC (fitted on all data, for visualisation only)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
Xraw = StandardScaler().fit_transform(F_STAT)
Xdec = StandardScaler().fit_transform(DEC().fit(np.hstack([F_STAT.to_numpy(), COV]), y).transform(np.hstack([F_STAT.to_numpy(), COV])))
for ax, Xp, title in [(axes[0], Xraw, "Statistical features (raw)"), (axes[1], Xdec, "After drift/environment compensation")]:
    pca = PCA(2).fit(Xp); Z = pca.transform(Xp)
    for cls in CLASSES:
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=12, color=COLORS[cls], marker=MARKERS[cls], alpha=0.75, edgecolor="white", linewidth=0.4, label=cls)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
axes[0].legend(fontsize=7)
fig.tight_layout(); savefig(fig, "fig6_pca.png")

# Fig. 7 — accuracy: cycle-wise vs sample-wise, by strategy and model
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
model_cols = {"KNN": "#2a78d6", "SVM": "#eb6834", "ANN": "#1baf7a", "LDA": "#eda100"}
n_mod = len(MODELS)
strats = list(STRATEGIES)
for ax, scheme in zip(axes, ["cycle-wise", "sample-wise"]):
    for j, m in enumerate(MODELS):
        sub = summary[(summary.Scheme == scheme) & (summary.Model == m)].set_index("Strategy").loc[strats]
        xs = np.arange(len(strats)) + (j - (n_mod - 1) / 2) * 0.18
        ax.errorbar(xs, 100 * sub.Acc_mean, yerr=100 * sub.Acc_sd, fmt="o",
                    color=model_cols[m], ms=4, lw=1, capsize=0, label=m)
    ax.axhline(100 / 3, color=MUTED, lw=1, ls="--"); ax.text(len(strats) - 0.6, 100 / 3 + 1.5, "chance", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(range(len(strats))); ax.set_xticklabels(strats, rotation=35, ha="right", fontsize=7)
    ax.set_title(f"{scheme.capitalize()} validation", fontsize=9); ax.set_ylim(20, 100)
axes[0].set_ylabel("Accuracy (%) — mean ± SD over folds"); axes[0].legend(fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(); savefig(fig, "fig7_accuracy_comparison.png")

# Fig. 8 — confusion matrices (best sample-wise configuration vs original Stat+SVM), pooled over the 125 splits
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
for ax, key in zip(axes, [("sample-wise", "Stat", "SVM"), ("sample-wise", BEST_STRAT, BEST_MODEL)]):
    yt, yp, *_ = per_class_metrics(RESULTS[key])
    cm = confusion_matrix(yt, yp, labels=CLASSES, normalize="true") * 100
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        os.path.join(OUT, f"confusion_{key[1]}_{key[2]}.csv".replace("+", "-")))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=7); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    ax.set_title(f"{key[1]} + {key[2]}", fontsize=9)
fig.tight_layout(); savefig(fig, "fig8_confusion_matrices.png")

# Fig. 9 — learning curve
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for (strategy, model_name), g in lc.groupby(["Strategy", "Model"], sort=False):
    col = {"Stat": "#9a9890", "Stat+DEC": "#eb6834"}.get(strategy, "#2a78d6")
    ax.errorbar(g.n_train_samples_per_class, 100 * g.Acc_mean, yerr=100 * g.Acc_sd, color=col, marker="o", ms=4, lw=1.5,
                capsize=0, label=f"{strategy} + {model_name}")
ax.axhline(100 / 3, color=MUTED, lw=1, ls="--")
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Independent training samples per class"); ax.set_ylabel("Sample-wise accuracy (%)")
ax.legend(fontsize=7)
savefig(fig, "fig9_learning_curve.png")

# %% 14. Save key numbers for the manuscript
key = {
    "dataset": dataset_summary,
    "dwt_params": dwt_params,
    "drift_increase_cycle1_to_20_pct": drift_increase_pct,
    "anova_F_by_cycle_mean_over_sensors": f_by_cycle,
    "best_sample_wise": {"strategy": BEST_STRAT, "model": BEST_MODEL},
    "runtime_s": time.time() - T0,
}
with open(os.path.join(OUT, "key_numbers.json"), "w") as fh:
    json.dump(key, fh, indent=2, default=float)
log("done")
