"""Train and compare the candidate classifiers.

Run:  python -m src.models.train_compare            (all models)
      python -m src.models.train_compare logreg     (only some)

Protocol (same for every model):
  * Training data: 80 train trips, only points whose true arc is in the candidate set
    (otherwise the point has no correct row and would only add misleading negatives),
    every 3rd GPS point (1 Hz points are near-duplicates; keeps KNN/RF tractable).
  * 5-fold GroupKFold with trip as the group -> no trip appears in both train and validation.
  * Refit on all selected train rows, then evaluate on the 10 student_test trips (ALL points).
Outputs: experiments/model_comparison.csv, experiments/logs/<model>.json, reports/figures/*.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_trip
from src.features.candidates import FEATURES
from src.models import metrics as M

ROOT = Path(__file__).resolve().parents[2]
EXP, FIG = ROOT / "experiments", ROOT / "reports" / "figures"
SEED = 42


def make_models():
    """name -> (family, sklearn estimator, short config string). Scaled where distance/gradient based."""
    return {
        "logreg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
                   "L2, C=1, balanced"),
        "gnb": ("Probabilistic", GaussianNB(), "default var_smoothing"),
        "knn": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
                "k=25, distance-weighted"),
        "rf": ("Ensemble (bagging)", RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5,
                                                             class_weight="balanced", n_jobs=-1, random_state=SEED),
               "200 trees, depth 14"),
        "lgbm": ("Boosting", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                            class_weight="balanced", random_state=SEED, verbose=-1),
                 "300 trees, lr 0.05, 31 leaves"),
        "mlp": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                               max_iter=60, random_state=SEED)),
                "64-32 ReLU, Adam, early stop"),
    }


def load_train():
    f = pd.read_parquet(EXP / "features_train.parquet")
    usable = f.groupby(["trip", "point_idx"]).label.transform("max") == 1
    f = f[usable & (f.point_idx % 3 == 0)]
    return f.reset_index(drop=True)


def evaluate_student_test(model, name):
    f = pd.read_parquet(EXP / "features_student_test.parquet")
    pts = pd.read_parquet(EXP / "points_student_test.parquet")
    t0 = time.time()
    proba = model.predict_proba(f[FEATURES].to_numpy())[:, 1]
    infer_s = time.time() - t0
    res = M.candidate_metrics(f.label.to_numpy(), proba)
    pp = M.attach_predictions(pts, M.top1_predictions(f, proba))
    res["top1_acc"] = M.top1_accuracy(pp)
    res["route_continuity"] = np.mean([
        M.route_continuity(g.pred_arc, g.seg_id, load_trip("student_test", t)["arcs"]) for t, g in pp.groupby("trip")])
    res["infer_ms_per_1k_rows"] = 1000 * infer_s / len(f) * 1000
    # confusion matrix (candidate level)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ConfusionMatrixDisplay.from_predictions(f.label, (proba >= 0.5).astype(int), ax=ax, colorbar=False,
                                            display_labels=["wrong arc", "true arc"], values_format="d")
    ax.set_title(f"{name} — student_test")
    plt.tight_layout(); plt.savefig(FIG / f"cm_{name}.png", dpi=120); plt.close()
    return res, proba


def run(name, family, model, config, train):
    X, y, g = train[FEATURES].to_numpy(), train.label.to_numpy(), train.trip.to_numpy()
    fold_f1 = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        model.fit(X[tr], y[tr])
        fold_f1.append(f1_score(y[va], model.predict(X[va]), average="macro"))
        print(f"  {name} fold {k}: macro F1 {fold_f1[-1]:.4f}")
    t0 = time.time()
    model.fit(X, y)
    train_s = time.time() - t0
    res, _ = evaluate_student_test(model, name)
    row = {"model": name, "family": family, "config": config,
           "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
           "train_s": train_s, **{f"st_{k}": v for k, v in res.items()}}
    (EXP / "logs" / f"{name}.json").write_text(json.dumps({**row, "cv_folds": fold_f1, "seed": SEED}, indent=2))
    if name == "mlp":  # only model with epoch-wise history
        mlp = model[-1]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(mlp.loss_curve_, label="training loss")
        ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy")
        ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax2.set_ylabel("val accuracy")
        fig.legend(loc="upper right"); plt.title("MLP training history"); plt.tight_layout()
        plt.savefig(FIG / "mlp_history.png", dpi=120); plt.close()
    return row


def plot_comparison(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(df.model, df.cv_macro_f1_mean, yerr=df.cv_macro_f1_std, capsize=4); ax[0].set_title("CV macro F1 (5-fold, by trip)")
    x = np.arange(len(df)); w = 0.4
    ax[1].bar(x - w / 2, df.st_top1_acc, w, label="Top-1 acc"); ax[1].bar(x + w / 2, df.st_macro_f1, w, label="Macro F1")
    ax[1].set_xticks(x, df.model); ax[1].legend(); ax[1].set_title("Student-test")
    ax[2].bar(df.model, df.st_log_loss, color="tab:red"); ax[2].set_title("Student-test log loss (lower = better)")
    for a in ax: a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(FIG / "model_comparison.png", dpi=120); plt.close()


def main(names):
    train = load_train()
    print(f"train rows: {len(train)}, trips: {train.trip.nunique()}, positive rate {train.label.mean():.3f}")
    models = make_models()
    out = EXP / "model_comparison.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for n in names or list(models):
        fam, mdl, cfg = models[n]
        print(f"== {n}")
        row = run(n, fam, mdl, cfg, train)
        prev = pd.concat([prev[prev.model != n] if len(prev) else prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(out, index=False)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    plot_comparison(prev)


if __name__ == "__main__":
    main(sys.argv[1:])
