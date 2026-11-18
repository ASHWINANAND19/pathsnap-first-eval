"""Evaluation metrics.

Candidate level (one row per point-candidate pair, binary label):
  macro F1, precision, recall, log loss  -> how well the classifier separates right/wrong arcs.
Observation level (one value per cleaned GPS point):
  Top-1 accuracy  -> pick the highest-probability candidate per point; correct if it equals the
                     point's true arc. Points with no candidates / candidate misses count as WRONG.
Route level:
  route continuity -> share of consecutive point pairs (same segment) whose predicted arcs are the
                      same arc or directly connected (prev.v == next.u). 1.0 = a drivable path.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_predictions(cands: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Best candidate per (trip, point). Returns trip, point_idx, pred_arc, pred_proba."""
    c = cands[["trip", "point_idx", "arc_id"]].copy()
    c["p"] = proba
    best = c.loc[c.groupby(["trip", "point_idx"]).p.idxmax()]
    return best.rename(columns={"arc_id": "pred_arc", "p": "pred_proba"}).reset_index(drop=True)


def attach_predictions(points: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Join top-1 predictions onto ALL cleaned points (missing -> pred_arc = -1)."""
    out = points[["trip", "seg_id", "true_arc"]].copy()
    out["point_idx"] = points.groupby("trip").cumcount().to_numpy()
    out = out.merge(best, on=["trip", "point_idx"], how="left")
    out["pred_arc"] = out.pred_arc.fillna(-1).astype(int)
    return out


def top1_accuracy(pred_points: pd.DataFrame, col="pred_arc") -> float:
    return float((pred_points[col] == pred_points.true_arc).mean())


def route_continuity(pred_arcs, seg_ids, arcs: pd.DataFrame) -> float:
    """Share of consecutive pairs (within a segment, both predicted) that are same/connected arcs."""
    a, s = np.asarray(pred_arcs), np.asarray(seg_ids)
    ok_pair = (a[:-1] >= 0) & (a[1:] >= 0) & (s[:-1] == s[1:])
    if ok_pair.sum() == 0:
        return float("nan")
    p, n = a[:-1][ok_pair], a[1:][ok_pair]
    same = p == n
    connected = arcs.v.to_numpy()[p] == arcs.u.to_numpy()[n]
    return float(np.mean(same | connected))
