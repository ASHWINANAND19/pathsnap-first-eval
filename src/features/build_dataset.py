"""Build the candidate-level ML dataset for train and student_test.

Run: python -m src.features.build_dataset
Outputs:
  experiments/features_<partition>.parquet  one row per (GPS point, candidate arc)
  experiments/points_<partition>.parquet    one row per cleaned GPS point (true arc, distance)
  experiments/candidate_stats.csv           per-trip candidate miss rate etc.
"""
import time
from pathlib import Path

import pandas as pd

from src.data.loader import list_trips, load_trip
from src.features.candidates import build_candidates, candidate_miss_rate

OUT = Path(__file__).resolve().parents[2] / "experiments"


def main():
    stats = []
    for part in ["train", "student_test"]:
        feats, points = [], []
        for tid in list_trips(part):
            t0 = time.time()
            clean, c = build_candidates(load_trip(part, tid))
            c.insert(0, "trip", tid)
            clean.insert(0, "trip", tid)
            feats.append(c)
            points.append(clean)
            stats.append({
                "partition": part, "trip": tid, "points": len(clean), "rows": len(c),
                "pts_no_candidates": len(clean) - c.point_idx.nunique(),
                "miss_rate": candidate_miss_rate(clean, c),
                "median_true_dist_m": clean.true_dist_m.median(),
                "seconds": round(time.time() - t0, 2),
            })
            print(f"{part}/{tid}: {len(clean)} pts, {len(c)} rows, miss={stats[-1]['miss_rate']:.3f}, {stats[-1]['seconds']}s")
        pd.concat(feats).to_parquet(OUT / f"features_{part}.parquet", index=False)
        pd.concat(points).to_parquet(OUT / f"points_{part}.parquet", index=False)
    s = pd.DataFrame(stats)
    s.to_csv(OUT / "candidate_stats.csv", index=False)
    w = lambda g: (g.miss_rate * g.points).sum() / g.points.sum()
    print("\nPoint-weighted candidate miss rate:", s.groupby("partition").apply(w, include_groups=False).round(4).to_dict())
    print("Worst trips:\n", s.sort_values("miss_rate", ascending=False).head(6).to_string(index=False))


if __name__ == "__main__":
    main()
