"""Dataset inventory: one row per trip. Run: python -m src.data.inventory"""
import numpy as np
import pandas as pd

from src.data.loader import DATA_DIR, PARTITIONS, list_trips, load_trip


def trip_stats(partition: str, trip_id: str) -> dict:
    t = load_trip(partition, trip_id)
    tr, nodes, arcs, route = t["track"], t["nodes"], t["arcs"], t["route"]
    dt = np.diff(tr["t"].to_numpy())
    route_ids = route["arc_id"].to_numpy() if route is not None else np.array([])
    return {
        "partition": partition,
        "trip": trip_id,
        "gps_points": len(tr),
        "duration_s": tr["t"].iloc[-1] - tr["t"].iloc[0],
        "pct_dt_1s": float(np.mean(dt == 1) * 100),
        "max_gap_s": float(dt.max()),
        "non_monotonic_t": int((dt <= 0).sum()),
        "nan_rows": int(tr.isna().any(axis=1).sum()),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_len": len(route_ids),
        "route_arcs_valid": bool(((route_ids >= 0) & (route_ids < len(arcs))).all()),
        "lon_range": (tr.lon.min(), tr.lon.max()),
        "lat_range": (tr.lat.min(), tr.lat.max()),
    }


def main():
    rows = [trip_stats(p, tid) for p in PARTITIONS for tid in list_trips(p)]
    df = pd.DataFrame(rows)
    out = DATA_DIR.parent / "experiments" / "dataset_inventory.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("partition")[["gps_points", "duration_s", "max_gap_s", "nodes", "arcs", "route_len"]]
          .describe().T.round(1))
    print("\nNaN rows:", df.nan_rows.sum(), "| non-monotonic timestamps:", df.non_monotonic_t.sum(),
          "| invalid route arc ids:", (~df.route_arcs_valid).sum())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
