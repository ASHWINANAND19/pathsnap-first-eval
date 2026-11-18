"""Raw PathSnap data loading.

Each trip folder <trip_id>/ contains:
  <trip_id>.track  lon lat t         (GPS, ~1 Hz)
  <trip_id>.nodes  lon lat           (row index = node_id)
  <trip_id>.arcs   from_node to_node (row index = arc_id)
  <trip_id>.route  arc_id            (ground-truth arc sequence, chronological)
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PARTITIONS = ("train", "student_test")


def list_trips(partition: str) -> list[str]:
    return sorted(p.name for p in (DATA_DIR / partition).iterdir() if p.is_dir())


def load_trip(partition: str, trip_id: str) -> dict[str, pd.DataFrame]:
    d = DATA_DIR / partition / trip_id
    read = lambda ext, cols: pd.read_csv(d / f"{trip_id}.{ext}", sep=r"\s+", header=None, names=cols)
    track = read("track", ["lon", "lat", "t"])
    nodes = read("nodes", ["lon", "lat"])
    arcs = read("arcs", ["u", "v"])
    route_path = d / f"{trip_id}.route"
    route = read("route", ["arc_id"]) if route_path.exists() else None
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"
    return {"track": track, "nodes": nodes, "arcs": arcs, "route": route}
