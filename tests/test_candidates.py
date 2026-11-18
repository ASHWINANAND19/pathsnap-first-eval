import numpy as np
import pandas as pd

from src.data.geo import clean_track
from src.features.candidates import build_candidates, candidate_miss_rate

EPSG = 32632  # toy data sits near 9°E, 0°N... any zone works for relative distances


def _toy_trip():
    """Three parallel west->east roads 0.0005° (~55 m) apart; the car drives on the middle one."""
    lats = [0.0, 0.0005, 0.0010]
    nodes = pd.DataFrame({"lon": [9.0, 9.01] * 3, "lat": [la for la in lats for _ in range(2)]})
    arcs = pd.DataFrame({"u": [0, 2, 4, 3], "v": [1, 3, 5, 2]})  # arc 3 = middle road reversed
    route = pd.DataFrame({"arc_id": [1]})
    lon = np.linspace(9.001, 9.009, 20)
    track = pd.DataFrame({"lon": lon, "lat": 0.0005 + 0.00002, "t": np.arange(20.0)})
    return {"track": track, "nodes": nodes, "arcs": arcs, "route": route}


def test_split_on_gap():
    tr = pd.DataFrame({"lon": [9.0, 9.001, 9.002], "lat": [0.0, 0.0, 0.0], "t": [0.0, 1.0, 62.0]})
    assert clean_track(tr, EPSG).seg_id.tolist() == [0, 0, 1]


def test_stationary_filter():
    tr = pd.DataFrame({"lon": [9.0, 9.0000001, 9.001], "lat": [0.0, 0.0, 0.0], "t": [0.0, 1.0, 2.0]})
    assert clean_track(tr, EPSG).orig_idx.tolist() == [0, 2]


def test_label_is_nearest_route_arc():
    clean, c = build_candidates(_toy_trip(), R=50, k=8)
    # every point's single positive row is arc 1 (middle road, same direction as travel)
    assert (c[c.label == 1].arc_id == 1).all()
    assert c.groupby("point_idx").label.sum().eq(1).all()
    assert candidate_miss_rate(clean, c) == 0.0
    # reversed twin arc 3 has same distance but ~180° heading difference
    rev = c[c.arc_id == 3].heading_diff_deg
    assert (rev > 170).all()


def test_twin_tie_break_uses_direction_of_travel():
    trip = _toy_trip()
    trip["route"] = pd.DataFrame({"arc_id": [3, 1]})  # street driven both ways: both twins on route
    _, c = build_candidates(trip, R=50, k=8)
    assert (c[c.label == 1].arc_id == 1).all()        # car moves west->east = arc 1, not reversed arc 3
