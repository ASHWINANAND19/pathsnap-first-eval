"""Candidate generation + labelling (option A) + per-candidate features.

For every cleaned GPS point we collect the road arcs within R metres (max K
nearest). Each (point, candidate arc) pair becomes ONE ROW of the ML dataset.

LABEL (option A, "nearest route arc"):
  The .route file is an ordered list of arcs, not one label per GPS point.
  For each GPS point we find the route arc that is physically closest to it;
  that arc is the point's "true arc". The candidate row whose arc_id equals the
  true arc gets label 1, all others 0.
  If the true arc is not among the candidates (too far / outside K) the point is
  a "candidate miss": no row can be correct, so the classifier cannot fix it.

Note on two-way roads: most streets are stored as two arcs (u->v and v->u) with
identical geometry and identical distance. Only the heading feature can tell them
apart, which is why heading_diff_deg is important.
"""
import numpy as np
import pandas as pd
import shapely

from src.data.geo import clean_track, project, trip_epsg

R_M = 50.0  # search radius (m). EDA: median GPS->route distance 7 m, 90th pct 34 m
K = 8       # max candidates per point

FEATURES = [
    "dist_m",            # perpendicular distance GPS point -> arc
    "dist_rank",         # 0 = closest candidate of this point
    "dist_minus_min",    # extra distance compared with the closest candidate
    "heading_diff_deg",  # angle between GPS direction of travel and arc direction u->v (0..180)
    "arc_len_m",         # length of the arc
    "speed_mps",         # GPS speed at this point
    "n_candidates",      # how crowded the area is (junctions have many)
    "frac_along",        # where the projection falls on the arc (0=start u, 1=end v)
]


def _arc_geometry(nodes_xy, arcs):
    """Vectorised LineStrings for every arc, plus arc bearings (deg) and lengths (m)."""
    u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
    a, b = nodes_xy[u], nodes_xy[v]
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    d = b - a
    bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360  # 0 = north, clockwise
    length = np.hypot(d[:, 0], d[:, 1])
    return lines, bearing, length


def _gps_motion(clean):
    """Heading (deg, 0=north) and speed (m/s) per point from neighbours in the same segment."""
    x, y, t, seg = clean.x.to_numpy(), clean.y.to_numpy(), clean.t.to_numpy(), clean.seg_id.to_numpy()
    n = len(x)
    prev = np.arange(n) - 1
    nxt = np.arange(n) + 1
    prev[0] = 0
    nxt[-1] = n - 1
    # do not look across a segment break
    prev = np.where(seg[prev] == seg, prev, np.arange(n))
    nxt = np.where(seg[nxt] == seg, nxt, np.arange(n))
    dx, dy = x[nxt] - x[prev], y[nxt] - y[prev]
    dt = np.maximum(t[nxt] - t[prev], 1e-6)
    heading = np.degrees(np.arctan2(dx, dy)) % 360
    speed = np.hypot(dx, dy) / dt
    return heading, speed


TIE_M = 1.0  # route arcs within this many metres of the nearest one count as "equally close"


def true_arcs(clean, nodes_xy, arcs, route):
    """Option A label: for each point, the closest arc on the reference route and its distance.

    Tie-break: if the car drove a street in both directions, the route contains both twin arcs
    (u->v and v->u) with identical geometry, so "closest" is a coin-flip. Among route arcs within
    TIE_M of the closest one, we pick the one whose direction best matches the GPS direction of travel.
    (Without this, ~8% of labels pointed backwards.)
    """
    route_ids = np.unique(route.arc_id.to_numpy())
    u, v = arcs.u.to_numpy()[route_ids], arcs.v.to_numpy()[route_ids]
    a, b = nodes_xy[u], nodes_xy[v]
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    bearing = np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) % 360
    pts = shapely.points(clean.x.to_numpy(), clean.y.to_numpy())
    tree = shapely.STRtree(lines)
    nearest = tree.nearest(pts)
    d0 = shapely.distance(pts, lines[nearest])

    # all route arcs that are (almost) as close as the nearest one
    p_idx, l_idx = tree.query(pts, predicate="dwithin", distance=float(d0.max()) + TIE_M)
    d = shapely.distance(pts[p_idx], lines[l_idx])
    keep = d <= d0[p_idx] + TIE_M
    p_idx, l_idx = p_idx[keep], l_idx[keep]
    heading, _ = _gps_motion(clean)
    hd = np.abs(heading[p_idx] - bearing[l_idx]) % 360
    hd = np.minimum(hd, 360 - hd)
    tie = pd.DataFrame({"p": p_idx, "l": l_idx, "hd": hd}).sort_values(["p", "hd"], kind="mergesort")
    best = tie.drop_duplicates("p").set_index("p").l
    chosen = nearest.copy()
    chosen[best.index.to_numpy()] = best.to_numpy()
    return route_ids[chosen], d0


def build_candidates(trip, R=R_M, k=K, with_labels=True):
    """Return (clean_track_df, candidates_df) for one trip dict from loader.load_trip.

    candidates_df columns: point_idx, seg_id, arc_id, FEATURES..., and if labels: label, true_arc.
    """
    epsg = trip_epsg(trip["track"])
    clean = clean_track(trip["track"], epsg)
    nodes, arcs = trip["nodes"], trip["arcs"]
    nodes_xy = np.column_stack(project(nodes.lon.to_numpy(), nodes.lat.to_numpy(), epsg))
    lines, arc_bearing, arc_len = _arc_geometry(nodes_xy, arcs)

    pts = shapely.points(clean.x.to_numpy(), clean.y.to_numpy())
    tree = shapely.STRtree(lines)
    p_idx, a_idx = tree.query(pts, predicate="dwithin", distance=R)  # all pairs within R
    dist = shapely.distance(pts[p_idx], lines[a_idx])
    frac = shapely.line_locate_point(lines[a_idx], pts[p_idx], normalized=True)

    df = pd.DataFrame({"point_idx": p_idx, "arc_id": a_idx, "dist_m": dist, "frac_along": frac})
    df = df.sort_values(["point_idx", "dist_m", "arc_id"], kind="mergesort")
    df["dist_rank"] = df.groupby("point_idx").cumcount()
    df = df[df.dist_rank < k].copy()

    heading, speed = _gps_motion(clean)
    df["seg_id"] = clean.seg_id.to_numpy()[df.point_idx]
    df["dist_minus_min"] = df.dist_m - df.groupby("point_idx").dist_m.transform("min")
    hd = np.abs(heading[df.point_idx] - arc_bearing[df.arc_id]) % 360
    df["heading_diff_deg"] = np.minimum(hd, 360 - hd)
    df["arc_len_m"] = arc_len[df.arc_id]
    df["speed_mps"] = speed[df.point_idx]
    df["n_candidates"] = df.groupby("point_idx").arc_id.transform("size")

    if with_labels and trip["route"] is not None:
        t_arc, t_dist = true_arcs(clean, nodes_xy, arcs, trip["route"])
        clean["true_arc"], clean["true_dist_m"] = t_arc, t_dist
        df["true_arc"] = t_arc[df.point_idx]
        df["label"] = (df.arc_id == df.true_arc).astype(np.int8)

    return clean, df.reset_index(drop=True)


def candidate_miss_rate(clean, cands):
    """Fraction of points whose true arc is NOT in their candidate set (incl. points with no candidates)."""
    hit = cands.groupby("point_idx").label.max()
    return 1.0 - hit.reindex(range(len(clean)), fill_value=0).mean()
