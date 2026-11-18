"""Coordinate projection and track cleaning.

Why project? lon/lat are angles; 1 degree of longitude shrinks with latitude
(~75 km at 47°N, ~111 km at the equator) while 1 degree of latitude is ~111 km,
so Euclidean distance in degrees is distorted.

The trips are spread across the world (25 different UTM zones, both
hemispheres), so each trip is projected into ITS OWN UTM zone, chosen from the
trip's mean longitude/latitude. Inside one zone, UTM distortion is < 0.1%,
and a single trip never spans more than a few km. After projecting, x/y are metres.
"""
import numpy as np
import pandas as pd
from pyproj import Transformer

from functools import lru_cache

STATIONARY_M = 3.0   # drop points that moved less than this from the last kept point
GAP_S = 60.0         # start a new segment when the time gap exceeds this


def utm_epsg(lon_mean: float, lat_mean: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx = north, 327xx = south."""
    zone = int((lon_mean + 180) // 6) + 1
    return (32600 if lat_mean >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project(lon, lat, epsg: int):
    """lon/lat (degrees) -> x/y (metres) in the given UTM zone."""
    x, y = _transformer(epsg).transform(np.asarray(lon, float), np.asarray(lat, float))
    return x, y


def trip_epsg(track: pd.DataFrame) -> int:
    """Pick the UTM zone for a trip from its GPS points (same zone used for its road network)."""
    return utm_epsg(float(track.lon.mean()), float(track.lat.mean()))


def clean_track(track: pd.DataFrame, epsg: int, stationary_m=STATIONARY_M, gap_s=GAP_S) -> pd.DataFrame:
    """Return kept GPS points with columns lon, lat, t, x, y, seg_id, orig_idx.

    1. Stationary filter: when the car is parked, GPS jitters around one spot and
       adds no route information, so we skip points < stationary_m from the last kept point.
    2. Gap split: if two kept points are > gap_s apart in time we cannot trust any
       route between them, so the HMM treats them as separate segments.
    """
    x, y = project(track.lon.to_numpy(), track.lat.to_numpy(), epsg)
    t = track.t.to_numpy()
    keep = [0]
    for i in range(1, len(t)):
        j = keep[-1]
        if np.hypot(x[i] - x[j], y[i] - y[j]) >= stationary_m:
            keep.append(i)
    keep = np.array(keep)
    out = track.iloc[keep].copy()
    out["x"], out["y"] = x[keep], y[keep]
    out["orig_idx"] = keep
    dt = np.diff(out.t.to_numpy(), prepend=out.t.iloc[0])
    out["seg_id"] = np.cumsum(dt > gap_s)
    return out.reset_index(drop=True)
