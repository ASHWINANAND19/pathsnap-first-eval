"""Folium map helpers.

Layer colours (kept consistent everywhere, incl. the report):
  blue dots  = raw GPS observations
  grey lines = road network near the trip
  green line = reference (ground-truth) route
  red line   = predicted route
"""
import folium
import numpy as np


def _arc_latlons(arc_ids, nodes, arcs):
    """Return [[(lat, lon), (lat, lon)], ...] for each arc id."""
    u = arcs.loc[arc_ids, "u"].to_numpy()
    v = arcs.loc[arc_ids, "v"].to_numpy()
    a = nodes.loc[u, ["lat", "lon"]].to_numpy()
    b = nodes.loc[v, ["lat", "lon"]].to_numpy()
    return [[tuple(p), tuple(q)] for p, q in zip(a, b)]


def nearby_arc_ids(track, nodes, arcs, margin_deg=0.003):
    """Arcs with at least one end inside the track's bounding box (+margin).
    Used only to keep maps light — the full network can have 300k arcs."""
    lo_lon, hi_lon = track.lon.min() - margin_deg, track.lon.max() + margin_deg
    lo_lat, hi_lat = track.lat.min() - margin_deg, track.lat.max() + margin_deg
    inside = nodes.lon.between(lo_lon, hi_lon) & nodes.lat.between(lo_lat, hi_lat)
    mask = inside.to_numpy()[arcs.u.to_numpy()] | inside.to_numpy()[arcs.v.to_numpy()]
    return arcs.index[mask].to_numpy()


def trip_map(trip, pred_arcs=None, show_route=True, show_network=True, gps_every=1):
    """Build an interactive map for one trip dict (from loader.load_trip)."""
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=14, tiles="OpenStreetMap")

    if show_network:
        fg = folium.FeatureGroup(name="Road network (nearby)")
        for seg in _arc_latlons(nearby_arc_ids(track, nodes, arcs), nodes, arcs):
            folium.PolyLine(seg, color="#9e9e9e", weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)

    if show_route and route is not None:
        fg = folium.FeatureGroup(name="Reference route (ground truth)")
        for seg in _arc_latlons(route.arc_id.to_numpy(), nodes, arcs):
            folium.PolyLine(seg, color="green", weight=6, opacity=0.7).add_to(fg)
        fg.add_to(m)

    if pred_arcs is not None:
        ids = np.unique(np.asarray(pred_arcs)[np.asarray(pred_arcs) >= 0])
        fg = folium.FeatureGroup(name="Predicted route")
        for seg in _arc_latlons(ids, nodes, arcs):
            folium.PolyLine(seg, color="red", weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS points")
    for lat, lon in track[["lat", "lon"]].to_numpy()[::gps_every]:
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True, weight=0, fill_opacity=0.7).add_to(fg)
    fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
    return m
