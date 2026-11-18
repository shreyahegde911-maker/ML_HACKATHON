"""Strict raw readers. load_trip never opens a .route file."""

from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

DEVELOPMENT_PARTITIONS = ("train", "student_test")


def trip_directory(data_root, partition, trip_id):
    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError("Only train and student_test are permitted during development")
    trip_id = str(trip_id)
    if not trip_id.isdigit():
        raise ValueError("trip_id must contain only digits")
    trip_id = trip_id.zfill(8)
    directory = Path(data_root) / partition / trip_id
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def _read(path, columns, integer=False):
    # Read as strings first: reject fractional/overflowing IDs before conversion.
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != columns:
                raise ValueError(
                    f"{path}:{number}: expected {columns} columns, got {len(fields)}"
                )
            rows.append(fields)
    if not rows:
        raise ValueError(f"{path}: empty file")
    if integer:
        try:
            values = np.asarray(
                [[int(value) for value in row] for row in rows], dtype=np.int64
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{path}: IDs must be signed 64-bit integers") from exc
    else:
        try:
            values = np.asarray(rows, dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{path}: nonnumeric value") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: nonfinite value")
    return values


def _coordinates(values, path):
    if ((np.abs(values[:, 0]) > 180) | (np.abs(values[:, 1]) > 90)).any():
        raise ValueError(f"{path}: invalid longitude/latitude bounds")


def _project(transformer, values):
    # pyproj's one-element fast path expects scalars with current NumPy.
    if len(values) == 1:
        x, y = transformer.transform(float(values[0, 0]), float(values[0, 1]))
        return np.array([x]), np.array([y])
    return transformer.transform(values[:, 0], values[:, 1])


def load_trip(data_root, partition, trip_id, *, gap_threshold_s=30.0):
    """Return observations/nodes/arcs DataFrames, CRS metadata and validation.

    IDs and observation order are source row numbers. No rows are dropped or
    sorted, no scaling is fitted, and reference files are never read here.
    """
    if gap_threshold_s <= 0:
        raise ValueError("gap_threshold_s must be positive")
    directory = trip_directory(data_root, partition, trip_id)
    return load_trip_directory(
        directory, partition=partition, gap_threshold_s=gap_threshold_s
    )


def load_trip_directory(directory, *, partition="evaluator", gap_threshold_s=30.0):
    """Explicit inference entry point for a server-configured trip directory.

    Does not load references. Development partition restrictions remain on
    load_trip; this separate function allows a reserved external evaluator root.
    """
    directory = Path(directory)
    if not directory.name.isdigit() or not directory.is_dir():
        raise ValueError("Expected an existing numeric trip directory")
    trip_id = directory.name
    track = _read(directory / f"{trip_id}.track", 3)
    coordinates = _read(directory / f"{trip_id}.nodes", 2)
    edges = _read(directory / f"{trip_id}.arcs", 2, integer=True)
    _coordinates(track, directory / f"{trip_id}.track")
    _coordinates(coordinates, directory / f"{trip_id}.nodes")
    if ((edges < 0) | (edges >= len(coordinates))).any():
        raise ValueError(f"{directory}: arc references a nonexistent node")

    lon, lat = np.median(track[:, :2], axis=0)
    if not -80 <= lat <= 84:
        raise ValueError("This UTM reader supports trip centres between 80 S and 84 N")
    zone = min(60, max(1, int(np.floor((lon + 180) / 6)) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    observations = pd.DataFrame(track, columns=["longitude", "latitude", "timestamp"])
    observations.insert(0, "observation_id", np.arange(len(track), dtype=np.int64))
    observations["x_m"], observations["y_m"] = _project(transformer, track)
    nodes = pd.DataFrame(coordinates, columns=["longitude", "latitude"])
    nodes.insert(0, "node_id", np.arange(len(coordinates), dtype=np.int64))
    nodes["x_m"], nodes["y_m"] = _project(transformer, coordinates)
    if (
        not np.isfinite(observations[["x_m", "y_m"]]).all().all()
        or not np.isfinite(nodes[["x_m", "y_m"]]).all().all()
    ):
        raise ValueError("Projection produced nonfinite coordinates")
    arcs = pd.DataFrame(edges, columns=["from_node_id", "to_node_id"])
    arcs.insert(0, "arc_id", np.arange(len(edges), dtype=np.int64))
    dt = np.diff(track[:, 2])
    large = np.flatnonzero(dt > gap_threshold_s) + 1
    nonpositive = np.flatnonzero(dt <= 0) + 1
    projected_extent = np.ptp(
        np.vstack([observations[["x_m", "y_m"]], nodes[["x_m", "y_m"]]]), axis=0
    )
    warnings = []
    if len(large):
        warnings.append(
            f"{len(large)} timestamp gaps exceed {gap_threshold_s:g} seconds"
        )
    if len(nonpositive):
        warnings.append(
            f"{len(nonpositive)} nonpositive timestamp steps; source order preserved"
        )
    if projected_extent.max() > 200_000:
        warnings.append(
            "Network/trip extent exceeds 200 km; inspect UTM distortion before using distances"
        )
    used_nodes = np.unique(edges)
    return {
        "trip_id": trip_id,
        "partition": partition,
        "observations": observations,
        "nodes": nodes,
        "arcs": arcs,
        "crs": {
            "source": "EPSG:4326",
            "projected": f"EPSG:{epsg}",
            "epsg": epsg,
            "units": "metre",
            "always_xy": True,
            "wkt": crs.to_wkt(),
            "selection": "UTM zone from median observed longitude/latitude",
        },
        "validation": {
            "observation_count": len(track),
            "node_count": len(nodes),
            "arc_count": len(arcs),
            "large_gap_observation_ids": large.tolist(),
            "nonpositive_time_observation_ids": nonpositive.tolist(),
            "max_gap_s": float(dt.max()) if len(dt) else 0.0,
            "one_second_steps": int((dt == 1).sum()),
            "timestamp_steps": len(dt),
            "isolated_node_count": len(nodes) - len(used_nodes),
            "longitude_min": float(track[:, 0].min()),
            "longitude_max": float(track[:, 0].max()),
            "latitude_min": float(track[:, 1].min()),
            "latitude_max": float(track[:, 1].max()),
            "warnings": warnings,
        },
    }


def load_reference_route(data_root, partition, trip_id, *, arcs=None):
    """Read reference occurrences separately; retain repeated arc IDs and order.

    section_id increments whenever the next arc does not start at the previous
    arc's end node. Such a break is reported, not repaired or interpolated.
    """
    directory = trip_directory(data_root, partition, trip_id)
    values = _read(directory / f"{directory.name}.route", 1, integer=True).ravel()
    if arcs is None:
        edges = _read(directory / f"{directory.name}.arcs", 2, integer=True)
    else:
        if not np.array_equal(arcs["arc_id"].to_numpy(), np.arange(len(arcs))):
            raise ValueError(
                "arcs must retain original zero-based IDs and source order"
            )
        edges = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    if ((values < 0) | (values >= len(edges))).any():
        raise ValueError(f"{directory}: reference arc ID out of bounds")
    endpoints = edges[values]
    breaks = np.r_[False, endpoints[:-1, 1] != endpoints[1:, 0]]
    return pd.DataFrame(
        {
            "route_position": np.arange(len(values), dtype=np.int64),
            "arc_id": values,
            "section_id": np.cumsum(breaks),
            "discontinuity_before": breaks,
        }
    )
