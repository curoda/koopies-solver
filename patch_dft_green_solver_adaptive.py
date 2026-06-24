#!/usr/bin/env python3
"""
patch_dft_green_solver.py

Prototype generalized 3-D acoustic radiation solver using a patch-block
DFT-compressed Green boundary formulation.

Working equation
----------------
    (1/2 I - D_DFT) p = i k S_DFT v_n

where:
    p      = unknown complex surface pressure vector
    v_n    = prescribed complex normal velocity vector
    S_DFT  = single-layer / monopole-like Green operator, patch compressed
    D_DFT  = double-layer / dipole-like Green operator, patch compressed
    k      = acoustic wave number = ka / a

Patch-block compression
-----------------------
For each receiver patch a and source patch b, an exact Green block A_ab
is either kept dense, if the two patches are near, or compressed as

    A_ab ~= G_a C_ab G_b^H

where:
    G_b^H  projects source-patch values into local plane-wave/DFT components,
    C_ab   is a small dense spectral core that carries Green-function propagation,
    G_a    reconstructs the field on the receiver patch.

This is a research prototype. It is intended to make the logic transparent
and to support parametric studies. It is not yet a production BEM code.

Adaptive patch/version notes
----------------------------
This version keeps the original validated formulation but adds:
    1. Patch-diameter control: patches are recursively split until
       d_patch <= lambda_star / patches_per_wavelength, where
       lambda_star = min(lambda_acoustic, lambda_velocity) if lambda_velocity
       is supplied.
    2. Feature-aware patch protection: optional face, edge, corner, feature
       type, and feature ID columns are canonicalized into hard patch-boundary
       groups. Edge/corner groups are separated by face side when face IDs are
       available. If no metadata is supplied, geometric controls remain active.
    3. Adaptive M: each patch may use its own local basis size. The code
       increases M through a user-selectable schedule until the measured
       far-block projection error for both S and D is below tolerance, or
       until the schedule / number of points is exhausted.

Expected CSV input columns
--------------------------
Required:
    N or index          point index
    x, y, z             physical coordinates, in the same length units as a
    vn                  complex normal velocity, or vn_real and vn_imag

Recommended:
    nx, ny, nz          outward unit normals. If absent, normals are estimated.
    area                surface area weight for each point. If absent, equal
                        area weights 4*pi*a^2/N are used as a first estimate.

Optional preprocessor topology metadata:
    face_id              parent face/panel/region identifier
    edge_id              sharp-edge/crease/seam identifier
    corner_id            corner/vertex/apex identifier
    feature_type         surface, edge, corner, or another feature class
    feature_id           general feature entity identifier
    is_edge, is_corner   Boolean feature flags when IDs are unavailable

Optional / carried through for compatibility with the original DFT notation:
    l, m, n              digitized geometry coordinates

Example run
-----------
    python patch_dft_green_solver.py input.csv --ka 1.0 --a 0.25 --W 100 --out results

Demo sphere input generation
----------------------------
    python patch_dft_green_solver.py --make-demo-sphere demo_sphere.csv --demo-N 720
    python patch_dft_green_solver.py demo_sphere.csv --ka 1.0 --a 1.0 --out demo_out

Dependencies: numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, gmres


# -----------------------------
# Utility printing / logging
# -----------------------------

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def complex_from_columns(df: pd.DataFrame, base: str) -> np.ndarray:
    """Parse complex vector from either base or base_real/base_imag columns."""
    if base in df.columns:
        # pandas can parse strings like "1+2j" if needed
        return df[base].apply(lambda x: complex(x)).to_numpy(dtype=np.complex128)
    real_col = f"{base}_real"
    imag_col = f"{base}_imag"
    if real_col in df.columns:
        re = df[real_col].to_numpy(dtype=float)
        im = df[imag_col].to_numpy(dtype=float) if imag_col in df.columns else np.zeros_like(re)
        return re + 1j * im
    raise ValueError(f"Missing velocity column '{base}' or '{base}_real'/'{base}_imag'.")


# -----------------------------
# Demo data
# -----------------------------

def fibonacci_sphere_points(n: int, radius: float = 1.0) -> np.ndarray:
    """Nearly equal-area points on a sphere."""
    i = np.arange(n, dtype=float)
    phi = (1 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = 2.0 * np.pi * i / phi
    rxy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack((rxy * np.cos(theta), rxy * np.sin(theta), z))
    return radius * pts


def write_demo_sphere_csv(path: Path, n: int, radius: float, verbose: bool = True) -> None:
    pts = fibonacci_sphere_points(n, radius)
    normals = pts / np.linalg.norm(pts, axis=1)[:, None]
    area = np.full(n, 4.0 * np.pi * radius * radius / n)
    df = pd.DataFrame({
        "N": np.arange(1, n + 1),
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "l": np.round(100 * normals[:, 0]).astype(int),
        "m": np.round(100 * normals[:, 1]).astype(int),
        "n": np.round(100 * normals[:, 2]).astype(int),
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "vn_real": np.ones(n), "vn_imag": np.zeros(n),
        "area": area,
    })
    df.to_csv(path, index=False)
    log(f"Wrote demo pulsating-sphere CSV: {path}", verbose)


# -----------------------------
# Geometry loading / normals
# -----------------------------

@dataclass
class FeatureMetadata:
    """Preprocessor topology/feature metadata carried with every surface point.

    Values are stored as cleaned strings so identifiers may be numeric or text.
    ``group_code`` is the hard patch-boundary code used by the legacy patching
    routines. A compression patch is never allowed to contain more than one
    group code.
    """

    face_id: np.ndarray
    edge_id: np.ndarray
    corner_id: np.ndarray
    feature_id: np.ndarray
    feature_type_raw: np.ndarray
    feature_class: np.ndarray
    group_code: np.ndarray
    group_label: np.ndarray
    detected_columns: Dict[str, Optional[str]]
    boundary_mode: str
    connectivity_radius: float
    feature_type_map: Dict[str, str]
    zero_feature_id_is_missing: bool


# Public compatibility name used by the feature-aware solver/front-end.
# FeatureMetadata is the implementation dataclass; both names refer to the
# same type so existing imports continue to work.
GeometryFeatures = FeatureMetadata


@dataclass
class Geometry:
    index: np.ndarray
    xyz: np.ndarray
    normals: np.ndarray
    area: np.ndarray
    vn: np.ndarray
    digital_lmn: Optional[np.ndarray]
    patch_region: Optional[np.ndarray] = None
    patch_region_name: Optional[str] = None
    features: Optional[FeatureMetadata] = None


def estimate_normals_pca(xyz: np.ndarray, k_neighbors: int = 20) -> np.ndarray:
    """Estimate outward normals using local PCA and orient from centroid.

    This is only a fallback. For real arbitrary radiators, provide nx,ny,nz.
    """
    n = xyz.shape[0]
    tree = cKDTree(xyz)
    k = min(k_neighbors, n)
    _, idx = tree.query(xyz, k=k)
    normals = np.zeros_like(xyz)
    centroid = xyz.mean(axis=0)
    for i in range(n):
        cloud = xyz[idx[i]]
        cloud = cloud - cloud.mean(axis=0)
        cov = cloud.T @ cloud
        vals, vecs = np.linalg.eigh(cov)
        normal = vecs[:, np.argmin(vals)]
        # orient outward approximately relative to global centroid
        if np.dot(normal, xyz[i] - centroid) < 0:
            normal *= -1.0
        normals[i] = normal / (np.linalg.norm(normal) + 1e-15)
    return normals


def _normalized_column_name(value: str) -> str:
    text = str(value).strip().lower()
    for token in (" ", "-", ".", "/", "\\"):
        text = text.replace(token, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _resolve_optional_column(
    df: pd.DataFrame,
    override: Optional[str],
    aliases: Sequence[str],
    role: str,
) -> Optional[str]:
    """Resolve a preprocessor column by explicit name or case-insensitive alias."""
    normalized = {_normalized_column_name(col): str(col) for col in df.columns}
    if override is not None and str(override).strip() not in {"", "auto", "none"}:
        requested = str(override).strip()
        if requested in df.columns:
            return requested
        match = normalized.get(_normalized_column_name(requested))
        if match is not None:
            return match
        raise ValueError(
            f"Requested {role} column '{requested}' was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    for alias in aliases:
        match = normalized.get(_normalized_column_name(alias))
        if match is not None:
            return match
    return None


def _clean_identifier_values(
    series: Optional[pd.Series],
    n: int,
    zero_is_missing: bool = False,
) -> np.ndarray:
    """Return identifier values as strings; blanks/NaN/negative sentinels become ''."""
    if series is None:
        return np.full(n, "", dtype=object)
    output = np.empty(n, dtype=object)
    missing_tokens = {"", "nan", "none", "null", "na", "n/a", "-1", "-1.0"}
    if zero_is_missing:
        missing_tokens.update({"0", "0.0"})
    for i, value in enumerate(series.to_numpy(dtype=object)):
        if pd.isna(value):
            output[i] = ""
            continue
        if isinstance(value, (int, np.integer)):
            integer = int(value)
            output[i] = "" if integer < 0 or (zero_is_missing and integer == 0) else str(integer)
            continue
        if isinstance(value, (float, np.floating)):
            if (
                not np.isfinite(value)
                or float(value) < 0
                or (zero_is_missing and float(value) == 0.0)
            ):
                output[i] = ""
            elif float(value).is_integer():
                output[i] = str(int(value))
            else:
                output[i] = f"{float(value):.15g}"
            continue
        text = str(value).strip()
        output[i] = "" if text.lower() in missing_tokens else text
    return output


def _clean_text_values(series: Optional[pd.Series], n: int) -> np.ndarray:
    if series is None:
        return np.full(n, "", dtype=object)
    output = np.empty(n, dtype=object)
    for i, value in enumerate(series.to_numpy(dtype=object)):
        output[i] = "" if pd.isna(value) else str(value).strip()
    return output


def _boolean_values(series: Optional[pd.Series], n: int) -> np.ndarray:
    if series is None:
        return np.zeros(n, dtype=bool)
    truthy = {"1", "true", "yes", "y", "on", "edge", "corner", "sharp"}
    values = np.zeros(n, dtype=bool)
    for i, value in enumerate(series.to_numpy(dtype=object)):
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            values[i] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            values[i] = bool(value)
        else:
            values[i] = str(value).strip().lower() in truthy
    return values


def _looks_boolean_series(series: pd.Series) -> bool:
    values = []
    for value in series.to_numpy(dtype=object):
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            values.append("true" if bool(value) else "false")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)):
                continue
            values.append(str(int(float(value))) if float(value).is_integer() else str(float(value)))
        else:
            values.append(str(value).strip().lower())
    if not values:
        return False
    allowed = {"0", "1", "-1", "true", "false", "yes", "no", "y", "n", "on", "off"}
    return set(values).issubset(allowed)


def parse_feature_type_map(value: Optional[object]) -> Dict[str, str]:
    """Parse feature-type code mappings such as '0:surface,1:edge,2:corner'."""
    if value is None:
        return {}
    if isinstance(value, dict):
        raw_mapping = value
    else:
        text = str(value).strip()
        if not text:
            return {}
        if text.startswith("{"):
            raw_mapping = json.loads(text)
        else:
            raw_mapping = {}
            for part in text.replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                separator = ":" if ":" in part else ("=" if "=" in part else None)
                if separator is None:
                    raise ValueError(
                        "feature type map entries must use code:class or code=class."
                    )
                key, target = part.split(separator, 1)
                raw_mapping[key.strip()] = target.strip()
    canonical: Dict[str, str] = {}
    target_aliases = {
        "surface": "surface", "face": "surface", "panel": "surface", "smooth": "surface",
        "edge": "edge", "crease": "edge", "seam": "edge", "sharp_edge": "edge",
        "corner": "corner", "vertex": "corner", "apex": "corner",
        "feature": "feature", "other": "feature",
    }
    for key, target in raw_mapping.items():
        normalized_target = _normalized_column_name(target)
        if normalized_target not in target_aliases:
            raise ValueError(
                f"Unsupported feature class '{target}' in feature type map; "
                "use surface, edge, corner, or feature."
            )
        canonical[_normalized_column_name(key)] = target_aliases[normalized_target]
    return canonical


def _canonical_feature_classes(
    feature_type_raw: np.ndarray,
    edge_id: np.ndarray,
    corner_id: np.ndarray,
    is_edge: np.ndarray,
    is_corner: np.ndarray,
    feature_type_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    corner_words = ("corner", "vertex", "apex", "point_feature")
    edge_words = (
        "edge", "crease", "seam", "ridge", "rim", "sharp", "boundary_curve",
        "hard_line", "feature_line",
    )
    surface_words = (
        "surface", "face", "smooth", "panel", "shell", "area", "exterior",
        "interior", "body",
    )
    mapping = feature_type_map or {}
    result = np.empty(len(feature_type_raw), dtype=object)
    for i, raw in enumerate(feature_type_raw):
        text = _normalized_column_name(raw) if raw else ""
        mapped = mapping.get(text)
        if corner_id[i] or is_corner[i]:
            result[i] = "corner"
        elif edge_id[i] or is_edge[i]:
            result[i] = "edge"
        elif mapped is not None:
            result[i] = mapped
        elif any(word in text for word in corner_words):
            result[i] = "corner"
        elif any(word in text for word in edge_words):
            result[i] = "edge"
        elif not text or any(word in text for word in surface_words):
            result[i] = "surface"
        else:
            result[i] = "feature"
    return result


def _median_point_spacing(xyz: np.ndarray) -> float:
    if len(xyz) <= 1:
        return 0.0
    distances, _ = cKDTree(xyz).query(xyz, k=min(2, len(xyz)))
    nearest = np.asarray(distances)[:, -1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size:
        return float(np.median(nearest))
    extent = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    return extent / max(math.sqrt(len(xyz)), 1.0)


def _connected_component_names(
    xyz: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    radius: float,
) -> np.ndarray:
    """Name connected feature components when the preprocessor omitted IDs."""
    names = np.full(len(xyz), "", dtype=object)
    global_indices = np.flatnonzero(mask)
    if global_indices.size == 0:
        return names
    if global_indices.size == 1 or radius <= 0.0:
        for component, index in enumerate(global_indices):
            names[index] = f"{prefix}_{component + 1}"
        return names

    points = xyz[global_indices]
    tree = cKDTree(points)
    parent = np.arange(len(points), dtype=int)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in tree.query_pairs(radius):
        union(int(left), int(right))
    roots = [find(i) for i in range(len(points))]
    root_to_component: Dict[int, int] = {}
    for local_index, root in enumerate(roots):
        component = root_to_component.setdefault(root, len(root_to_component) + 1)
        names[global_indices[local_index]] = f"{prefix}_{component}"
    return names


def build_feature_metadata(
    df: pd.DataFrame,
    xyz: np.ndarray,
    face_column: Optional[str] = None,
    edge_column: Optional[str] = None,
    corner_column: Optional[str] = None,
    feature_type_column: Optional[str] = None,
    feature_id_column: Optional[str] = None,
    is_edge_column: Optional[str] = None,
    is_corner_column: Optional[str] = None,
    feature_boundary_mode: str = "auto",
    feature_connectivity_factor: float = 2.5,
    zero_feature_id_is_missing: bool = True,
    feature_type_map: Optional[object] = None,
    verbose: bool = True,
) -> Optional[FeatureMetadata]:
    """Read and canonicalize topology information generated by a preprocessor.

    Supported metadata can be numeric or textual. Explicit column options take
    precedence over aliases. In ``strict`` mode surface faces, edges, corners,
    and other feature groups become hard compression-patch boundaries.
    """
    aliases = {
        "face": (
            "face_id", "face", "surface_id", "surface", "region_id", "region",
            "panel_id", "panel", "component_id", "component", "part_id", "part",
            "patch_region", "body_region",
        ),
        "edge": (
            "edge_id", "edge", "edge_group", "sharp_edge_id", "crease_id",
            "seam_id", "boundary_edge_id", "feature_edge_id",
        ),
        "corner": (
            "corner_id", "corner", "corner_group", "vertex_id", "vertex",
            "apex_id", "feature_corner_id",
        ),
        "feature_type": (
            "feature_type", "feature_class", "feature_kind", "geometry_feature",
            "topology_type", "topology_class", "entity_type",
        ),
        "feature_id": (
            "feature_id", "feature_group_id", "feature_group", "feature_label",
            "feature_name", "topology_id", "entity_id",
        ),
        "is_edge": ("is_edge", "edge_flag", "on_edge", "sharp_edge_flag"),
        "is_corner": ("is_corner", "corner_flag", "is_vertex", "vertex_flag"),
    }
    detected = {
        "face": _resolve_optional_column(df, face_column, aliases["face"], "face/region"),
        "edge": _resolve_optional_column(df, edge_column, aliases["edge"], "edge"),
        "corner": _resolve_optional_column(df, corner_column, aliases["corner"], "corner"),
        "feature_type": _resolve_optional_column(
            df, feature_type_column, aliases["feature_type"], "feature type"
        ),
        "feature_id": _resolve_optional_column(
            df, feature_id_column, aliases["feature_id"], "feature ID"
        ),
        "is_edge": _resolve_optional_column(df, is_edge_column, aliases["is_edge"], "edge flag"),
        "is_corner": _resolve_optional_column(
            df, is_corner_column, aliases["is_corner"], "corner flag"
        ),
    }
    # Resolve ambiguous columns named simply "edge" or "corner". If they
    # contain Boolean-like values, treat them as flags rather than identifiers.
    if (
        detected["edge"] is not None
        and detected["is_edge"] is None
        and _normalized_column_name(detected["edge"]) == "edge"
    ):
        if _looks_boolean_series(df[detected["edge"]]):
            detected["is_edge"] = detected["edge"]
            detected["edge"] = None
    if (
        detected["corner"] is not None
        and detected["is_corner"] is None
        and _normalized_column_name(detected["corner"]) == "corner"
    ):
        if _looks_boolean_series(df[detected["corner"]]):
            detected["is_corner"] = detected["corner"]
            detected["corner"] = None

    n = len(df)
    any_metadata = any(value is not None for value in detected.values())
    mode = str(feature_boundary_mode).strip().lower().replace("_", "-")
    aliases_mode = {"none": "off", "region": "face-only", "face": "face-only"}
    mode = aliases_mode.get(mode, mode)
    if mode not in {"auto", "strict", "face-only", "off"}:
        raise ValueError(
            "feature_boundary_mode must be one of auto, strict, face-only, or off."
        )
    if mode == "auto":
        if any(detected[key] is not None for key in ("edge", "corner", "feature_type", "feature_id", "is_edge", "is_corner")):
            mode = "strict"
        elif detected["face"] is not None:
            mode = "face-only"
        else:
            mode = "off"

    if not any_metadata and mode == "off":
        log(
            "No preprocessor face/edge/corner/feature columns detected. "
            "Patch boundaries will use geometric controls only.",
            verbose,
        )
        return None

    face_id = _clean_identifier_values(
        df[detected["face"]] if detected["face"] else None,
        n,
        zero_is_missing=False,
    )
    edge_id = _clean_identifier_values(
        df[detected["edge"]] if detected["edge"] else None,
        n,
        zero_is_missing=zero_feature_id_is_missing,
    )
    corner_id = _clean_identifier_values(
        df[detected["corner"]] if detected["corner"] else None,
        n,
        zero_is_missing=zero_feature_id_is_missing,
    )
    feature_id = _clean_identifier_values(
        df[detected["feature_id"]] if detected["feature_id"] else None,
        n,
        zero_is_missing=zero_feature_id_is_missing,
    )
    feature_type_raw = _clean_text_values(
        df[detected["feature_type"]] if detected["feature_type"] else None, n
    )
    is_edge = _boolean_values(df[detected["is_edge"]] if detected["is_edge"] else None, n)
    is_corner = _boolean_values(
        df[detected["is_corner"]] if detected["is_corner"] else None, n
    )
    parsed_feature_type_map = parse_feature_type_map(feature_type_map)
    feature_class = _canonical_feature_classes(
        feature_type_raw,
        edge_id,
        corner_id,
        is_edge,
        is_corner,
        feature_type_map=parsed_feature_type_map,
    )
    nonempty_types = [str(value).strip() for value in feature_type_raw if str(value).strip()]
    if nonempty_types and not parsed_feature_type_map:
        numeric_types = True
        for value in set(nonempty_types):
            try:
                float(value)
            except ValueError:
                numeric_types = False
                break
        if numeric_types:
            log(
                "WARNING: feature_type contains numeric codes but no mapping was supplied. "
                "Use --feature-type-map, for example '0:surface,1:edge,2:corner'.",
                verbose,
            )

    spacing = _median_point_spacing(xyz)
    connectivity_radius = max(float(feature_connectivity_factor), 0.0) * spacing
    unnamed_edge = (feature_class == "edge") & (edge_id == "") & (feature_id == "")
    unnamed_corner = (feature_class == "corner") & (corner_id == "") & (feature_id == "")
    edge_components = _connected_component_names(
        xyz, unnamed_edge, "auto_edge", connectivity_radius
    )
    corner_components = _connected_component_names(
        xyz, unnamed_corner, "auto_corner", connectivity_radius
    )

    group_label = np.empty(n, dtype=object)
    for i in range(n):
        # Generic feature IDs are intentionally not used to partition ordinary
        # surface points: many preprocessors emit a unique element/entity ID per
        # sample, which would otherwise create thousands of one-point patches.
        face_only_identifier = face_id[i] or "default"
        surface_identifier = face_id[i] or feature_type_raw[i] or "default"
        if mode == "off":
            group_label[i] = "all"
        elif mode == "face-only":
            group_label[i] = f"face:{face_only_identifier}"
        elif feature_class[i] == "corner":
            identifier = corner_id[i] or feature_id[i] or corner_components[i] or f"point_{i + 1}"
            side = f":face:{face_id[i]}" if face_id[i] else ""
            group_label[i] = f"corner:{identifier}{side}"
        elif feature_class[i] == "edge":
            identifier = edge_id[i] or feature_id[i] or edge_components[i] or f"point_{i + 1}"
            side = f":face:{face_id[i]}" if face_id[i] else ""
            group_label[i] = f"edge:{identifier}{side}"
        elif feature_class[i] == "feature":
            identifier = feature_id[i] or feature_type_raw[i] or surface_identifier
            side = f":face:{face_id[i]}" if face_id[i] else ""
            group_label[i] = f"feature:{identifier}{side}"
        else:
            group_label[i] = f"surface:{surface_identifier}"

    group_code, unique_groups = pd.factorize(group_label, sort=True)
    metadata = FeatureMetadata(
        face_id=face_id,
        edge_id=edge_id,
        corner_id=corner_id,
        feature_id=feature_id,
        feature_type_raw=feature_type_raw,
        feature_class=feature_class,
        group_code=group_code.astype(int),
        group_label=group_label,
        detected_columns=detected,
        boundary_mode=mode,
        connectivity_radius=float(connectivity_radius),
        feature_type_map=parsed_feature_type_map,
        zero_feature_id_is_missing=bool(zero_feature_id_is_missing),
    )
    class_counts = {
        value: int(np.sum(feature_class == value))
        for value in ("surface", "edge", "corner", "feature")
    }
    detected_text = ", ".join(
        f"{role}={column}" for role, column in detected.items() if column is not None
    ) or "none"
    log(
        f"Preprocessor feature metadata: mode={mode}; columns [{detected_text}]; "
        f"hard groups={len(unique_groups)}; class counts={class_counts}; "
        f"auto-connectivity radius={connectivity_radius:.6g}.",
        verbose,
    )
    return metadata


def feature_subset_summary(geom: Geometry, inds: np.ndarray) -> Dict[str, object]:
    """Compact feature/topology summary for one compression patch."""
    if geom.features is None or len(inds) == 0:
        return {
            "feature_group_code": -1,
            "feature_group_label": "",
            "feature_class": "",
            "feature_type": "",
            "face_id": "",
            "edge_id": "",
            "corner_id": "",
            "feature_id": "",
            "surface_points": 0,
            "edge_points": 0,
            "corner_points": 0,
            "other_feature_points": 0,
        }

    features = geom.features

    def joined(values: np.ndarray, limit: int = 12) -> str:
        unique = sorted({str(value) for value in values if str(value)})
        if len(unique) > limit:
            return "|".join(unique[:limit]) + f"|...(+{len(unique) - limit})"
        return "|".join(unique)

    classes = features.feature_class[inds]
    groups = np.unique(features.group_code[inds])
    return {
        "feature_group_code": int(groups[0]) if len(groups) == 1 else -1,
        "feature_group_label": joined(features.group_label[inds]),
        "feature_class": joined(classes),
        "feature_type": joined(features.feature_type_raw[inds]),
        "face_id": joined(features.face_id[inds]),
        "edge_id": joined(features.edge_id[inds]),
        "corner_id": joined(features.corner_id[inds]),
        "feature_id": joined(features.feature_id[inds]),
        "surface_points": int(np.sum(classes == "surface")),
        "edge_points": int(np.sum(classes == "edge")),
        "corner_points": int(np.sum(classes == "corner")),
        "other_feature_points": int(np.sum(classes == "feature")),
    }


def _clean_join_key_values(series: pd.Series) -> pd.Series:
    """Canonical string keys for joining a separate preprocessor metadata table."""
    cleaned: List[str] = []
    for value in series.to_numpy(dtype=object):
        if pd.isna(value):
            cleaned.append("")
        elif isinstance(value, (int, np.integer)):
            cleaned.append(str(int(value)))
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                cleaned.append("")
            elif float(value).is_integer():
                cleaned.append(str(int(value)))
            else:
                cleaned.append(f"{float(value):.15g}")
        else:
            cleaned.append(str(value).strip())
    return pd.Series(cleaned, index=series.index, dtype="object")


def merge_feature_metadata_table(
    geometry_df: pd.DataFrame,
    metadata_csv: Path,
    geometry_key_column: Optional[str] = None,
    metadata_key_column: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Join a separate preprocessor metadata CSV onto the acoustic point table.

    The join is one-to-one by point identifier. Core acoustic columns are never
    overwritten by the metadata file. Feature/topology columns in the metadata
    table take precedence over same-named non-core columns in the geometry CSV.
    """
    metadata_path = Path(metadata_csv)
    metadata_df = pd.read_csv(metadata_path)
    key_aliases = ("node_id", "N", "index", "point_id", "point", "id")
    geometry_key = _resolve_optional_column(
        geometry_df, geometry_key_column, key_aliases, "geometry join key"
    )
    metadata_key = _resolve_optional_column(
        metadata_df, metadata_key_column, key_aliases, "feature metadata join key"
    )
    if geometry_key is None or metadata_key is None:
        raise ValueError(
            "A separate feature metadata CSV requires a point-ID key in both files. "
            "Use --feature-geometry-key-column and --feature-metadata-key-column "
            "when automatic node_id/N/index detection is insufficient."
        )

    geometry_keys = _clean_join_key_values(geometry_df[geometry_key])
    metadata_keys = _clean_join_key_values(metadata_df[metadata_key])
    if np.any(geometry_keys == ""):
        raise ValueError(f"Geometry join key '{geometry_key}' contains blank values.")
    if np.any(metadata_keys == ""):
        raise ValueError(f"Metadata join key '{metadata_key}' contains blank values.")
    if geometry_keys.duplicated().any():
        duplicates = geometry_keys[geometry_keys.duplicated()].unique()[:8]
        raise ValueError(
            f"Geometry join key '{geometry_key}' is not unique; examples: {duplicates.tolist()}"
        )
    if metadata_keys.duplicated().any():
        duplicates = metadata_keys[metadata_keys.duplicated()].unique()[:8]
        raise ValueError(
            f"Metadata join key '{metadata_key}' is not unique; examples: {duplicates.tolist()}"
        )

    metadata_indexed = metadata_df.copy()
    metadata_indexed["__feature_join_key__"] = metadata_keys.to_numpy()
    metadata_indexed = metadata_indexed.set_index("__feature_join_key__", drop=True)
    result = geometry_df.copy()
    protected = {
        "x", "y", "z", "nx", "ny", "nz", "area", "vn", "vn_real", "vn_imag",
        "l", "m", "n", "frequency_hz", "case_id",
        _normalized_column_name(geometry_key),
    }
    imported_columns: List[str] = []
    skipped_columns: List[str] = []
    for column in metadata_df.columns:
        if column == metadata_key:
            continue
        if _normalized_column_name(column) in protected:
            skipped_columns.append(str(column))
            continue
        mapped = geometry_keys.map(metadata_indexed[column])
        if column in result.columns:
            result[column] = mapped.combine_first(result[column])
        else:
            result[column] = mapped
        imported_columns.append(str(column))

    matched = int(geometry_keys.isin(metadata_indexed.index).sum())
    log(
        f"Joined feature metadata '{metadata_path.name}' using geometry key "
        f"'{geometry_key}' and metadata key '{metadata_key}': matched "
        f"{matched}/{len(result)} acoustic points; imported columns="
        f"{imported_columns or ['none']}; skipped core columns={skipped_columns or ['none']}.",
        verbose,
    )
    if matched == 0:
        raise ValueError("Feature metadata join matched zero acoustic points.")
    if matched < len(result):
        log(
            f"WARNING: {len(result) - matched} acoustic points have no row in the "
            "feature metadata table; they will use blank/default feature values.",
            verbose,
        )
    result.attrs["feature_metadata_join"] = {
        "metadata_source": metadata_path.name,
        "geometry_join_key": geometry_key,
        "metadata_join_key": metadata_key,
        "matched_points": matched,
        "geometry_points": int(len(result)),
    }
    return result


def load_geometry(
    csv_path: Path,
    a: float,
    verbose: bool = True,
    face_column: Optional[str] = None,
    edge_column: Optional[str] = None,
    corner_column: Optional[str] = None,
    feature_type_column: Optional[str] = None,
    feature_id_column: Optional[str] = None,
    is_edge_column: Optional[str] = None,
    is_corner_column: Optional[str] = None,
    feature_boundary_mode: str = "auto",
    feature_connectivity_factor: float = 2.5,
    feature_metadata_csv: Optional[Path] = None,
    feature_geometry_key_column: Optional[str] = None,
    feature_metadata_key_column: Optional[str] = None,
    zero_feature_id_is_missing: bool = True,
    feature_type_map: Optional[object] = None,
) -> Geometry:
    df = pd.read_csv(csv_path)
    if feature_metadata_csv is not None:
        df = merge_feature_metadata_table(
            geometry_df=df,
            metadata_csv=Path(feature_metadata_csv),
            geometry_key_column=feature_geometry_key_column,
            metadata_key_column=feature_metadata_key_column,
            verbose=verbose,
        )
    columns = set(df.columns)

    idx_col = "N" if "N" in columns else ("node_id" if "node_id" in columns else ("index" if "index" in columns else None))
    if idx_col is None:
        log("No N/node_id/index column found; using row number as index.", verbose)
        index = np.arange(1, len(df) + 1)
    else:
        index = df[idx_col].to_numpy()

    for c in ["x", "y", "z"]:
        if c not in df.columns:
            raise ValueError(f"Input CSV must contain coordinate column '{c}'.")
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)

    vn = complex_from_columns(df, "vn")

    if all(c in df.columns for c in ["nx", "ny", "nz"]):
        normals = df[["nx", "ny", "nz"]].to_numpy(dtype=float)
        normals /= (np.linalg.norm(normals, axis=1)[:, None] + 1e-15)
        log("Using user-supplied surface normals nx, ny, nz.", verbose)
    else:
        log("WARNING: nx, ny, nz not found. Estimating normals by local PCA. ", verbose)
        log("         For production arbitrary geometries, supply outward normals.", verbose)
        normals = estimate_normals_pca(xyz)

    if "area" in df.columns:
        area = df["area"].to_numpy(dtype=float)
        log("Using user-supplied area weights.", verbose)
    else:
        total_area = 4.0 * np.pi * a * a
        area = np.full(len(df), total_area / len(df))
        log("WARNING: area column not found. Using equal weights 4*pi*a^2/N.", verbose)
        log("         For arbitrary radiators, provide local surface area weights.", verbose)

    digital_lmn = None
    if all(c in df.columns for c in ["l", "m", "n"]):
        digital_lmn = df[["l", "m", "n"]].to_numpy(dtype=float)
        log("Found digitized l,m,n coordinates; they will be carried to output.", verbose)

    features = build_feature_metadata(
        df=df,
        xyz=xyz,
        face_column=face_column,
        edge_column=edge_column,
        corner_column=corner_column,
        feature_type_column=feature_type_column,
        feature_id_column=feature_id_column,
        is_edge_column=is_edge_column,
        is_corner_column=is_corner_column,
        feature_boundary_mode=feature_boundary_mode,
        feature_connectivity_factor=feature_connectivity_factor,
        zero_feature_id_is_missing=zero_feature_id_is_missing,
        feature_type_map=feature_type_map,
        verbose=verbose,
    )
    if features is not None:
        join_info = df.attrs.get("feature_metadata_join", {})
        features.detected_columns["metadata_source"] = (
            join_info.get("metadata_source", "embedded")
        )
        features.detected_columns["geometry_join_key"] = join_info.get(
            "geometry_join_key"
        )
        features.detected_columns["metadata_join_key"] = join_info.get(
            "metadata_join_key"
        )
        if join_info:
            features.detected_columns["metadata_join_matched_points"] = str(
                join_info.get("matched_points")
            )
    patch_region = (
        features.group_code
        if features is not None and features.boundary_mode != "off"
        else None
    )
    patch_region_name = "feature_group" if patch_region is not None else None

    return Geometry(
        index=index,
        xyz=xyz,
        normals=normals,
        area=area,
        vn=vn,
        digital_lmn=digital_lmn,
        patch_region=patch_region,
        patch_region_name=patch_region_name,
        features=features,
    )


# -----------------------------
# Patch selection
# -----------------------------

def auto_patch_count(n_points: int) -> int:
    """Empirical automatic patch count.

    Chosen to reproduce the successful validation scale (N=1440 -> B~64)
    while growing sublinearly for larger geometries.
    """
    b = int(round(1.7 * math.sqrt(n_points)))
    b = max(8, b)
    b = min(b, max(1, n_points // 8))  # keep at least ~8 points per patch
    return b


def farthest_point_patches(xyz: np.ndarray, b: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Patch points using farthest point sampling for centers, then nearest center assignment."""
    n = xyz.shape[0]
    b = min(b, n)
    centers_idx = np.empty(b, dtype=int)
    centroid = xyz.mean(axis=0)
    centers_idx[0] = int(np.argmin(np.linalg.norm(xyz - centroid, axis=1)))
    min_d2 = np.sum((xyz - xyz[centers_idx[0]]) ** 2, axis=1)
    for c in range(1, b):
        centers_idx[c] = int(np.argmax(min_d2))
        d2 = np.sum((xyz - xyz[centers_idx[c]]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    centers = xyz[centers_idx]
    tree = cKDTree(centers)
    _, labels = tree.query(xyz, k=1)
    counts = np.bincount(labels, minlength=b)
    log(f"Patch assignment complete: B={b}, min/mean/max patch sizes = "
        f"{counts.min()}/{counts.mean():.1f}/{counts.max()}", verbose)
    return labels, centers_idx


def max_pairwise_diameter(xyz: np.ndarray, inds: np.ndarray) -> float:
    """Exact maximum Euclidean point-to-point diameter of one patch."""
    if len(inds) <= 1:
        return 0.0
    pts = xyz[inds]
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt(np.max(np.sum(diff * diff, axis=2))))


def normal_spread_angle(normals: np.ndarray, inds: np.ndarray) -> float:
    """Maximum angular spread of point normals inside one patch, radians."""
    if len(inds) <= 1:
        return 0.0
    n = normals[inds]
    mean = n.mean(axis=0)
    mean /= np.linalg.norm(mean) + 1e-15
    dots = np.clip(n @ mean, -1.0, 1.0)
    return float(np.max(np.arccos(dots)))


def split_indices_by_farthest_pair(xyz: np.ndarray, inds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a point set into two children using the farthest pair as seeds."""
    if len(inds) <= 1:
        return inds, np.array([], dtype=int)
    pts = xyz[inds]
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    i0, i1 = np.unravel_index(int(np.argmax(d2)), d2.shape)
    seed0 = pts[i0]
    seed1 = pts[i1]
    d0 = np.sum((pts - seed0) ** 2, axis=1)
    d1 = np.sum((pts - seed1) ** 2, axis=1)
    mask = d0 <= d1
    if np.all(mask) or not np.any(mask):
        order = np.argsort(pts[:, 0] + 0.37 * pts[:, 1] + 0.19 * pts[:, 2])
        half = len(inds) // 2
        left = inds[order[:half]]
        right = inds[order[half:]]
    else:
        left = inds[mask]
        right = inds[~mask]
    return left.astype(int), right.astype(int)


def relabel_from_patch_lists(patch_lists: Sequence[np.ndarray], n_points: int) -> np.ndarray:
    labels = np.empty(n_points, dtype=int)
    for b, inds in enumerate(patch_lists):
        labels[np.asarray(inds, dtype=int)] = b
    return labels


def initial_patch_lists_by_region(geom: Geometry, requested_B: int, verbose: bool = True) -> List[np.ndarray]:
    """Create initial patch lists, respecting explicit region/face boundaries if present."""
    N = geom.xyz.shape[0]
    patch_lists: List[np.ndarray] = []
    if geom.patch_region is None:
        labels, _ = farthest_point_patches(geom.xyz, requested_B, verbose=verbose)
        return [np.where(labels == b)[0] for b in range(len(np.unique(labels)))]

    # Allocate a requested number of initial patches to each hard feature group
    # in proportion to point count, with at least one patch per nonempty group.
    regions = np.unique(geom.patch_region)
    if requested_B < len(regions):
        log(
            f"Requested B={requested_B} is smaller than the {len(regions)} hard "
            "preprocessor feature groups; raising the initial patch count so no "
            "edge/corner/feature boundary is crossed.",
            verbose,
        )
        requested_B = len(regions)
    counts = {int(r): int(np.sum(geom.patch_region == r)) for r in regions}
    total = float(N)
    allocation = {r: max(1, int(round(requested_B * counts[int(r)] / total))) for r in regions}
    # Correct any rounding excess/deficit while preserving at least one per region.
    while sum(allocation.values()) > requested_B and any(v > 1 for v in allocation.values()):
        r = max(allocation, key=lambda rr: allocation[rr])
        if allocation[r] > 1:
            allocation[r] -= 1
        else:
            break
    while sum(allocation.values()) < requested_B:
        r = max(allocation, key=lambda rr: counts[int(rr)] / allocation[rr])
        allocation[r] += 1

    for r in regions:
        inds = np.where(geom.patch_region == r)[0]
        br = min(allocation[int(r)], len(inds))
        if br <= 1:
            patch_lists.append(inds.astype(int))
        else:
            labels_local, _ = farthest_point_patches(geom.xyz[inds], br, verbose=False)
            for b in range(br):
                patch_lists.append(inds[np.where(labels_local == b)[0]].astype(int))
    log(f"Initial feature-group-respecting patch assignment complete: B={len(patch_lists)}", verbose)
    return [p for p in patch_lists if len(p) > 0]


def refine_patches_by_diameter_and_edges(
    geom: Geometry,
    initial_B: int,
    max_patch_diameter: Optional[float],
    min_points_per_patch: int = 1,
    max_normal_angle_rad: float = math.radians(35.0),
    max_refine_passes: int = 50,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Refine patch lists until wavelength diameter and edge criteria are met.

    Supplied face/edge/corner/feature metadata are canonicalized into hard
    groups, and initial patches are created within those groups. Without hard
    metadata groups, patches with large normal spread are split as a practical
    edge/corner safeguard.
    """
    patch_lists = initial_patch_lists_by_region(geom, initial_B, verbose=verbose)
    for pass_id in range(max_refine_passes):
        changed = False
        new_lists: List[np.ndarray] = []
        for inds in patch_lists:
            d = max_pairwise_diameter(geom.xyz, inds)
            angle = normal_spread_angle(geom.normals, inds)
            too_large = max_patch_diameter is not None and d > max_patch_diameter
            crosses_region = False
            if geom.patch_region is not None:
                crosses_region = len(np.unique(geom.patch_region[inds])) > 1
            crosses_edge_by_normals = geom.patch_region is None and angle > max_normal_angle_rad
            can_split = len(inds) >= max(2, 2 * min_points_per_patch)
            if (too_large or crosses_region or crosses_edge_by_normals) and can_split:
                if crosses_region:
                    for r in np.unique(geom.patch_region[inds]):
                        rinds = inds[geom.patch_region[inds] == r]
                        if len(rinds) > 0:
                            new_lists.append(rinds.astype(int))
                else:
                    left, right = split_indices_by_farthest_pair(geom.xyz, inds)
                    if len(left) >= min_points_per_patch and len(right) >= min_points_per_patch:
                        new_lists.extend([left, right])
                    else:
                        new_lists.append(inds)
                changed = True
            else:
                new_lists.append(inds)
        patch_lists = [p for p in new_lists if len(p) > 0]
        if not changed:
            break
    labels = relabel_from_patch_lists(patch_lists, geom.xyz.shape[0])
    summary = []
    for b, inds in enumerate(patch_lists):
        summary.append({
            "patch": int(b),
            "n_points": int(len(inds)),
            "diameter": float(max_pairwise_diameter(geom.xyz, inds)),
            "normal_spread_deg": float(math.degrees(normal_spread_angle(geom.normals, inds))),
            "region": int(geom.patch_region[inds[0]]) if geom.patch_region is not None and len(inds) else -1,
        })
    counts = np.bincount(labels, minlength=len(patch_lists))
    diameters = np.array([r["diameter"] for r in summary]) if summary else np.array([0.0])
    log(f"Refined patches: B={len(patch_lists)}, min/mean/max sizes = "
        f"{counts.min()}/{counts.mean():.1f}/{counts.max()}", verbose)
    if max_patch_diameter is not None:
        log(f"Patch diameter target={max_patch_diameter:.6g}; max observed={diameters.max():.6g}", verbose)
    return labels, summary


# -----------------------------
# Local DFT / plane-wave directions
# -----------------------------

def fibonacci_directions(m: int) -> np.ndarray:
    """Well-spaced directions on the unit sphere."""
    i = np.arange(m, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / m
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * i
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def digital_pqr_from_dirs(dirs: np.ndarray, W: int) -> np.ndarray:
    return np.round(W * dirs).astype(int)


def local_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two tangents and a unit normal."""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, ref)) > 0.85:
        ref = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(ref, n)
    t1 /= np.linalg.norm(t1) + 1e-15
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-15
    return t1, t2, n


def rotated_directions(base_dirs: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Rotate base directions from local coordinates into global coordinates."""
    t1, t2, n = local_frame(normal)
    return base_dirs[:, 0:1] * t1 + base_dirs[:, 1:2] * t2 + base_dirs[:, 2:3] * n


@dataclass
class PatchBasis:
    indices: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    basis: np.ndarray        # Q matrix, shape n_patch x r
    directions: np.ndarray   # global directions, shape M x 3
    M_requested: int
    diameter: float = 0.0
    normal_spread_deg: float = 0.0
    region: int = -1


# -----------------------------
# Green blocks and compression
# -----------------------------

@dataclass
class Block:
    receiver_patch: int
    source_patch: int
    kind: str                 # 'near' or 'far'
    S: Optional[np.ndarray]   # dense block or core
    D: Optional[np.ndarray]


@dataclass
class SolverModel:
    geom: Geometry
    ka: float
    a: float
    k: float
    W: int
    B: int
    M: int
    labels: np.ndarray
    patches: List[PatchBasis]
    blocks: List[Block]
    near_blocks: int
    far_blocks: int
    pqr: np.ndarray
    lambda_acoustic: float
    lambda_velocity: Optional[float]
    lambda_star: float
    max_patch_diameter_target: Optional[float]
    patch_summary: List[Dict[str, float]]
    far_error_summary: Dict[str, float]


def self_radius_from_area(area: np.ndarray) -> np.ndarray:
    # Equivalent disk radius R0 such that pi R0^2 = area.
    return np.sqrt(np.maximum(area, 0.0) / np.pi)


def S_self_value(k: float, r0: np.ndarray) -> np.ndarray:
    """Self-cell approximation for integral of exp(-ikR)/(4piR) over local disk."""
    out = np.empty_like(r0, dtype=np.complex128)
    small = np.abs(k * r0) < 1e-8
    out[small] = r0[small] / 2.0
    out[~small] = (1.0 - np.exp(-1j * k * r0[~small])) / (2j * k)
    return out


def D_self_value(k: float, r0: np.ndarray, model: str = "cap") -> np.ndarray:
    """Finite self-cell correction for D. The 1/2 jump is handled separately.

    'zero' is the conservative baseline. 'cap' is the local cap/disk correction
    used in the sphere validation studies. For arbitrary sharp-edged radiators,
    this model should be checked with refinement or replaced by element quadrature.
    """
    if model == "zero":
        return np.zeros_like(r0, dtype=np.complex128)
    if model != "cap":
        raise ValueError("self_d_model must be 'zero' or 'cap'.")
    out = np.empty_like(r0, dtype=np.complex128)
    small = np.abs(k * r0) < 1e-8
    # Low-frequency limiting behavior of the cap expression is small; use zero.
    out[small] = 0.0
    rr = r0[~small]
    out[~small] = 0.25 * (rr * np.exp(-1j * k * rr) + (2.0 / (1j * k)) * (np.exp(-1j * k * rr) - 1.0))
    return out


def green_blocks_exact(
    xyz: np.ndarray,
    normals: np.ndarray,
    area: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    k: float,
    self_d_model: str = "cap",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build exact dense S and D block for receiver indices I and source indices J."""
    X = xyz[I]
    Y = xyz[J]
    nY = normals[J]
    aY = area[J]

    Rvec = X[:, None, :] - Y[None, :, :]
    R = np.linalg.norm(Rvec, axis=2)

    S = np.zeros((len(I), len(J)), dtype=np.complex128)
    D = np.zeros_like(S)

    mask = R > 1e-14
    Rm = R[mask]
    phase = np.exp(-1j * k * Rm)
    # S_ij = area_j G_ij
    area_cols = np.broadcast_to(aY[None, :], R.shape)
    S[mask] = area_cols[mask] * phase / (4.0 * np.pi * Rm)

    # D_ij = area_j * dG/dn_source. Sign convention matched to
    # (1/2 I - D) p = i k S v_n.
    # dG/dn_y = (n_y dot (x-y)) (1+ikR) exp(-ikR) / (4pi R^3)
    dot = np.einsum("ijc,jc->ij", Rvec, nY)
    D[mask] = area_cols[mask] * dot[mask] * (1.0 + 1j * k * Rm) * phase / (4.0 * np.pi * Rm**3)

    # Replace exact self pairs, if any, with finite cell corrections.
    # This only applies when the same global point appears in I and J.
    if np.intersect1d(I, J).size > 0:
        # Map source global index to local source column.
        source_pos = {int(g): jj for jj, g in enumerate(J)}
        r0_all = self_radius_from_area(area)
        for ii, gi in enumerate(I):
            jj = source_pos.get(int(gi))
            if jj is not None:
                r0 = np.array([r0_all[gi]])
                S[ii, jj] = S_self_value(k, r0)[0]
                D[ii, jj] = D_self_value(k, r0, self_d_model)[0]
    return S, D


def build_one_patch_basis(
    geom: Geometry,
    inds: np.ndarray,
    M: int,
    k: float,
) -> PatchBasis:
    """Build one local QR basis for a patch with its own M."""
    inds = np.asarray(inds, dtype=int)
    center = geom.xyz[inds].mean(axis=0)
    normal = geom.normals[inds].mean(axis=0)
    normal /= np.linalg.norm(normal) + 1e-15
    M_eff = int(max(1, min(M, len(inds))))
    base_dirs = fibonacci_directions(M_eff)
    dirs = rotated_directions(base_dirs, normal)
    local = geom.xyz[inds] - center
    G = np.exp(-1j * k * (local @ dirs.T))
    Q, _ = np.linalg.qr(G, mode="reduced")
    return PatchBasis(
        indices=inds,
        center=center,
        normal=normal,
        basis=Q,
        directions=dirs,
        M_requested=M_eff,
        diameter=max_pairwise_diameter(geom.xyz, inds),
        normal_spread_deg=math.degrees(normal_spread_angle(geom.normals, inds)),
        region=int(geom.patch_region[inds[0]]) if geom.patch_region is not None and len(inds) else -1,
    )


def build_patch_basis(
    geom: Geometry,
    labels: np.ndarray,
    B: int,
    M_per_patch: Sequence[int],
    k: float,
    verbose: bool = True,
) -> List[PatchBasis]:
    patches: List[PatchBasis] = []
    for b in range(B):
        inds = np.where(labels == b)[0]
        if len(inds) == 0:
            continue
        patches.append(build_one_patch_basis(geom, inds, int(M_per_patch[b]), k))
    if len(patches) != B:
        log(f"WARNING: requested B={B}, but only built {len(patches)} non-empty patches.", verbose)
    return patches


def parse_m_schedule(text: str) -> List[int]:
    vals = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("M schedule cannot be empty.")
    return vals


def nearest_schedule_value(m: int, schedule: Sequence[int], nmax: int) -> int:
    candidates = [min(v, nmax) for v in schedule if v >= m]
    if candidates:
        return int(max(1, min(candidates[0], nmax)))
    return int(max(1, min(max(schedule), nmax)))


def next_schedule_value(m: int, schedule: Sequence[int], nmax: int) -> int:
    for v in schedule:
        vv = int(min(v, nmax))
        if vv > m:
            return vv
    return int(m)


def initial_m_for_patch(n_points: int, k: float, diameter: float, schedule: Sequence[int], M_user: Optional[int]) -> int:
    if M_user is not None:
        return int(max(1, min(M_user, n_points)))
    kd = abs(k) * diameter
    estimate = int(math.ceil(4.0 * (1.0 + kd) ** 2))
    estimate = max(8, estimate)
    return nearest_schedule_value(estimate, schedule, n_points)


def projection_error(A: np.ndarray, Qa: np.ndarray, Qb: np.ndarray) -> float:
    denom = np.linalg.norm(A, ord="fro") + 1e-30
    Acomp = Qa @ (Qa.conj().T @ A @ Qb) @ Qb.conj().T
    return float(np.linalg.norm(A - Acomp, ord="fro") / denom)


def adapt_patch_bases_by_far_error(
    geom: Geometry,
    labels: np.ndarray,
    patches: List[PatchBasis],
    k: float,
    near_threshold: float,
    self_d_model: str,
    m_schedule: Sequence[int],
    far_error_tol: float,
    max_adapt_passes: int = 8,
    verbose: bool = True,
) -> Tuple[List[PatchBasis], Dict[str, float]]:
    """Increase per-patch M until far-block S and D projection errors satisfy tolerance."""
    centroid = geom.xyz.mean(axis=0)
    B = len(patches)
    last_errors: List[Tuple[float, float]] = []
    for pass_id in range(max_adapt_passes):
        changed = False
        worst_S = 0.0
        worst_D = 0.0
        sum_S = 0.0
        sum_D = 0.0
        n_far = 0
        for a_idx, pa in enumerate(patches):
            for b_idx, pb in enumerate(patches):
                sep = patch_center_angle_or_distance(pa, pb, centroid)
                is_near = (a_idx == b_idx) or (sep < near_threshold)
                if is_near:
                    continue
                S_exact, D_exact = green_blocks_exact(
                    geom.xyz, geom.normals, geom.area,
                    pa.indices, pb.indices, k, self_d_model=self_d_model,
                )
                eS = projection_error(S_exact, pa.basis, pb.basis)
                eD = projection_error(D_exact, pa.basis, pb.basis)
                err = max(eS, eD)
                worst_S = max(worst_S, eS)
                worst_D = max(worst_D, eD)
                sum_S += eS
                sum_D += eD
                n_far += 1
                if err > far_error_tol:
                    new_a = next_schedule_value(pa.M_requested, m_schedule, len(pa.indices))
                    new_b = next_schedule_value(pb.M_requested, m_schedule, len(pb.indices))
                    # Increase the smaller/available side first, then both if possible.
                    if new_a > pa.M_requested and (pa.M_requested <= pb.M_requested or new_b == pb.M_requested):
                        patches[a_idx] = build_one_patch_basis(geom, pa.indices, new_a, k)
                        changed = True
                    elif new_b > pb.M_requested:
                        patches[b_idx] = build_one_patch_basis(geom, pb.indices, new_b, k)
                        changed = True
                    elif new_a > pa.M_requested:
                        patches[a_idx] = build_one_patch_basis(geom, pa.indices, new_a, k)
                        changed = True
        mean_S = sum_S / max(n_far, 1)
        mean_D = sum_D / max(n_far, 1)
        last_errors.append((worst_S, worst_D))
        log(f"Adaptive M pass {pass_id + 1}: far blocks={n_far}, "
            f"max error S={worst_S:.3e}, D={worst_D:.3e}; "
            f"mean S={mean_S:.3e}, D={mean_D:.3e}; "
            f"M range={min(p.M_requested for p in patches)}-{max(p.M_requested for p in patches)}", verbose)
        if not changed or max(worst_S, worst_D) <= far_error_tol:
            break
    Mvals = np.array([p.M_requested for p in patches], dtype=int)
    summary = {
        "far_error_tol": float(far_error_tol),
        "far_error_max_S": float(last_errors[-1][0] if last_errors else 0.0),
        "far_error_max_D": float(last_errors[-1][1] if last_errors else 0.0),
        "M_min": int(Mvals.min()) if len(Mvals) else 0,
        "M_mean": float(Mvals.mean()) if len(Mvals) else 0.0,
        "M_max": int(Mvals.max()) if len(Mvals) else 0,
        "adaptive_passes": int(len(last_errors)),
    }
    return patches, summary


def patch_center_angle_or_distance(pa: PatchBasis, pb: PatchBasis, centroid: np.ndarray) -> float:
    """Use angle about the centroid when meaningful; fallback to Euclidean distance."""
    va = pa.center - centroid
    vb = pb.center - centroid
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na > 1e-12 and nb > 1e-12:
        cosang = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
        return float(np.arccos(cosang))
    return float(np.linalg.norm(pa.center - pb.center))


def build_model(
    geom: Geometry,
    ka: float,
    a: float,
    W: int,
    B_user: Optional[int] = None,
    M_user: Optional[int] = None,
    near_threshold: float = 0.75,
    self_d_model: str = "cap",
    lambda_velocity: Optional[float] = None,
    patches_per_wavelength: float = 6.0,
    max_patch_diameter: Optional[float] = None,
    max_normal_angle_deg: float = 35.0,
    min_points_per_patch: int = 1,
    far_error_tol: float = 1e-4,
    m_schedule: Sequence[int] = (8, 12, 16, 24, 32, 48, 64),
    disable_adaptive_M: bool = False,
    verbose: bool = True,
) -> SolverModel:
    N = geom.xyz.shape[0]
    k = ka / a
    lambda_acoustic = 2.0 * np.pi / abs(k) if abs(k) > 0 else np.inf
    lambda_star = min(lambda_acoustic, lambda_velocity) if lambda_velocity is not None else lambda_acoustic
    if max_patch_diameter is None and np.isfinite(lambda_star):
        max_patch_diameter = lambda_star / patches_per_wavelength

    B0 = B_user if B_user is not None else auto_patch_count(N)
    log("\n--- Patch-DFT Green setup ---", verbose)
    log(f"Input points N={N}", verbose)
    log(f"ka={ka:g}, a={a:g}, physical wave number k=ka/a={k:g}", verbose)
    log(f"lambda_acoustic={lambda_acoustic:.6g}; lambda_velocity={lambda_velocity}; lambda_star={lambda_star:.6g}", verbose)
    log(f"Patch-diameter target d_max <= {max_patch_diameter:.6g} "
        f"({patches_per_wavelength:g} patches per shortest wavelength)", verbose)
    log(f"Digital multiplier W={W}", verbose)
    log(f"Initial automatic/requested patch count B0={B0}", verbose)

    labels, patch_summary = refine_patches_by_diameter_and_edges(
        geom=geom,
        initial_B=B0,
        max_patch_diameter=max_patch_diameter,
        min_points_per_patch=min_points_per_patch,
        max_normal_angle_rad=math.radians(max_normal_angle_deg),
        verbose=verbose,
    )
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[x] for x in labels], dtype=int)
    B = len(unique)
    counts = np.bincount(labels, minlength=B)
    log(f"Final patch sizes min/mean/max = {counts.min()}/{counts.mean():.1f}/{counts.max()}", verbose)

    # Per-patch initial M. This replaces the old global M-only rule, but still
    # respects --M when supplied by using it as the fixed requested starting value.
    m_schedule = sorted(set(int(v) for v in m_schedule if int(v) > 0))
    if not m_schedule:
        raise ValueError("m_schedule cannot be empty")
    M_per_patch = []
    for b in range(B):
        inds = np.where(labels == b)[0]
        d = max_pairwise_diameter(geom.xyz, inds)
        M_per_patch.append(initial_m_for_patch(len(inds), k, d, m_schedule, M_user))
    log(f"Initial M range = {min(M_per_patch)}-{max(M_per_patch)}", verbose)

    patches = build_patch_basis(geom, labels, B, M_per_patch, k, verbose=verbose)
    if disable_adaptive_M:
        far_error_summary = {
            "far_error_tol": float(far_error_tol), "far_error_max_S": np.nan, "far_error_max_D": np.nan,
            "M_min": int(min(M_per_patch)), "M_mean": float(np.mean(M_per_patch)),
            "M_max": int(max(M_per_patch)), "adaptive_passes": 0,
        }
        log("Adaptive far-block M control disabled by user.", verbose)
    else:
        patches, far_error_summary = adapt_patch_bases_by_far_error(
            geom=geom, labels=labels, patches=patches, k=k,
            near_threshold=near_threshold, self_d_model=self_d_model,
            m_schedule=m_schedule, far_error_tol=far_error_tol, verbose=verbose,
        )

    # Keep a global pqr catalog for compatibility/output. Actual patches may use
    # fewer or more directions according to PatchBasis.M_requested.
    M_report = int(max(p.M_requested for p in patches)) if patches else 0
    pqr = digital_pqr_from_dirs(fibonacci_directions(max(M_report, 1)), W)[:M_report]
    log(f"Final adaptive M range = {min(p.M_requested for p in patches)}-{max(p.M_requested for p in patches)}", verbose)

    centroid = geom.xyz.mean(axis=0)
    blocks: List[Block] = []
    near_count = 0
    far_count = 0
    log("Building final patch-patch blocks...", verbose)
    for a_idx, pa in enumerate(patches):
        for b_idx, pb in enumerate(patches):
            sep = patch_center_angle_or_distance(pa, pb, centroid)
            is_near = (a_idx == b_idx) or (sep < near_threshold)
            S_exact, D_exact = green_blocks_exact(
                geom.xyz, geom.normals, geom.area,
                pa.indices, pb.indices, k, self_d_model=self_d_model,
            )
            if is_near:
                blocks.append(Block(a_idx, b_idx, "near", S_exact, D_exact))
                near_count += 1
            else:
                Ga = pa.basis
                Gb = pb.basis
                S_core = Ga.conj().T @ S_exact @ Gb
                D_core = Ga.conj().T @ D_exact @ Gb
                blocks.append(Block(a_idx, b_idx, "far", S_core, D_core))
                far_count += 1
    log(f"Blocks built: near/self dense={near_count}, far DFT-compressed={far_count}", verbose)

    # Refresh patch summary with final M values.
    patch_summary = []
    for b, patch in enumerate(patches):
        record: Dict[str, object] = {
            "patch": int(b),
            "n_points": int(len(patch.indices)),
            "diameter": float(patch.diameter),
            "normal_spread_deg": float(patch.normal_spread_deg),
            "region": int(patch.region),
            "M_requested": int(patch.M_requested),
            "rank_stored": int(patch.basis.shape[1]),
        }
        record.update(feature_subset_summary(geom, patch.indices))
        patch_summary.append(record)

    return SolverModel(
        geom=geom, ka=ka, a=a, k=k, W=W, B=B, M=M_report,
        labels=labels, patches=patches, blocks=blocks,
        near_blocks=near_count, far_blocks=far_count, pqr=pqr,
        lambda_acoustic=float(lambda_acoustic), lambda_velocity=lambda_velocity,
        lambda_star=float(lambda_star), max_patch_diameter_target=max_patch_diameter,
        patch_summary=patch_summary, far_error_summary=far_error_summary,
    )


def apply_operator(model: SolverModel, x: np.ndarray, which: str) -> np.ndarray:
    """Apply S_DFT or D_DFT to vector x using stored patch blocks."""
    assert which in ("S", "D")
    y = np.zeros_like(x, dtype=np.complex128)
    patches = model.patches
    for block in model.blocks:
        pa = patches[block.receiver_patch]
        pb = patches[block.source_patch]
        I = pa.indices
        J = pb.indices
        A = block.S if which == "S" else block.D
        if A is None:
            continue
        if block.kind == "near":
            y[I] += A @ x[J]
        else:
            # y_a += G_a C_ab G_b^H x_b
            tmp = pb.basis.conj().T @ x[J]
            tmp = A @ tmp
            y[I] += pa.basis @ tmp
    return y


def solve_pressure(
    model: SolverModel,
    rtol: float = 1e-5,
    maxiter: int = 200,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve (1/2 I - D)p = i k S vn by GMRES."""
    vn = model.geom.vn
    rhs = 1j * model.k * apply_operator(model, vn, "S")
    N = len(vn)

    def matvec(p: np.ndarray) -> np.ndarray:
        return 0.5 * p - apply_operator(model, p, "D")

    Aop = LinearOperator((N, N), matvec=matvec, dtype=np.complex128)
    residuals: List[float] = []

    def cb(residual):
        # scipy may pass scalar residual norm for callback_type='pr_norm'.
        try:
            residuals.append(float(residual))
        except Exception:
            pass

    log("Solving closed-body boundary equation by GMRES...", verbose)
    try:
        p, info = gmres(Aop, rhs, rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb, callback_type="pr_norm")
    except TypeError:
        # Older scipy fallback.
        p, info = gmres(Aop, rhs, tol=rtol, maxiter=maxiter, callback=cb)

    res = np.linalg.norm(matvec(p) - rhs) / (np.linalg.norm(rhs) + 1e-30)
    if info == 0:
        log(f"GMRES converged. Relative residual={res:.3e}, iterations={len(residuals)}", verbose)
    else:
        log(f"WARNING: GMRES returned info={info}. Relative residual={res:.3e}, iterations={len(residuals)}", verbose)
    stats = {"gmres_info": float(info), "relative_residual": float(res), "iterations": float(len(residuals))}
    return p, stats


def compute_outputs(model: SolverModel, p: np.ndarray) -> Dict[str, complex | float]:
    area = model.geom.area
    vn = model.geom.vn
    denom = np.sum(area * np.abs(vn) ** 2) + 1e-30
    Z = np.sum(area * p * np.conj(vn)) / denom
    power = 0.5 * np.real(np.sum(area * p * np.conj(vn)))
    return {"surface_impedance": Z, "radiated_power_rhoc1": float(power)}


def save_outputs(model: SolverModel, p: np.ndarray, stats: Dict[str, float], metrics: Dict[str, complex | float], out_prefix: Path, verbose: bool = True) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    g = model.geom
    df = pd.DataFrame({
        "N": g.index,
        "x": g.xyz[:, 0], "y": g.xyz[:, 1], "z": g.xyz[:, 2],
        "nx": g.normals[:, 0], "ny": g.normals[:, 1], "nz": g.normals[:, 2],
        "area": g.area,
        "vn_real": np.real(g.vn), "vn_imag": np.imag(g.vn),
        "p_real": np.real(p), "p_imag": np.imag(p),
        "p_abs": np.abs(p), "p_phase_rad": np.angle(p),
        "patch": model.labels,
    })
    if g.digital_lmn is not None:
        df.insert(4, "l", g.digital_lmn[:, 0])
        df.insert(5, "m", g.digital_lmn[:, 1])
        df.insert(6, "n", g.digital_lmn[:, 2])
    if g.features is not None:
        df["feature_group_code"] = g.features.group_code
        df["feature_group_label"] = g.features.group_label
        df["feature_class"] = g.features.feature_class
        df["feature_type"] = g.features.feature_type_raw
        df["face_id"] = g.features.face_id
        df["edge_id"] = g.features.edge_id
        df["corner_id"] = g.features.corner_id
        df["feature_id"] = g.features.feature_id
    pressure_path = out_prefix.with_suffix("_pressure.csv") if False else Path(str(out_prefix) + "_pressure.csv")
    df.to_csv(pressure_path, index=False)

    pqr_df = pd.DataFrame(model.pqr, columns=["p", "q", "r"])
    pqr_path = Path(str(out_prefix) + "_fibonacci_pqr.csv")
    pqr_df.to_csv(pqr_path, index=False)

    patch_path = Path(str(out_prefix) + "_patch_summary.csv")
    pd.DataFrame(model.patch_summary).to_csv(patch_path, index=False)

    Z = metrics["surface_impedance"]
    report = {
        "N_points": int(len(g.index)),
        "B_patches": int(model.B),
        "M_plane_wave_terms_max_per_far_block": int(model.M),
        "W_multiplier": int(model.W),
        "ka": float(model.ka),
        "a": float(model.a),
        "k": float(model.k),
        "lambda_acoustic": float(model.lambda_acoustic),
        "lambda_velocity": None if model.lambda_velocity is None else float(model.lambda_velocity),
        "lambda_star": float(model.lambda_star),
        "max_patch_diameter_target": None if model.max_patch_diameter_target is None else float(model.max_patch_diameter_target),
        "near_blocks_dense": int(model.near_blocks),
        "far_blocks_dft_compressed": int(model.far_blocks),
        "far_block_error_control": model.far_error_summary,
        "gmres": stats,
        "surface_impedance_real": float(np.real(Z)),
        "surface_impedance_imag": float(np.imag(Z)),
        "radiated_power_rhoc1": metrics["radiated_power_rhoc1"],
        "notes": [
            "This version uses dense near/self Green blocks and DFT-compressed far blocks.",
            "Patch count is refined by maximum patch diameter and sharp-edge/normal-spread controls.",
            "Far-block M is selected adaptively from measured S and D projection error unless disabled.",
            "If nx,ny,nz or area were not supplied, estimates were used; provide them for production accuracy.",
            "Pressure normalization assumes rho*c = 1. Multiply by rho*c for physical pressure units if velocity is in m/s.",
            "Preprocessor face/edge/corner/feature metadata are carried through and used as hard patch boundaries according to feature_boundary_mode.",
        ],
    }
    report_path = Path(str(out_prefix) + "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log(f"Saved pressure results: {pressure_path}", verbose)
    log(f"Saved Fibonacci digital directions: {pqr_path}", verbose)
    log(f"Saved patch summary: {patch_path}", verbose)
    log(f"Saved run report: {report_path}", verbose)


# -----------------------------
# CLI
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Patch-block DFT-compressed Green acoustic radiation solver")
    parser.add_argument("input_csv", nargs="?", help="Input CSV with N,x,y,z,vn or vn_real/vn_imag; recommended nx,ny,nz,area")
    parser.add_argument("--feature-metadata-csv", default=None)
    parser.add_argument("--feature-geometry-key-column", default=None)
    parser.add_argument("--feature-metadata-key-column", default=None)
    parser.add_argument("--feature-boundary-mode", choices=["auto", "strict", "face-only", "off"], default="auto")
    parser.add_argument("--face-column", default=None)
    parser.add_argument("--edge-column", default=None)
    parser.add_argument("--corner-column", default=None)
    parser.add_argument("--feature-type-column", default=None)
    parser.add_argument("--feature-type-map", default="")
    parser.add_argument("--feature-id-column", default=None)
    parser.add_argument("--is-edge-column", default=None)
    parser.add_argument("--is-corner-column", default=None)
    parser.add_argument("--feature-zero-id-is-valid", action="store_true")
    parser.add_argument("--feature-connectivity-factor", type=float, default=2.5)
    parser.add_argument("--ka", type=float, default=1.0, help="Dimensionless wave number ka")
    parser.add_argument("--a", type=float, default=1.0, help="Typical radiator length/radius a, same units as x,y,z")
    parser.add_argument("--W", type=int, default=100, help="Digital DFT rounding multiplier for p,q,r directions")
    parser.add_argument("--B", type=int, default=None, help="Number of patches. If omitted, chosen automatically from N")
    parser.add_argument("--M", type=int, default=None, help="Initial/requested local plane-wave modes. If omitted, chosen from k*d_patch and then adapted")
    parser.add_argument("--lambda-v", type=float, default=None, help="Shortest wavelength in the prescribed surface-velocity pattern, same units as x,y,z")
    parser.add_argument("--patches-per-wavelength", type=float, default=6.0, help="Patch-diameter rule: d_patch <= lambda_star / this value")
    parser.add_argument("--max-patch-diameter", type=float, default=None, help="Direct patch-diameter limit. Overrides lambda_star/patches_per_wavelength when supplied")
    parser.add_argument("--max-normal-angle-deg", type=float, default=35.0, help="Fallback sharp-edge split: split patches whose normal spread exceeds this angle when no face/region column exists")
    parser.add_argument("--min-points-per-patch", type=int, default=1, help="Do not split below this many points per child patch")
    parser.add_argument("--far-error-tol", type=float, default=1e-4, help="Measured far-block projection error tolerance for both S and D")
    parser.add_argument("--M-schedule", default="8,12,16,24,32,48,64", help="Comma-separated candidate M values for adaptive far-block control")
    parser.add_argument("--disable-adaptive-M", action="store_true", help="Use initial M values only; do not increase M by measured far-block error")
    parser.add_argument("--near-threshold", type=float, default=0.75, help="Near-block threshold. For closed bodies near sphere, radians about centroid")
    parser.add_argument("--self-d-model", choices=["zero", "cap"], default="cap", help="Self-cell correction for D operator")
    parser.add_argument("--rtol", type=float, default=1e-5, help="GMRES relative tolerance")
    parser.add_argument("--maxiter", type=int, default=200, help="GMRES maximum iterations")
    parser.add_argument("--out", default="patch_dft_output", help="Output prefix, without extension")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    parser.add_argument("--make-demo-sphere", type=str, default=None, help="Write a demo pulsating sphere CSV to this path and exit")
    parser.add_argument("--demo-N", type=int, default=720, help="Number of demo sphere points")
    parser.add_argument("--demo-radius", type=float, default=1.0, help="Demo sphere radius")
    args = parser.parse_args(argv)
    verbose = not args.quiet

    if args.make_demo_sphere:
        write_demo_sphere_csv(Path(args.make_demo_sphere), args.demo_N, args.demo_radius, verbose=verbose)
        return 0

    if not args.input_csv:
        parser.error("input_csv is required unless --make-demo-sphere is used")

    geom = load_geometry(
        Path(args.input_csv),
        a=args.a,
        verbose=verbose,
        face_column=args.face_column,
        edge_column=args.edge_column,
        corner_column=args.corner_column,
        feature_type_column=args.feature_type_column,
        feature_id_column=args.feature_id_column,
        is_edge_column=args.is_edge_column,
        is_corner_column=args.is_corner_column,
        feature_boundary_mode=args.feature_boundary_mode,
        feature_connectivity_factor=args.feature_connectivity_factor,
        feature_metadata_csv=(Path(args.feature_metadata_csv) if args.feature_metadata_csv else None),
        feature_geometry_key_column=args.feature_geometry_key_column,
        feature_metadata_key_column=args.feature_metadata_key_column,
        zero_feature_id_is_missing=not args.feature_zero_id_is_valid,
        feature_type_map=args.feature_type_map,
    )
    model = build_model(
        geom=geom, ka=args.ka, a=args.a, W=args.W, B_user=args.B, M_user=args.M,
        near_threshold=args.near_threshold, self_d_model=args.self_d_model,
        lambda_velocity=args.lambda_v, patches_per_wavelength=args.patches_per_wavelength,
        max_patch_diameter=args.max_patch_diameter, max_normal_angle_deg=args.max_normal_angle_deg,
        min_points_per_patch=args.min_points_per_patch, far_error_tol=args.far_error_tol,
        m_schedule=parse_m_schedule(args.M_schedule), disable_adaptive_M=args.disable_adaptive_M,
        verbose=verbose,
    )
    p, stats = solve_pressure(model, rtol=args.rtol, maxiter=args.maxiter, verbose=verbose)
    metrics = compute_outputs(model, p)
    Z = metrics["surface_impedance"]
    log("\n--- Results ---", verbose)
    log(f"Surface impedance Z = {np.real(Z): .8e} + {np.imag(Z): .8e} i", verbose)
    log(f"Radiated power, rho*c=1 convention = {metrics['radiated_power_rhoc1']:.8e}", verbose)
    save_outputs(model, p, stats, metrics, Path(args.out), verbose=verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
