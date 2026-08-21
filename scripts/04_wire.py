#!/usr/bin/env python3
"""Self-contained 04_wire stage built from backup solid/dash detectors.

Input/output paths and JSON shape match the previous 04_wire.py:
  * outputs/03_rect/<pdf_stem>/circuit_images/page_NNN.png
  * outputs/04_wire/<pdf_stem>/{images,json,debug,review.pdf,result.pdf}

The detector implementations are embedded verbatim from:
  * scripts/backup/02_solid_wire.py
  * scripts/backup/03_dash_wire.py
"""
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import time
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import fitz
import numpy as np
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / Path(__file__).stem
SOURCE_STAGE = "03_rect"
STAGE_NAME = Path(__file__).stem
GREEN = (0, 190, 0)
ORANGE = (255, 145, 0)
_SOLID_WIRE_SOURCE = r'''#!/usr/bin/env python3
"""Minimal rectangle-only wire probe.

This script intentionally stops before higher-level symbol interpretation. It
renders the Phase 1 page, binarizes it, and visualizes:

1. long horizontal/vertical projected rectangles in green;
2. short horizontal/vertical projected rectangles in cyan;
3. short-line groups in yellow-green.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import fitz
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs"
STAGE_NAME = Path(__file__).stem
PHASE1_STAGE = "01_circuit"


@dataclass
class PageInfo:
    pdf: str
    page: int
    dpi: int
    width: int
    height: int

    @property
    def mapping(self) -> dict[str, Any]:
        return {
            "image_origin": "top-left",
            "pixel_units": "pixels",
            "source": f"outputs/{PHASE1_STAGE}/<pdf>/clean_image",
        }


@dataclass
class DetectedObject:
    id: str
    type: str
    confidence: float
    bbox: list[int]
    geometry: dict[str, Any]
    attributes: dict[str, Any] = field(default_factory=dict)
    source_phase: str = STAGE_NAME

    def json(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [], "")}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def resolve_pdf(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, ROOT_DIR / path, INPUT_DIR / path]
    if path.suffix.lower() != ".pdf":
        candidates.append(INPUT_DIR / f"{value}.pdf")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"PDF not found: {value}. Looked directly and in {INPUT_DIR}")


def resolve_phase1_clean_dir(pdf_path: Path) -> Path:
    clean_dir = OUTPUT_DIR / PHASE1_STAGE / pdf_path.stem / "clean_image"
    if not clean_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Phase 1 clean images: {clean_dir}. "
            f"Run: python scripts/01_circuit.py --pdf {pdf_path.name}"
        )
    return clean_dir


def available_clean_pages(clean_dir: Path) -> list[int]:
    pages: list[int] = []
    for path in sorted(clean_dir.glob("page_*_clean.png")):
        try:
            pages.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not pages:
        raise FileNotFoundError(f"No page_*_clean.png files found under: {clean_dir}")
    return pages


def parse_pages(spec: str | None, available_pages: Sequence[int]) -> list[int]:
    available = set(int(page) for page in available_pages)
    if not spec:
        return sorted(available)
    requested: set[int] = set()
    try:
        for token in spec.split(","):
            token = token.strip()
            if "-" in token:
                start, end = (int(part) for part in token.split("-", 1))
            else:
                start = end = int(token)
            if start < 1 or end < start:
                raise ValueError
            requested.update(range(start, end + 1))
    except ValueError as exc:
        raise ValueError(f"Invalid --pages '{spec}'; use 3 or 1-3,7,10-12") from exc
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Requested page(s) {missing} are missing from Phase 1 clean images")
    return sorted(requested)


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    resolved_output = root.resolve()
    resolved_stage = (OUTPUT_DIR / STAGE_NAME).resolve()
    if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
    shutil.rmtree(root)


def make_output_dirs(stem: str) -> tuple[Path, Path, Path]:
    root = OUTPUT_DIR / STAGE_NAME / stem
    clear_output_root(root)
    image_dir = root / "image"
    json_dir = root / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return root, image_dir, json_dir


def load_clean_image(clean_dir: Path, page_number: int) -> np.ndarray:
    path = clean_dir / f"page_{page_number:03d}_clean.png"
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb(path: Path, img: np.ndarray, dpi: int = 300) -> None:
    Image.fromarray(img).save(path, dpi=(dpi, dpi))


def images_to_pdf(paths: Sequence[Path], dest: Path, dpi: int) -> None:
    doc = fitz.open()
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        page = doc.new_page(width=width * 72 / dpi, height=height * 72 / dpi)
        page.insert_image(page.rect, filename=str(path), keep_proportion=True)
    doc.save(dest, deflate=True)
    doc.close()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def page_json(info: PageInfo, objects: Sequence[DetectedObject], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "pdf": info.pdf,
        "page": info.page,
        "dpi": info.dpi,
        "image_size": {"width": info.width, "height": info.height},
        "coordinate_mapping": info.mapping,
        "objects": [obj.json() for obj in objects],
    }
    if extra:
        data.update(extra)
    return data


def foreground(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < 128


def mask_to_relative_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for y, row in enumerate(mask.astype(bool)):
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        start = int(xs[0])
        previous = int(xs[0])
        for value in xs[1:]:
            current = int(value)
            if current == previous + 1:
                previous = current
                continue
            runs.append([int(y), start, previous + 1])
            start = current
            previous = current
        runs.append([int(y), start, previous + 1])
    return runs


def relative_runs_to_mask(runs: Sequence[Sequence[int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for run in runs:
        if len(run) != 3:
            continue
        y, x1, x2 = [int(value) for value in run]
        if 0 <= y < shape[0]:
            mask[y, max(0, x1):min(shape[1], x2)] = True
    return mask


def run_pixel_count(runs: Sequence[Sequence[int]]) -> int:
    return int(sum(max(0, int(run[2]) - int(run[1])) for run in runs if len(run) == 3))


def foreground_with_antialias_edges(
    rgb: np.ndarray,
    strict_mask: np.ndarray | None = None,
    gray_threshold: int = 175,
) -> np.ndarray:
    if strict_mask is None:
        strict_mask = foreground(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (strict_mask > 0) | (gray < int(gray_threshold))


def expand_axis_segment_edge_pixels(
    rgb: np.ndarray,
    strict_mask: np.ndarray,
    segment: dict[str, Any],
    gray_threshold: int = 175,
    normal_padding: int = 2,
) -> dict[str, Any] | None:
    orientation = str(segment.get("orientation", ""))
    if orientation not in {"horizontal", "vertical"}:
        return None
    h, w = strict_mask.shape
    x1, y1, x2, y2 = [int(v) for v in segment["bbox"]]
    pad = max(0, int(normal_padding))
    if orientation == "horizontal":
        px1, py1, px2, py2 = x1, y1 - pad, x2, y2 + pad
    else:
        px1, py1, px2, py2 = x1 - pad, y1, x2 + pad, y2
    px1 = max(0, min(w, px1))
    px2 = max(0, min(w, px2))
    py1 = max(0, min(h, py1))
    py2 = max(0, min(h, py2))
    if px2 <= px1 or py2 <= py1:
        return None
    wide = foreground_with_antialias_edges(rgb, strict_mask, gray_threshold)
    local_wide = wide[py1:py2, px1:px2]
    seed = np.zeros_like(local_wide, dtype=bool)
    sx1, sx2 = max(px1, x1) - px1, min(px2, x2) - px1
    sy1, sy2 = max(py1, y1) - py1, min(py2, y2) - py1
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    seed[sy1:sy2, sx1:sx2] = strict_mask[max(py1, y1):min(py2, y2), max(px1, x1):min(px2, x2)] > 0
    seed &= local_wide
    if not bool(np.any(seed)):
        return None
    _labels_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(local_wide.astype(np.uint8), connectivity=8)
    seed_labels = {int(value) for value in np.unique(labels[seed]) if int(value) > 0}
    if not seed_labels:
        return None
    merged = np.isin(labels, list(seed_labels))
    ys, xs = np.nonzero(merged)
    if xs.size == 0:
        return None
    bx1 = int(px1 + int(np.min(xs)))
    by1 = int(py1 + int(np.min(ys)))
    bx2 = int(px1 + int(np.max(xs)) + 1)
    by2 = int(py1 + int(np.max(ys)) + 1)
    crop = merged[by1 - py1:by2 - py1, bx1 - px1:bx2 - px1]
    runs = mask_to_relative_runs(crop)
    strict_count = int(np.count_nonzero(seed))
    return {
        "bbox": [bx1, by1, bx2, by2],
        "pixel_runs": runs,
        "added_pixels": max(0, run_pixel_count(runs) - strict_count),
        "gray_threshold": int(gray_threshold),
        "normal_padding": int(pad),
    }


def apply_axis_line_edge_expansion(
    rgb: np.ndarray,
    strict_mask: np.ndarray,
    segments: Iterable[dict[str, Any]],
    gray_threshold: int = 175,
    normal_padding: int = 2,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for segment in segments:
        expansion = expand_axis_segment_edge_pixels(rgb, strict_mask, segment, gray_threshold, normal_padding)
        if expansion is None:
            continue
        old_bbox = [int(v) for v in segment["bbox"]]
        new_bbox = [int(v) for v in expansion["bbox"]]
        segment["pre_edge_expansion_bbox"] = old_bbox
        segment["edge_expansion_bbox"] = new_bbox
        segment["edge_expansion_pixel_runs"] = list(expansion["pixel_runs"])
        segment["edge_expansion_added_pixels"] = int(expansion["added_pixels"])
        segment["edge_expansion_gray_threshold"] = int(expansion["gray_threshold"])
        segment["edge_expansion_normal_padding"] = int(expansion["normal_padding"])
        history.append(
            {
                "segment_id": str(segment.get("segment_id", "")),
                "orientation": str(segment.get("orientation", "")),
                "pre_edge_expansion_bbox": old_bbox,
                "edge_expansion_bbox": new_bbox,
                "added_pixels": int(expansion["added_pixels"]),
                "gray_threshold": int(expansion["gray_threshold"]),
                "normal_padding": int(expansion["normal_padding"]),
            }
        )
    return history


def edge_expansion_attributes(segment: dict[str, Any]) -> dict[str, Any]:
    runs = list(segment.get("edge_expansion_pixel_runs", []))
    if not runs:
        return {}
    return {
        "pre_edge_expansion_bbox": list(segment.get("pre_edge_expansion_bbox", [])),
        "edge_expansion_bbox": list(segment.get("edge_expansion_bbox", [])),
        "edge_expansion_pixel_runs": runs,
        "edge_expansion_added_pixels": int(segment.get("edge_expansion_added_pixels", 0)),
        "edge_expansion_gray_threshold": int(segment.get("edge_expansion_gray_threshold", 175)),
        "edge_expansion_normal_padding": int(segment.get("edge_expansion_normal_padding", 0)),
    }


def wire_pixel_attributes(segment: dict[str, Any]) -> dict[str, Any]:
    runs = list(segment.get("wire_pixel_runs", []))
    if not runs:
        return {}
    return {
        "wire_pixel_bbox": list(segment.get("wire_pixel_bbox", [])),
        "wire_pixel_runs": runs,
        "wire_pixel_count": int(segment.get("wire_pixel_count", 0)),
        "body_width": round(float(segment.get("body_width", segment.get("width", 0.0))), 3),
        "bbox_width": round(float(segment.get("bbox_width", segment.get("width", 0.0))), 3),
        "wire_body_initial_core_rows": list(segment.get("wire_body_initial_core_rows", [])),
        "wire_body_core_rows": list(segment.get("wire_body_core_rows", [])),
        "wire_body_edge_expanded_pixels": int(segment.get("wire_body_edge_expanded_pixels", 0)),
        "wire_body_removed_spur_pixels": int(segment.get("wire_body_removed_spur_pixels", 0)),
        "wire_body_kept_spur_pixels": int(segment.get("wire_body_kept_spur_pixels", 0)),
        "external_endpoint_orthogonal_trimmed_pixels": int(
            segment.get("external_endpoint_orthogonal_trimmed_pixels", 0)
        ),
        "external_endpoint_orthogonal_trim_events": list(
            segment.get("external_endpoint_orthogonal_trim_events", [])
        ),
        "connected_endpoint_length_trimmed_pixels": int(
            segment.get("connected_endpoint_length_trimmed_pixels", 0)
        ),
        "connected_endpoint_length_trim_events": list(
            segment.get("connected_endpoint_length_trim_events", [])
        ),
        "junction_dot_trimmed_pixels": int(segment.get("junction_dot_trimmed_pixels", 0)),
        "junction_dot_trim_events": list(segment.get("junction_dot_trim_events", [])),
        "width_profile_split": bool(segment.get("width_profile_split", False)),
        "width_profile_parent_bbox": list(segment.get("width_profile_parent_bbox", [])),
        "local_width_anomaly": bool(segment.get("local_width_anomaly", False)),
        "local_width_anomaly_parent_bbox": list(segment.get("local_width_anomaly_parent_bbox", [])),
    }


def merged_segment_attributes(segment: dict[str, Any]) -> dict[str, Any]:
    merged_ids = list(segment.get("merged_segment_ids", []))
    if not merged_ids:
        return {}
    return {
        "merged_segment_ids": merged_ids,
        "merged_segment_count": int(segment.get("merged_segment_count", len(merged_ids))),
    }


def extension_output_attributes(segment: dict[str, Any]) -> dict[str, Any]:
    source_type = str(segment.get("source_segment_type", ""))
    if source_type == "diagonal_endpoint_extension":
        return {
            "parent_segment_id": str(segment.get("parent_segment_id", "")),
            "parent_endpoint_index": int(segment.get("parent_endpoint_index", -1)),
            "extension_angle_degrees": round(float(segment.get("extension_angle_degrees", 0.0)), 3),
            "extension_direction_vector": list(segment.get("extension_direction_vector", [])),
            "extension_axis_coverage": round(float(segment.get("extension_axis_coverage", 0.0)), 3),
            "diagonal_forward_extra_pixels": round(float(segment.get("diagonal_forward_extra_pixels", 0.0)), 3),
            "low_angle_continuation_applied": bool(segment.get("low_angle_continuation_applied", False)),
            "extension_search_distance_pixels": int(segment.get("extension_search_distance_pixels", 0)),
            "track_cross_center": round(float(segment.get("track_cross_center", 0.0)), 3),
            "track_cross_half": round(float(segment.get("track_cross_half", 0.0)), 3),
            "track_middle_cross_half": round(float(segment.get("track_middle_cross_half", 0.0)), 3),
            "track_radial_connected": bool(segment.get("track_radial_connected", False)),
            "near_endpoint_axis": dict(segment.get("near_endpoint_axis", {})),
            "oriented_rect_fit": dict(segment.get("oriented_rect_fit", {})),
            "final_region_metrics": dict(segment.get("final_region_metrics", {})),
            "direction_seed_bbox": list(segment.get("direction_seed_bbox", [])),
            "direction_seed_pixel_runs": list(segment.get("direction_seed_pixel_runs", [])),
            "direction_seed_angle_degrees": round(float(segment.get("direction_seed_angle_degrees", 0.0)), 3),
            "direction_seed_source": str(segment.get("direction_seed_source", "")),
            "direction_seed_pixel_count": int(segment.get("direction_seed_pixel_count", 0)),
            "direction_seed_scan_radius_pixels": round(float(segment.get("direction_seed_scan_radius_pixels", 0.0)), 3),
            "direction_seed_scan_recent_width": round(float(segment.get("direction_seed_scan_recent_width", 0.0)), 3),
            "direction_seed_scan_robust_width": round(float(segment.get("direction_seed_scan_robust_width", 0.0)), 3),
            "direction_seed_scan_axis_coverage": round(float(segment.get("direction_seed_scan_axis_coverage", 0.0)), 3),
            "direction_seed_fan_width_threshold": round(float(segment.get("direction_seed_fan_width_threshold", 0.0)), 3),
            "direction_seed_normal_width_limit": round(float(segment.get("direction_seed_normal_width_limit", 0.0)), 3),
            "track_truncated_at_projection": segment.get("track_truncated_at_projection"),
            "track_truncated_pixels": int(segment.get("track_truncated_pixels", 0)),
            "diagonal_core_pixel_count": int(segment.get("diagonal_core_pixel_count", 0)),
            "diagonal_core_width_padding_px": round(float(segment.get("diagonal_core_width_padding_px", 0.0)), 3),
            "new_endpoint_parent_distance": round(float(segment.get("new_endpoint_parent_distance", 0.0)), 3),
            "completion_pixels": int(segment.get("completion_pixels", 0)),
            "far_endpoint_dot_trimmed": bool(segment.get("far_endpoint_dot_trimmed", False)),
            "far_endpoint_dot_trimmed_pixels": int(segment.get("far_endpoint_dot_trimmed_pixels", 0)),
            "solid_rectangle_tail_trimmed": bool(segment.get("solid_rectangle_tail_trimmed", False)),
            "solid_rectangle_tail_trimmed_pixels": int(segment.get("solid_rectangle_tail_trimmed_pixels", 0)),
            "solid_rectangle_tail_trim_cut_projection": round(
                float(segment.get("solid_rectangle_tail_trim_cut_projection", 0.0)),
                3,
            ),
            "topology_metrics": dict(segment.get("topology_metrics", {})),
            "solid_rectangle_like_metrics": dict(segment.get("solid_rectangle_like_metrics", {})),
            "unclaimed_pixel_ratio": round(float(segment.get("unclaimed_pixel_ratio", 0.0)), 3),
        }
    if source_type == "perpendicular_endpoint_extension":
        return {
            "parent_segment_id": str(segment.get("parent_segment_id", "")),
            "parent_endpoint_index": int(segment.get("parent_endpoint_index", -1)),
            "extension_depth": int(segment.get("extension_depth", 0)),
            "extension_direction": list(segment.get("extension_direction", [])),
            "extension_search_distance_pixels": int(segment.get("extension_search_distance_pixels", 0)),
            "extension_touches_search_far_edge": bool(segment.get("extension_touches_search_far_edge", False)),
            "unclaimed_pixel_ratio": round(float(segment.get("unclaimed_pixel_ratio", 0.0)), 3),
            "pre_extension_width_expand_bbox": list(segment.get("pre_extension_width_expand_bbox", [])),
            "extension_width_expanded_pixels": int(segment.get("extension_width_expanded_pixels", 0)),
        }
    return {}


def estimate_wire_width(mask: np.ndarray) -> float:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    skel = np.zeros_like(mask, np.uint8)
    work = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(64):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel |= cv2.subtract(work, opened)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    vals = 2 * dist[skel > 0]
    vals = vals[(vals >= 1) & (vals <= max(12, mask.shape[1] * 0.01))]
    return float(np.clip(np.percentile(vals, 35) if vals.size else 2.0, 1.0, 12.0))


JUNCTION_DOT_VISUAL_COLOR: tuple[int, int, int] = (0, 0, 255)

# Keep junction dots out of the default overlay while preserving their default color.
VISUAL_COLORS: dict[str, tuple[int, int, int]] = {
    "solid_wire": (0, 190, 0),
    "dash_member": (182, 215, 0),
}

VISUAL_LEGEND: list[tuple[str, str, tuple[int, int, int]]] = [
    ("solid_wire", "solid horizontal/vertical wire segment", VISUAL_COLORS["solid_wire"]),
    ("dash_member", "member segment in a dashed-wire group", VISUAL_COLORS["dash_member"]),
]


@dataclass(frozen=True)
class JustWireConfig:
    dpi: int = 300
    min_wire_width_px: int = 2
    max_wire_widths: float = 4.5
    min_long_text_height_ratio: float = 3.0
    min_long_wire_widths: float = 8.0
    short_min_text_height_ratio: float = 1.3
    short_min_wire_widths: float = 4.0
    short_max_text_height_ratio: float = 4.8
    short_min_aspect_ratio: float = 3.5
    min_projected_fill_ratio: float = 0.35
    min_width_coord_coverage: float = 0.80
    strict_min_rect_fill_ratio: float = 0.95
    strict_min_axis_slice_coverage: float = 0.95
    strict_min_cross_row_coverage: float = 0.90
    strict_min_aspect_ratio: float = 3.5
    projected_nms_overlap: float = 0.72
    width_profile_split_min_delta_px: int = 2
    width_profile_split_noise_px: int = 12
    band_group_min_length_ratio: float = 0.65
    horizontal_run_recovery_min_rows: int = 2
    horizontal_run_recovery_min_fill_ratio: float = 0.90
    horizontal_run_recovery_min_axis_coverage: float = 0.90
    horizontal_run_recovery_min_cross_coverage: float = 0.90
    local_width_anomaly_min_px: int = 24
    local_width_anomaly_min_text_heights: float = 1.5
    local_width_anomaly_min_wire_widths: float = 12.0
    axis_run_bridge_gap_px: int = 1
    short_group_centerline_wire_widths: float = 1.5
    short_group_gap_text_height_ratio: float = 1.0
    short_group_length_min_ratio: float = 0.5
    short_group_length_max_ratio: float = 3.0
    short_group_terminal_min_ratio: float = 0.05
    short_group_terminal_max_ratio: float = 1.2
    short_group_terminal_min_aspect_ratio: float = 3.5
    terminal_dot_min_diameter_widths: float = 2.0
    terminal_dot_max_rectangular_fill: float = 0.92
    junction_dot_core_min_radius_wire_widths: float = 1.6
    junction_dot_weak_core_min_radius_wire_widths: float = 1.2
    junction_dot_weak_min_peak_radius_px: float = 4.0
    junction_dot_clean_min_core_area: int = 3
    junction_dot_weak_min_core_area: int = 1
    junction_dot_min_diameter_wire_widths: float = 1.2
    junction_dot_max_diameter_wire_widths: float = 6.0
    junction_dot_max_diameter_px: int = 28
    junction_dot_max_diameter_extra_px: float = 1.5
    junction_dot_prominent_max_diameter_wire_widths: float = 8.0
    junction_dot_prominent_max_diameter_px: int = 34
    junction_dot_min_prominence_ratio: float = 1.55
    junction_dot_weak_min_prominence_ratio: float = 1.85
    junction_dot_max_aspect_ratio: float = 1.60
    junction_dot_min_fill_ratio: float = 0.35
    junction_dot_max_fill_ratio: float = 0.96
    junction_dot_neighbor_suppression_radius_widths: float = 4.0
    junction_dot_trim_padding_wire_widths: float = 0.0
    junction_dot_min_diameter_line_width_ratio: float = 1.5
    junction_dot_non_axis_min_diameter_line_width_ratio: float = 2.0
    junction_dot_non_axis_angle_tolerance_degrees: float = 12.0
    junction_dot_four_way_min_core_area: int = 24
    junction_dot_l_or_t_min_core_area: int = 24
    enable_weak_junction_dot_recovery: bool = False
    ultrashort_min_wire_widths: float = 1.2
    ultrashort_min_aspect_ratio: float = 2.0
    endpoint_dot_min_diameter_wire_widths: float = 1.2
    endpoint_dot_max_diameter_wire_widths: float = 4.5
    endpoint_dot_max_aspect_ratio: float = 1.45
    endpoint_dot_min_fill_ratio: float = 0.45
    perpendicular_extension_search_text_heights: float = 6.0
    perpendicular_extension_search_wire_widths: float = 30.0
    perpendicular_extension_min_wire_widths: float = 3.5
    perpendicular_extension_max_total: int = 600
    perpendicular_extension_max_depth: int = 24
    perpendicular_extension_external_end_trim_wire_widths: float = 1.5
    perpendicular_extension_min_unclaimed_ratio: float = 0.25
    perpendicular_extension_max_search_text_heights: float = 30.0
    perpendicular_extension_max_search_wire_widths: float = 180.0
    perpendicular_extension_max_search_pixels: int = 600
    body_width_trim_wire_widths: float = 1.5
    merge_parallel_min_axis_overlap_ratio: float = 0.85
    merge_parallel_max_cross_gap_wire_widths: float = 1.25
    perpendicular_extension_short_requires_connection_px: int = 24
    perpendicular_extension_short_requires_connection_wire_widths: float = 12.0
    perpendicular_extension_unconnected_min_px: int = 12
    perpendicular_extension_unconnected_min_wire_widths: float = 6.0
    perpendicular_extension_min_width_px: int = 1
    perpendicular_extension_width_expand_px: int = 3
    perpendicular_extension_width_expand_min_coverage: float = 0.80
    perpendicular_extension_slanted_drift_min_px: float = 2.0
    perpendicular_extension_slanted_drift_wire_widths: float = 0.75
    perpendicular_extension_far_diagonal_probe_px: int = 64
    perpendicular_extension_far_diagonal_max_span_px: int = 56
    perpendicular_extension_far_diagonal_min_slope: float = 0.10
    diagonal_extension_search_text_heights: float = 2.5
    diagonal_extension_min_length_text_heights: float = 1.0
    diagonal_extension_min_length_wire_widths: float = 4.0
    diagonal_extension_external_end_trim_wire_widths: float = 1.5
    diagonal_extension_min_angle_degrees: float = 12.0
    diagonal_extension_max_width_ratio: float = 1.9
    diagonal_extension_width_abs_tol_px: float = 4.0
    diagonal_extension_min_axis_coverage: float = 0.62
    diagonal_extension_max_initial_gap_widths: float = 1.25
    diagonal_extension_core_trim_fraction: float = 0.20
    diagonal_extension_core_width_padding_px: float = 0.25
    diagonal_extension_low_angle_continuation_degrees: float = 25.0
    diagonal_extension_low_angle_continuation_text_heights: float = 14.0
    diagonal_extension_low_angle_continuation_length_ratio: float = 2.4
    diagonal_extension_low_angle_continuation_wire_widths: float = 45.0
    diagonal_extension_near_seed_wire_widths: float = 2.5
    diagonal_extension_fan_probe_wire_widths: float = 3.0
    diagonal_extension_fan_width_ratio: float = 3.5
    diagonal_extension_direction_scan_min_wire_widths: float = 3.0
    diagonal_extension_direction_scan_width_ratio: float = 2.2
    diagonal_extension_endpoint_bridge_wire_widths: float = 2.5
    diagonal_extension_oriented_rect_min_axis_coverage: float = 0.60
    diagonal_extension_oriented_rect_min_fill_ratio: float = 0.50
    diagonal_extension_oriented_rect_fill_width_extra_px: float = 1.0
    diagonal_extension_oriented_rect_fill_width_shrink_px: float = 1.0
    diagonal_extension_oriented_rect_max_width_ratio: float = 3.50
    diagonal_extension_oriented_rect_max_width_cv: float = 0.65
    diagonal_extension_oriented_rect_fit_padding_wire_widths: float = 2.0
    diagonal_extension_final_max_components: int = 1
    diagonal_extension_final_projection_max_run_count: int = 1
    diagonal_extension_final_projection_max_gap_bins: int = 0
    diagonal_extension_final_rect_min_axis_coverage: float = 0.74
    diagonal_extension_final_rect_min_fill_ratio: float = 0.60
    diagonal_extension_final_rect_max_width_ratio: float = 2.60
    diagonal_extension_final_rect_max_width_cv: float = 0.42
    diagonal_extension_angle_cluster_degrees: float = 8.0
    diagonal_extension_angle_cluster_min_pixels: int = 18
    diagonal_extension_fan_angle_cluster_degrees: float = 15.0
    diagonal_extension_dominant_cluster_min_weight_ratio: float = 0.62
    diagonal_extension_dominant_cluster_min_margin_ratio: float = 1.75
    diagonal_extension_topology_anchor_ignore_wire_widths: float = 2.0
    diagonal_extension_topology_anchor_ignore_px: float = 4.0
    diagonal_extension_topology_min_skeleton_pixels: int = 8
    diagonal_extension_topology_max_components: int = 1
    diagonal_extension_topology_min_endpoint_clusters: int = 2
    diagonal_extension_topology_max_branch_points: int = 0
    diagonal_extension_topology_max_endpoints: int = 2
    diagonal_extension_topology_min_largest_component_ratio: float = 0.9
    diagonal_extension_connected_topology_min_endpoint_clusters: int = 2
    diagonal_extension_connected_topology_max_branch_points: int = 1
    diagonal_extension_connected_topology_max_endpoints: int = 3
    diagonal_extension_branch_fallback_trigger_count: int = 6
    diagonal_extension_branch_fallback_radius_text_heights: float = 1.25
    diagonal_extension_branch_fallback_radius_wire_widths: float = 8.0
    diagonal_extension_no_seed_recovery_min_angle_degrees: float = 8.0
    diagonal_extension_no_seed_recovery_forward_wire_widths: float = 9.0
    diagonal_extension_no_seed_recovery_forward_text_heights: float = 1.2
    diagonal_extension_junction_backfill_radius_wire_widths: float = 8.0
    diagonal_extension_junction_backfill_max_radius_px: float = 18.0
    diagonal_extension_max_total: int = 300
    wire_body_edge_min_coverage: float = 0.80
    wire_spur_max_outward_px: int = 2
    wire_spur_max_area_px: int = 12
    wire_spur_max_axis_span_px: int = 8
    external_endpoint_orthogonal_trim_allowed_wire_widths: float = 1.15
    external_endpoint_orthogonal_trim_max_wire_widths: float = 12.0
    external_endpoint_orthogonal_trim_probe_min_px: int = 6
    external_endpoint_orthogonal_trim_probe_wire_widths: float = 4.0
    connected_endpoint_length_trim_max_text_heights: float = 14.0
    connected_endpoint_length_trim_parent_min_ratio: float = 2.0
    connected_endpoint_length_trim_max_wire_widths: float = 5.0

def estimate_text_height(binary: np.ndarray) -> float:
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    heights: list[int] = []
    h, w = binary.shape
    max_h = max(12, int(h * 0.035))
    max_w = max(20, int(w * 0.08))
    for label in range(1, num_labels):
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if bh < 5 or bh > max_h or bw < 2 or bw > max_w:
            continue
        fill = area / max(1.0, float(bw * bh))
        if fill < 0.08 or fill > 0.85:
            continue
        heights.append(bh)
    if not heights:
        return 12.0
    return float(np.clip(np.percentile(np.array(heights, dtype=float), 60), 8.0, 28.0))


def true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[0, values.astype(np.int8), 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def bbox_overlap_fraction(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)))
    return inter / area


def suppress_overlapping_segments(
    candidates: list[dict[str, Any]],
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item["span"]),
            -float(item["width"]),
            -float(item["projected_fill_ratio"]),
            item["bbox"][1],
            item["bbox"][0],
        ),
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if any(
            candidate["orientation"] == existing["orientation"]
            and bbox_overlap_fraction(candidate["bbox"], existing["bbox"]) >= overlap_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(key=lambda s: (s["bbox"][1], s["bbox"][0], s["orientation"]))
    return kept


def refine_projected_width_runs(
    window: np.ndarray,
    min_cross: int,
    cfg: JustWireConfig,
    min_axis_span: int = 1,
    max_axis_span: int | None = None,
) -> list[tuple[int, int, int, int, float, int]]:
    span = int(window.shape[1])
    if span <= 0:
        return []
    coverage = window.sum(axis=1) / max(1.0, float(span))
    valid = coverage >= cfg.min_width_coord_coverage
    if not bool(np.any(valid)):
        return []
    runs: list[tuple[int, int, int, int, float, int]] = []
    for start, end in true_runs(valid):
        width = int(end - start)
        if width < min_cross:
            continue
        refined_window = window[start:end, :]
        validation_window = bridge_axis_gaps_for_validation(refined_window, cfg)
        if is_strict_axis_rectangle(validation_window, cfg):
            ink = int(refined_window.sum())
            fill = int(validation_window.sum()) / max(1.0, float(width * span))
            runs.append((0, int(span), int(start), int(end), float(fill), ink))
            continue

        axis_counts = refined_window.sum(axis=0)
        axis_good = axis_counts / max(1.0, float(width)) >= float(cfg.strict_min_axis_slice_coverage)
        for axis_start, axis_end in true_runs(axis_good):
            sub_span = int(axis_end - axis_start)
            if sub_span < int(min_axis_span):
                continue
            if max_axis_span is not None and sub_span > int(max_axis_span):
                continue
            sub_window = refined_window[:, axis_start:axis_end]
            validation_sub_window = bridge_axis_gaps_for_validation(sub_window, cfg)
            if not is_strict_axis_rectangle(validation_sub_window, cfg):
                continue
            ink = int(sub_window.sum())
            fill = int(validation_sub_window.sum()) / max(1.0, float(width * sub_span))
            runs.append((int(axis_start), int(axis_end), int(start), int(end), float(fill), ink))
    return runs


def is_strict_axis_rectangle(window: np.ndarray, cfg: JustWireConfig) -> bool:
    """Validate a candidate in scan coordinates: rows=cross width, cols=axis span."""
    if window.ndim != 2 or window.size == 0 or not bool(np.any(window)):
        return False
    cross_width, axis_span = window.shape
    if cross_width <= 0 or axis_span <= 0:
        return False
    if axis_span / max(1.0, float(cross_width)) < float(cfg.strict_min_aspect_ratio):
        return False

    fill = float(window.sum()) / max(1.0, float(cross_width * axis_span))
    if fill < float(cfg.strict_min_rect_fill_ratio):
        return False

    axis_counts = window.sum(axis=0)
    axis_coverage = axis_counts / max(1.0, float(cross_width))
    if np.any(axis_coverage < float(cfg.strict_min_axis_slice_coverage)):
        return False

    cross_counts = window.sum(axis=1)
    active_cross_rows = cross_counts > 0
    if not bool(np.any(active_cross_rows)):
        return False
    if np.any(cross_counts[active_cross_rows] / max(1.0, float(axis_span)) < float(cfg.strict_min_cross_row_coverage)):
        return False

    for axis_index in range(axis_span):
        rows = np.flatnonzero(window[:, axis_index])
        if rows.size == 0:
            return False
        if int(rows[-1] - rows[0] + 1) != int(rows.size):
            return False

    component_count, _labels = cv2.connectedComponents(window.astype(np.uint8), connectivity=8)
    return int(component_count) == 2


def make_projected_segment(
    orientation: str,
    x_offset: int,
    y_offset: int,
    axis_start: int,
    axis_end: int,
    cross_start: int,
    cross_end: int,
    fill: float,
    ink: int,
) -> dict[str, Any]:
    if orientation == "horizontal":
        bbox = [
            int(x_offset + axis_start),
            int(y_offset + cross_start),
            int(x_offset + axis_end),
            int(y_offset + cross_end),
        ]
        center = (bbox[1] + bbox[3] - 1) / 2.0
        points = [[bbox[0], int(round(center))], [bbox[2] - 1, int(round(center))]]
    else:
        bbox = [
            int(x_offset + cross_start),
            int(y_offset + axis_start),
            int(x_offset + cross_end),
            int(y_offset + axis_end),
        ]
        center = (bbox[0] + bbox[2] - 1) / 2.0
        points = [[int(round(center)), bbox[1]], [int(round(center)), bbox[3] - 1]]
    return {
        "orientation": orientation,
        "points": points,
        "bbox": bbox,
        "span": float(axis_end - axis_start),
        "width": float(cross_end - cross_start),
        "area": int(ink),
        "centerline": float(center),
        "projected_fill_ratio": float(fill),
        "num_members": 1,
    }


def interval_overlap_int(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def merge_close_axis_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[tuple[int, int]] = [runs[0]]
    gap_limit = max(0, int(max_gap))
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if int(start) - int(prev_end) <= gap_limit:
            merged[-1] = (prev_start, int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def bridge_axis_gaps_for_validation(window: np.ndarray, cfg: JustWireConfig) -> np.ndarray:
    max_gap = int(max(0, cfg.axis_run_bridge_gap_px))
    if max_gap <= 0 or window.ndim != 2 or window.size == 0:
        return window
    bridged = window.copy()
    cross_width = int(window.shape[0])
    if cross_width <= 0:
        return bridged
    axis_coverage = window.sum(axis=0) / max(1.0, float(cross_width))
    good = axis_coverage >= float(cfg.strict_min_axis_slice_coverage)
    for start, end in true_runs(~good):
        if start == 0 or end == int(good.size):
            continue
        if end - start > max_gap:
            continue
        if bool(good[start - 1]) and bool(good[end]):
            bridged[:, start:end] = True
    return bridged


def band_row_runs(
    scan_mask: np.ndarray,
    seed_min_len: int,
    max_len: int | None,
    bridge_gap: int,
) -> list[list[tuple[int, int]]]:
    row_runs: list[list[tuple[int, int]]] = []
    row_counts = np.sum(scan_mask, axis=1)
    for row in range(scan_mask.shape[0]):
        runs: list[tuple[int, int]] = []
        if int(row_counts[row]) < int(seed_min_len):
            row_runs.append(runs)
            continue
        raw_runs = merge_close_axis_runs(true_runs(scan_mask[row, :]), bridge_gap)
        for start, end in raw_runs:
            span = int(end - start)
            if span < int(seed_min_len):
                continue
            if max_len is not None and span > int(max_len):
                continue
            runs.append((int(start), int(end)))
        row_runs.append(runs)
    return row_runs


def grouped_axis_bands(
    scan_mask: np.ndarray,
    min_len: int,
    max_len: int | None,
    min_cross: int,
    cfg: JustWireConfig,
    seed_min_len: int | None = None,
) -> list[tuple[int, int, int, int]]:
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    row_min_len = int(seed_min_len if seed_min_len is not None else min_len)
    for row, runs in enumerate(band_row_runs(scan_mask, row_min_len, max_len, cfg.axis_run_bridge_gap_px)):
        next_active: list[dict[str, Any]] = []
        used_active: set[int] = set()
        for start, end in runs:
            best_index = -1
            best_score = 0.0
            for index, group in enumerate(active):
                if index in used_active:
                    continue
                overlap = interval_overlap_int((start, end), (int(group["axis_start"]), int(group["axis_end"])))
                current_len = int(end - start)
                group_len = int(group["axis_end"]) - int(group["axis_start"])
                shorter = max(1, min(current_len, group_len))
                longer = max(1, max(current_len, group_len))
                if shorter / float(longer) < float(cfg.band_group_min_length_ratio):
                    continue
                score = overlap / float(shorter)
                if score >= 0.80 and score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0:
                group = dict(active[best_index])
                used_active.add(best_index)
                group["cross_end"] = row + 1
                group["axis_start"] = max(int(group["axis_start"]), start)
                group["axis_end"] = min(int(group["axis_end"]), end)
                group["raw_axis_start"] = min(int(group["raw_axis_start"]), start)
                group["raw_axis_end"] = max(int(group["raw_axis_end"]), end)
                next_active.append(group)
            else:
                next_active.append(
                    {
                        "cross_start": row,
                        "cross_end": row + 1,
                        "axis_start": start,
                        "axis_end": end,
                        "raw_axis_start": start,
                        "raw_axis_end": end,
                    }
                )
        for index, group in enumerate(active):
            if index not in used_active:
                finished.append(group)
        active = next_active
    finished.extend(active)

    bands: list[tuple[int, int, int, int]] = []
    for group in finished:
        cross_start = int(group["cross_start"])
        cross_end = int(group["cross_end"])
        if cross_end - cross_start < int(min_cross):
            continue
        axis_start = int(group["axis_start"])
        axis_end = int(group["axis_end"])
        if axis_end - axis_start < int(min_len):
            continue
        bands.append((axis_start, axis_end, cross_start, cross_end))
    return bands


def recover_horizontal_stable_run_segments(
    component: np.ndarray,
    x_offset: int,
    y_offset: int,
    min_len: int,
    max_len: int | None,
    min_cross: int,
    max_cross: float,
    cfg: JustWireConfig,
) -> list[dict[str, Any]]:
    if component.ndim != 2 or not bool(np.any(component)):
        return []
    max_width = int(max(min_cross, round(max_cross)))
    max_width = min(max_width, int(component.shape[0]))
    min_rows = max(2, int(cfg.horizontal_run_recovery_min_rows))
    if max_width < min_rows:
        return []

    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    for row, runs in enumerate(band_row_runs(component, min_len, max_len, cfg.axis_run_bridge_gap_px)):
        next_active: list[dict[str, Any]] = []
        used: set[int] = set()
        for start, end in runs:
            best_index = -1
            best_score = 0.0
            run_len = max(1, int(end - start))
            for index, group in enumerate(active):
                if index in used:
                    continue
                overlap = interval_overlap_int((start, end), (int(group["axis_start"]), int(group["axis_end"])))
                group_len = max(1, int(group["axis_end"]) - int(group["axis_start"]))
                score = overlap / float(max(1, min(run_len, group_len)))
                if score >= 0.85 and score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0:
                group = dict(active[best_index])
                used.add(best_index)
                group["cross_end"] = row + 1
                group["axis_start"] = max(int(group["axis_start"]), int(start))
                group["axis_end"] = min(int(group["axis_end"]), int(end))
                group["raw_axis_start"] = min(int(group["raw_axis_start"]), int(start))
                group["raw_axis_end"] = max(int(group["raw_axis_end"]), int(end))
                next_active.append(group)
            else:
                next_active.append(
                    {
                        "cross_start": int(row),
                        "cross_end": int(row + 1),
                        "axis_start": int(start),
                        "axis_end": int(end),
                        "raw_axis_start": int(start),
                        "raw_axis_end": int(end),
                    }
                )
        for index, group in enumerate(active):
            if index not in used:
                finished.append(group)
        active = next_active
    finished.extend(active)

    candidates: list[dict[str, Any]] = []
    for group in finished:
        cross_start = int(group["cross_start"])
        cross_end = int(group["cross_end"])
        width = int(cross_end - cross_start)
        if width < min_rows or width > max_width:
            continue
        axis_start = int(group["axis_start"])
        axis_end = int(group["axis_end"])
        span = int(axis_end - axis_start)
        if span < int(min_len):
            continue
        if max_len is not None and span > int(max_len):
            continue
        window = component[cross_start:cross_end, axis_start:axis_end]
        if window.size == 0:
            continue
        fill = float(np.count_nonzero(window)) / max(1.0, float(window.size))
        if fill < float(cfg.horizontal_run_recovery_min_fill_ratio):
            continue
        axis_counts = np.sum(window, axis=0)
        if np.any(axis_counts / max(1.0, float(width)) < float(cfg.horizontal_run_recovery_min_axis_coverage)):
            continue
        cross_counts = np.sum(window, axis=1)
        if np.any(cross_counts / max(1.0, float(span)) < float(cfg.horizontal_run_recovery_min_cross_coverage)):
            continue
        component_count, _labels = cv2.connectedComponents(window.astype(np.uint8), connectivity=8)
        if int(component_count) != 2:
            continue
        segment = make_projected_segment(
            "horizontal",
            x_offset,
            y_offset,
            axis_start,
            axis_end,
            cross_start,
            cross_end,
            fill,
            int(np.count_nonzero(window)),
        )
        segment["source"] = "horizontal_stable_run_recovery"
        segment["horizontal_run_recovery"] = True
        segment["horizontal_run_recovery_raw_axis_bbox"] = make_projected_segment(
            "horizontal",
            x_offset,
            y_offset,
            int(group["raw_axis_start"]),
            int(group["raw_axis_end"]),
            cross_start,
            cross_end,
            fill,
            int(np.count_nonzero(window)),
        )["bbox"]
        candidates.append(segment)
    return candidates


def value_runs(values: Sequence[tuple[int, int]]) -> list[tuple[tuple[int, int], int, int]]:
    if not values:
        return []
    runs: list[tuple[tuple[int, int], int, int]] = []
    start = 0
    current = values[0]
    for index in range(1, len(values)):
        value = values[index]
        if value == current:
            continue
        runs.append((current, start, index))
        start = index
        current = value
    runs.append((current, start, len(values)))
    return runs


def smooth_short_profile_runs(
    profiles: list[tuple[int, int]],
    noise_px: int,
) -> list[tuple[int, int]]:
    if len(profiles) <= 1 or noise_px <= 1:
        return list(profiles)
    smoothed = list(profiles)
    for _ in range(3):
        runs = value_runs(smoothed)
        changed = False
        for run_index, (_value, start, end) in enumerate(runs):
            if end - start > noise_px or len(runs) == 1:
                continue
            if run_index == 0:
                replacement = runs[run_index + 1][0]
            elif run_index == len(runs) - 1:
                replacement = runs[run_index - 1][0]
            else:
                left_value, left_start, left_end = runs[run_index - 1]
                right_value, right_start, right_end = runs[run_index + 1]
                if left_value == right_value:
                    replacement = left_value
                else:
                    replacement = left_value if (left_end - left_start) >= (right_end - right_start) else right_value
            for index in range(start, end):
                smoothed[index] = replacement
            changed = True
        if not changed:
            break
    return smoothed


def profile_difference(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(
        abs(int(first[0]) - int(second[0])),
        abs(int(first[1]) - int(second[1])),
        abs((int(first[1]) - int(first[0])) - (int(second[1]) - int(second[0]))),
    )


def expanded_width_profiles(
    scan_mask: np.ndarray,
    axis_start: int,
    axis_end: int,
    cross_start: int,
    cross_end: int,
    cfg: JustWireConfig,
) -> list[tuple[int, int]]:
    profiles: list[tuple[int, int]] = []
    for axis_index in range(axis_start, axis_end):
        lo = int(cross_start)
        hi = int(cross_end)
        while lo > 0 and bool(scan_mask[lo - 1, axis_index]):
            lo -= 1
        while hi < scan_mask.shape[0] and bool(scan_mask[hi, axis_index]):
            hi += 1
        profiles.append((lo, hi))
    noise_px = int(max(1, cfg.width_profile_split_noise_px))
    return smooth_short_profile_runs(profiles, noise_px)


def candidate_matches_full_width_profile(
    scan_mask: np.ndarray,
    axis_start: int,
    axis_end: int,
    cross_start: int,
    cross_end: int,
    cfg: JustWireConfig,
) -> bool:
    profiles = expanded_width_profiles(scan_mask, axis_start, axis_end, cross_start, cross_end, cfg)
    if not profiles:
        return False
    target = (int(cross_start), int(cross_end))
    min_delta = int(max(1, cfg.width_profile_split_min_delta_px))
    minor_anomaly_max_span = int(max(cfg.width_profile_split_noise_px, cfg.local_width_anomaly_min_px))
    for profile, start, end in value_runs(profiles):
        if profile_difference(profile, target) < min_delta:
            continue
        if int(end - start) <= minor_anomaly_max_span:
            continue
        return False
    return True


def split_axis_rectangle_by_width_profile(
    scan_mask: np.ndarray,
    axis_start: int,
    axis_end: int,
    cross_start: int,
    cross_end: int,
    min_axis_span: int,
    max_axis_span: int | None,
    cfg: JustWireConfig,
) -> list[tuple[int, int, int, int, float, int]]:
    span = int(axis_end - axis_start)
    if span <= 0:
        return []

    profiles = expanded_width_profiles(scan_mask, axis_start, axis_end, cross_start, cross_end, cfg)
    runs = value_runs(profiles)
    stable_runs = [(value, start, end) for value, start, end in runs if end - start >= int(min_axis_span)]
    if len(stable_runs) < 2:
        return []

    min_delta = int(max(1, cfg.width_profile_split_min_delta_px))
    has_real_difference = any(
        profile_difference(first[0], second[0]) >= min_delta
        for index, first in enumerate(stable_runs)
        for second in stable_runs[index + 1 :]
    )
    if not has_real_difference:
        return []

    pieces: list[tuple[int, int, int, int, float, int]] = []
    for _value, local_start, local_end in runs:
        sub_span = int(local_end - local_start)
        if sub_span < int(min_axis_span):
            continue
        if max_axis_span is not None and sub_span > int(max_axis_span):
            continue
        absolute_axis_start = int(axis_start + local_start)
        absolute_axis_end = int(axis_start + local_end)
        sub_profiles = profiles[local_start:local_end]
        sub_lo = int(round(float(np.median([item[0] for item in sub_profiles]))))
        sub_hi = int(round(float(np.median([item[1] for item in sub_profiles]))))
        if sub_hi <= sub_lo:
            continue
        sub_window = scan_mask[sub_lo:sub_hi, absolute_axis_start:absolute_axis_end]
        if not is_strict_axis_rectangle(sub_window, cfg):
            continue
        ink = int(sub_window.sum())
        fill = ink / max(1.0, float((sub_hi - sub_lo) * sub_span))
        pieces.append((absolute_axis_start, absolute_axis_end, sub_lo, sub_hi, float(fill), ink))
    return pieces


def local_width_anomaly_rectangles(
    scan_mask: np.ndarray,
    axis_start: int,
    axis_end: int,
    cross_start: int,
    cross_end: int,
    anomaly_min_axis_span: int,
    max_axis_span: int | None,
    max_cross: float,
    cfg: JustWireConfig,
) -> list[tuple[int, int, int, int, float, int]]:
    span = int(axis_end - axis_start)
    base_width = int(cross_end - cross_start)
    if span <= 0 or base_width <= 0 or anomaly_min_axis_span <= 0:
        return []

    max_width = int(max(base_width, round(max_cross)))
    min_delta = int(max(1, cfg.width_profile_split_min_delta_px))
    profiles: list[tuple[int, int] | None] = []
    for axis_index in range(axis_start, axis_end):
        lo = int(cross_start)
        hi = int(cross_end)
        while lo > 0 and bool(scan_mask[lo - 1, axis_index]):
            lo -= 1
        while hi < scan_mask.shape[0] and bool(scan_mask[hi, axis_index]):
            hi += 1
        width = int(hi - lo)
        if width >= base_width + min_delta and width <= max_width:
            profiles.append((lo, hi))
        else:
            profiles.append(None)

    pieces: list[tuple[int, int, int, int, float, int]] = []
    start = 0
    current = profiles[0] if profiles else None
    for index in range(1, len(profiles) + 1):
        value = profiles[index] if index < len(profiles) else None
        if value == current:
            continue
        if current is not None:
            sub_span = int(index - start)
            if sub_span >= int(anomaly_min_axis_span):
                absolute_axis_start = int(axis_start + start)
                absolute_axis_end = int(axis_start + index)
                if max_axis_span is None or sub_span <= int(max_axis_span):
                    sub_lo, sub_hi = int(current[0]), int(current[1])
                    sub_window = scan_mask[sub_lo:sub_hi, absolute_axis_start:absolute_axis_end]
                    if is_strict_axis_rectangle(sub_window, cfg):
                        ink = int(sub_window.sum())
                        fill = ink / max(1.0, float((sub_hi - sub_lo) * sub_span))
                        pieces.append((absolute_axis_start, absolute_axis_end, sub_lo, sub_hi, float(fill), ink))
        start = index
        current = value
    return pieces


def component_projected_rectangles(
    component: np.ndarray,
    orientation: str,
    x_offset: int,
    y_offset: int,
    min_len: int,
    max_len: int | None,
    min_cross: int,
    max_cross: float,
    cfg: JustWireConfig,
    local_anomaly_min_len: int | None = None,
) -> list[dict[str, Any]]:
    scan_mask = component if orientation == "horizontal" else component.T
    if not bool(np.any(scan_mask)):
        return []

    max_width = int(max(min_cross, round(max_cross)))
    max_width = min(max_width, scan_mask.shape[0])
    candidates: list[dict[str, Any]] = []
    for axis_start, axis_end, cross_start, cross_end in grouped_axis_bands(
        scan_mask,
        min_len,
        max_len,
        min_cross,
        cfg,
        seed_min_len=local_anomaly_min_len,
    ):
        width = int(cross_end - cross_start)
        if width > max_width:
            continue
        window = scan_mask[cross_start:cross_end, axis_start:axis_end]
        for local_axis_start, local_axis_end, local_cross_start, local_cross_end, refined_fill, refined_ink in refine_projected_width_runs(
            window,
            min_cross,
            cfg,
            min_len,
            max_len,
        ):
            refined_axis_start = axis_start + local_axis_start
            refined_axis_end = axis_start + local_axis_end
            refined_span = int(refined_axis_end - refined_axis_start)
            if refined_span < min_len:
                continue
            if max_len is not None and refined_span > max_len:
                continue
            refined_cross_start = cross_start + local_cross_start
            refined_cross_end = cross_start + local_cross_end
            split_pieces = split_axis_rectangle_by_width_profile(
                scan_mask,
                refined_axis_start,
                refined_axis_end,
                refined_cross_start,
                refined_cross_end,
                min_len,
                max_len,
                cfg,
            )
            profile_split = bool(split_pieces)
            if not split_pieces:
                if not candidate_matches_full_width_profile(
                    scan_mask,
                    refined_axis_start,
                    refined_axis_end,
                    refined_cross_start,
                    refined_cross_end,
                    cfg,
                ):
                    continue
                split_pieces = [
                    (
                        refined_axis_start,
                        refined_axis_end,
                        refined_cross_start,
                        refined_cross_end,
                        refined_fill,
                        refined_ink,
                    )
                ]
            parent_bbox: list[int] | None = None
            if profile_split:
                parent_bbox = make_projected_segment(
                    orientation,
                    x_offset,
                    y_offset,
                    refined_axis_start,
                    refined_axis_end,
                    refined_cross_start,
                    refined_cross_end,
                    refined_fill,
                    refined_ink,
                )["bbox"]
            for (
                piece_axis_start,
                piece_axis_end,
                piece_cross_start,
                piece_cross_end,
                piece_fill,
                piece_ink,
            ) in split_pieces:
                piece_width = int(piece_cross_end - piece_cross_start)
                if piece_width > max_width:
                    continue
                segment = make_projected_segment(
                    orientation,
                    x_offset,
                    y_offset,
                    piece_axis_start,
                    piece_axis_end,
                    piece_cross_start,
                    piece_cross_end,
                    piece_fill,
                    piece_ink,
                )
                if profile_split and parent_bbox is not None:
                    segment["width_profile_split"] = True
                    segment["width_profile_parent_bbox"] = parent_bbox
                candidates.append(segment)
            if local_anomaly_min_len is not None:
                anomaly_pieces = local_width_anomaly_rectangles(
                    scan_mask,
                    refined_axis_start,
                    refined_axis_end,
                    refined_cross_start,
                    refined_cross_end,
                    int(local_anomaly_min_len),
                    max_len,
                    max_cross,
                    cfg,
                )
                if anomaly_pieces:
                    anomaly_parent_bbox = make_projected_segment(
                        orientation,
                        x_offset,
                        y_offset,
                        refined_axis_start,
                        refined_axis_end,
                        refined_cross_start,
                        refined_cross_end,
                        refined_fill,
                        refined_ink,
                    )["bbox"]
                    for (
                        anomaly_axis_start,
                        anomaly_axis_end,
                        anomaly_cross_start,
                        anomaly_cross_end,
                        anomaly_fill,
                        anomaly_ink,
                    ) in anomaly_pieces:
                        anomaly_segment = make_projected_segment(
                            orientation,
                            x_offset,
                            y_offset,
                            anomaly_axis_start,
                            anomaly_axis_end,
                            anomaly_cross_start,
                            anomaly_cross_end,
                            anomaly_fill,
                            anomaly_ink,
                        )
                        anomaly_segment["local_width_anomaly"] = True
                        anomaly_segment["local_width_anomaly_parent_bbox"] = anomaly_parent_bbox
                        candidates.append(anomaly_segment)
    return suppress_overlapping_segments(candidates, cfg.projected_nms_overlap)


def extract_axis_rectangles(
    binary: np.ndarray,
    orientation: str,
    min_len: int,
    max_len: int | None,
    wire_width: float,
    cfg: JustWireConfig,
    local_anomaly_min_len: int | None = None,
) -> list[dict[str, Any]]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    min_cross = int(max(1, cfg.min_wire_width_px))
    max_cross = max(float(cfg.min_wire_width_px), wire_width * cfg.max_wire_widths)
    segments: list[dict[str, Any]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if orientation == "horizontal":
            if bw < min_len or bh < min_cross:
                continue
        else:
            if bh < min_len or bw < min_cross:
                continue
        component = labels[y : y + bh, x : x + bw] == label
        component_segments = component_projected_rectangles(
            component,
            orientation,
            x,
            y,
            min_len,
            max_len,
            min_cross,
            max_cross,
            cfg,
            local_anomaly_min_len=local_anomaly_min_len,
        )
        if orientation == "horizontal":
            component_segments = suppress_overlapping_segments(
                component_segments
                + recover_horizontal_stable_run_segments(
                    component,
                    x,
                    y,
                    min_len,
                    max_len,
                    min_cross,
                    max_cross,
                    cfg,
                ),
                cfg.projected_nms_overlap,
            )
        component_bbox = [x, y, x + bw, y + bh]
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        for segment in component_segments:
            segment.update(
                {
                    "source_component_label": int(label),
                    "source_component_bbox": component_bbox,
                    "source_component_area": component_area,
                }
            )
        segments.extend(component_segments)
    if local_anomaly_min_len is not None:
        global_segments = component_projected_rectangles(
            binary,
            orientation,
            0,
            0,
            min_len,
            max_len,
            min_cross,
            max_cross,
            cfg,
            local_anomaly_min_len=local_anomaly_min_len,
        )
        for segment in global_segments:
            segment.update(
                {
                    "source": "global_axis_gap_bridged_wire_segment",
                    "source_component_label": -1,
                    "source_component_bbox": [0, 0, int(binary.shape[1]), int(binary.shape[0])],
                    "source_component_area": int(np.count_nonzero(binary)),
                    "axis_gap_bridged_global": True,
                }
            )
        segments.extend(global_segments)
        segments = suppress_overlapping_segments(segments, cfg.projected_nms_overlap)
    segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0], s["orientation"]))
    return segments





def foreground_component_lookup(binary: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    components: dict[int, dict[str, Any]] = {}
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        components[int(label)] = {
            "label": int(label),
            "bbox": [x, y, x + width, y + height],
            "width": width,
            "height": height,
            "area": area,
            "fill": float(area) / max(1.0, float(width * height)),
            "centroid": [float(centroids[label][0]), float(centroids[label][1])],
        }
    return labels, components



def detect_junction_dots(
    binary: np.ndarray,
    wire_width: float,
    cfg: JustWireConfig,
) -> list[dict[str, Any]]:
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    representative_width = max(float(wire_width), float(cfg.min_wire_width_px), 3.0)
    clean_core_radius = max(3.0, representative_width * float(cfg.junction_dot_core_min_radius_wire_widths))
    weak_core_radius = max(3.0, representative_width * float(cfg.junction_dot_weak_core_min_radius_wire_widths))
    min_diameter = max(5.0, representative_width * float(cfg.junction_dot_min_diameter_wire_widths))
    max_diameter = max(
        min_diameter,
        min(float(cfg.junction_dot_max_diameter_px), representative_width * float(cfg.junction_dot_max_diameter_wire_widths)),
    )
    diameter_tolerance = max(
        0.0,
        float(cfg.junction_dot_max_diameter_extra_px),
        representative_width * 0.5,
    )
    non_axis_angle_tolerance = max(0.0, min(44.0, float(cfg.junction_dot_non_axis_angle_tolerance_degrees)))

    def angle_bin_axis_deviation(angle_start: int) -> float:
        center = float(angle_start) + 7.5
        normalized = ((center + 180.0) % 360.0) - 180.0
        return min(
            abs(normalized),
            abs(normalized - 90.0),
            abs(normalized + 90.0),
            abs(abs(normalized) - 180.0),
        )

    def structure_metrics(cx: float, cy: float, radius: float) -> dict[str, Any]:
        outer_radius = max(radius * 3.5, radius + 8.0)
        x1 = max(0, int(math.floor(cx - outer_radius)))
        y1 = max(0, int(math.floor(cy - outer_radius)))
        x2 = min(binary.shape[1], int(math.ceil(cx + outer_radius + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + outer_radius + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return {
                "annulus_pixels": 0,
                "angle_bin_count": 0,
                "axis_angle_bin_count": 0,
                "non_axis_angle_bin_count": 0,
                "has_non_axis_extension": False,
                "fanout_score": 0.0,
            }
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx.astype(float) - float(cx)
        dy = yy.astype(float) - float(cy)
        rr = np.sqrt(dx * dx + dy * dy)
        annulus = (rr >= radius + 1.0) & (rr <= outer_radius) & binary[y1:y2, x1:x2]
        annulus_pixels = int(np.count_nonzero(annulus))
        angles = np.degrees(np.arctan2(dy[annulus], dx[annulus])) if annulus_pixels > 0 else np.array([], dtype=float)
        angle_bins: list[dict[str, int]] = []
        axis_angle_bins: list[dict[str, Any]] = []
        non_axis_angle_bins: list[dict[str, Any]] = []
        for lower in range(-180, 180, 15):
            count = int(np.count_nonzero((angles >= float(lower)) & (angles < float(lower + 15)))) if angles.size else 0
            if count >= max(6, int(round(representative_width * 2.0))):
                entry = {
                    "angle_start": int(lower),
                    "angle_center_degrees": round(float(lower) + 7.5, 1),
                    "pixel_count": int(count),
                }
                angle_bins.append(entry)
                if angle_bin_axis_deviation(lower) <= non_axis_angle_tolerance:
                    axis_angle_bins.append(entry)
                else:
                    non_axis_angle_bins.append(entry)
        fanout_score = float(annulus_pixels) + float(len(angle_bins)) * 20.0
        return {
            "annulus_pixels": int(annulus_pixels),
            "angle_bin_count": int(len(angle_bins)),
            "angle_bins": angle_bins[:24],
            "axis_angle_bin_count": int(len(axis_angle_bins)),
            "non_axis_angle_bin_count": int(len(non_axis_angle_bins)),
            "non_axis_angle_bins": non_axis_angle_bins[:24],
            "has_non_axis_extension": bool(non_axis_angle_bins),
            "fanout_score": round(float(fanout_score), 3),
        }

    def cardinal_extension_metrics(cx: float, cy: float, radius: float) -> dict[str, Any]:
        outer_radius = max(radius * 3.5, radius + 8.0)
        x1 = max(0, int(math.floor(cx - outer_radius)))
        y1 = max(0, int(math.floor(cy - outer_radius)))
        x2 = min(binary.shape[1], int(math.ceil(cx + outer_radius + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + outer_radius + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return {"all_four_cardinal_extensions": False, "cardinal_extension_count": 0}
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx.astype(float) - float(cx)
        dy = yy.astype(float) - float(cy)
        local = binary[y1:y2, x1:x2]
        start = float(radius) + 1.0
        band = max(2.5, representative_width * 0.85)
        min_pixels = max(18, int(round(representative_width * 12.0)))
        masks = {
            "left": (dx <= -start) & (dx >= -outer_radius) & (np.abs(dy) <= band),
            "right": (dx >= start) & (dx <= outer_radius) & (np.abs(dy) <= band),
            "up": (dy <= -start) & (dy >= -outer_radius) & (np.abs(dx) <= band),
            "down": (dy >= start) & (dy <= outer_radius) & (np.abs(dx) <= band),
        }
        counts = {key: int(np.count_nonzero(local & mask)) for key, mask in masks.items()}
        present = {key: int(value) >= int(min_pixels) for key, value in counts.items()}
        extension_count = int(sum(1 for value in present.values() if value))
        return {
            "cardinal_counts": counts,
            "cardinal_present": present,
            "cardinal_min_pixels": int(min_pixels),
            "cardinal_extension_count": int(extension_count),
            "all_four_cardinal_extensions": bool(extension_count == 4),
        }

    def dot_prominence_metrics(cx: float, cy: float, radius: float) -> dict[str, Any]:
        outer_radius = max(radius * 2.8, radius + representative_width * 5.0, radius + 10.0)
        inner_radius = radius + max(1.0, representative_width * 0.35)
        x1 = max(0, int(math.floor(cx - outer_radius)))
        y1 = max(0, int(math.floor(cy - outer_radius)))
        x2 = min(binary.shape[1], int(math.ceil(cx + outer_radius + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + outer_radius + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return {
                "ok": True,
                "reason": "empty_context",
                "dot_diameter_pixels": round(float(radius * 2.0), 3),
                "surrounding_width_pixels": 0.0,
                "prominence_ratio": 999.0,
            }
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx.astype(float) - float(cx)
        dy = yy.astype(float) - float(cy)
        rr = np.sqrt(dx * dx + dy * dy)
        annulus = (rr >= inner_radius) & (rr <= outer_radius) & binary[y1:y2, x1:x2]
        annulus_pixels = int(np.count_nonzero(annulus))
        if annulus_pixels <= 0:
            return {
                "ok": True,
                "reason": "no_surrounding_pixels",
                "dot_diameter_pixels": round(float(radius * 2.0), 3),
                "surrounding_width_pixels": 0.0,
                "prominence_ratio": 999.0,
                "annulus_pixels": 0,
            }
        local_dist = dist[y1:y2, x1:x2]
        surrounding_radius = float(np.percentile(local_dist[annulus], 95.0))
        surrounding_width = float(surrounding_radius * 2.0)
        dot_diameter = float(radius * 2.0)
        ratio = dot_diameter / max(1.0, surrounding_width)
        return {
            "ok": bool(ratio >= float(cfg.junction_dot_min_prominence_ratio)),
            "reason": "ok" if ratio >= float(cfg.junction_dot_min_prominence_ratio) else "not_prominent_vs_surrounding_structure",
            "dot_diameter_pixels": round(float(dot_diameter), 3),
            "surrounding_width_pixels": round(float(surrounding_width), 3),
            "prominence_ratio": round(float(ratio), 3),
            "annulus_pixels": int(annulus_pixels),
            "inner_radius_pixels": round(float(inner_radius), 3),
            "outer_radius_pixels": round(float(outer_radius), 3),
        }

    def weak_core_radial_balance_ok(local: np.ndarray, cx: float, cy: float, x1: int, y1: int) -> bool:
        yy, xx = np.nonzero(local)
        if yy.size <= 0:
            return False
        dx = xx.astype(float) + float(x1) - float(cx)
        dy = yy.astype(float) + float(y1) - float(cy)
        angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        sector_counts = np.bincount(np.floor(angles / 45.0).astype(int).clip(0, 7), minlength=8)
        min_sector_pixels = max(2, int(round(float(representative_width))))
        occupied = int(np.count_nonzero(sector_counts >= min_sector_pixels))
        if occupied < 5:
            return False
        opposite_pairs = [
            int(sector_counts[index] + sector_counts[index + 4])
            for index in range(4)
        ]
        strong_pairs = int(sum(value >= max(4, int(round(float(representative_width) * 2.0))) for value in opposite_pairs))
        return strong_pairs >= 2

    def candidate_from_component(
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        label: int,
        core_radius: float,
        min_core_area: int,
        source: str,
    ) -> dict[str, Any] | None:
        core_area = int(stats[label, cv2.CC_STAT_AREA])
        if core_area < int(min_core_area):
            return None
        label_mask = labels == label
        max_radius = float(np.max(dist[label_mask]))
        if max_radius < float(core_radius):
            return None
        if source == "weak_core_recovery" and max_radius < float(cfg.junction_dot_weak_min_peak_radius_px):
            return None
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        radius = max(float(max_radius), float(min_diameter) / 2.0)
        pad = max(1.5, float(wire_width) * 0.5)
        x1 = max(0, int(math.floor(cx - radius - pad)))
        y1 = max(0, int(math.floor(cy - radius - pad)))
        x2 = min(binary.shape[1], int(math.ceil(cx + radius + pad + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + radius + pad + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return None
        width = float(x2 - x1)
        height = float(y2 - y1)
        if width < min_diameter or height < min_diameter:
            return None
        strict_max_diameter = max_diameter + diameter_tolerance
        prominent_max_diameter = max(
            strict_max_diameter,
            min(
                float(cfg.junction_dot_prominent_max_diameter_px),
                representative_width * float(cfg.junction_dot_prominent_max_diameter_wire_widths),
            )
            + diameter_tolerance,
        )
        if width > prominent_max_diameter or height > prominent_max_diameter:
            return None
        exceeds_strict_diameter = bool(width > strict_max_diameter or height > strict_max_diameter)
        aspect = max(width, height) / max(1.0, min(width, height))
        if aspect > float(cfg.junction_dot_max_aspect_ratio):
            return None
        yy, xx = np.mgrid[y1:y2, x1:x2]
        circle = ((xx.astype(float) - cx) ** 2 + (yy.astype(float) - cy) ** 2) <= (radius + pad) ** 2
        local = binary[y1:y2, x1:x2] & circle
        pixel_count = int(np.count_nonzero(local))
        if pixel_count <= 0:
            return None
        fill = float(pixel_count) / max(1.0, float(np.count_nonzero(circle)))
        if fill < float(cfg.junction_dot_min_fill_ratio) or fill > float(cfg.junction_dot_max_fill_ratio):
            return None
        prominence_metrics = dot_prominence_metrics(cx, cy, radius)
        if exceeds_strict_diameter and not bool(prominence_metrics.get("ok", False)):
            return None
        if source == "weak_core_recovery":
            weak_prominence_ratio = float(prominence_metrics.get("prominence_ratio", 0.0))
            if weak_prominence_ratio < float(cfg.junction_dot_weak_min_prominence_ratio):
                return None
        if source == "weak_core_recovery":
            weak_min_core_area = max(
                int(cfg.junction_dot_weak_min_core_area),
                6,
                int(round(representative_width * representative_width * 1.5)),
            )
            if int(core_area) < int(weak_min_core_area):
                return None
            if not weak_core_radial_balance_ok(local, cx, cy, x1, y1):
                return None
        metrics = structure_metrics(cx, cy, radius)
        cardinal_metrics = cardinal_extension_metrics(cx, cy, radius)
        dot_diameter = float(prominence_metrics.get("dot_diameter_pixels", radius * 2.0))
        local_line_width = max(
            float(prominence_metrics.get("surrounding_width_pixels", 0.0)),
            float(representative_width),
        )
        non_axis_min_diameter = local_line_width * float(cfg.junction_dot_non_axis_min_diameter_line_width_ratio)
        has_non_axis_extension = bool(metrics.get("has_non_axis_extension", False))
        if has_non_axis_extension and dot_diameter < non_axis_min_diameter:
            return None
        four_way_min_core_area = max(
            int(cfg.junction_dot_four_way_min_core_area),
            int(round(representative_width * representative_width * 2.5)),
        )
        if bool(cardinal_metrics.get("all_four_cardinal_extensions", False)):
            if source != "clean_core" or int(core_area) < int(four_way_min_core_area):
                return None
        l_or_t_min_core_area = max(
            int(cfg.junction_dot_l_or_t_min_core_area),
            int(round(representative_width * representative_width * 2.5)),
        )
        if (
            source == "weak_core_recovery"
            and int(cardinal_metrics.get("cardinal_extension_count", 0)) >= 2
            and int(core_area) < int(l_or_t_min_core_area)
        ):
            return None
        return {
            "dot_id": "",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "centroid": [round(float(cx), 3), round(float(cy), 3)],
            "radius_pixels": round(float(radius), 3),
            "core_radius_threshold_pixels": round(float(core_radius), 3),
            "core_area": int(core_area),
            "pixel_count": int(pixel_count),
            "fill": round(float(fill), 3),
            "aspect": round(float(aspect), 3),
            "detection_source": source,
            "structure_metrics": metrics,
            "cardinal_extension_metrics": cardinal_metrics,
            "prominence_metrics": prominence_metrics,
            "fanout_score": float(metrics.get("fanout_score", 0.0)),
            "non_axis_extension_present": bool(has_non_axis_extension),
            "non_axis_line_width_pixels": round(float(local_line_width), 3),
            "non_axis_min_diameter_pixels": round(float(non_axis_min_diameter), 3),
            "pixel_runs": mask_to_relative_runs(local),
        }

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    candidate_sources = [
        (clean_core_radius, int(cfg.junction_dot_clean_min_core_area), "clean_core"),
    ]
    if bool(cfg.enable_weak_junction_dot_recovery):
        candidate_sources.append(
            (weak_core_radius, int(cfg.junction_dot_weak_min_core_area), "weak_core_recovery")
        )

    for core_radius, min_core_area, source in candidate_sources:
        core = dist >= float(core_radius)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):
            candidate = candidate_from_component(labels, stats, centroids, label, core_radius, min_core_area, source)
            if candidate is None:
                continue
            key = (
                int(round(float(candidate["centroid"][0]))),
                int(round(float(candidate["centroid"][1]))),
                int(round(float(candidate["radius_pixels"]))),
                int(candidate["core_area"]),
                str(source),
            )
            if key in seen:
                continue
            seen.add(key)
            if source == "weak_core_recovery":
                candidate["fanout_score"] = float(candidate.get("fanout_score", 0.0)) + 10.0
            candidates.append(candidate)

    if not candidates:
        return []

    suppression_radius = max(
        float(min_diameter),
        representative_width * float(cfg.junction_dot_neighbor_suppression_radius_widths),
    )
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(candidates):
        fx, fy = [float(v) for v in first["centroid"]]
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            sx, sy = [float(v) for v in second["centroid"]]
            distance = math.hypot(fx - sx, fy - sy)
            local_limit = max(
                suppression_radius,
                (float(first["radius_pixels"]) + float(second["radius_pixels"])) * 2.05,
            )
            if distance <= local_limit:
                union(first_index, second_index)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        clusters.setdefault(find(index), []).append(candidate)

    kept: list[dict[str, Any]] = []
    for members in clusters.values():
        best = min(
            members,
            key=lambda item: (
                float(item.get("fanout_score", 0.0)),
                0 if str(item.get("detection_source", "")) == "clean_core" else 1,
                -int(item.get("core_area", 0)),
                item["bbox"][1],
                item["bbox"][0],
            ),
        )
        if len(members) > 1:
            best["suppressed_nearby_dot_candidates"] = [
                {
                    "bbox": list(member.get("bbox", [])),
                    "centroid": list(member.get("centroid", [])),
                    "detection_source": str(member.get("detection_source", "")),
                    "fanout_score": round(float(member.get("fanout_score", 0.0)), 3),
                }
                for member in members
                if member is not best
            ]
        kept.append(best)

    dots = sorted(kept, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    dots.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, dot in enumerate(dots, start=1):
        dot["dot_id"] = f"junction_dot_{index:04d}"
    return dots


def dot_mask_in_bbox(dot: dict[str, Any], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    local = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return local
    db = [int(v) for v in dot.get("bbox", [])]
    if len(db) != 4:
        return local
    dx1, dy1, dx2, dy2 = db
    ix1, iy1 = max(x1, dx1), max(y1, dy1)
    ix2, iy2 = min(x2, dx2), min(y2, dy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return local
    dot_local = relative_runs_to_mask(dot.get("pixel_runs", []), (dy2 - dy1, dx2 - dx1))
    local[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = dot_local[iy1 - dy1:iy2 - dy1, ix1 - dx1:ix2 - dx1]
    return local


def junction_dot_is_hard_boundary(dot: dict[str, Any]) -> bool:
    source = str(dot.get("detection_source", ""))
    if source == "clean_core":
        return True
    if source != "weak_core_recovery":
        return False
    metrics = dict(dot.get("structure_metrics", {}))
    has_clean_support = any(
        str(candidate.get("detection_source", "")) == "clean_core"
        for candidate in dot.get("suppressed_nearby_dot_candidates", [])
    )
    return (
        float(dot.get("aspect", 99.0)) <= 1.35
        and 0.45 <= float(dot.get("fill", 0.0)) <= 0.92
        and (
            has_clean_support
            or (
                float(dot.get("fanout_score", metrics.get("fanout_score", 0.0))) <= 450.0
                and int(metrics.get("angle_bin_count", 0)) <= 8
            )
        )
    )


def build_junction_dot_stop_mask(
    shape: tuple[int, int],
    junction_dots: list[dict[str, Any]],
    padding_px: int = 1,
    hard_only: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mask = np.zeros(shape, dtype=bool)
    included_dots: list[dict[str, Any]] = []
    for dot in junction_dots:
        dot["hard_boundary"] = bool(junction_dot_is_hard_boundary(dot))
        if hard_only and not bool(dot["hard_boundary"]):
            continue
        bbox = [int(v) for v in dot.get("bbox", [])]
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = clipped_bbox(bbox, shape)
        if x2 <= x1 or y2 <= y1:
            continue
        local = dot_mask_in_bbox(dot, (x1, y1, x2, y2))
        if not bool(np.any(local)):
            continue
        mask[y1:y2, x1:x2] |= local
        included_dots.append(dot)
    pad = max(0, int(padding_px))
    if pad > 0 and bool(np.any(mask)):
        kernel = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask, included_dots


def collect_junction_dot_boundaries(
    junction_dots: list[dict[str, Any]],
    hard_only: bool = True,
) -> list[dict[str, Any]]:
    included_dots: list[dict[str, Any]] = []
    for dot in junction_dots:
        dot["hard_boundary"] = bool(junction_dot_is_hard_boundary(dot))
        if hard_only and not bool(dot["hard_boundary"]):
            continue
        included_dots.append(dot)
    return included_dots


def axis_segment_crosses_dot_center(
    segment: dict[str, Any],
    dot: dict[str, Any],
    wire_width: float,
) -> bool:
    orientation = str(segment.get("orientation", ""))
    if orientation not in {"horizontal", "vertical"}:
        return True
    centroid = list(dot.get("centroid", []))
    if len(centroid) < 2:
        return False
    dot_x = float(centroid[0])
    dot_y = float(centroid[1])
    x1, y1, x2, y2 = [float(v) for v in segment.get("wire_pixel_bbox", segment.get("bbox", []))]
    center_tolerance = max(1.5, float(wire_width) * 0.75)
    axis_tolerance = max(1.0, float(wire_width) * 0.5)
    if orientation == "horizontal":
        line_y = float(segment.get("centerline", (y1 + y2 - 1.0) / 2.0))
        return abs(line_y - dot_y) <= center_tolerance and (x1 - axis_tolerance) <= dot_x <= (x2 + axis_tolerance)
    line_x = float(segment.get("centerline", (x1 + x2 - 1.0) / 2.0))
    return abs(line_x - dot_x) <= center_tolerance and (y1 - axis_tolerance) <= dot_y <= (y2 + axis_tolerance)


def junction_dot_effective_diameter(dot: dict[str, Any]) -> float:
    radius = float(dot.get("radius_pixels", 0.0))
    if radius > 0.0:
        return float(radius * 2.0)
    bbox = [float(v) for v in dot.get("bbox", [])]
    if len(bbox) == 4:
        return float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return 0.0


def junction_dot_large_enough_for_segment(
    dot: dict[str, Any],
    segment: dict[str, Any],
    wire_width: float,
    cfg: JustWireConfig,
) -> bool:
    line_width = max(float(wire_width), segment_effective_width(segment))
    diameter = junction_dot_effective_diameter(dot)
    min_diameter = line_width * float(cfg.junction_dot_min_diameter_line_width_ratio)
    ok = diameter > min_diameter
    dot.setdefault("line_width_checks", []).append(
        {
            "segment_id": str(segment.get("segment_id", "")),
            "segment_bbox": [int(v) for v in segment.get("bbox", [])],
            "segment_width_pixels": round(float(line_width), 3),
            "dot_diameter_pixels": round(float(diameter), 3),
            "min_required_diameter_pixels": round(float(min_diameter), 3),
            "ok": bool(ok),
        }
    )
    if not ok:
        dot["line_width_rejected"] = True
    return ok


def axis_segment_dot_stop_mask(
    binary: np.ndarray,
    segment: dict[str, Any],
    junction_dots: list[dict[str, Any]],
    wire_width: float,
    bbox: tuple[int, int, int, int],
    cfg: JustWireConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x1, y1, x2, y2 = bbox
    local = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=bool)
    touched_dots: list[dict[str, Any]] = []
    if x2 <= x1 or y2 <= y1:
        return local, touched_dots
    for dot in junction_dots:
        dot_bbox = [int(v) for v in dot.get("bbox", [])]
        if len(dot_bbox) != 4:
            continue
        if dot_bbox[2] <= x1 or x2 <= dot_bbox[0] or dot_bbox[3] <= y1 or y2 <= dot_bbox[1]:
            continue
        if not axis_segment_crosses_dot_center(segment, dot, wire_width):
            continue
        if not junction_dot_large_enough_for_segment(dot, segment, wire_width, cfg):
            continue
        dot_local = dot_mask_in_bbox(dot, bbox)
        if not bool(np.any(dot_local)):
            continue
        local |= dot_local
        touched_dots.append(dot)
    return local, touched_dots


def mark_dot_blocked_endpoints(
    binary: np.ndarray,
    segment: dict[str, Any],
    dot_mask: np.ndarray,
) -> list[int]:
    blocked: list[int] = []
    if not bool(np.any(dot_mask)):
        segment["dot_blocked_endpoint_indices"] = blocked
        return blocked
    for endpoint_index in range(2):
        if segment_endpoint_touches_mask(binary, segment, endpoint_index, dot_mask):
            blocked.append(int(endpoint_index))
    segment["dot_blocked_endpoint_indices"] = blocked
    return blocked


def dot_mask_full(shape: tuple[int, int], dot: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    bbox = [int(v) for v in dot.get("bbox", [])]
    if len(bbox) != 4:
        return mask
    x1, y1, x2, y2 = clipped_bbox(bbox, shape)
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1:y2, x1:x2] = dot_mask_in_bbox(dot, (x1, y1, x2, y2))
    return mask


def build_line_width_filtered_dot_stop_mask(
    shape: tuple[int, int],
    junction_dots: list[dict[str, Any]],
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mask = np.zeros(shape, dtype=bool)
    valid_dots: list[dict[str, Any]] = []
    full_dot_masks: dict[int, np.ndarray] = {}
    for dot in junction_dots:
        dot.pop("line_width_rejected", None)
        dot.pop("line_width_checks", None)
        bbox = [int(v) for v in dot.get("bbox", [])]
        if len(bbox) != 4:
            continue
        touched_segments: list[dict[str, Any]] = []
        full_dot_mask: np.ndarray | None = None
        for segment in segments:
            orientation = str(segment.get("orientation", ""))
            if orientation in {"horizontal", "vertical"}:
                if not axis_segment_crosses_dot_center(segment, dot, wire_width):
                    continue
            else:
                cache_key = id(dot)
                full_dot_mask = full_dot_masks.get(cache_key)
                if full_dot_mask is None:
                    full_dot_mask = dot_mask_full(shape, dot)
                    full_dot_masks[cache_key] = full_dot_mask
                if not segment_endpoint_touches_mask(binary, segment, 0, full_dot_mask) and not segment_endpoint_touches_mask(
                    binary,
                    segment,
                    1,
                    full_dot_mask,
                ):
                    continue
            touched_segments.append(segment)
        if touched_segments:
            if not all(junction_dot_large_enough_for_segment(dot, segment, wire_width, cfg) for segment in touched_segments):
                continue
        x1, y1, x2, y2 = clipped_bbox(bbox, shape)
        if x2 <= x1 or y2 <= y1:
            continue
        local = dot_mask_in_bbox(dot, (x1, y1, x2, y2))
        if not bool(np.any(local)):
            continue
        mask[y1:y2, x1:x2] |= local
        valid_dots.append(dot)
    return mask, valid_dots


def split_axis_segment_at_stop_mask(
    binary: np.ndarray,
    segment: dict[str, Any],
    junction_dots: list[dict[str, Any]],
    wire_width: float,
    min_len: int,
    cfg: JustWireConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    orientation = str(segment.get("orientation", ""))
    if orientation not in {"horizontal", "vertical"}:
        return [segment], None
    source = segment_wire_pixel_source(binary, segment)
    if source is None:
        return [segment], None
    bbox, local = source
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return [segment], None
    local_stop, touched_dots = axis_segment_dot_stop_mask(binary, segment, junction_dots, wire_width, bbox, cfg)
    overlap = local & local_stop
    overlap_pixels = int(np.count_nonzero(overlap))
    if overlap_pixels <= 0:
        return [segment], None
    remaining = local & ~local_stop
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(remaining.astype(np.uint8), connectivity=8)
    full_stop_mask = np.zeros(binary.shape, dtype=bool)
    full_stop_mask[y1:y2, x1:x2] = local_stop
    pieces: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        lx = int(stats[label, cv2.CC_STAT_LEFT])
        ly = int(stats[label, cv2.CC_STAT_TOP])
        lw = int(stats[label, cv2.CC_STAT_WIDTH])
        lh = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp = labels[ly:ly + lh, lx:lx + lw] == label
        span = lw if orientation == "horizontal" else lh
        if int(span) < int(min_len):
            continue
        piece = dict(segment)
        px1, py1, px2, py2 = x1 + lx, y1 + ly, x1 + lx + lw, y1 + ly + lh
        piece["bbox"] = [int(px1), int(py1), int(px2), int(py2)]
        if orientation == "horizontal":
            center_y = float(py1 + py2 - 1) / 2.0
            piece["points"] = [[int(px1), int(round(center_y))], [int(px2 - 1), int(round(center_y))]]
            piece["span"] = float(px2 - px1)
            piece["width"] = float(py2 - py1)
            piece["centerline"] = float(center_y)
        else:
            center_x = float(px1 + px2 - 1) / 2.0
            piece["points"] = [[int(round(center_x)), int(py1)], [int(round(center_x)), int(py2 - 1)]]
            piece["span"] = float(py2 - py1)
            piece["width"] = float(px2 - px1)
            piece["centerline"] = float(center_x)
        piece["area"] = int(area)
        piece["wire_pixel_bbox"] = [int(px1), int(py1), int(px2), int(py2)]
        piece["wire_pixel_runs"] = mask_to_relative_runs(comp)
        piece["wire_pixel_count"] = int(area)
        piece["junction_dot_split"] = True
        piece["junction_dot_split_parent_bbox"] = [int(v) for v in segment.get("bbox", [])]
        mark_dot_blocked_endpoints(binary, piece, full_stop_mask)
        piece.pop("segment_id", None)
        pieces.append(piece)
    pieces.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    event = {
        "source_segment_id": str(segment.get("segment_id", "")),
        "source_bbox": [int(v) for v in segment.get("bbox", [])],
        "orientation": orientation,
        "overlap_pixels": int(overlap_pixels),
        "junction_dots": [
            {
                "dot_id": str(dot.get("dot_id", "")),
                "dot_bbox": [int(v) for v in dot.get("bbox", [])],
                "centroid": list(dot.get("centroid", [])),
            }
            for dot in touched_dots
        ],
        "num_pieces": int(len(pieces)),
        "piece_bboxes": [[int(v) for v in piece["bbox"]] for piece in pieces],
    }
    return pieces, event


def split_axis_segments_at_junction_dots(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    junction_dots: list[dict[str, Any]],
    wire_width: float,
    min_len: int,
    cfg: JustWireConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not junction_dots:
        return segments, []
    split_segments: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for segment in segments:
        pieces, event = split_axis_segment_at_stop_mask(binary, segment, junction_dots, wire_width, min_len, cfg)
        split_segments.extend(pieces)
        if event is not None:
            events.append(event)
    split_segments.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["orientation"]))
    return split_segments, events


def set_segment_wire_pixel_mask(
    segment: dict[str, Any],
    bbox: tuple[int, int, int, int],
    local: np.ndarray,
) -> None:
    ys, xs = np.nonzero(local)
    if xs.size == 0:
        segment["wire_pixel_bbox"] = []
        segment["wire_pixel_runs"] = []
        segment["wire_pixel_count"] = 0
        return
    lx1 = int(np.min(xs))
    lx2 = int(np.max(xs)) + 1
    ly1 = int(np.min(ys))
    ly2 = int(np.max(ys)) + 1
    cropped = local[ly1:ly2, lx1:lx2]
    segment["wire_pixel_bbox"] = [int(bbox[0] + lx1), int(bbox[1] + ly1), int(bbox[0] + lx2), int(bbox[1] + ly2)]
    segment["wire_pixel_runs"] = mask_to_relative_runs(cropped)
    segment["wire_pixel_count"] = int(np.count_nonzero(cropped))




def assign_segment_ids(segments: list[dict[str, Any]], prefix: str) -> None:
    for index, segment in enumerate(segments, start=1):
        segment["segment_id"] = f"{prefix}_{index:04d}"




def segment_endpoints(segment: dict[str, Any]) -> list[tuple[float, float]]:
    points = segment.get("points", [])
    if len(points) >= 2:
        return [
            (float(points[0][0]), float(points[0][1])),
            (float(points[-1][0]), float(points[-1][1])),
        ]
    x1, y1, x2, y2 = [float(v) for v in segment["bbox"]]
    if segment.get("orientation") == "horizontal":
        center_y = (y1 + y2 - 1.0) / 2.0
        return [(x1, center_y), (x2 - 1.0, center_y)]
    center_x = (x1 + x2 - 1.0) / 2.0
    return [(center_x, y1), (center_x, y2 - 1.0)]


def ensure_endpoint_roles(segment: dict[str, Any]) -> list[str]:
    roles = list(segment.get("endpoint_roles", []))
    while len(roles) < 2:
        roles.append("external_end")
    segment["endpoint_roles"] = roles[:2]
    return segment["endpoint_roles"]


def segment_endpoint_caps(segment: dict[str, Any], shape: tuple[int, int], tolerance: int = 0) -> list[tuple[int, int, int, int]]:
    if segment.get("orientation") == "diagonal":
        pad = max(2, int(round(segment_effective_width(segment))) + max(0, int(tolerance)))
        return [
            clipped_bbox(
                [
                    int(round(point[0])) - pad,
                    int(round(point[1])) - pad,
                    int(round(point[0])) + pad + 1,
                    int(round(point[1])) + pad + 1,
                ],
                shape,
            )
            for point in segment_endpoints(segment)
        ]
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in segment["bbox"]], shape)
    width = max(1, int(round(segment_effective_width(segment))))
    pad = max(0, int(tolerance))
    if segment["orientation"] == "horizontal":
        return [
            clipped_bbox([x1 - pad, y1 - pad, x1 + width + pad, y2 + pad], shape),
            clipped_bbox([x2 - width - pad, y1 - pad, x2 + pad, y2 + pad], shape),
        ]
    return [
        clipped_bbox([x1 - pad, y1 - pad, x2 + pad, y1 + width + pad], shape),
        clipped_bbox([x1 - pad, y2 - width - pad, x2 + pad, y2 + pad], shape),
    ]


def endpoint_perpendicular_directions(segment: dict[str, Any]) -> list[tuple[int, int]]:
    if segment.get("orientation") == "horizontal":
        return [(0, -1), (0, 1)]
    if segment.get("orientation") == "vertical":
        return [(-1, 0), (1, 0)]
    return []


def perpendicular_orientation(segment: dict[str, Any]) -> str:
    return "vertical" if segment.get("orientation") == "horizontal" else "horizontal"


def opposite_endpoint_index_for_direction(segment: dict[str, Any], direction: tuple[int, int]) -> int:
    projections = [
        float(point[0]) * float(direction[0]) + float(point[1]) * float(direction[1])
        for point in segment_endpoints(segment)
    ]
    return 0 if projections[0] > projections[1] else 1


def mask_to_scan(mask: np.ndarray, orientation: str) -> np.ndarray:
    return mask if orientation == "horizontal" else mask.T


def scan_to_mask(mask: np.ndarray, orientation: str) -> np.ndarray:
    return mask if orientation == "horizontal" else mask.T


def component_touches_black_outside_local(
    binary: np.ndarray,
    component_local: np.ndarray,
    origin: tuple[int, int],
    local_shape: tuple[int, int],
) -> bool:
    ys, xs = np.nonzero(component_local)
    if xs.size == 0:
        return False
    origin_x, origin_y = origin
    x1 = max(0, origin_x + int(np.min(xs)) - 1)
    y1 = max(0, origin_y + int(np.min(ys)) - 1)
    x2 = min(binary.shape[1], origin_x + int(np.max(xs)) + 2)
    y2 = min(binary.shape[0], origin_y + int(np.max(ys)) + 2)
    if x2 <= x1 or y2 <= y1:
        return False
    comp_window = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    comp_window[
        origin_y + ys - y1,
        origin_x + xs - x1,
    ] = True
    dilated = cv2.dilate(comp_window.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    local_x1 = max(x1, origin_x)
    local_y1 = max(y1, origin_y)
    local_x2 = min(x2, origin_x + local_shape[1])
    local_y2 = min(y2, origin_y + local_shape[0])
    inside_local = np.zeros_like(dilated)
    if local_x2 > local_x1 and local_y2 > local_y1:
        inside_local[
            local_y1 - y1:local_y2 - y1,
            local_x1 - x1:local_x2 - x1,
        ] = True
    external_black = binary[y1:y2, x1:x2] & ~inside_local
    return bool(np.any(dilated & external_black))


def clean_wire_body_mask(
    binary: np.ndarray,
    local: np.ndarray,
    origin: tuple[int, int],
    segment: dict[str, Any],
    cfg: JustWireConfig,
) -> np.ndarray:
    if local.ndim != 2 or not bool(np.any(local)):
        return local
    orientation = str(segment.get("orientation", "horizontal"))
    scan = mask_to_scan(local, orientation)
    cross, axis = scan.shape
    if cross <= 0 or axis <= 0:
        return local
    core_width = max(1, int(round(float(segment.get("width", 1.0)))))
    if core_width >= cross:
        return local
    row_counts = np.sum(scan, axis=1)
    best_start = 0
    best_score = -1
    for start in range(0, cross - core_width + 1):
        score = int(np.sum(row_counts[start:start + core_width]))
        if score > best_score:
            best_score = score
            best_start = start
    if best_score <= 0:
        return local
    initial_core_start = int(best_start)
    initial_core_end = int(best_start + core_width)
    core_start = int(initial_core_start)
    core_end = int(initial_core_end)
    edge_min_coverage = float(cfg.wire_body_edge_min_coverage)
    while core_start > 0 and row_counts[core_start - 1] / max(1.0, float(axis)) >= edge_min_coverage:
        core_start -= 1
    while core_end < cross and row_counts[core_end] / max(1.0, float(axis)) >= edge_min_coverage:
        core_end += 1
    keep_scan = np.zeros_like(scan, dtype=bool)
    keep_scan[core_start:core_end, :] = scan[core_start:core_end, :]

    outside_scan = scan.copy()
    outside_scan[core_start:core_end, :] = False
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        outside_scan.astype(np.uint8),
        connectivity=8,
    )
    removed_pixels = 0
    kept_spur_pixels = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp_scan = labels == label
        if y + h <= core_start:
            outward = core_start - (y + h - 1)
        elif y >= core_end:
            outward = y - (core_end - 1)
        else:
            outward = 0
        comp_local = scan_to_mask(comp_scan, orientation)
        keep_as_spur = (
            outward <= int(cfg.wire_spur_max_outward_px)
            and area <= int(cfg.wire_spur_max_area_px)
            and w <= int(cfg.wire_spur_max_axis_span_px)
            and not component_touches_black_outside_local(binary, comp_local, origin, local.shape)
        )
        if keep_as_spur:
            keep_scan |= comp_scan
            kept_spur_pixels += area
        else:
            removed_pixels += area
    cleaned = scan_to_mask(keep_scan, orientation)
    segment["wire_body_initial_core_rows"] = [int(initial_core_start), int(initial_core_end)]
    segment["wire_body_core_rows"] = [int(core_start), int(core_end)]
    segment["wire_body_edge_expanded_pixels"] = int(
        np.count_nonzero(scan[core_start:core_end, :])
        - np.count_nonzero(scan[initial_core_start:initial_core_end, :])
    )
    segment["wire_body_removed_spur_pixels"] = int(removed_pixels)
    segment["wire_body_kept_spur_pixels"] = int(kept_spur_pixels)
    return cleaned


def set_segment_wire_pixels(
    binary: np.ndarray,
    segment: dict[str, Any],
    cfg: JustWireConfig,
    dot_stop_mask: np.ndarray | None = None,
) -> None:
    runs = segment.get("edge_expansion_pixel_runs", [])
    if runs:
        x1, y1, x2, y2 = clipped_bbox(segment.get("edge_expansion_bbox", segment["bbox"]), binary.shape)
        if x2 <= x1 or y2 <= y1:
            return
        local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1)) & binary[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = clipped_bbox([int(v) for v in segment["bbox"]], binary.shape)
        if x2 <= x1 or y2 <= y1:
            return
        local = binary[y1:y2, x1:x2].copy()
    if dot_stop_mask is not None and bool(np.any(dot_stop_mask)):
        local &= ~dot_stop_mask[y1:y2, x1:x2]
    local = clean_wire_body_mask(binary, local, (x1, y1), segment, cfg)
    ys, xs = np.nonzero(local)
    if xs.size == 0:
        segment["wire_pixel_bbox"] = []
        segment["wire_pixel_runs"] = []
        segment["wire_pixel_count"] = 0
        return
    lx1 = int(np.min(xs))
    lx2 = int(np.max(xs)) + 1
    ly1 = int(np.min(ys))
    ly2 = int(np.max(ys)) + 1
    cropped = local[ly1:ly2, lx1:lx2]
    segment["wire_pixel_bbox"] = [int(x1 + lx1), int(y1 + ly1), int(x1 + lx2), int(y1 + ly2)]
    segment["wire_pixel_runs"] = mask_to_relative_runs(cropped)
    segment["wire_pixel_count"] = int(np.count_nonzero(cropped))


def strongest_scan_core_band(scan: np.ndarray, core_width: int) -> tuple[int, int]:
    cross, _axis = scan.shape
    width = max(1, min(int(core_width), max(1, int(cross))))
    if cross <= width:
        return 0, int(cross)
    row_counts = np.sum(scan, axis=1)
    best_start = 0
    best_score = -1
    for start in range(0, cross - width + 1):
        score = int(np.sum(row_counts[start:start + width]))
        if score > best_score:
            best_start = int(start)
            best_score = int(score)
    return best_start, int(best_start + width)


def orthogonal_binary_run_width(
    binary: np.ndarray,
    orientation: str,
    bbox: tuple[int, int, int, int],
    axis_index: int,
    core_start: int,
    core_end: int,
) -> tuple[int, tuple[int, int]]:
    x1, y1, _x2, _y2 = bbox
    if orientation == "horizontal":
        x = x1 + int(axis_index)
        if x < 0 or x >= binary.shape[1]:
            return 0, (0, 0)
        line = binary[:, x]
        center_start = max(0, y1 + int(core_start))
        center_end = min(binary.shape[0], y1 + int(core_end))
    else:
        y = y1 + int(axis_index)
        if y < 0 or y >= binary.shape[0]:
            return 0, (0, 0)
        line = binary[y, :]
        center_start = max(0, x1 + int(core_start))
        center_end = min(binary.shape[1], x1 + int(core_end))
    if center_end <= center_start:
        return 0, (0, 0)
    anchors = np.flatnonzero(line[center_start:center_end])
    if anchors.size == 0:
        center = int(round((center_start + center_end - 1) / 2.0))
        if center < 0 or center >= line.shape[0] or not bool(line[center]):
            return 0, (0, 0)
        left = center
        right = center + 1
    else:
        left = center_start + int(np.min(anchors))
        right = center_start + int(np.max(anchors)) + 1
    while left > 0 and bool(line[left - 1]):
        left -= 1
    while right < line.shape[0] and bool(line[right]):
        right += 1
    return int(right - left), (int(left), int(right))


def trim_external_endpoint_orthogonal_intrusions(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        orientation = str(segment.get("orientation", ""))
        if orientation not in {"horizontal", "vertical"}:
            continue
        roles = ensure_endpoint_roles(segment)
        if "external_end" not in roles[:2]:
            continue
        source = segment_wire_pixel_source(binary, segment)
        if source is None:
            continue
        bbox, local = source
        if not bool(np.any(local)):
            continue
        scan = mask_to_scan(local, orientation).copy()
        cross, axis = scan.shape
        if cross <= 0 or axis <= 1:
            continue
        axis_counts = np.sum(scan, axis=0)
        nonzero_axis = np.flatnonzero(axis_counts > 0)
        if nonzero_axis.size == 0:
            continue
        first_axis = int(np.min(nonzero_axis))
        last_axis = int(np.max(nonzero_axis))
        effective_width = max(float(wire_width), segment_effective_width(segment))
        core_start, core_end = strongest_scan_core_band(scan, int(math.ceil(effective_width)))
        allowed_width = max(1, int(math.ceil(effective_width * float(cfg.external_endpoint_orthogonal_trim_allowed_wire_widths))))
        max_trim = max(1, int(math.ceil(effective_width * float(cfg.external_endpoint_orthogonal_trim_max_wire_widths))))
        for endpoint_index, role in enumerate(roles[:2]):
            if role != "external_end":
                continue
            endpoint_axes = list(
                range(first_axis, min(last_axis + 1, first_axis + max_trim))
                if endpoint_index == 0
                else range(last_axis, max(first_axis - 1, last_axis - max_trim), -1)
            )
            if not endpoint_axes:
                continue
            probe_limit = max(
                int(cfg.external_endpoint_orthogonal_trim_probe_min_px),
                int(math.ceil(float(wire_width) * float(cfg.external_endpoint_orthogonal_trim_probe_wire_widths))),
            )
            probe_len = min(len(endpoint_axes), max(1, int(probe_limit)))
            triggered = False
            trim_axis_pixels = 0
            observed_widths: list[int] = []
            observed_bounds: list[tuple[int, int]] = []
            stop_width = 0
            seen_wide_run = False
            for probe_axis in endpoint_axes[:probe_len]:
                run_width, _run_bounds = orthogonal_binary_run_width(
                    binary,
                    orientation,
                    bbox,
                    int(probe_axis),
                    core_start,
                    core_end,
                )
                if run_width > allowed_width:
                    triggered = True
                    break
            if not triggered:
                continue
            for axis_index in endpoint_axes:
                run_width, run_bounds = orthogonal_binary_run_width(
                    binary,
                    orientation,
                    bbox,
                    int(axis_index),
                    core_start,
                    core_end,
                )
                observed_widths.append(int(run_width))
                observed_bounds.append(run_bounds)
                if run_width > allowed_width:
                    seen_wide_run = True
                    trim_axis_pixels += 1
                    continue
                if not seen_wide_run:
                    trim_axis_pixels += 1
                    continue
                stop_width = int(run_width)
                break
            if trim_axis_pixels <= 0:
                continue
            trimmed_scan = scan.copy()
            if endpoint_index == 0:
                trimmed_scan[:, :first_axis + trim_axis_pixels] = False
            else:
                trimmed_scan[:, last_axis - trim_axis_pixels + 1:] = False
            trimmed_local = scan_to_mask(trimmed_scan, orientation)
            ys, xs = np.nonzero(trimmed_local)
            if xs.size == 0:
                continue
            lx1 = int(np.min(xs))
            lx2 = int(np.max(xs)) + 1
            ly1 = int(np.min(ys))
            ly2 = int(np.max(ys)) + 1
            cropped = trimmed_local[ly1:ly2, lx1:lx2]
            old_count = int(np.count_nonzero(local))
            new_count = int(np.count_nonzero(cropped))
            trimmed_pixels = max(0, old_count - new_count)
            if trimmed_pixels <= 0:
                continue
            old_bbox = [int(v) for v in segment.get("wire_pixel_bbox", segment.get("bbox", []))]
            new_bbox = [int(bbox[0] + lx1), int(bbox[1] + ly1), int(bbox[0] + lx2), int(bbox[1] + ly2)]
            segment["wire_pixel_bbox"] = new_bbox
            segment["wire_pixel_runs"] = mask_to_relative_runs(cropped)
            segment["wire_pixel_count"] = int(new_count)
            segment["external_endpoint_orthogonal_trimmed_pixels"] = int(
                segment.get("external_endpoint_orthogonal_trimmed_pixels", 0)
            ) + int(trimmed_pixels)
            event = {
                "segment_index": int(segment_index),
                "segment_id": str(segment.get("segment_id", "")),
                "endpoint_index": int(endpoint_index),
                "orientation": orientation,
                "old_wire_pixel_bbox": old_bbox,
                "new_wire_pixel_bbox": new_bbox,
                "trimmed_axis_pixels": int(trim_axis_pixels),
                "trimmed_wire_pixels": int(trimmed_pixels),
                "probe_pixels": int(probe_len),
                "probe_limit_pixels": int(probe_limit),
                "allowed_orthogonal_run_pixels": int(allowed_width),
                "max_observed_orthogonal_run_pixels": int(max(observed_widths) if observed_widths else 0),
                "stop_orthogonal_run_pixels": int(stop_width),
                "observed_orthogonal_run_bounds": [
                    [int(start), int(end)] for start, end in observed_bounds[:trim_axis_pixels + 1]
                ],
            }
            segment.setdefault("external_endpoint_orthogonal_trim_events", []).append(event)
            events.append(event)
            local = cropped
            bbox = (new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3])
            scan = mask_to_scan(local, orientation).copy()
            axis_counts = np.sum(scan, axis=0)
            nonzero_axis = np.flatnonzero(axis_counts > 0)
            if nonzero_axis.size == 0:
                break
            first_axis = int(np.min(nonzero_axis))
            last_axis = int(np.max(nonzero_axis))
            core_start, core_end = strongest_scan_core_band(scan, int(math.ceil(effective_width)))
    return events


def trim_connected_endpoint_short_orthogonal_overlaps(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    text_height: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_id = {str(segment.get("segment_id", "")): segment for segment in segments}
    max_connected_span = max(
        float(text_height) * float(cfg.connected_endpoint_length_trim_max_text_heights),
        float(wire_width) * 80.0,
        48.0,
    )
    max_trim = max(1, int(math.ceil(float(wire_width) * float(cfg.connected_endpoint_length_trim_max_wire_widths))))

    for segment_index, segment in enumerate(segments):
        orientation = str(segment.get("orientation", ""))
        if orientation not in {"horizontal", "vertical"}:
            continue
        roles = ensure_endpoint_roles(segment)
        endpoints = segment.get("endpoints", [])
        if len(endpoints) < 2:
            continue
        source = segment_wire_pixel_source(binary, segment)
        if source is None:
            continue
        bbox, local = source
        if not bool(np.any(local)):
            continue
        scan = mask_to_scan(local, orientation).copy()
        _cross, axis = scan.shape
        if axis <= 1:
            continue
        axis_counts = np.sum(scan, axis=0)
        nonzero_axis = np.flatnonzero(axis_counts > 0)
        if nonzero_axis.size == 0:
            continue
        first_axis = int(np.min(nonzero_axis))
        last_axis = int(np.max(nonzero_axis))
        segment_span = max(1.0, float(segment.get("span", segment.get("length", 0.0))))

        for endpoint_index, role in enumerate(roles[:2]):
            if role != "connected_end":
                continue
            connected_ids = [
                str(value)
                for value in endpoints[endpoint_index].get("connected_segment_ids", [])
                if str(value) != str(segment.get("segment_id", ""))
            ]
            if not connected_ids:
                continue
            trim_axis_pixels = 0
            connected_segment_id = ""
            connected_bbox: list[int] = []
            connected_span_value = 0.0
            for connected_id in connected_ids:
                other = by_id.get(connected_id)
                if other is None:
                    continue
                other_orientation = str(other.get("orientation", ""))
                if other_orientation not in {"horizontal", "vertical"} or other_orientation == orientation:
                    continue
                other_span = max(1.0, float(other.get("span", other.get("length", 0.0))))
                if other_span > max_connected_span:
                    continue
                if segment_span < other_span * float(cfg.connected_endpoint_length_trim_parent_min_ratio):
                    continue
                other_bbox = [int(v) for v in other.get("wire_pixel_bbox", other.get("bbox", []))]
                if len(other_bbox) != 4:
                    continue
                if orientation == "horizontal":
                    overlap_start = max(int(bbox[0]), other_bbox[0])
                    overlap_end = min(int(bbox[2]), other_bbox[2])
                    if endpoint_index == 0:
                        candidate_trim = max(0, overlap_end - int(bbox[0]))
                    else:
                        candidate_trim = max(0, int(bbox[2]) - overlap_start)
                else:
                    overlap_start = max(int(bbox[1]), other_bbox[1])
                    overlap_end = min(int(bbox[3]), other_bbox[3])
                    if endpoint_index == 0:
                        candidate_trim = max(0, overlap_end - int(bbox[1]))
                    else:
                        candidate_trim = max(0, int(bbox[3]) - overlap_start)
                candidate_trim = min(int(candidate_trim), int(max_trim))
                if candidate_trim > trim_axis_pixels:
                    trim_axis_pixels = int(candidate_trim)
                    connected_segment_id = connected_id
                    connected_bbox = other_bbox
                    connected_span_value = float(other_span)
            if trim_axis_pixels <= 0:
                continue

            trimmed_scan = scan.copy()
            if endpoint_index == 0:
                trimmed_scan[:, :first_axis + trim_axis_pixels] = False
            else:
                trimmed_scan[:, last_axis - trim_axis_pixels + 1:] = False
            trimmed_local = scan_to_mask(trimmed_scan, orientation)
            ys, xs = np.nonzero(trimmed_local)
            if xs.size == 0:
                continue
            lx1 = int(np.min(xs))
            lx2 = int(np.max(xs)) + 1
            ly1 = int(np.min(ys))
            ly2 = int(np.max(ys)) + 1
            cropped = trimmed_local[ly1:ly2, lx1:lx2]
            old_count = int(np.count_nonzero(local))
            new_count = int(np.count_nonzero(cropped))
            trimmed_pixels = max(0, old_count - new_count)
            if trimmed_pixels <= 0:
                continue
            old_bbox = [int(v) for v in segment.get("wire_pixel_bbox", segment.get("bbox", []))]
            new_bbox = [int(bbox[0] + lx1), int(bbox[1] + ly1), int(bbox[0] + lx2), int(bbox[1] + ly2)]
            segment["wire_pixel_bbox"] = new_bbox
            segment["wire_pixel_runs"] = mask_to_relative_runs(cropped)
            segment["wire_pixel_count"] = int(new_count)
            segment["connected_endpoint_length_trimmed_pixels"] = int(
                segment.get("connected_endpoint_length_trimmed_pixels", 0)
            ) + int(trimmed_pixels)
            event = {
                "segment_index": int(segment_index),
                "segment_id": str(segment.get("segment_id", "")),
                "endpoint_index": int(endpoint_index),
                "orientation": orientation,
                "connected_segment_id": connected_segment_id,
                "connected_segment_bbox": connected_bbox,
                "connected_segment_length_pixels": round(float(connected_span_value), 3),
                "old_wire_pixel_bbox": old_bbox,
                "new_wire_pixel_bbox": new_bbox,
                "trimmed_axis_pixels": int(trim_axis_pixels),
                "trimmed_wire_pixels": int(trimmed_pixels),
                "max_connected_segment_length_pixels": round(float(max_connected_span), 3),
            }
            segment.setdefault("connected_endpoint_length_trim_events", []).append(event)
            events.append(event)
            local = cropped
            bbox = (new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3])
            scan = mask_to_scan(local, orientation).copy()
            axis_counts = np.sum(scan, axis=0)
            nonzero_axis = np.flatnonzero(axis_counts > 0)
            if nonzero_axis.size == 0:
                break
            first_axis = int(np.min(nonzero_axis))
            last_axis = int(np.max(nonzero_axis))
    return events


def trim_solid_wires_at_junction_dots(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    junction_dots: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not junction_dots:
        return events
    pad = max(0, int(math.ceil(float(wire_width) * float(cfg.junction_dot_trim_padding_wire_widths))))
    kernel: np.ndarray | None = None
    if pad > 0:
        kernel_size = max(1, pad * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for segment_index, segment in enumerate(segments):
        orientation = str(segment.get("orientation", ""))
        if orientation not in {"horizontal", "vertical"}:
            continue
        source = segment_wire_pixel_source(binary, segment)
        if source is None:
            continue
        bbox, local = source
        if not bool(np.any(local)):
            continue
        current = local.copy()
        old_bbox = [int(v) for v in segment.get("wire_pixel_bbox", segment.get("bbox", []))]
        old_count = int(np.count_nonzero(current))
        touched_dots: list[dict[str, Any]] = []
        for dot in junction_dots:
            dot_bbox = [int(v) for v in dot.get("bbox", [])]
            if len(dot_bbox) != 4:
                continue
            if not axis_segment_crosses_dot_center(segment, dot, wire_width):
                continue
            if not junction_dot_large_enough_for_segment(dot, segment, wire_width, cfg):
                continue
            if dot_bbox[2] <= bbox[0] or bbox[2] <= dot_bbox[0] or dot_bbox[3] <= bbox[1] or bbox[3] <= dot_bbox[1]:
                continue
            dot_local = dot_mask_in_bbox(dot, bbox)
            if not bool(np.any(dot_local)):
                continue
            cut_local = (
                cv2.dilate(dot_local.astype(np.uint8), kernel, iterations=1).astype(bool)
                if kernel is not None
                else dot_local
            )
            overlap = current & cut_local
            overlap_pixels = int(np.count_nonzero(overlap))
            if overlap_pixels <= 0:
                continue
            current &= ~cut_local
            touched_dots.append(
                {
                    "dot_id": str(dot.get("dot_id", "")),
                    "dot_bbox": [int(v) for v in dot_bbox],
                    "overlap_pixels": int(overlap_pixels),
                }
            )
        new_count = int(np.count_nonzero(current))
        trimmed_pixels = max(0, old_count - new_count)
        if trimmed_pixels <= 0:
            continue
        set_segment_wire_pixel_mask(segment, bbox, current)
        new_bbox = [int(v) for v in segment.get("wire_pixel_bbox", [])]
        event = {
            "segment_index": int(segment_index),
            "segment_id": str(segment.get("segment_id", "")),
            "orientation": orientation,
            "old_wire_pixel_bbox": old_bbox,
            "new_wire_pixel_bbox": new_bbox,
            "trimmed_wire_pixels": int(trimmed_pixels),
            "dot_trim_padding_pixels": int(pad),
            "junction_dots": touched_dots,
        }
        segment["junction_dot_trimmed_pixels"] = int(segment.get("junction_dot_trimmed_pixels", 0)) + int(trimmed_pixels)
        segment.setdefault("junction_dot_trim_events", []).append(event)
        events.append(event)
    return events


def segment_body_width_from_pixels(
    binary: np.ndarray,
    segment: dict[str, Any],
    fallback_width: float,
    cfg: JustWireConfig,
) -> float:
    source = segment_wire_pixel_source(binary, segment)
    if source is None:
        return float(fallback_width)
    _bbox, local = source
    if not bool(np.any(local)):
        return float(fallback_width)
    if segment.get("orientation") == "horizontal":
        axis_counts = np.sum(local, axis=0)
    else:
        axis_counts = np.sum(local, axis=1)
    axis_counts = axis_counts[axis_counts > 0]
    if axis_counts.size == 0:
        return float(fallback_width)
    trim = int(round(max(1.0, float(fallback_width) * float(cfg.body_width_trim_wire_widths))))
    if axis_counts.size > trim * 2 + 4:
        axis_counts = axis_counts[trim:-trim]
    stable_counts = axis_counts[axis_counts <= np.percentile(axis_counts, 75)]
    if stable_counts.size == 0:
        stable_counts = axis_counts
    return float(np.clip(np.median(stable_counts), 1.0, max(1.0, float(segment.get("width", fallback_width)))))


def set_segment_body_width(binary: np.ndarray, segment: dict[str, Any], fallback_width: float, cfg: JustWireConfig) -> None:
    body_width = segment_body_width_from_pixels(binary, segment, fallback_width, cfg)
    segment["body_width"] = float(body_width)
    segment["bbox_width"] = float(segment.get("width", body_width))


def segment_effective_width(segment: dict[str, Any]) -> float:
    return float(segment.get("body_width", segment.get("width", 1.0)))


def diagonal_parent_effective_width(segment: dict[str, Any]) -> float:
    width = max(1.0, segment_effective_width(segment))
    if str(segment.get("source_segment_type", "")) != "perpendicular_endpoint_extension":
        return width
    pre_bbox = list(segment.get("pre_extension_width_expand_bbox", []))
    if len(pre_bbox) != 4:
        return width
    orientation = str(segment.get("orientation", ""))
    if orientation == "horizontal":
        original_width = max(1.0, float(int(pre_bbox[3]) - int(pre_bbox[1])))
    else:
        original_width = max(1.0, float(int(pre_bbox[2]) - int(pre_bbox[0])))
    return max(1.0, min(width, original_width + 1.0))


def segment_axis_bounds(segment: dict[str, Any]) -> tuple[float, float]:
    bbox = [float(v) for v in segment["bbox"]]
    if segment["orientation"] == "horizontal":
        return bbox[0], bbox[2]
    return bbox[1], bbox[3]


def segment_cross_bounds(segment: dict[str, Any]) -> tuple[float, float]:
    bbox = [float(v) for v in segment["bbox"]]
    if segment["orientation"] == "horizontal":
        return bbox[1], bbox[3]
    return bbox[0], bbox[2]


def interval_overlap_length(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0.0


def segments_should_merge_parallel(
    binary: np.ndarray,
    first: dict[str, Any],
    second: dict[str, Any],
    cfg: JustWireConfig,
) -> bool:
    if first.get("orientation") != second.get("orientation"):
        return False
    if list(first.get("source_component_bbox", [])) != list(second.get("source_component_bbox", [])):
        return False
    first_axis = segment_axis_bounds(first)
    second_axis = segment_axis_bounds(second)
    shorter_axis = max(1.0, min(first_axis[1] - first_axis[0], second_axis[1] - second_axis[0]))
    axis_overlap = interval_overlap_length(first_axis, second_axis)
    if axis_overlap / shorter_axis < float(cfg.merge_parallel_min_axis_overlap_ratio):
        return False
    first_cross = segment_cross_bounds(first)
    second_cross = segment_cross_bounds(second)
    cross_gap = interval_gap(first_cross, second_cross)
    max_cross_gap = max(
        1.0,
        max(segment_effective_width(first), segment_effective_width(second))
        * float(cfg.merge_parallel_max_cross_gap_wire_widths),
    )
    if cross_gap > max_cross_gap:
        return False
    x1 = int(min(first["bbox"][0], second["bbox"][0]))
    y1 = int(min(first["bbox"][1], second["bbox"][1]))
    x2 = int(max(first["bbox"][2], second["bbox"][2]))
    y2 = int(max(first["bbox"][3], second["bbox"][3]))
    bbox = clipped_bbox([x1, y1, x2, y2], binary.shape)
    first_mask = segment_mask_in_bbox(binary, first, bbox)
    second_mask = segment_mask_in_bbox(binary, second, bbox)
    if not masks_touch(first_mask, second_mask):
        return False
    merged_local = first_mask | second_mask
    merged_scan = mask_to_scan(merged_local, str(first.get("orientation", "")))
    return is_strict_axis_rectangle(merged_scan, cfg)


def merge_parallel_group(
    binary: np.ndarray,
    group: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    ordered = sorted(group, key=lambda item: (item["bbox"][1], item["bbox"][0], item["bbox"][2], item["bbox"][3]))
    merged = dict(ordered[0])
    bbox = [
        min(int(segment["bbox"][0]) for segment in ordered),
        min(int(segment["bbox"][1]) for segment in ordered),
        max(int(segment["bbox"][2]) for segment in ordered),
        max(int(segment["bbox"][3]) for segment in ordered),
    ]
    orientation = str(merged["orientation"])
    if orientation == "horizontal":
        center = (bbox[1] + bbox[3] - 1) / 2.0
        points = [[bbox[0], int(round(center))], [bbox[2] - 1, int(round(center))]]
        span = float(bbox[2] - bbox[0])
        width = float(bbox[3] - bbox[1])
    else:
        center = (bbox[0] + bbox[2] - 1) / 2.0
        points = [[int(round(center)), bbox[1]], [int(round(center)), bbox[3] - 1]]
        span = float(bbox[3] - bbox[1])
        width = float(bbox[2] - bbox[0])
    merged.update(
        {
            "bbox": bbox,
            "points": points,
            "span": span,
            "width": width,
            "centerline": float(center),
            "area": int(sum(int(segment.get("area", 0)) for segment in ordered)),
            "projected_fill_ratio": float(np.mean([float(segment.get("projected_fill_ratio", 0.0)) for segment in ordered])),
            "merged_segment_ids": [str(segment.get("segment_id", "")) for segment in ordered],
            "merged_segment_count": int(len(ordered)),
            "source": "merged_parallel_projected_axis_wire_segment",
        }
    )
    for key in [
        "wire_pixel_bbox",
        "wire_pixel_runs",
        "wire_pixel_count",
        "body_width",
        "bbox_width",
        "edge_expansion_bbox",
        "edge_expansion_pixel_runs",
        "edge_expansion_added_pixels",
    ]:
        merged.pop(key, None)
    set_segment_wire_pixels(binary, merged, cfg)
    set_segment_body_width(binary, merged, wire_width, cfg)
    blocked_points: list[np.ndarray] = []
    for segment in ordered:
        for endpoint_index in segment.get("dot_blocked_endpoint_indices", []):
            try:
                blocked_points.append(np.array(segment_endpoints(segment)[int(endpoint_index)], dtype=float))
            except (IndexError, TypeError, ValueError):
                continue
    if blocked_points:
        merged_blocked: list[int] = []
        distance_limit = max(3.0, float(wire_width) * 2.0)
        for endpoint_index, endpoint in enumerate(segment_endpoints(merged)):
            endpoint_array = np.array(endpoint, dtype=float)
            if any(float(np.linalg.norm(endpoint_array - point)) <= distance_limit for point in blocked_points):
                merged_blocked.append(int(endpoint_index))
        merged["dot_blocked_endpoint_indices"] = merged_blocked
    return merged


def merge_parallel_adjacent_segments(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for segment in segments:
        set_segment_wire_pixels(binary, segment, cfg)
        set_segment_body_width(binary, segment, wire_width, cfg)
    parent = list(range(len(segments)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index in range(len(segments)):
        for second_index in range(first_index + 1, len(segments)):
            if segments_should_merge_parallel(binary, segments[first_index], segments[second_index], cfg):
                union(first_index, second_index)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, segment in enumerate(segments):
        grouped.setdefault(find(index), []).append(segment)

    merged_segments: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for group in grouped.values():
        if len(group) == 1:
            merged_segments.append(group[0])
            continue
        merged = merge_parallel_group(binary, group, wire_width, cfg)
        merged_segments.append(merged)
        history.append(
            {
                "merged_segment_ids": list(merged["merged_segment_ids"]),
                "merged_segment_count": int(merged["merged_segment_count"]),
                "orientation": str(merged["orientation"]),
                "bbox": [int(v) for v in merged["bbox"]],
                "wire_pixel_bbox": list(merged.get("wire_pixel_bbox", [])),
                "body_width": round(float(merged.get("body_width", merged.get("width", 0.0))), 3),
            }
        )
    merged_segments.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["orientation"]))
    return merged_segments, history


def segment_wire_pixel_source(
    binary: np.ndarray,
    segment: dict[str, Any],
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    runs = segment.get("wire_pixel_runs", [])
    if runs:
        x1, y1, x2, y2 = clipped_bbox(segment.get("wire_pixel_bbox", segment["bbox"]), binary.shape)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2), relative_runs_to_mask(runs, (y2 - y1, x2 - x1)) & binary[y1:y2, x1:x2]
    runs = segment.get("edge_expansion_pixel_runs", [])
    if runs:
        x1, y1, x2, y2 = clipped_bbox(segment.get("edge_expansion_bbox", segment["bbox"]), binary.shape)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2), relative_runs_to_mask(runs, (y2 - y1, x2 - x1)) & binary[y1:y2, x1:x2]
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in segment["bbox"]], binary.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2), binary[y1:y2, x1:x2]


def segment_pixel_mask(binary: np.ndarray, segment: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(binary.shape, dtype=bool)
    source = segment_wire_pixel_source(binary, segment)
    if source is None:
        return mask
    (x1, y1, x2, y2), local = source
    mask[y1:y2, x1:x2] = local
    return mask


def segment_mask_in_bbox(binary: np.ndarray, segment: dict[str, Any], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    local = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return local
    source = segment_wire_pixel_source(binary, segment)
    if source is None:
        return local
    (sx1, sy1, sx2, sy2), source_mask = source
    ix1, iy1 = max(x1, sx1), max(y1, sy1)
    ix2, iy2 = min(x2, sx2), min(y2, sy2)
    if ix2 > ix1 and iy2 > iy1:
        local[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = source_mask[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1]
    return local


def paint_segment_mask(target: np.ndarray, binary: np.ndarray, segment: dict[str, Any]) -> None:
    source = segment_wire_pixel_source(binary, segment)
    if source is None:
        return
    (x1, y1, x2, y2), local = source
    target[y1:y2, x1:x2] |= local


def build_segment_union_mask(binary: np.ndarray, segments: list[dict[str, Any]]) -> np.ndarray:
    union_mask = np.zeros(binary.shape, dtype=bool)
    for segment in segments:
        paint_segment_mask(union_mask, binary, segment)
    return union_mask


def segment_endpoint_touches_mask(
    binary: np.ndarray,
    segment: dict[str, Any],
    endpoint_index: int,
    target_mask: np.ndarray,
) -> bool:
    cap = segment_endpoint_cap_mask(binary, segment, endpoint_index)
    if not bool(np.any(cap)):
        return False
    dilated_cap = cv2.dilate(cap.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return bool(np.any(dilated_cap & target_mask))


def external_endpoint_trim_bbox(
    segment: dict[str, Any],
    endpoint_index: int,
    trim_length: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clipped_bbox(
        [int(v) for v in segment.get("wire_pixel_bbox", segment.get("edge_expansion_bbox", segment["bbox"]))],
        shape,
    )
    trim = max(1, int(trim_length))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    if segment["orientation"] == "horizontal":
        if endpoint_index == 0:
            return clipped_bbox([x1, y1, min(x2, x1 + trim), y2], shape)
        return clipped_bbox([max(x1, x2 - trim), y1, x2, y2], shape)
    if endpoint_index == 0:
        return clipped_bbox([x1, y1, x2, min(y2, y1 + trim)], shape)
    return clipped_bbox([x1, max(y1, y2 - trim), x2, y2], shape)


def unclaim_external_endpoint_caps(
    claimed: np.ndarray,
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    wire_width: float,
    trim_wire_widths: float,
) -> list[dict[str, Any]]:
    trim_events: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        roles = ensure_endpoint_roles(segment)
        trim_length = int(math.ceil(
            max(float(wire_width), segment_effective_width(segment))
            * float(trim_wire_widths)
        ))
        for endpoint_index, role in enumerate(roles[:2]):
            if role != "external_end":
                continue
            tx1, ty1, tx2, ty2 = external_endpoint_trim_bbox(segment, endpoint_index, trim_length, binary.shape)
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            before = int(np.count_nonzero(claimed[ty1:ty2, tx1:tx2]))
            claimed[ty1:ty2, tx1:tx2] = False
            if before <= 0:
                continue
            trim_events.append(
                {
                    "segment_index": int(segment_index),
                    "segment_id": str(segment.get("segment_id", "")),
                    "endpoint_index": int(endpoint_index),
                    "trim_bbox": [int(tx1), int(ty1), int(tx2), int(ty2)],
                    "trim_length_pixels": int(trim_length),
                    "trimmed_claimed_pixels": int(before),
                }
            )
    return trim_events


def build_extension_claimed_mask(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    claimed = np.zeros(binary.shape, dtype=bool)
    for segment in segments:
        paint_segment_mask(claimed, binary, segment)

    trim_events = unclaim_external_endpoint_caps(
        claimed,
        binary,
        segments,
        wire_width,
        cfg.perpendicular_extension_external_end_trim_wire_widths,
    )
    return claimed, trim_events


def build_diagonal_extension_claimed_mask(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    claimed = np.zeros(binary.shape, dtype=bool)
    for segment in segments:
        paint_segment_mask(claimed, binary, segment)

    trim_events = unclaim_external_endpoint_caps(
        claimed,
        binary,
        segments,
        wire_width,
        cfg.diagonal_extension_external_end_trim_wire_widths,
    )
    return claimed, trim_events


def segment_endpoint_cap_mask(binary: np.ndarray, segment: dict[str, Any], endpoint_index: int) -> np.ndarray:
    cap = np.zeros(binary.shape, dtype=bool)
    x1, y1, x2, y2 = segment_endpoint_caps(segment, binary.shape)[endpoint_index]
    if x2 <= x1 or y2 <= y1:
        return cap
    segment_mask = segment_pixel_mask(binary, segment)
    cap[y1:y2, x1:x2] = segment_mask[y1:y2, x1:x2]
    return cap



def masks_touch(first: np.ndarray, second: np.ndarray) -> bool:
    if not bool(np.any(first)) or not bool(np.any(second)):
        return False
    dilated = cv2.dilate(first.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return bool(np.any(dilated & second))


def candidate_runs_mask_on_canvas(
    shape: tuple[int, int],
    bbox: Sequence[int],
    runs: Sequence[Sequence[int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(bbox) != 4:
        return mask
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in bbox], shape)
    if x2 <= x1 or y2 <= y1:
        return mask
    local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    mask[y1:y2, x1:x2] = local
    return mask


def diagonal_completion_touches_parent_endpoint(
    binary: np.ndarray,
    completion: dict[str, Any],
    segment: dict[str, Any],
    endpoint_index: int,
) -> bool:
    runs = list(completion.get("runs", []))
    bbox = [int(v) for v in completion.get("bbox", [])]
    if not runs or len(bbox) != 4:
        return False
    completion_mask = candidate_runs_mask_on_canvas(binary.shape, bbox, runs)
    endpoint_cap = segment_endpoint_cap_mask(binary, segment, endpoint_index)
    return masks_touch(endpoint_cap, completion_mask)




def classify_segment_endpoints(binary: np.ndarray, segments: list[dict[str, Any]]) -> None:
    extension_sources = {
        (int(segment.get("parent_segment_index", -1)), int(segment.get("parent_endpoint_index", -1))): str(
            segment.get("source_segment_type", "")
        )
        for segment in segments
        if segment.get("source_segment_type") in {"perpendicular_endpoint_extension", "diagonal_endpoint_extension"}
    }
    union_mask = np.zeros(binary.shape, dtype=bool)
    for segment in segments:
        paint_segment_mask(union_mask, binary, segment)
    for index, segment in enumerate(segments):
        roles = ensure_endpoint_roles(segment)
        endpoints: list[dict[str, Any]] = []
        for endpoint_index, point in enumerate(segment_endpoints(segment)):
            cap_bbox = segment_endpoint_caps(segment, binary.shape, tolerance=1)[endpoint_index]
            x1, y1, x2, y2 = cap_bbox
            touches = False
            touch_indices: list[int] = []
            if x2 > x1 and y2 > y1:
                own = segment_mask_in_bbox(binary, segment, cap_bbox)
                other = union_mask[y1:y2, x1:x2].copy()
                other &= ~own
                cap = segment_mask_in_bbox(binary, segment, segment_endpoint_caps(segment, binary.shape)[endpoint_index])
                local_cap = np.zeros_like(other)
                cx1, cy1, cx2, cy2 = segment_endpoint_caps(segment, binary.shape)[endpoint_index]
                ix1, iy1 = max(x1, cx1), max(y1, cy1)
                ix2, iy2 = min(x2, cx2), min(y2, cy2)
                if ix2 > ix1 and iy2 > iy1:
                    local_cap[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = cap[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1]
                if bool(np.any(local_cap)):
                    dilated_cap = cv2.dilate(local_cap.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                    touches = bool(np.any(dilated_cap & other))
                    if touches:
                        for other_index, other_segment in enumerate(segments):
                            if other_index == index:
                                continue
                            ox1, oy1, ox2, oy2 = clipped_bbox(
                                other_segment.get("wire_pixel_bbox", other_segment.get("edge_expansion_bbox", other_segment["bbox"])),
                                binary.shape,
                            )
                            if ox2 < x1 or x2 < ox1 or oy2 < y1 or y2 < oy1:
                                continue
                            other_local = segment_mask_in_bbox(binary, other_segment, cap_bbox)
                            if bool(np.any(dilated_cap & other_local)):
                                touch_indices.append(other_index)
            extension_source_type = extension_sources.get((index, endpoint_index), "")
            dot_blocked_endpoint_indices = {
                int(value)
                for value in segment.get("dot_blocked_endpoint_indices", [])
                if isinstance(value, (int, np.integer, float))
            }
            if endpoint_index in dot_blocked_endpoint_indices:
                role = "junction_dot_end"
            elif extension_source_type == "perpendicular_endpoint_extension":
                role = "external_end_extended_perpendicular"
            elif extension_source_type == "diagonal_endpoint_extension":
                role = "external_end_extended_diagonal"
            elif touches:
                role = "connected_end"
            else:
                role = "external_end"
            roles[endpoint_index] = role
            endpoints.append(
                {
                    "point": [round(float(point[0]), 3), round(float(point[1]), 3)],
                    "role": role,
                    "connected_segment_ids": [
                        str(segments[touch].get("segment_id", f"segment_{touch + 1:04d}"))
                        for touch in touch_indices
                    ],
                }
            )
        segment["endpoint_roles"] = roles
        segment["endpoints"] = endpoints


def assign_pixel_net_ids(binary: np.ndarray, segments: list[dict[str, Any]], page: int) -> dict[str, Any]:
    union_mask = np.zeros(binary.shape, dtype=bool)
    for segment in segments:
        paint_segment_mask(union_mask, binary, segment)
    _count, labels = cv2.connectedComponents(union_mask.astype(np.uint8), connectivity=8)
    parents = list(range(len(segments)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        if first < 0 or second < 0 or first >= len(parents) or second >= len(parents):
            return
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    label_to_indices: dict[int, list[int]] = {}
    for index, segment in enumerate(segments):
        x1, y1, x2, y2 = clipped_bbox(
            segment.get("wire_pixel_bbox", segment.get("edge_expansion_bbox", segment["bbox"])),
            binary.shape,
        )
        local = segment_mask_in_bbox(binary, segment, (x1, y1, x2, y2))
        values = labels[y1:y2, x1:x2][local]
        values = values[values > 0]
        if values.size == 0:
            continue
        else:
            unique, counts = np.unique(values, return_counts=True)
            label = int(unique[int(np.argmax(counts))])
            label_to_indices.setdefault(label, []).append(index)
    for indices in label_to_indices.values():
        if not indices:
            continue
        first = int(indices[0])
        for other in indices[1:]:
            union(first, int(other))
    root_to_net: dict[int, str] = {}
    for index, segment in enumerate(segments):
        root = find(index)
        if root not in root_to_net:
            root_to_net[root] = f"p{page:03d}_net_{len(root_to_net) + 1:04d}"
        net_id = root_to_net[root]
        segment["net_id"] = net_id
        segment["wire_net_id"] = net_id
        for endpoint in segment.get("endpoints", []):
            endpoint["net_id"] = net_id
    return {
        "num_wire_nets": int(len(set(root_to_net.values()))),
        "net_ids": sorted(set(root_to_net.values())),
    }


def extension_search_bbox(
    segment: dict[str, Any],
    endpoint_index: int,
    direction: tuple[int, int],
    shape: tuple[int, int],
    search_distance: int,
) -> tuple[int, int, int, int]:
    endpoint = segment_endpoints(segment)[endpoint_index]
    endpoint_x = int(round(float(endpoint[0])))
    endpoint_y = int(round(float(endpoint[1])))
    anchor_x, anchor_y = extension_search_anchor(segment, endpoint_index, direction, shape)
    half_width = int(math.ceil(max(segment_effective_width(segment) * 2.5, 6.0)))
    if direction[0] > 0:
        return clipped_bbox([anchor_x, endpoint_y - half_width, anchor_x + search_distance + 1, endpoint_y + half_width + 1], shape)
    if direction[0] < 0:
        return clipped_bbox([anchor_x - search_distance, endpoint_y - half_width, anchor_x + 1, endpoint_y + half_width + 1], shape)
    if direction[1] > 0:
        return clipped_bbox([endpoint_x - half_width, anchor_y, endpoint_x + half_width + 1, anchor_y + search_distance + 1], shape)
    return clipped_bbox([endpoint_x - half_width, anchor_y - search_distance, endpoint_x + half_width + 1, anchor_y + 1], shape)


def extension_search_anchor(
    segment: dict[str, Any],
    endpoint_index: int,
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[int, int]:
    x1, y1, x2, y2 = clipped_bbox(
        [int(v) for v in segment.get("wire_pixel_bbox", segment.get("edge_expansion_bbox", segment["bbox"]))],
        shape,
    )
    endpoint = segment_endpoints(segment)[endpoint_index]
    endpoint_x = int(round(float(endpoint[0])))
    endpoint_y = int(round(float(endpoint[1])))
    if direction[0] > 0:
        return x2 - 1, endpoint_y
    if direction[0] < 0:
        return x1, endpoint_y
    if direction[1] > 0:
        return endpoint_x, y2 - 1
    return endpoint_x, y1


def direction_start_offset(
    bbox: list[int],
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    endpoint_x = int(anchor[0])
    endpoint_y = int(anchor[1])
    if direction[0] > 0:
        return int(bbox[0] - endpoint_x)
    if direction[0] < 0:
        return int(endpoint_x - (bbox[2] - 1))
    if direction[1] > 0:
        return int(bbox[1] - endpoint_y)
    return int(endpoint_y - (bbox[3] - 1))


def candidate_touches_search_far_edge(
    bbox: list[int],
    search_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tolerance: int = 1,
) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    sx1, sy1, sx2, sy2 = [int(v) for v in search_bbox]
    tol = max(0, int(tolerance))
    if direction[0] > 0:
        return x2 >= sx2 - tol
    if direction[0] < 0:
        return x1 <= sx1 + tol
    if direction[1] > 0:
        return y2 >= sy2 - tol
    return y1 <= sy1 + tol


def extension_candidate_center_delta(
    candidate: dict[str, Any],
    endpoint: tuple[float, float],
) -> float:
    bbox = [int(v) for v in candidate["bbox"]]
    if candidate["orientation"] == "horizontal":
        centerline = (bbox[1] + bbox[3] - 1) / 2.0
        return abs(centerline - float(endpoint[1]))
    centerline = (bbox[0] + bbox[2] - 1) / 2.0
    return abs(centerline - float(endpoint[0]))


def extension_strict_window_ok(binary: np.ndarray, candidate: dict[str, Any], cfg: JustWireConfig) -> bool:
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in candidate["bbox"]], binary.shape)
    if x2 <= x1 or y2 <= y1:
        return False
    window = binary[y1:y2, x1:x2]
    scan_window = window if candidate["orientation"] == "horizontal" else window.T
    return is_strict_axis_rectangle(scan_window, cfg)


def extension_candidate_slanted_drift_metrics(
    binary: np.ndarray,
    candidate: dict[str, Any],
    cfg: JustWireConfig,
) -> dict[str, Any]:
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in candidate["bbox"]], binary.shape)
    if x2 <= x1 or y2 <= y1:
        return {"reject": False, "reason": "empty_bbox"}
    orientation = str(candidate.get("orientation", "horizontal"))
    window = binary[y1:y2, x1:x2]
    scan = window if orientation == "horizontal" else window.T
    axis_len = int(scan.shape[1])
    cross_len = int(scan.shape[0])
    if axis_len < max(8, cross_len * 3):
        return {"reject": False, "reason": "too_short_for_drift"}

    centers: list[float] = []
    positions: list[int] = []
    for axis_index in range(axis_len):
        cross_values = np.flatnonzero(scan[:, axis_index])
        if cross_values.size <= 0:
            continue
        centers.append(float(np.mean(cross_values)))
        positions.append(int(axis_index))
    min_present = max(5, int(round(axis_len * 0.45)))
    if len(centers) < min_present:
        return {"reject": False, "reason": "sparse_axis_samples", "sample_count": int(len(centers))}

    positions_array = np.array(positions, dtype=float)
    centers_array = np.array(centers, dtype=float)
    quintile = max(2, int(round(len(centers_array) * 0.2)))
    start_center = float(np.median(centers_array[:quintile]))
    end_center = float(np.median(centers_array[-quintile:]))
    drift = float(end_center - start_center)
    drift_abs = abs(drift)
    drift_limit = max(
        float(cfg.perpendicular_extension_slanted_drift_min_px),
        float(cross_len) * float(cfg.perpendicular_extension_slanted_drift_wire_widths),
    )
    if drift_abs < drift_limit:
        return {
            "reject": False,
            "drift_pixels": round(float(drift), 3),
            "drift_limit_pixels": round(float(drift_limit), 3),
            "sample_count": int(len(centers)),
        }

    if float(np.std(centers_array)) <= 1e-6:
        return {"reject": False, "reason": "flat_centers"}
    corr = float(np.corrcoef(positions_array, centers_array)[0, 1]) if len(centers_array) >= 3 else 0.0
    reject = bool(abs(corr) >= 0.55)
    return {
        "reject": reject,
        "drift_pixels": round(float(drift), 3),
        "drift_limit_pixels": round(float(drift_limit), 3),
        "center_position_correlation": round(float(corr), 3),
        "sample_count": int(len(centers)),
        "axis_len": int(axis_len),
        "cross_len": int(cross_len),
    }


def extension_candidate_far_diagonal_continuation_metrics(
    binary: np.ndarray,
    candidate: dict[str, Any],
    direction: tuple[int, int],
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in candidate["bbox"]], binary.shape)
    if x2 <= x1 or y2 <= y1:
        return {"reject": False, "reason": "empty_bbox"}
    span = float(candidate.get("span", 0.0))
    max_span = max(
        float(cfg.perpendicular_extension_far_diagonal_max_span_px),
        float(wire_width) * 24.0,
    )
    if span > max_span:
        return {"reject": False, "reason": "candidate_too_long", "span": round(float(span), 3)}

    orientation = str(candidate.get("orientation", "horizontal"))
    probe_axis = max(
        int(cfg.perpendicular_extension_far_diagonal_probe_px),
        int(math.ceil(float(wire_width) * 28.0)),
    )
    probe_cross = max(10, int(math.ceil(float(wire_width) * 10.0)))
    if orientation == "horizontal":
        center_cross = (float(y1) + float(y2) - 1.0) * 0.5
        if int(direction[0]) < 0:
            px1, px2 = max(0, x1 - probe_axis), min(binary.shape[1], x1 + 1)
            far_axis = float(x1)
            axis_sign = -1.0
        else:
            px1, px2 = max(0, x2 - 1), min(binary.shape[1], x2 + probe_axis)
            far_axis = float(x2 - 1)
            axis_sign = 1.0
        py1 = max(0, int(math.floor(center_cross - probe_cross)))
        py2 = min(binary.shape[0], int(math.ceil(center_cross + probe_cross + 1.0)))
    else:
        center_cross = (float(x1) + float(x2) - 1.0) * 0.5
        if int(direction[1]) < 0:
            py1, py2 = max(0, y1 - probe_axis), min(binary.shape[0], y1 + 1)
            far_axis = float(y1)
            axis_sign = -1.0
        else:
            py1, py2 = max(0, y2 - 1), min(binary.shape[0], y2 + probe_axis)
            far_axis = float(y2 - 1)
            axis_sign = 1.0
        px1 = max(0, int(math.floor(center_cross - probe_cross)))
        px2 = min(binary.shape[1], int(math.ceil(center_cross + probe_cross + 1.0)))
    if px2 <= px1 or py2 <= py1:
        return {"reject": False, "reason": "empty_probe"}

    local = binary[py1:py2, px1:px2].copy()
    # Keep only the component that is actually attached to the far end of the orthogonal stub.
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return {"reject": False, "reason": "no_probe_component"}
    if orientation == "horizontal":
        touch_column = 0 if int(direction[0]) > 0 else local.shape[1] - 1
        far_band = np.zeros(local.shape, dtype=bool)
        band_y1 = max(0, int(math.floor(center_cross - float(wire_width) * 2.0)) - py1)
        band_y2 = min(local.shape[0], int(math.ceil(center_cross + float(wire_width) * 2.0 + 1.0)) - py1)
        far_band[band_y1:band_y2, max(0, touch_column - 1):min(local.shape[1], touch_column + 2)] = True
    else:
        touch_row = 0 if int(direction[1]) > 0 else local.shape[0] - 1
        far_band = np.zeros(local.shape, dtype=bool)
        band_x1 = max(0, int(math.floor(center_cross - float(wire_width) * 2.0)) - px1)
        band_x2 = min(local.shape[1], int(math.ceil(center_cross + float(wire_width) * 2.0 + 1.0)) - px1)
        far_band[max(0, touch_row - 1):min(local.shape[0], touch_row + 2), band_x1:band_x2] = True
    touching_labels = {
        int(value)
        for value in np.unique(labels[far_band & local])
        if int(value) != 0
    }
    if not touching_labels:
        return {"reject": False, "reason": "no_component_touching_far_end"}
    component_mask = np.isin(labels, list(touching_labels))
    yy, xx = np.nonzero(component_mask)
    if yy.size < max(8, int(round(float(wire_width) * 4.0))):
        return {"reject": False, "reason": "too_few_probe_pixels", "probe_pixels": int(yy.size)}

    if orientation == "horizontal":
        axis_values = (xx.astype(float) + float(px1) - far_axis) * axis_sign
        cross_values = yy.astype(float) + float(py1) - center_cross
    else:
        axis_values = (yy.astype(float) + float(py1) - far_axis) * axis_sign
        cross_values = xx.astype(float) + float(px1) - center_cross
    selector = axis_values >= -1.0
    axis_values = axis_values[selector]
    cross_values = cross_values[selector]
    if axis_values.size < max(8, int(round(float(wire_width) * 4.0))):
        return {"reject": False, "reason": "too_few_forward_pixels", "probe_pixels": int(axis_values.size)}
    axis_span = float(np.max(axis_values) - np.min(axis_values) + 1.0)
    if axis_span < max(8.0, float(wire_width) * 5.0):
        return {"reject": False, "reason": "probe_axis_span_too_short", "axis_span": round(float(axis_span), 3)}
    if float(np.std(axis_values)) <= 1e-6:
        return {"reject": False, "reason": "flat_axis"}
    bin_centers: list[float] = []
    bin_positions: list[float] = []
    rounded_axis = np.floor(axis_values + 0.5).astype(int)
    for bin_value in range(int(np.min(rounded_axis)), int(np.max(rounded_axis)) + 1):
        selector = rounded_axis == bin_value
        if np.count_nonzero(selector) <= 0:
            continue
        bin_positions.append(float(bin_value))
        bin_centers.append(float(np.median(cross_values[selector])))
    if len(bin_centers) < max(6, int(round(axis_span * 0.35))):
        return {
            "reject": False,
            "reason": "too_few_center_bins",
            "axis_span": round(float(axis_span), 3),
            "center_bins": int(len(bin_centers)),
        }
    fit_axis = np.array(bin_positions, dtype=float)
    fit_cross = np.array(bin_centers, dtype=float)
    if float(np.std(fit_axis)) <= 1e-6:
        return {"reject": False, "reason": "flat_center_axis"}
    slope, intercept = np.polyfit(fit_axis, fit_cross, deg=1)
    predicted = fit_axis * float(slope) + float(intercept)
    residual = float(np.median(np.abs(fit_cross - predicted)))
    max_residual = max(2.0, float(wire_width) * 1.5)
    min_slope = float(cfg.perpendicular_extension_far_diagonal_min_slope)
    reject = bool(abs(float(slope)) >= min_slope and residual <= max_residual)
    return {
        "reject": reject,
        "slope": round(float(slope), 3),
        "min_slope": round(float(min_slope), 3),
        "median_residual_pixels": round(float(residual), 3),
        "max_residual_pixels": round(float(max_residual), 3),
        "axis_span": round(float(axis_span), 3),
        "probe_pixels": int(axis_values.size),
        "center_bins": int(len(bin_centers)),
    }


def refresh_axis_candidate_geometry(binary: np.ndarray, candidate: dict[str, Any]) -> None:
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in candidate["bbox"]], binary.shape)
    if x2 <= x1 or y2 <= y1:
        return
    orientation = str(candidate.get("orientation", "horizontal"))
    area = int(np.count_nonzero(binary[y1:y2, x1:x2]))
    if orientation == "horizontal":
        span = int(x2 - x1)
        width = int(y2 - y1)
        center = (y1 + y2 - 1) / 2.0
        points = [[x1, int(round(center))], [x2 - 1, int(round(center))]]
    else:
        span = int(y2 - y1)
        width = int(x2 - x1)
        center = (x1 + x2 - 1) / 2.0
        points = [[int(round(center)), y1], [int(round(center)), y2 - 1]]
    candidate["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
    candidate["points"] = points
    candidate["span"] = float(span)
    candidate["width"] = float(width)
    candidate["area"] = int(area)
    candidate["centerline"] = float(center)
    candidate["projected_fill_ratio"] = float(area) / max(1.0, float(span * width))


def expand_extension_candidate_width(
    binary: np.ndarray,
    candidate: dict[str, Any],
    max_cross: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    expanded = dict(candidate)
    x1, y1, x2, y2 = clipped_bbox([int(v) for v in expanded["bbox"]], binary.shape)
    if x2 <= x1 or y2 <= y1:
        return expanded
    orientation = str(expanded.get("orientation", "horizontal"))
    max_extra = max(0, int(cfg.perpendicular_extension_width_expand_px))
    max_width = max(1, int(round(max_cross)))
    min_coverage = float(cfg.perpendicular_extension_width_expand_min_coverage)
    old_bbox = [int(x1), int(y1), int(x2), int(y2)]

    if orientation == "horizontal":
        axis_span = max(1, x2 - x1)
        while y1 > 0 and old_bbox[1] - y1 < max_extra and y2 - y1 < max_width:
            coverage = float(np.count_nonzero(binary[y1 - 1, x1:x2])) / float(axis_span)
            if coverage < min_coverage:
                break
            y1 -= 1
        while y2 < binary.shape[0] and y2 - old_bbox[3] < max_extra and y2 - y1 < max_width:
            coverage = float(np.count_nonzero(binary[y2, x1:x2])) / float(axis_span)
            if coverage < min_coverage:
                break
            y2 += 1
    else:
        axis_span = max(1, y2 - y1)
        while x1 > 0 and old_bbox[0] - x1 < max_extra and x2 - x1 < max_width:
            coverage = float(np.count_nonzero(binary[y1:y2, x1 - 1])) / float(axis_span)
            if coverage < min_coverage:
                break
            x1 -= 1
        while x2 < binary.shape[1] and x2 - old_bbox[2] < max_extra and x2 - x1 < max_width:
            coverage = float(np.count_nonzero(binary[y1:y2, x2])) / float(axis_span)
            if coverage < min_coverage:
                break
            x2 += 1

    expanded["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
    refresh_axis_candidate_geometry(binary, expanded)
    if expanded["bbox"] != old_bbox:
        expanded["pre_extension_width_expand_bbox"] = old_bbox
        expanded["extension_width_expanded_pixels"] = max(
            0,
            int(expanded.get("area", 0)) - int(candidate.get("area", 0)),
        )
    else:
        expanded["extension_width_expanded_pixels"] = 0
    return expanded


def extract_perpendicular_extension_candidates(
    binary: np.ndarray,
    claimed: np.ndarray,
    segment: dict[str, Any],
    endpoint_index: int,
    direction: tuple[int, int],
    text_height: float,
    wire_width: float,
    cfg: JustWireConfig,
    dot_stop_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    initial_search_distance = int(round(max(
        float(text_height) * float(cfg.perpendicular_extension_search_text_heights),
        float(wire_width) * float(cfg.perpendicular_extension_search_wire_widths),
        80.0,
    )))
    max_search_distance = int(round(max(
        float(initial_search_distance),
        float(text_height) * float(cfg.perpendicular_extension_max_search_text_heights),
        float(wire_width) * float(cfg.perpendicular_extension_max_search_wire_widths),
        float(cfg.perpendicular_extension_max_search_pixels),
    )))
    orientation = perpendicular_orientation(segment)
    min_len = int(math.ceil(max(
        8.0,
        float(wire_width) * float(cfg.perpendicular_extension_min_wire_widths),
        float(cfg.min_wire_width_px) * float(cfg.strict_min_aspect_ratio),
    )))
    min_cross = int(max(1, cfg.perpendicular_extension_min_width_px))
    max_cross = max(float(cfg.min_wire_width_px), float(wire_width) * float(cfg.max_wire_widths))
    endpoint = segment_endpoints(segment)[endpoint_index]
    anchor = extension_search_anchor(segment, endpoint_index, direction, binary.shape)
    extension_source = binary
    if dot_stop_mask is not None and bool(np.any(dot_stop_mask)):
        if segment_endpoint_touches_mask(binary, segment, endpoint_index, dot_stop_mask):
            return []
        extension_source = binary & ~dot_stop_mask
    endpoint_cap = segment_endpoint_cap_mask(binary, segment, endpoint_index)
    endpoint_cap_dilated = cv2.dilate(endpoint_cap.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    search_distance = initial_search_distance
    last_candidates: list[dict[str, Any]] = []
    while True:
        search_bbox = extension_search_bbox(segment, endpoint_index, direction, binary.shape, search_distance)
        x1, y1, x2, y2 = search_bbox
        if x2 <= x1 or y2 <= y1:
            return last_candidates
        local_binary = extension_source[y1:y2, x1:x2]
        if not bool(np.any(local_binary)):
            return last_candidates
        candidates: list[dict[str, Any]] = []
        projected = component_projected_rectangles(
            local_binary,
            orientation,
            x1,
            y1,
            min_len,
            search_distance + max(local_binary.shape),
            min_cross,
            max_cross,
            cfg,
        )
        for candidate in projected:
            candidate = expand_extension_candidate_width(binary, candidate, max_cross, cfg)
            if not extension_strict_window_ok(binary, candidate, cfg):
                continue
            slanted_drift_metrics = extension_candidate_slanted_drift_metrics(binary, candidate, cfg)
            if bool(slanted_drift_metrics.get("reject", False)):
                continue
            candidate["slanted_drift_metrics"] = dict(slanted_drift_metrics)
            far_diagonal_metrics = extension_candidate_far_diagonal_continuation_metrics(
                binary,
                candidate,
                direction,
                wire_width,
                cfg,
            )
            if bool(far_diagonal_metrics.get("reject", False)):
                continue
            candidate["far_diagonal_continuation_metrics"] = dict(far_diagonal_metrics)
            bbox = [int(v) for v in candidate["bbox"]]
            start_offset = direction_start_offset(bbox, anchor, direction)
            if start_offset < -1:
                continue
            if start_offset > 1:
                continue
            if extension_candidate_center_delta(candidate, endpoint) > max(1.0, segment_effective_width(segment) * 0.75):
                continue
            cx1, cy1, cx2, cy2 = clipped_bbox(bbox, binary.shape)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            candidate_mask = np.zeros(binary.shape, dtype=bool)
            candidate_mask[cy1:cy2, cx1:cx2] = extension_source[cy1:cy2, cx1:cx2]
            if not bool(np.any(endpoint_cap_dilated & candidate_mask)):
                continue
            area = int(np.count_nonzero(candidate_mask[cy1:cy2, cx1:cx2]))
            unclaimed_area = int(np.count_nonzero(candidate_mask[cy1:cy2, cx1:cx2] & ~claimed[cy1:cy2, cx1:cx2]))
            if unclaimed_area <= 0:
                continue
            if unclaimed_area / max(1.0, float(area)) < float(cfg.perpendicular_extension_min_unclaimed_ratio):
                continue
            candidate["direction"] = [int(direction[0]), int(direction[1])]
            candidate["run_start_offset"] = int(start_offset)
            candidate["unclaimed_pixel_ratio"] = float(unclaimed_area) / max(1.0, float(area))
            candidate["search_distance_pixels"] = int(search_distance)
            candidate["search_bbox"] = [int(v) for v in search_bbox]
            candidate["touches_search_far_edge"] = bool(
                candidate_touches_search_far_edge(bbox, search_bbox, direction)
            )
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                abs(int(item.get("run_start_offset", 0))),
                -float(item["span"]),
                -float(item.get("projected_fill_ratio", 0.0)),
            )
        )
        if not candidates:
            return last_candidates
        last_candidates = candidates
        if not bool(candidates[0].get("touches_search_far_edge", False)):
            return candidates
        if search_distance >= max_search_distance:
            return candidates
        next_distance = min(max_search_distance, max(search_distance + initial_search_distance, int(math.ceil(search_distance * 1.6))))
        if next_distance <= search_distance:
            return candidates
        search_distance = next_distance


def make_perpendicular_extension_segment(
    candidate: dict[str, Any],
    parent_segment: dict[str, Any],
    parent_segment_index: int,
    parent_endpoint_index: int,
    depth: int,
    next_index: int,
) -> dict[str, Any]:
    return {
        "segment_id": f"perp_ext_{next_index:04d}",
        "orientation": candidate["orientation"],
        "points": candidate["points"],
        "bbox": [int(v) for v in candidate["bbox"]],
        "span": float(candidate["span"]),
        "width": float(candidate["width"]),
        "area": int(candidate["area"]),
        "centerline": float(candidate["centerline"]),
        "projected_fill_ratio": float(candidate.get("projected_fill_ratio", 0.0)),
        "slanted_drift_metrics": dict(candidate.get("slanted_drift_metrics", {})),
        "far_diagonal_continuation_metrics": dict(candidate.get("far_diagonal_continuation_metrics", {})),
        "num_members": 1,
        "source_segment_type": "perpendicular_endpoint_extension",
        "source": "iterative_perpendicular_endpoint_extension",
        "parent_segment_index": int(parent_segment_index),
        "parent_segment_id": str(parent_segment.get("segment_id", "")),
        "parent_endpoint_index": int(parent_endpoint_index),
        "extension_depth": int(depth),
        "extension_direction": [int(v) for v in candidate["direction"]],
        "extension_run_start_offset": int(candidate.get("run_start_offset", 0)),
        "extension_search_distance_pixels": int(candidate.get("search_distance_pixels", 0)),
        "extension_touches_search_far_edge": bool(candidate.get("touches_search_far_edge", False)),
        "unclaimed_pixel_ratio": float(candidate.get("unclaimed_pixel_ratio", 0.0)),
        "pre_extension_width_expand_bbox": list(candidate.get("pre_extension_width_expand_bbox", [])),
        "extension_width_expanded_pixels": int(candidate.get("extension_width_expanded_pixels", 0)),
    }


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return float(math.hypot(px - sx, py - sy))
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return float(math.hypot(px - closest_x, py - closest_y))


def fit_diagonal_axis_from_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if coords.shape[0] < 2:
        return None
    mean = coords.mean(axis=0)
    centered = coords - mean
    covariance = np.cov(centered.T)
    if not np.all(np.isfinite(covariance)):
        return None
    _values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return None
    return mean, axis / norm


def diagonal_component_metrics(
    component_mask: np.ndarray,
    bbox: list[int],
    anchor_point: tuple[float, float] | None = None,
    cfg: JustWireConfig | None = None,
) -> dict[str, Any]:
    ys, xs = np.nonzero(component_mask)
    if xs.size < 2:
        return {"ok": False}
    x1, y1, _x2, _y2 = [int(v) for v in bbox]
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    fitted = fit_diagonal_axis_from_coords(coords)
    if fitted is None:
        return {"ok": False}
    mean, axis = fitted
    if anchor_point is not None:
        anchor = np.array(anchor_point, dtype=float)
        far_index = int(np.argmax(np.linalg.norm(coords - anchor.reshape(1, 2), axis=1)))
        if float((coords[far_index] - anchor) @ axis) < 0.0:
            axis = -axis

    trim_fraction = float(cfg.diagonal_extension_core_trim_fraction) if cfg is not None else 0.20
    trim_fraction = float(np.clip(trim_fraction, 0.05, 0.35))
    padding_px = float(cfg.diagonal_extension_core_width_padding_px) if cfg is not None else 2.5
    padding_px = float(np.clip(padding_px, 0.0, 2.5))

    projections = (coords - mean) @ axis
    perpendicular = np.array([-axis[1], axis[0]], dtype=float)
    cross = (coords - mean) @ perpendicular
    core_low = float(np.percentile(projections, trim_fraction * 100.0))
    core_high = float(np.percentile(projections, (1.0 - trim_fraction) * 100.0))
    core_selector = (projections >= core_low) & (projections <= core_high)
    if anchor_point is not None:
        anchor_dist = np.linalg.norm(coords - np.array(anchor_point, dtype=float).reshape(1, 2), axis=1)
        near_trim = max(2.0, float(np.percentile(anchor_dist, 15)))
        far_trim = float(np.percentile(anchor_dist, 92))
        core_selector &= (anchor_dist >= near_trim) & (anchor_dist <= far_trim)
    core_coords = coords[core_selector]
    if core_coords.shape[0] >= max(8, int(coords.shape[0] * 0.18)):
        fitted_core = fit_diagonal_axis_from_coords(core_coords)
        if fitted_core is not None:
            mean, axis = fitted_core
            if anchor_point is not None:
                anchor = np.array(anchor_point, dtype=float)
                if float((mean - anchor) @ axis) < 0.0:
                    axis = -axis
            perpendicular = np.array([-axis[1], axis[0]], dtype=float)

    projections = (coords - mean) @ axis
    cross = (coords - mean) @ perpendicular
    length = float(np.percentile(projections, 98) - np.percentile(projections, 2) + 1.0)
    raw_width = float(np.percentile(cross, 95) - np.percentile(cross, 5) + 1.0)
    core_low = float(np.percentile(projections, trim_fraction * 100.0))
    core_high = float(np.percentile(projections, (1.0 - trim_fraction) * 100.0))
    core_selector = (projections >= core_low) & (projections <= core_high)
    core_cross = cross[core_selector]
    if core_cross.size >= max(6, int(xs.size * 0.18)):
        core_center = float(np.median(core_cross))
        core_half = float(
            max(
                np.percentile(np.abs(core_cross - core_center), 80),
                np.percentile(np.abs(core_cross - core_center), 60) + padding_px,
                0.5,
            )
        )
        connected_middle = core_selector & (np.abs(cross - core_center) <= core_half)
        if np.count_nonzero(connected_middle) >= max(6, int(xs.size * 0.12)):
            fitted_middle = fit_diagonal_axis_from_coords(coords[connected_middle])
            if fitted_middle is not None:
                mean, axis = fitted_middle
                if anchor_point is not None:
                    anchor = np.array(anchor_point, dtype=float)
                    if float((mean - anchor) @ axis) < 0.0:
                        axis = -axis
                perpendicular = np.array([-axis[1], axis[0]], dtype=float)
                projections = (coords - mean) @ axis
                cross = (coords - mean) @ perpendicular
                core_selector = connected_middle
                core_cross = cross[core_selector]
    if core_cross.size >= max(6, int(xs.size * 0.12)):
        width = float(np.percentile(core_cross, 90) - np.percentile(core_cross, 10) + 1.0)
        core_pixel_count = int(core_cross.size)
    else:
        width = raw_width
        core_pixel_count = 0
    if length <= 0.0 or width <= 0.0:
        return {"ok": False}
    angle = abs(math.degrees(math.atan2(float(axis[1]), float(axis[0]))))
    angle = min(angle, 180.0 - angle)
    axis_min = int(math.floor(float(np.min(projections))))
    axis_max = int(math.ceil(float(np.max(projections))))
    bins = max(1, axis_max - axis_min + 1)
    occupied = np.zeros(bins, dtype=bool)
    for value in projections:
        occupied[int(round(float(value))) - axis_min] = True
    start_point = mean + axis * float(np.percentile(projections, 2))
    end_point = mean + axis * float(np.percentile(projections, 98))
    return {
        "ok": True,
        "axis": [float(axis[0]), float(axis[1])],
        "angle_degrees": float(angle),
        "length": float(length),
        "width": float(width),
        "raw_width": float(raw_width),
        "core_pixel_count": int(core_pixel_count),
        "core_width_padding_px": float(padding_px),
        "axis_coverage": float(np.count_nonzero(occupied)) / float(bins),
        "start_point": [float(start_point[0]), float(start_point[1])],
        "end_point": [float(end_point[0]), float(end_point[1])],
    }


def skeletonize_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or not bool(np.any(mask)):
        return np.zeros_like(mask, dtype=bool)
    work = mask.astype(np.uint8) * 255
    skel = np.zeros_like(work, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(mask.shape) + 8):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel |= cv2.subtract(work, opened)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skel > 0


def diagonal_branch_topology_metrics(
    mask: np.ndarray,
    origin: tuple[int, int],
    anchor_point: tuple[float, float],
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    skeleton = skeletonize_binary_mask(mask.astype(bool))
    if not bool(np.any(skeleton)):
        return {"ok": False, "reason": "empty_skeleton"}

    origin_x, origin_y = int(origin[0]), int(origin[1])
    anchor_x = float(anchor_point[0]) - float(origin_x)
    anchor_y = float(anchor_point[1]) - float(origin_y)
    ignore_radius = max(
        float(cfg.diagonal_extension_topology_anchor_ignore_px),
        float(wire_width) * float(cfg.diagonal_extension_topology_anchor_ignore_wire_widths),
    )
    yy, xx = np.mgrid[0 : skeleton.shape[0], 0 : skeleton.shape[1]]
    keep = ((xx.astype(float) - anchor_x) ** 2 + (yy.astype(float) - anchor_y) ** 2) > (ignore_radius ** 2)
    pruned = skeleton & keep
    skeleton_pixels = int(np.count_nonzero(pruned))
    if skeleton_pixels < int(cfg.diagonal_extension_topology_min_skeleton_pixels):
        return {
            "ok": True,
            "reason": "too_short_for_topology",
            "skeleton_pixels": int(skeleton_pixels),
            "anchor_ignore_radius_pixels": round(float(ignore_radius), 3),
        }

    neighbor_kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    neighbor_count = cv2.filter2D(pruned.astype(np.uint8), cv2.CV_16S, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    endpoint_mask = pruned & (neighbor_count == 1)
    branch_mask = pruned & (neighbor_count >= 3)
    endpoints = int(np.count_nonzero(endpoint_mask))
    branch_points = int(np.count_nonzero(branch_mask))
    endpoint_labels, _endpoint_components = cv2.connectedComponents(endpoint_mask.astype(np.uint8), connectivity=8)
    branch_labels, _branch_components = cv2.connectedComponents(branch_mask.astype(np.uint8), connectivity=8)
    endpoint_cluster_count = max(0, int(endpoint_labels) - 1)
    branch_cluster_count = max(0, int(branch_labels) - 1)

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(pruned.astype(np.uint8), connectivity=8)
    component_sizes = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, labels_count)
        if int(stats[label, cv2.CC_STAT_AREA]) > 0
    ]
    component_count = int(len(component_sizes))
    largest_component = int(max(component_sizes) if component_sizes else 0)
    largest_component_ratio = float(largest_component) / max(1.0, float(sum(component_sizes)))

    ok = (
        component_count <= int(cfg.diagonal_extension_topology_max_components)
        and endpoint_cluster_count >= int(cfg.diagonal_extension_topology_min_endpoint_clusters)
        and endpoint_cluster_count <= int(cfg.diagonal_extension_topology_max_endpoints)
        and branch_cluster_count <= int(cfg.diagonal_extension_topology_max_branch_points)
        and largest_component_ratio >= float(cfg.diagonal_extension_topology_min_largest_component_ratio)
    )
    reason = "ok"
    if component_count > int(cfg.diagonal_extension_topology_max_components):
        reason = "multi_component_skeleton"
    elif endpoint_cluster_count < int(cfg.diagonal_extension_topology_min_endpoint_clusters):
        reason = "too_few_endpoint_clusters"
    elif endpoint_cluster_count > int(cfg.diagonal_extension_topology_max_endpoints):
        reason = "too_many_endpoints"
    elif branch_cluster_count > int(cfg.diagonal_extension_topology_max_branch_points):
        reason = "branching_skeleton"
    elif largest_component_ratio < float(cfg.diagonal_extension_topology_min_largest_component_ratio):
        reason = "fragmented_skeleton"
    return {
        "ok": bool(ok),
        "reason": reason,
        "skeleton_pixels": int(skeleton_pixels),
        "anchor_ignore_radius_pixels": round(float(ignore_radius), 3),
        "component_count": int(component_count),
        "largest_component_pixels": int(largest_component),
        "largest_component_ratio": round(float(largest_component_ratio), 3),
        "endpoint_count": int(endpoints),
        "endpoint_cluster_count": int(endpoint_cluster_count),
        "branch_point_count": int(branch_points),
        "branch_cluster_count": int(branch_cluster_count),
    }


def connected_diagonal_topology_relaxation_ok(
    topology_metrics: dict[str, Any],
    cfg: JustWireConfig,
) -> bool:
    if not topology_metrics:
        return False
    component_count = int(topology_metrics.get("component_count", 0))
    branch_clusters = int(topology_metrics.get("branch_cluster_count", 0))
    endpoint_clusters = int(topology_metrics.get("endpoint_cluster_count", 0))
    largest_component_ratio = float(topology_metrics.get("largest_component_ratio", 0.0))
    return (
        component_count <= int(cfg.diagonal_extension_topology_max_components)
        and endpoint_clusters >= int(cfg.diagonal_extension_connected_topology_min_endpoint_clusters)
        and endpoint_clusters <= int(cfg.diagonal_extension_connected_topology_max_endpoints)
        and branch_clusters <= int(cfg.diagonal_extension_connected_topology_max_branch_points)
        and largest_component_ratio >= float(cfg.diagonal_extension_topology_min_largest_component_ratio)
    )


def oriented_rect_fit_metrics(
    mask: np.ndarray,
    origin: tuple[int, int],
    anchor_point: tuple[float, float],
    direction_vector: np.ndarray,
    cross_center: float,
    expected_width: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"ok": False, "reason": "empty"}
    direction = np.array(direction_vector, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return {"ok": False, "reason": "bad_direction"}
    direction = direction / norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    origin_x, origin_y = int(origin[0]), int(origin[1])
    coords = np.column_stack([xs.astype(float) + float(origin_x), ys.astype(float) + float(origin_y)])
    anchor = np.array(anchor_point, dtype=float)
    projections = (coords - anchor.reshape(1, 2)) @ direction
    cross_delta = (coords @ perpendicular) - float(cross_center)

    forward_selector = projections >= -max(float(wire_width) * 1.5, 4.0)
    if np.count_nonzero(forward_selector) < max(4, int(round(float(expected_width)))):
        forward_selector = np.ones(projections.shape, dtype=bool)
    fit_proj = projections[forward_selector]
    fit_cross = cross_delta[forward_selector]
    if fit_proj.size == 0:
        return {"ok": False, "reason": "empty_forward"}

    axis_min = int(math.floor(float(np.min(fit_proj))))
    axis_max = int(math.ceil(float(np.max(fit_proj))))
    axis_bins = max(1, axis_max - axis_min + 1)
    widths: list[float] = []
    occupied_bins: set[int] = set()
    for bin_value in range(axis_min, axis_max + 1):
        selector = np.abs(fit_proj - float(bin_value)) <= 0.5
        if not bool(np.any(selector)):
            continue
        occupied_bins.add(int(bin_value))
        widths.append(float(np.max(fit_cross[selector]) - np.min(fit_cross[selector]) + 1.0))
    if not widths:
        return {"ok": False, "reason": "empty_bins"}

    width_array = np.array(widths, dtype=float)
    robust_width = float(np.percentile(width_array, 80))
    mean_width = float(np.mean(width_array))
    width_cv = float(np.std(width_array) / max(1.0, mean_width))
    axis_coverage = float(len(occupied_bins)) / float(axis_bins)
    model_width = max(1.0, float(expected_width), float(wire_width))
    width_ratio = robust_width / model_width
    rect_width = min(
        max(model_width, robust_width),
        model_width + max(0.0, float(cfg.diagonal_extension_oriented_rect_fill_width_extra_px)),
    )
    fill_rect_width = max(
        1.0,
        rect_width - max(0.0, float(cfg.diagonal_extension_oriented_rect_fill_width_shrink_px)),
    )
    rect_area = max(1.0, float(axis_bins) * fill_rect_width)
    fill_ratio = float(fit_proj.size) / rect_area

    ok = (
        axis_coverage >= float(cfg.diagonal_extension_oriented_rect_min_axis_coverage)
        and fill_ratio >= float(cfg.diagonal_extension_oriented_rect_min_fill_ratio)
        and width_ratio <= float(cfg.diagonal_extension_oriented_rect_max_width_ratio)
        and width_cv <= float(cfg.diagonal_extension_oriented_rect_max_width_cv)
    )
    reason = "ok"
    if axis_coverage < float(cfg.diagonal_extension_oriented_rect_min_axis_coverage):
        reason = "low_axis_coverage"
    elif fill_ratio < float(cfg.diagonal_extension_oriented_rect_min_fill_ratio):
        reason = "low_fill_ratio"
    elif width_ratio > float(cfg.diagonal_extension_oriented_rect_max_width_ratio):
        reason = "high_width_ratio"
    elif width_cv > float(cfg.diagonal_extension_oriented_rect_max_width_cv):
        reason = "high_width_cv"
    return {
        "ok": bool(ok),
        "reason": reason,
        "axis_coverage": float(axis_coverage),
        "fill_ratio": float(fill_ratio),
        "width_ratio": float(width_ratio),
        "width_cv": float(width_cv),
        "robust_width": float(robust_width),
        "rect_width": float(rect_width),
        "fill_rect_width": float(fill_rect_width),
        "axis_bins": int(axis_bins),
        "occupied_bins": int(len(occupied_bins)),
    }


def axis_aligned_span_consistency_metrics(mask: np.ndarray) -> dict[str, Any]:
    pixels = int(np.count_nonzero(mask))
    if pixels <= 0 or mask.size <= 0:
        return {
            "ok": False,
            "pixel_count": 0,
            "fill_ratio": 0.0,
            "row_span_cv": 0.0,
            "column_span_cv": 0.0,
            "row_span_mean": 0.0,
            "column_span_mean": 0.0,
            "row_span_max": 0,
            "column_span_max": 0,
        }

    def nonzero_span_values(values: Sequence[int]) -> np.ndarray:
        return np.array([int(value) for value in values if int(value) > 0], dtype=float)

    row_spans = nonzero_span_values([int(np.count_nonzero(row)) for row in mask.astype(bool)])
    column_spans = nonzero_span_values(
        [int(np.count_nonzero(mask[:, index])) for index in range(int(mask.shape[1]))]
    )

    def span_cv(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.std(values) / max(1.0, float(np.mean(values))))

    return {
        "ok": True,
        "pixel_count": int(pixels),
        "fill_ratio": float(pixels) / max(1.0, float(mask.shape[0] * mask.shape[1])),
        "row_span_cv": span_cv(row_spans),
        "column_span_cv": span_cv(column_spans),
        "row_span_mean": float(np.mean(row_spans)) if row_spans.size else 0.0,
        "column_span_mean": float(np.mean(column_spans)) if column_spans.size else 0.0,
        "row_span_max": int(np.max(row_spans)) if row_spans.size else 0,
        "column_span_max": int(np.max(column_spans)) if column_spans.size else 0,
    }


def diagonal_solid_rectangle_like_metrics(
    completion: dict[str, Any],
    width: float,
    seed: dict[str, Any],
) -> dict[str, Any]:
    runs = list(completion.get("runs", []))
    bbox = [int(value) for value in completion.get("bbox", [])]
    if not runs or len(bbox) != 4:
        return {"ok": False, "reject": False, "reason": "empty"}
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return {"ok": False, "reject": False, "reason": "bad_bbox"}

    local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    spans = axis_aligned_span_consistency_metrics(local)
    fit = dict(completion.get("oriented_rect_fit", {}))
    width_value = max(1.0, float(width))
    fit_width_ratio = float(fit.get("width_ratio", 0.0))
    fit_fill_ratio = float(fit.get("fill_ratio", 0.0))
    fit_width_cv = float(fit.get("width_cv", 0.0))
    seed_recent_width = float(seed.get("scan_recent_width", 0.0))
    seed_source = str(seed.get("direction_source", ""))

    min_projection_cv = min(float(spans["row_span_cv"]), float(spans["column_span_cv"]))
    max_projection_mean = max(float(spans["row_span_mean"]), float(spans["column_span_mean"]))
    local_recovery_solid_rect = (
        seed_source == "local_no_seed_recovery"
        and width_value >= 7.5
        and fit_width_ratio >= 1.55
        and fit_fill_ratio >= 1.10
        and min_projection_cv <= 0.33
        and max_projection_mean >= width_value * 1.80
    )
    local_recovery_block_edge_rect = (
        seed_source == "local_no_seed_recovery"
        and not local_recovery_solid_rect
        and width_value >= 7.5
        and fit_width_ratio >= 1.55
        and fit_fill_ratio >= 1.15
        and fit_width_cv <= 0.42
        and float(spans["row_span_cv"]) <= 0.52
        and float(spans["column_span_cv"]) <= 0.42
        and float(spans["fill_ratio"]) >= 0.40
    )
    wide_seed_solid_rect = (
        seed_source == "width_scan_before_fan"
        and fit_fill_ratio >= 0.95
        and fit_width_cv >= 0.50
        and (
            (
                width_value >= 7.5
                and fit_width_ratio >= 1.70
                and seed_recent_width >= width_value * 1.05
            )
            or (
                width_value >= 4.5
                and fit_width_ratio >= 2.10
                and seed_recent_width >= width_value * 1.05
            )
        )
    )
    trim = bool(local_recovery_solid_rect or local_recovery_block_edge_rect)
    reject = bool(wide_seed_solid_rect)
    reason = "solid_rectangle_like" if (trim or reject) else "ok"
    if local_recovery_solid_rect:
        reason = "local_recovery_solid_rectangle_like"
    elif local_recovery_block_edge_rect:
        reason = "local_recovery_block_edge_rectangle_like"
    elif wide_seed_solid_rect:
        reason = "wide_seed_solid_rectangle_like"
    metrics = {
        "ok": True,
        "trim": trim,
        "reject": reject,
        "reason": reason,
        "width": float(width_value),
        "seed_source": seed_source,
        "seed_recent_width": float(seed_recent_width),
        "fit_width_ratio": float(fit_width_ratio),
        "fit_fill_ratio": float(fit_fill_ratio),
        "fit_width_cv": float(fit_width_cv),
    }
    metrics.update({f"axis_aligned_{key}": value for key, value in spans.items() if key != "ok"})
    return metrics


def directional_track_pixels(
    source_mask: np.ndarray,
    seed_mask: np.ndarray,
    seed_bbox: list[int],
    anchor_point: tuple[float, float],
    direction_vector: np.ndarray,
    width: float,
    wire_width: float,
    forward_extra: float,
    backward_extra: float,
    normal_padding: float = 0.0,
    cfg: JustWireConfig | None = None,
) -> dict[str, Any]:
    seed_x1, seed_y1, seed_x2, seed_y2 = [int(v) for v in seed_bbox]
    ys, xs = np.nonzero(seed_mask.astype(bool))
    if xs.size == 0:
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}
    direction = np.array(direction_vector, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}
    direction = direction / norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)

    seed_coords = np.column_stack([xs.astype(float) + float(seed_x1), ys.astype(float) + float(seed_y1)])
    seed_projections = seed_coords @ direction
    anchor = np.array(anchor_point, dtype=float)
    anchor_projection = float(anchor @ direction)
    cross_center = float(np.median(seed_coords @ perpendicular))
    width_tolerance = max(0.75, float(wire_width) * 0.20)
    base_cross_half = float(max(width * 0.5 + width_tolerance, 2.25))
    padded_cross_half = base_cross_half + max(0.0, min(2.5, float(normal_padding)))
    axis_start = min(float(np.min(seed_projections)), anchor_projection) - float(backward_extra)
    axis_end = max(float(np.max(seed_projections)), anchor_projection) + float(forward_extra)
    seed_mid_start = float(np.percentile(seed_projections, 20))
    seed_mid_end = float(np.percentile(seed_projections, 80))

    corners = []
    for projection in (axis_start, axis_end):
        for cross_value in (cross_center - padded_cross_half, cross_center + padded_cross_half):
            corners.append(direction * projection + perpendicular * cross_value)
    corners.extend([np.array([seed_x1, seed_y1], dtype=float), np.array([seed_x2, seed_y2], dtype=float), anchor])
    corner_array = np.vstack(corners)
    margin = int(math.ceil(max(3.0, float(wire_width) * 2.0)))
    x1 = max(0, int(math.floor(float(np.min(corner_array[:, 0])))) - margin)
    y1 = max(0, int(math.floor(float(np.min(corner_array[:, 1])))) - margin)
    x2 = min(source_mask.shape[1], int(math.ceil(float(np.max(corner_array[:, 0])))) + margin + 1)
    y2 = min(source_mask.shape[0], int(math.ceil(float(np.max(corner_array[:, 1])))) + margin + 1)
    if x2 <= x1 or y2 <= y1:
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}

    yy, xx = np.mgrid[y1:y2, x1:x2]
    coords = np.stack([xx.astype(float), yy.astype(float)], axis=-1)
    axial = coords @ direction
    cross = coords @ perpendicular
    seed_cross_limit = np.full(axial.shape, base_cross_half, dtype=float)
    middle_axis = (axial >= seed_mid_start) & (axial <= seed_mid_end)
    seed_cross_limit[middle_axis] = padded_cross_half
    seed_strip = (
        source_mask[y1:y2, x1:x2].astype(bool)
        & (axial >= axis_start)
        & (axial <= axis_end)
        & (np.abs(cross - cross_center) <= seed_cross_limit)
    )
    radial_strip = (
        source_mask[y1:y2, x1:x2].astype(bool)
        & (axial >= axis_start)
        & (axial <= axis_end)
        & (np.abs(cross - cross_center) <= padded_cross_half)
    )
    fit_cfg = cfg if cfg is not None else JustWireConfig()

    local_seed = np.zeros_like(radial_strip, dtype=bool)
    sx1 = max(0, seed_x1 - x1)
    sy1 = max(0, seed_y1 - y1)
    sx2 = min(x2 - x1, seed_x2 - x1)
    sy2 = min(y2 - y1, seed_y2 - y1)
    if sx2 <= sx1 or sy2 <= sy1:
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}
    local_seed[sy1:sy2, sx1:sx2] = seed_mask[
        sy1 + y1 - seed_y1 : sy2 + y1 - seed_y1,
        sx1 + x1 - seed_x1 : sx2 + x1 - seed_x1,
    ]
    local_seed &= seed_strip | radial_strip
    if not bool(np.any(local_seed)):
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}

    labels_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(radial_strip.astype(np.uint8), connectivity=8)
    seed_labels = {int(value) for value in np.unique(labels[local_seed]) if int(value) > 0}
    if not seed_labels:
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}
    merged = np.isin(labels, list(seed_labels))
    continuity_metrics = directional_projection_continuity_metrics(
        merged,
        (x1, y1),
        direction,
        wire_width,
    )
    if not bool(continuity_metrics.get("ok", False)):
        return {
            "bbox": [int(v) for v in seed_bbox],
            "runs": [],
            "completion_pixels": 0,
            "far_point": list(anchor_point),
            "projection_continuity": continuity_metrics,
        }
    fit_metrics = oriented_rect_fit_metrics(
        merged,
        (x1, y1),
        anchor_point,
        direction,
        cross_center,
        width,
        wire_width,
        fit_cfg,
    )
    if not bool(fit_metrics.get("ok", False)):
        return {
            "bbox": [int(v) for v in seed_bbox],
            "runs": [],
            "completion_pixels": 0,
            "far_point": list(anchor_point),
            "oriented_rect_fit": fit_metrics,
        }
    merged_y, merged_x = np.nonzero(merged)
    if merged_x.size == 0:
        return {"bbox": [int(v) for v in seed_bbox], "runs": [], "completion_pixels": 0, "far_point": list(anchor_point)}

    global_coords = np.column_stack([merged_x.astype(float) + float(x1), merged_y.astype(float) + float(y1)])
    projections = global_coords @ direction
    far_projection = float(np.max(projections))
    terminal_band = max(float(wire_width), float(width))
    far_coords = global_coords[projections >= far_projection - terminal_band]
    if far_coords.size == 0:
        far_coords = global_coords[[int(np.argmax(projections))]]
    far_point = np.mean(far_coords, axis=0)

    bx1 = int(x1 + int(np.min(merged_x)))
    by1 = int(y1 + int(np.min(merged_y)))
    bx2 = int(x1 + int(np.max(merged_x)) + 1)
    by2 = int(y1 + int(np.max(merged_y)) + 1)
    crop = merged[by1 - y1 : by2 - y1, bx1 - x1 : bx2 - x1]
    runs = mask_to_relative_runs(crop)
    return {
        "bbox": [bx1, by1, bx2, by2],
        "runs": runs,
        "completion_pixels": max(0, run_pixel_count(runs) - int(np.count_nonzero(seed_mask))),
        "far_point": [float(far_point[0]), float(far_point[1])],
        "track_cross_center": float(cross_center),
        "track_cross_half": float(base_cross_half),
        "track_middle_cross_half": float(padded_cross_half),
        "track_radial_connected": True,
        "projection_continuity": continuity_metrics,
        "oriented_rect_fit": fit_metrics,
    }


def directional_projection_continuity_metrics(
    mask: np.ndarray,
    origin: tuple[int, int],
    direction_vector: np.ndarray,
    wire_width: float,
) -> dict[str, Any]:
    ys, xs = np.nonzero(mask.astype(bool))
    if xs.size == 0:
        return {"ok": False, "reason": "empty_mask"}
    direction = np.array(direction_vector, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return {"ok": False, "reason": "zero_direction"}
    direction = direction / norm
    coords = np.column_stack([xs.astype(float) + float(origin[0]), ys.astype(float) + float(origin[1])])
    projections = coords @ direction
    min_projection = float(np.min(projections))
    bins = np.floor(projections - min_projection + 0.5).astype(int)
    if bins.size == 0:
        return {"ok": False, "reason": "empty_bins"}
    occupied = np.zeros(int(np.max(bins)) + 1, dtype=bool)
    occupied[bins] = True
    runs = true_runs(occupied)
    if not runs:
        return {"ok": False, "reason": "no_occupied_runs"}
    gaps: list[int] = []
    for (_start, previous_end), (next_start, _next_end) in zip(runs, runs[1:]):
        gaps.append(max(0, int(next_start) - int(previous_end)))
    max_gap = max(gaps) if gaps else 0
    axis_span = max(1, int(occupied.size))
    largest_run = max(int(end) - int(start) for start, end in runs)
    max_allowed_gap = max(2, int(round(float(wire_width) * 1.25)))
    largest_run_ratio = float(largest_run) / float(axis_span)
    ok = int(max_gap) <= int(max_allowed_gap) and largest_run_ratio >= 0.62
    return {
        "ok": bool(ok),
        "reason": "continuous" if ok else "projection_gap",
        "axis_bins": int(axis_span),
        "run_count": int(len(runs)),
        "max_gap_bins": int(max_gap),
        "max_allowed_gap_bins": int(max_allowed_gap),
        "largest_run_ratio": round(float(largest_run_ratio), 3),
    }


def diagonal_completion_region_metrics(
    completion: dict[str, Any],
    anchor_point: tuple[float, float],
    direction_vector: np.ndarray,
    width: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    runs = list(completion.get("runs", []))
    bbox = [int(value) for value in completion.get("bbox", [])]
    if not runs or len(bbox) != 4:
        return {"ok": False, "reason": "empty"}
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return {"ok": False, "reason": "bad_bbox"}

    local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    ys, xs = np.nonzero(local)
    if xs.size == 0:
        return {"ok": False, "reason": "empty_mask"}

    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
    component_sizes = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, labels_count)
        if int(stats[label, cv2.CC_STAT_AREA]) > 0
    ]
    component_count = int(len(component_sizes))
    largest_component = int(max(component_sizes) if component_sizes else 0)
    largest_component_ratio = float(largest_component) / max(1.0, float(sum(component_sizes)))

    direction = np.array(direction_vector, dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return {"ok": False, "reason": "bad_direction"}
    direction = direction / direction_norm

    continuity_metrics = directional_projection_continuity_metrics(
        local,
        (x1, y1),
        direction,
        wire_width,
    )
    continuity_ok = (
        bool(continuity_metrics.get("ok", False))
        and int(continuity_metrics.get("run_count", 0)) <= int(cfg.diagonal_extension_final_projection_max_run_count)
        and int(continuity_metrics.get("max_gap_bins", 0)) <= int(cfg.diagonal_extension_final_projection_max_gap_bins)
    )

    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    cross_center = float(completion.get("track_cross_center", 0.0))
    if abs(cross_center) <= 1e-6:
        cross_center = float(np.median(coords @ perpendicular))
    fit_metrics = oriented_rect_fit_metrics(
        local,
        (x1, y1),
        anchor_point,
        direction,
        cross_center,
        width,
        wire_width,
        cfg,
    )
    fit_ok = (
        bool(fit_metrics.get("ok", False))
        and float(fit_metrics.get("axis_coverage", 0.0)) >= float(cfg.diagonal_extension_final_rect_min_axis_coverage)
        and float(fit_metrics.get("fill_ratio", 0.0)) >= float(cfg.diagonal_extension_final_rect_min_fill_ratio)
        and float(fit_metrics.get("width_ratio", float("inf"))) <= float(cfg.diagonal_extension_final_rect_max_width_ratio)
        and float(fit_metrics.get("width_cv", float("inf"))) <= float(cfg.diagonal_extension_final_rect_max_width_cv)
    )

    ok = (
        component_count <= int(cfg.diagonal_extension_final_max_components)
        and continuity_ok
        and fit_ok
    )
    reason = "ok"
    if component_count > int(cfg.diagonal_extension_final_max_components):
        reason = "disconnected_region"
    elif not bool(continuity_metrics.get("ok", False)):
        reason = str(continuity_metrics.get("reason", "projection_gap"))
    elif int(continuity_metrics.get("run_count", 0)) > int(cfg.diagonal_extension_final_projection_max_run_count):
        reason = "projection_multi_run"
    elif int(continuity_metrics.get("max_gap_bins", 0)) > int(cfg.diagonal_extension_final_projection_max_gap_bins):
        reason = "projection_gap"
    elif not bool(fit_metrics.get("ok", False)):
        reason = str(fit_metrics.get("reason", "bad_rect_fit"))
    elif float(fit_metrics.get("axis_coverage", 0.0)) < float(cfg.diagonal_extension_final_rect_min_axis_coverage):
        reason = "low_rect_axis_coverage"
    elif float(fit_metrics.get("fill_ratio", 0.0)) < float(cfg.diagonal_extension_final_rect_min_fill_ratio):
        reason = "low_rect_fill_ratio"
    elif float(fit_metrics.get("width_ratio", float("inf"))) > float(cfg.diagonal_extension_final_rect_max_width_ratio):
        reason = "high_rect_width_ratio"
    elif float(fit_metrics.get("width_cv", float("inf"))) > float(cfg.diagonal_extension_final_rect_max_width_cv):
        reason = "high_rect_width_cv"

    return {
        "ok": bool(ok),
        "reason": reason,
        "component_count": int(component_count),
        "largest_component_pixels": int(largest_component),
        "largest_component_ratio": round(float(largest_component_ratio), 3),
        "projection_continuity": dict(continuity_metrics),
        "oriented_rect_fit": dict(fit_metrics),
    }




def trim_diagonal_completion_far_dot(
    completion: dict[str, Any],
    anchor_point: tuple[float, float],
    direction_vector: np.ndarray,
    width: float,
    wire_width: float,
) -> dict[str, Any]:
    runs = list(completion.get("runs", []))
    if not runs:
        return completion
    bbox = [int(v) for v in completion.get("bbox", [])]
    if len(bbox) != 4:
        return completion
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return completion

    local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    ys, xs = np.nonzero(local)
    if xs.size == 0:
        return completion

    direction = np.array(direction_vector, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return completion
    direction = direction / norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    anchor = np.array(anchor_point, dtype=float)
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    projections = (coords - anchor) @ direction
    cross = (coords - anchor) @ perpendicular
    far_projection = float(np.max(projections))
    terminal_window = max(float(width) * 4.0, float(wire_width) * 8.0, 18.0)
    terminal_selector = projections >= far_projection - terminal_window
    if np.count_nonzero(terminal_selector) < max(10, int(round(float(width) * 2.0))):
        return completion

    wide_threshold = max(float(width) * 1.65, float(width) + 4.0, float(wire_width) * 3.5)
    terminal_bins = sorted({int(round(float(value))) for value in projections[terminal_selector]})
    wide_bins: list[int] = []
    for bin_value in terminal_bins:
        selector = terminal_selector & (np.abs(projections - float(bin_value)) <= 0.5)
        if np.count_nonzero(selector) < max(3, int(round(float(width) * 0.7))):
            continue
        span = float(np.max(cross[selector]) - np.min(cross[selector]) + 1.0)
        if span >= wide_threshold:
            wide_bins.append(int(bin_value))
    if not wide_bins:
        return completion

    max_wide = max(wide_bins)
    if far_projection - float(max_wide) > max(float(width) * 1.5, float(wire_width) * 4.0, 8.0):
        return completion

    cluster = [max_wide]
    for bin_value in sorted((value for value in wide_bins if value < max_wide), reverse=True):
        if cluster[-1] - bin_value <= 2:
            cluster.append(bin_value)
        else:
            break
    trim_start = float(min(cluster))
    cut_projection = trim_start - max(1.0, float(wire_width) * 0.5)
    keep = projections < cut_projection
    removed_pixels = int(np.count_nonzero(~keep))
    if removed_pixels < max(8, int(round(float(width) * 1.5))):
        return completion
    if np.count_nonzero(keep) <= 0:
        return completion

    trimmed_local = np.zeros_like(local, dtype=bool)
    trimmed_local[ys[keep], xs[keep]] = True
    trimmed_y, trimmed_x = np.nonzero(trimmed_local)
    if trimmed_x.size == 0:
        return completion
    tbx1 = int(x1 + int(np.min(trimmed_x)))
    tby1 = int(y1 + int(np.min(trimmed_y)))
    tbx2 = int(x1 + int(np.max(trimmed_x)) + 1)
    tby2 = int(y1 + int(np.max(trimmed_y)) + 1)
    crop = trimmed_local[tby1 - y1 : tby2 - y1, tbx1 - x1 : tbx2 - x1]
    trimmed_runs = mask_to_relative_runs(crop)
    trimmed_coords = np.column_stack(
        [
            trimmed_x.astype(float) + float(x1),
            trimmed_y.astype(float) + float(y1),
        ]
    )
    trimmed_projections = (trimmed_coords - anchor) @ direction
    new_far_projection = float(np.max(trimmed_projections))
    terminal_band = max(float(wire_width), float(width))
    far_coords = trimmed_coords[trimmed_projections >= new_far_projection - terminal_band]
    if far_coords.size == 0:
        far_coords = trimmed_coords[[int(np.argmax(trimmed_projections))]]
    far_point = np.mean(far_coords, axis=0)

    trimmed = dict(completion)
    trimmed["bbox"] = [tbx1, tby1, tbx2, tby2]
    trimmed["runs"] = trimmed_runs
    trimmed["far_point"] = [float(far_point[0]), float(far_point[1])]
    trimmed["completion_pixels"] = max(0, int(completion.get("completion_pixels", 0)) - removed_pixels)
    trimmed["far_endpoint_dot_trimmed"] = True
    trimmed["far_endpoint_dot_trimmed_pixels"] = int(removed_pixels)
    trimmed["far_endpoint_dot_trim_cut_projection"] = float(cut_projection)
    return trimmed


def trim_diagonal_completion_solid_rectangle_tail(
    completion: dict[str, Any],
    anchor_point: tuple[float, float],
    direction_vector: np.ndarray,
    width: float,
    wire_width: float,
    min_wide_bins: int | None = None,
    cut_before_wide_cluster: bool = False,
) -> dict[str, Any]:
    runs = list(completion.get("runs", []))
    bbox = [int(v) for v in completion.get("bbox", [])]
    if not runs or len(bbox) != 4:
        return completion
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return completion

    local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    ys, xs = np.nonzero(local)
    if xs.size == 0:
        return completion

    direction = np.array(direction_vector, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return completion
    direction = direction / norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    anchor = np.array(anchor_point, dtype=float)
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    projections = (coords - anchor) @ direction
    cross = (coords - anchor) @ perpendicular
    forward = projections >= -max(float(wire_width) * 1.5, 4.0)
    if np.count_nonzero(forward) < max(8, int(round(float(width) * 1.5))):
        return completion

    axis_min = int(math.floor(float(np.min(projections[forward]))))
    axis_max = int(math.ceil(float(np.max(projections[forward]))))
    if axis_max <= axis_min:
        return completion

    wide_threshold = max(float(width) * 1.45, float(width) + 4.0)
    min_tail_projection = max(float(width) * 1.1, float(wire_width) * 3.0, 6.0)
    wide_cluster: list[int] = []
    for bin_value in range(axis_min, axis_max + 1):
        selector = forward & (np.abs(projections - float(bin_value)) <= 0.5)
        if np.count_nonzero(selector) < max(3, int(round(float(width) * 0.5))):
            if wide_cluster:
                break
            continue
        span = float(np.max(cross[selector]) - np.min(cross[selector]) + 1.0)
        is_tail = float(bin_value) >= min_tail_projection and span >= wide_threshold
        if is_tail:
            wide_cluster.append(int(bin_value))
            continue
        if wide_cluster:
            break

    min_tail_bins = (
        max(1, int(min_wide_bins))
        if min_wide_bins is not None
        else max(3, int(round(max(float(width), float(wire_width)) * 0.45)))
    )
    if len(wide_cluster) < min_tail_bins:
        return completion
    if cut_before_wide_cluster:
        trim_start = float(min(wide_cluster))
        cut_projection = trim_start - max(1.0, float(wire_width) * 0.25)
    else:
        trim_start = float(max(wide_cluster))
        cut_projection = trim_start + max(1.0, float(wire_width) * 0.25)
    keep = projections <= cut_projection
    removed_pixels = int(np.count_nonzero(~keep))
    if removed_pixels < max(8, int(round(float(width) * 1.5))):
        return completion
    if np.count_nonzero(keep) <= max(6, int(round(float(width)))):
        return completion

    trimmed_local = np.zeros_like(local, dtype=bool)
    trimmed_local[ys[keep], xs[keep]] = True
    trimmed_y, trimmed_x = np.nonzero(trimmed_local)
    if trimmed_x.size == 0:
        return completion

    tbx1 = int(x1 + int(np.min(trimmed_x)))
    tby1 = int(y1 + int(np.min(trimmed_y)))
    tbx2 = int(x1 + int(np.max(trimmed_x)) + 1)
    tby2 = int(y1 + int(np.max(trimmed_y)) + 1)
    crop = trimmed_local[tby1 - y1 : tby2 - y1, tbx1 - x1 : tbx2 - x1]
    trimmed_runs = mask_to_relative_runs(crop)
    trimmed_coords = np.column_stack(
        [
            trimmed_x.astype(float) + float(x1),
            trimmed_y.astype(float) + float(y1),
        ]
    )
    trimmed_projections = (trimmed_coords - anchor) @ direction
    far_projection = float(np.max(trimmed_projections))
    terminal_band = max(float(wire_width), float(width))
    far_coords = trimmed_coords[trimmed_projections >= far_projection - terminal_band]
    if far_coords.size == 0:
        far_coords = trimmed_coords[[int(np.argmax(trimmed_projections))]]
    far_point = np.mean(far_coords, axis=0)

    trimmed = dict(completion)
    trimmed["bbox"] = [tbx1, tby1, tbx2, tby2]
    trimmed["runs"] = trimmed_runs
    trimmed["far_point"] = [float(far_point[0]), float(far_point[1])]
    trimmed["completion_pixels"] = max(0, int(completion.get("completion_pixels", 0)) - removed_pixels)
    trimmed["solid_rectangle_tail_trimmed"] = True
    trimmed["solid_rectangle_tail_trimmed_pixels"] = int(removed_pixels)
    trimmed["solid_rectangle_tail_trim_cut_projection"] = float(cut_projection)
    return trimmed


def backfill_diagonal_junction_pixels(
    binary: np.ndarray,
    completion: dict[str, Any],
    anchor_point: tuple[float, float],
    direction_vector: np.ndarray,
    width: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any]:
    runs = list(completion.get("runs", []))
    bbox = [int(v) for v in completion.get("bbox", [])]
    far_point = completion.get("far_point", [])
    if not runs or len(bbox) != 4 or len(far_point) != 2:
        return completion
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return completion
    direction = np.array(direction_vector, dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return completion
    direction = direction / direction_norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    anchor = np.array(anchor_point, dtype=float)
    radius = min(
        float(cfg.diagonal_extension_junction_backfill_max_radius_px),
        max(8.0, float(wire_width) * float(cfg.diagonal_extension_junction_backfill_radius_wire_widths)),
    )
    axis_padding = max(radius, float(wire_width) * 8.0, 12.0)
    anchor_projection = float(anchor @ direction)
    far_projection = float(np.array(far_point, dtype=float) @ direction)
    axis_min = min(anchor_projection, far_projection) - axis_padding
    axis_max = max(anchor_projection, far_projection) + axis_padding
    cross_center = float(completion.get("track_cross_center", float(anchor @ perpendicular)))
    cross_half = max(
        float(width) * 0.5 + max(1.25, float(wire_width) * 0.35),
        float(completion.get("track_middle_cross_half", 0.0)),
    )
    backfill_cross_half = cross_half + max(1.0, float(wire_width) * 0.9)

    fx = float(far_point[0])
    fy = float(far_point[1])
    wx1 = max(0, int(math.floor(fx - radius)) - 2)
    wy1 = max(0, int(math.floor(fy - radius)) - 2)
    wx2 = min(binary.shape[1], int(math.ceil(fx + radius)) + 3)
    wy2 = min(binary.shape[0], int(math.ceil(fy + radius)) + 3)
    if wx2 <= wx1 or wy2 <= wy1:
        return completion

    window_foreground = binary[wy1:wy2, wx1:wx2].astype(bool)
    if not bool(np.any(window_foreground)):
        return completion
    yy, xx = np.mgrid[wy1:wy2, wx1:wx2]
    coords = np.stack([xx.astype(float), yy.astype(float)], axis=-1)
    axial = coords @ direction
    cross = coords @ perpendicular
    same_axis_tube = (
        (axial >= axis_min)
        & (axial <= axis_max)
        & (np.abs(cross - cross_center) <= backfill_cross_half)
    )
    near_far = (xx.astype(float) - fx) ** 2 + (yy.astype(float) - fy) ** 2 <= radius ** 2
    current_global = np.zeros(binary.shape, dtype=bool)
    current_mask = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    current_global[y1:y2, x1:x2] = current_mask
    current_window = current_global[wy1:wy2, wx1:wx2]
    candidate_foreground = window_foreground & near_far & same_axis_tube & ~current_window
    if not bool(np.any(candidate_foreground)):
        return completion

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate_foreground.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros_like(candidate_foreground, dtype=bool)
    max_component_area = int(round(max(80.0, float(wire_width) * float(wire_width) * 90.0)))
    for label in range(1, labels_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max_component_area:
            continue
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill_ratio = float(area) / max(1.0, float(component_width * component_height))
        aspect = max(float(component_width), float(component_height)) / max(1.0, min(float(component_width), float(component_height)))
        if (
            area >= max(60.0, float(wire_width) * float(wire_width) * 16.0)
            and aspect <= 1.45
            and fill_ratio >= 0.62
        ):
            continue
        keep |= component
    if not bool(np.any(keep)):
        return completion

    before_pixels = int(np.count_nonzero(current_global))
    current_global[wy1:wy2, wx1:wx2] |= keep
    ys, xs = np.nonzero(current_global)
    if xs.size == 0:
        return completion
    nbx1 = int(np.min(xs))
    nby1 = int(np.min(ys))
    nbx2 = int(np.max(xs) + 1)
    nby2 = int(np.max(ys) + 1)
    cropped = current_global[nby1:nby2, nbx1:nbx2]
    new_runs = mask_to_relative_runs(cropped)
    after_pixels = int(run_pixel_count(new_runs))
    added_pixels = max(0, after_pixels - before_pixels)
    if added_pixels <= 0:
        return completion
    updated = dict(completion)
    updated["bbox"] = [nbx1, nby1, nbx2, nby2]
    updated["runs"] = new_runs
    updated["completion_pixels"] = int(completion.get("completion_pixels", 0)) + int(added_pixels)
    updated["junction_backfill_pixels"] = int(completion.get("junction_backfill_pixels", 0)) + int(added_pixels)
    updated["junction_backfill_radius_pixels"] = float(radius)
    return updated


def diagonal_candidate_touches_endpoint(
    component_mask: np.ndarray,
    component_origin: tuple[int, int],
    segment: dict[str, Any],
    endpoint_index: int,
    shape: tuple[int, int],
    tolerance: int = 0,
    bridge_mask: np.ndarray | None = None,
) -> bool:
    x0, y0 = [int(v) for v in component_origin]
    cap = segment_endpoint_caps(segment, shape, tolerance=0)[endpoint_index]
    cx1, cy1, cx2, cy2 = [int(v) for v in cap]
    lx1 = max(0, cx1 - x0)
    ly1 = max(0, cy1 - y0)
    lx2 = min(component_mask.shape[1], cx2 - x0)
    ly2 = min(component_mask.shape[0], cy2 - y0)
    if lx2 > lx1 and ly2 > ly1:
        cap_local = np.zeros_like(component_mask, dtype=bool)
        cap_local[ly1:ly2, lx1:lx2] = True
        # Pixel-level connectivity allows 8-neighbor contact with the parent's
        # endpoint cap, but does not allow wider cap/dilation gaps.
        dilated_cap = cv2.dilate(cap_local.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if bool(np.any(dilated_cap & component_mask)):
            return True

    bridge = max(0, int(tolerance))
    if bridge <= 0 or bridge_mask is None:
        return False

    ys, xs = np.nonzero(component_mask)
    if xs.size == 0:
        return False
    comp_x1 = int(x0 + int(np.min(xs)))
    comp_y1 = int(y0 + int(np.min(ys)))
    comp_x2 = int(x0 + int(np.max(xs)) + 1)
    comp_y2 = int(y0 + int(np.max(ys)) + 1)
    rx1, ry1, rx2, ry2 = clipped_bbox(
        [
            min(cx1, comp_x1) - bridge,
            min(cy1, comp_y1) - bridge,
            max(cx2, comp_x2) + bridge,
            max(cy2, comp_y2) + bridge,
        ],
        shape,
    )
    if rx2 <= rx1 or ry2 <= ry1:
        return False

    local_bridge = bridge_mask[ry1:ry2, rx1:rx2].astype(bool).copy()
    cap_bridge = np.zeros_like(local_bridge, dtype=bool)
    cap_bridge[max(0, cy1 - ry1):min(ry2 - ry1, cy2 - ry1), max(0, cx1 - rx1):min(rx2 - rx1, cx2 - rx1)] = True
    comp_bridge = np.zeros_like(local_bridge, dtype=bool)
    comp_y = ys + y0 - ry1
    comp_x = xs + x0 - rx1
    valid = (
        (comp_y >= 0)
        & (comp_y < local_bridge.shape[0])
        & (comp_x >= 0)
        & (comp_x < local_bridge.shape[1])
    )
    if not bool(np.any(valid)):
        return False
    comp_bridge[comp_y[valid], comp_x[valid]] = True
    local_bridge |= cap_bridge | comp_bridge
    labels_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        local_bridge.astype(np.uint8),
        connectivity=8,
    )
    if labels_count <= 1:
        return False
    cap_labels = {int(value) for value in np.unique(labels[cap_bridge]) if int(value) > 0}
    if not cap_labels:
        return False
    comp_labels = {int(value) for value in np.unique(labels[comp_bridge]) if int(value) > 0}
    return bool(cap_labels & comp_labels)


def diagonal_endpoint_bridge_tolerance(wire_width: float, cfg: JustWireConfig) -> int:
    return int(math.ceil(max(1.0, float(wire_width) * float(cfg.diagonal_extension_endpoint_bridge_wire_widths))))


def diagonal_branch_masks_from_anchor(
    component_mask: np.ndarray,
    pixel_bbox: list[int],
    anchor_point: tuple[float, float],
    cfg: JustWireConfig,
) -> list[np.ndarray]:
    ys, xs = np.nonzero(component_mask)
    if xs.size < int(cfg.diagonal_extension_angle_cluster_min_pixels):
        return [component_mask]
    x1, y1, _x2, _y2 = [int(v) for v in pixel_bbox]
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    anchor = np.array(anchor_point, dtype=float)
    vectors = coords - anchor.reshape(1, 2)
    distances = np.linalg.norm(vectors, axis=1)
    if not np.any(distances > 1.0):
        return [component_mask]

    # Ignore the endpoint-adjacent cluster and the far overlap cluster while
    # estimating branch angles. Those two regions are where several wires
    # often touch or nearly touch.
    near_cut = max(2.0, float(np.percentile(distances, 12)))
    far_cut = float(np.percentile(distances, 88))
    selector = (distances >= near_cut) & (distances <= far_cut)
    if np.count_nonzero(selector) < int(cfg.diagonal_extension_angle_cluster_min_pixels):
        return [component_mask]

    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angle_half = float(np.clip(cfg.diagonal_extension_angle_cluster_degrees, 4.0, 14.0))
    seed_angles = angles[selector]
    branch_masks: list[np.ndarray] = [component_mask]
    seen_centers: list[float] = []
    for center in seed_angles:
        if any(abs(((float(center) - prev + 180.0) % 360.0) - 180.0) <= angle_half for prev in seen_centers):
            continue
        delta = np.abs(((angles - float(center) + 180.0) % 360.0) - 180.0)
        cluster = delta <= angle_half
        if np.count_nonzero(cluster) < int(cfg.diagonal_extension_angle_cluster_min_pixels):
            continue
        mask = np.zeros_like(component_mask, dtype=bool)
        mask[ys[cluster], xs[cluster]] = True
        # Keep only the largest connected piece so that unrelated fan branches
        # that happen to share a similar angle do not get folded together.
        labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if labels_count <= 1:
            continue
        best_label = max(range(1, labels_count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
        best_area = int(stats[best_label, cv2.CC_STAT_AREA])
        if best_area < int(cfg.diagonal_extension_angle_cluster_min_pixels):
            continue
        branch_masks.append(labels == best_label)
        seen_centers.append(float(center))
    return branch_masks


def diagonal_branch_masks_for_endpoint(
    component_mask: np.ndarray,
    pixel_bbox: list[int],
    anchor_point: tuple[float, float],
    parent_width: float,
    wire_width: float,
    text_height: float,
    cfg: JustWireConfig,
) -> list[np.ndarray]:
    branch_masks = diagonal_branch_masks_from_anchor(component_mask, pixel_bbox, anchor_point, cfg)
    if len(branch_masks) < int(cfg.diagonal_extension_branch_fallback_trigger_count):
        return branch_masks

    ys, xs = np.nonzero(component_mask)
    if xs.size < int(cfg.diagonal_extension_angle_cluster_min_pixels):
        return branch_masks
    x1, y1, _x2, _y2 = [int(v) for v in pixel_bbox]
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    anchor = np.array(anchor_point, dtype=float)
    distances = np.linalg.norm(coords - anchor.reshape(1, 2), axis=1)
    fallback_radius = max(
        10.0,
        float(text_height) * float(cfg.diagonal_extension_branch_fallback_radius_text_heights),
        float(wire_width) * float(cfg.diagonal_extension_branch_fallback_radius_wire_widths),
        float(parent_width) * 4.0,
    )
    fallback_masks: list[np.ndarray] | None = None
    for radius in (fallback_radius, fallback_radius * 0.75, fallback_radius * 0.55):
        near_selector = distances <= float(radius)
        if np.count_nonzero(near_selector) < int(cfg.diagonal_extension_angle_cluster_min_pixels):
            continue

        near_mask = np.zeros_like(component_mask, dtype=bool)
        near_mask[ys[near_selector], xs[near_selector]] = True
        near_branch_masks = diagonal_branch_masks_from_anchor(near_mask, pixel_bbox, anchor_point, cfg)
        if not near_branch_masks:
            continue
        fallback_masks = near_branch_masks
        if len(near_branch_masks) < int(cfg.diagonal_extension_branch_fallback_trigger_count):
            return near_branch_masks
    return fallback_masks if fallback_masks is not None else branch_masks


def diagonal_endpoint_seed_from_near_support(
    support_mask: np.ndarray,
    pixel_bbox: list[int],
    endpoint: tuple[float, float],
    parent_width: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(support_mask)
    if xs.size < 2:
        return None
    x1, y1, _x2, _y2 = [int(v) for v in pixel_bbox]
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    anchor = np.array(endpoint, dtype=float)
    distances = np.linalg.norm(coords - anchor.reshape(1, 2), axis=1)
    seed_radius = max(
        4.0,
        float(wire_width) * float(cfg.diagonal_extension_near_seed_wire_widths),
        float(parent_width) * float(cfg.diagonal_extension_near_seed_wire_widths),
    )
    selector = distances <= seed_radius
    if np.count_nonzero(selector) < max(3, int(round(float(parent_width) * 1.5))):
        relaxed_radius = max(seed_radius, float(np.percentile(distances, 45)))
        selector = distances <= relaxed_radius
    if np.count_nonzero(selector) < 3:
        return None

    seed_mask = np.zeros_like(support_mask, dtype=bool)
    seed_mask[ys[selector], xs[selector]] = True
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(seed_mask.astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return None
    best_label = 0
    best_score = -1
    for label in range(1, labels_count):
        component = labels == label
        component_ys, component_xs = np.nonzero(component)
        if component_xs.size == 0:
            continue
        component_coords = np.column_stack([
            component_xs.astype(float) + float(x1),
            component_ys.astype(float) + float(y1),
        ])
        if float(np.min(np.linalg.norm(component_coords - anchor.reshape(1, 2), axis=1))) > max(2.0, float(wire_width)):
            continue
        score = int(stats[label, cv2.CC_STAT_AREA])
        if score > best_score:
            best_score = score
            best_label = int(label)
    if best_label <= 0:
        return None

    seed_mask = labels == best_label
    seed_ys, seed_xs = np.nonzero(seed_mask)
    if seed_xs.size < 3:
        return None
    seed_coords = np.column_stack([seed_xs.astype(float) + float(x1), seed_ys.astype(float) + float(y1)])
    seed_distances = np.linalg.norm(seed_coords - anchor.reshape(1, 2), axis=1)
    far_distance = float(np.max(seed_distances))
    far_band = seed_distances >= max(0.0, far_distance - max(1.0, float(parent_width)))
    if np.count_nonzero(far_band) < 2:
        far_band = seed_distances >= float(np.percentile(seed_distances, 70))
    far_coords = seed_coords[far_band]
    if far_coords.shape[0] == 0:
        return None
    target = far_coords.mean(axis=0)
    direction = target - anchor
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    direction = direction / norm
    angle = abs(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    angle = min(angle, 180.0 - angle)
    min_angle = float(cfg.diagonal_extension_min_angle_degrees)
    if angle < min_angle or angle > 90.0 - min_angle:
        return None

    bx1 = int(x1 + int(np.min(seed_xs)))
    by1 = int(y1 + int(np.min(seed_ys)))
    bx2 = int(x1 + int(np.max(seed_xs)) + 1)
    by2 = int(y1 + int(np.max(seed_ys)) + 1)
    cropped = seed_mask[by1 - y1 : by2 - y1, bx1 - x1 : bx2 - x1]
    return {
        "mask": cropped,
        "bbox": [bx1, by1, bx2, by2],
        "direction": direction,
        "angle_degrees": float(angle),
        "pixel_count": int(seed_xs.size),
        "seed_radius": float(seed_radius),
    }


def diagonal_endpoint_seed_from_local_recovery(
    support_mask: np.ndarray,
    pixel_bbox: list[int],
    endpoint: tuple[float, float],
    parent_width: float,
    wire_width: float,
    text_height: float,
    cfg: JustWireConfig,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(support_mask)
    min_pixels = max(10, int(round(float(parent_width) * 3.0)))
    if xs.size < min_pixels:
        return None
    x1, y1, _x2, _y2 = [int(v) for v in pixel_bbox]
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    anchor = np.array(endpoint, dtype=float)
    rel = coords - anchor.reshape(1, 2)
    distances = np.linalg.norm(rel, axis=1)
    local_radius = max(
        10.0,
        float(wire_width) * float(cfg.diagonal_extension_no_seed_recovery_forward_wire_widths),
        float(text_height) * float(cfg.diagonal_extension_no_seed_recovery_forward_text_heights),
        float(parent_width) * 4.0,
    )
    local_selector = distances <= local_radius
    if np.count_nonzero(local_selector) < min_pixels:
        local_selector = distances <= max(local_radius, float(np.percentile(distances, 45.0)))
    if np.count_nonzero(local_selector) < min_pixels:
        return None

    local_coords = coords[local_selector]
    fitted = fit_diagonal_axis_from_coords(local_coords)
    if fitted is None:
        return None
    _mean, direction = fitted
    far_selector = distances >= float(np.percentile(distances, 65.0))
    far_center = np.mean(coords[far_selector], axis=0) if np.any(far_selector) else np.mean(local_coords, axis=0)
    if float((far_center - anchor) @ direction) < 0.0:
        direction = -direction
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return None
    direction = direction / direction_norm
    angle = abs(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    angle = min(angle, 180.0 - angle)
    min_angle = float(cfg.diagonal_extension_no_seed_recovery_min_angle_degrees)
    if angle < min_angle or angle > 90.0 - float(cfg.diagonal_extension_min_angle_degrees):
        return None

    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    forward = rel @ direction
    cross = rel @ perpendicular
    forward_limit = max(
        10.0,
        float(wire_width) * float(cfg.diagonal_extension_no_seed_recovery_forward_wire_widths),
        float(text_height) * float(cfg.diagonal_extension_no_seed_recovery_forward_text_heights),
        float(parent_width) * 4.0,
    )
    forward_selector = (
        (forward >= -max(1.0, float(wire_width) * 0.5))
        & (forward <= forward_limit)
        & local_selector
    )
    if np.count_nonzero(forward_selector) < min_pixels:
        return None
    local_cross = cross[forward_selector]
    cross_center = float(np.median(local_cross))
    robust_width = float(np.percentile(local_cross, 90.0) - np.percentile(local_cross, 10.0) + 1.0)
    width_limit = max(
        float(parent_width) * float(cfg.diagonal_extension_direction_scan_width_ratio),
        float(parent_width) + max(3.0, float(wire_width) * 1.5),
    )
    if robust_width < max(1.0, float(parent_width) * 0.35) or robust_width > width_limit:
        return None
    cross_half = max(1.5, robust_width * 0.6, float(wire_width) * 1.2)
    seed_selector = forward_selector & (np.abs(cross - cross_center) <= cross_half)
    if np.count_nonzero(seed_selector) < min_pixels:
        return None

    seed_forward = forward[seed_selector]
    bin_min = int(math.floor(float(np.min(seed_forward))))
    bin_max = int(math.ceil(float(np.max(seed_forward))))
    occupied = 0
    for bin_value in range(bin_min, bin_max + 1):
        if np.count_nonzero(seed_selector & (np.abs(forward - float(bin_value)) <= 0.5)) > 0:
            occupied += 1
    axis_coverage = float(occupied) / max(1.0, float(bin_max - bin_min + 1))
    if axis_coverage < 0.45:
        return None

    selected_local_x = xs[seed_selector]
    selected_local_y = ys[seed_selector]
    if selected_local_x.size < min_pixels:
        return None
    bx1 = int(x1 + int(np.min(selected_local_x)))
    by1 = int(y1 + int(np.min(selected_local_y)))
    bx2 = int(x1 + int(np.max(selected_local_x)) + 1)
    by2 = int(y1 + int(np.max(selected_local_y)) + 1)
    cropped = np.zeros((by2 - by1, bx2 - bx1), dtype=bool)
    cropped[selected_local_y + y1 - by1, selected_local_x + x1 - bx1] = True
    runs = mask_to_relative_runs(cropped)
    return {
        "mask": cropped,
        "bbox": [bx1, by1, bx2, by2],
        "pixel_runs": runs,
        "direction": direction,
        "angle_degrees": float(angle),
        "rough_angle_degrees": float(angle),
        "pixel_count": int(selected_local_x.size),
        "seed_radius": float(forward_limit),
        "scan_radius_pixels": float(forward_limit),
        "scan_recent_width": float(robust_width),
        "scan_robust_width": float(robust_width),
        "scan_axis_coverage": float(axis_coverage),
        "local_axis_aligned_width": float(axis_aligned_component_width(support_mask)),
        "stable_seed_width": float(robust_width),
        "stable_seed_bins": float(max(0, bin_max - bin_min + 1)),
        "stable_seed_bin_start": float(bin_min),
        "stable_seed_bin_end": float(bin_max),
        "fan_width_threshold": float("inf"),
        "normal_width_limit": float(width_limit),
        "direction_source": "local_no_seed_recovery",
    }


def fit_direction_from_binned_centers(
    coords: np.ndarray,
    anchor: np.ndarray,
    rough_direction: np.ndarray,
    wire_width: float,
) -> np.ndarray | None:
    direction = np.array(rough_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6 or coords.shape[0] < 4:
        return None
    direction = direction / norm
    rel = coords - anchor.reshape(1, 2)
    forward = rel @ direction
    usable = forward >= 0.0
    if np.count_nonzero(usable) < 4:
        return None
    coords = coords[usable]
    forward = forward[usable]
    bin_width = max(1.0, float(wire_width), 2.0)
    bin_values = np.floor(forward / bin_width).astype(int)
    centers: list[np.ndarray] = []
    for value in range(int(np.min(bin_values)), int(np.max(bin_values)) + 1):
        selector = bin_values == value
        if np.count_nonzero(selector) < 2:
            continue
        centers.append(np.median(coords[selector], axis=0))
    if len(centers) < 3:
        return None
    center_coords = np.vstack(centers)
    fitted = fit_diagonal_axis_from_coords(center_coords)
    if fitted is None:
        return None
    _mean, axis = fitted
    far_center = center_coords[int(np.argmax(np.linalg.norm(center_coords - anchor.reshape(1, 2), axis=1)))]
    if float((far_center - anchor) @ axis) < 0.0:
        axis = -axis
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return None
    return axis / axis_norm


def axis_aligned_component_width(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    row_widths: list[float] = []
    for y in np.unique(ys):
        row_xs = xs[ys == y]
        if row_xs.size > 0:
            row_widths.append(float(np.max(row_xs) - np.min(row_xs) + 1))
    col_widths: list[float] = []
    for x in np.unique(xs):
        col_ys = ys[xs == x]
        if col_ys.size > 0:
            col_widths.append(float(np.max(col_ys) - np.min(col_ys) + 1))
    if not row_widths or not col_widths:
        return 0.0
    row_width = float(np.percentile(np.array(row_widths, dtype=float), 70.0))
    col_width = float(np.percentile(np.array(col_widths, dtype=float), 70.0))
    return max(1.0, min(row_width, col_width))


def stable_diagonal_seed_selector(
    coords: np.ndarray,
    distances: np.ndarray,
    anchor: np.ndarray,
    rough_direction: np.ndarray,
    parent_width: float,
    local_width: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> tuple[np.ndarray, dict[str, float]] | None:
    direction = np.array(rough_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6 or coords.shape[0] < max(8, int(round(float(parent_width) * 3.0))):
        return None
    direction = direction / norm
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    rel = coords - anchor.reshape(1, 2)
    forward = rel @ direction
    cross = rel @ perpendicular
    min_forward = max(
        max(1.0, float(wire_width) * 1.5, float(parent_width) * 0.75),
        float(np.percentile(forward, 5.0)),
    )
    max_forward = float(np.percentile(forward, 96.0))
    bin_min = int(math.floor(min_forward))
    bin_max = int(math.ceil(max_forward))
    if bin_max - bin_min < max(4, int(round(float(wire_width) * 2.0))):
        return None
    max_bin_width = max(float(parent_width) + 1.0, float(local_width) + 1.0, float(wire_width) * 2.5)
    bin_infos: list[dict[str, Any]] = []
    for bin_value in range(bin_min, bin_max + 1):
        selector = (
            (forward >= float(bin_value) - 0.75)
            & (forward <= float(bin_value) + 0.75)
            & (distances >= min_forward)
        )
        count = int(np.count_nonzero(selector))
        if count <= 0:
            bin_infos.append({"ok": False, "bin": bin_value, "selector": selector, "width": 0.0, "count": 0})
            continue
        values = cross[selector]
        width = float(np.percentile(values, 90.0) - np.percentile(values, 10.0) + 1.0)
        ok = count >= 1 and width <= max_bin_width
        bin_infos.append({"ok": ok, "bin": bin_value, "selector": selector, "width": width, "count": count})
    min_run_bins = max(5, int(round(float(wire_width) * 3.0)))
    best_range: tuple[int, int] | None = None
    best_score = -1.0
    index = 0
    while index < len(bin_infos):
        if not bool(bin_infos[index]["ok"]):
            index += 1
            continue
        start = index
        while index < len(bin_infos) and bool(bin_infos[index]["ok"]):
            index += 1
        end = index
        if end - start < min_run_bins:
            continue
        run_selector = np.zeros(coords.shape[0], dtype=bool)
        widths: list[float] = []
        counts = 0
        for item in bin_infos[start:end]:
            run_selector |= item["selector"]
            widths.append(float(item["width"]))
            counts += int(item["count"])
        if np.count_nonzero(run_selector) < max(8, int(round(float(parent_width) * 2.0))):
            continue
        run_forward = forward[run_selector]
        span = float(np.max(run_forward) - np.min(run_forward) + 1.0)
        score = span + counts * 0.03 - float(np.mean(widths)) * 0.2
        if score > best_score:
            best_score = score
            best_range = (start, end)
    if best_range is None:
        return None
    seed_selector = np.zeros(coords.shape[0], dtype=bool)
    widths = []
    for item in bin_infos[best_range[0] : best_range[1]]:
        seed_selector |= item["selector"]
        widths.append(float(item["width"]))
    if np.count_nonzero(seed_selector) < max(8, int(round(float(parent_width) * 2.0))):
        return None
    return seed_selector, {
        "stable_width": float(np.percentile(np.array(widths, dtype=float), 80.0)) if widths else 0.0,
        "stable_bins": float(best_range[1] - best_range[0]),
        "stable_bin_start": float(bin_infos[best_range[0]]["bin"]),
        "stable_bin_end": float(bin_infos[best_range[1] - 1]["bin"]),
    }


def diagonal_endpoint_seed_from_width_scan(
    support_mask: np.ndarray,
    pixel_bbox: list[int],
    endpoint: tuple[float, float],
    parent_width: float,
    wire_width: float,
    cfg: JustWireConfig,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(support_mask)
    if xs.size < max(6, int(round(float(parent_width) * 2.0))):
        return None
    x1, y1, _x2, _y2 = [int(v) for v in pixel_bbox]
    coords = np.column_stack([xs.astype(float) + float(x1), ys.astype(float) + float(y1)])
    anchor = np.array(endpoint, dtype=float)
    rel = coords - anchor.reshape(1, 2)
    distances = np.linalg.norm(rel, axis=1)
    if float(np.max(distances)) < max(5.0, float(wire_width) * 2.0):
        return None
    start_radius = max(1.0, float(wire_width) * 0.5)
    local_axis_width = axis_aligned_component_width(support_mask)
    scan_width_base = max(float(parent_width), float(local_axis_width))

    min_radius = max(
        5.0,
        float(scan_width_base) * float(cfg.diagonal_extension_direction_scan_min_wire_widths),
        float(wire_width) * float(cfg.diagonal_extension_direction_scan_min_wire_widths),
    )
    fan_width_threshold = max(
        float(scan_width_base) * float(cfg.diagonal_extension_fan_width_ratio),
        float(scan_width_base) + max(6.0, float(wire_width) * 3.0),
    )
    normal_width_limit = max(
        float(scan_width_base) * float(cfg.diagonal_extension_direction_scan_width_ratio),
        float(scan_width_base) + max(3.0, float(wire_width) * 1.5),
    )
    max_radius = float(np.percentile(distances, 98.0))
    if max_radius < min_radius:
        return None

    best: dict[str, Any] | None = None
    radius_values = np.arange(min_radius, max_radius + 1.0, 1.0)
    latest_band_width = max(3.0, float(parent_width) * 2.0, float(wire_width) * 2.0)
    for radius in radius_values:
        prefix_selector = (distances >= start_radius) & (distances <= float(radius))
        if np.count_nonzero(prefix_selector) < max(8, int(round(float(parent_width) * 3.0))):
            continue
        far_band = prefix_selector & (distances >= max(1.0, float(radius) - latest_band_width))
        if np.count_nonzero(far_band) < max(2, int(round(float(parent_width)))):
            continue
        target = np.mean(coords[far_band], axis=0)
        direction = target - anchor
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            continue
        direction = direction / direction_norm
        angle = abs(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
        angle = min(angle, 180.0 - angle)
        min_angle = float(cfg.diagonal_extension_min_angle_degrees)
        if angle < min_angle or angle > 90.0 - min_angle:
            continue
        perpendicular = np.array([-direction[1], direction[0]], dtype=float)
        forward = rel @ direction
        cross = rel @ perpendicular
        prefix = prefix_selector & (forward >= -1.0) & (forward <= float(radius) + 1.0)
        if np.count_nonzero(prefix) < max(8, int(round(float(parent_width) * 3.0))):
            continue
        max_forward = float(np.max(forward[prefix]))
        recent = prefix & (forward >= max_forward - latest_band_width)
        if np.count_nonzero(recent) < max(2, int(round(float(parent_width)))):
            continue
        recent_width = float(np.percentile(cross[recent], 95.0) - np.percentile(cross[recent], 5.0) + 1.0)
        if recent_width > fan_width_threshold and best is not None:
            break
        if recent_width > normal_width_limit:
            continue

        bin_min = int(math.floor(float(np.min(forward[prefix]))))
        bin_max = int(math.ceil(max_forward))
        occupied = 0
        widths: list[float] = []
        for bin_value in range(bin_min, bin_max + 1):
            selector = prefix & (np.abs(forward - float(bin_value)) <= 0.5)
            if np.count_nonzero(selector) <= 0:
                continue
            occupied += 1
            widths.append(float(np.max(cross[selector]) - np.min(cross[selector]) + 1.0))
        axis_bins = max(1, bin_max - bin_min + 1)
        axis_coverage = float(occupied) / float(axis_bins)
        if axis_coverage < 0.45 or not widths:
            continue
        robust_width = float(np.percentile(np.array(widths, dtype=float), 80.0))
        if robust_width > normal_width_limit:
            continue
        best = {
            "selector": prefix,
            "radius": float(radius),
            "rough_direction": direction,
            "rough_angle_degrees": float(angle),
            "recent_width": float(recent_width),
            "robust_width": float(robust_width),
            "axis_coverage": float(axis_coverage),
        }

    if best is None:
        return diagonal_endpoint_seed_from_near_support(
            support_mask,
            pixel_bbox,
            endpoint,
            parent_width,
            wire_width,
            cfg,
        )

    stable_seed = stable_diagonal_seed_selector(
        coords,
        distances,
        anchor,
        np.array(best["rough_direction"], dtype=float),
        parent_width,
        local_axis_width,
        wire_width,
        cfg,
    )
    stable_metrics: dict[str, float] = {}
    if stable_seed is not None:
        seed_selector, stable_metrics = stable_seed
    else:
        seed_selector = np.array(best["selector"], dtype=bool)
    seed_coords = coords[seed_selector]
    if seed_coords.shape[0] < max(6, int(round(float(parent_width) * 2.0))):
        return None
    cap_trim_radius = max(1.0, float(wire_width) * 1.5, float(parent_width) * 0.75)
    direction_selector = seed_selector & (distances >= cap_trim_radius)
    direction_coords = coords[direction_selector]
    if direction_coords.shape[0] < max(6, int(round(float(parent_width) * 2.0))):
        direction_coords = seed_coords
    direction = fit_direction_from_binned_centers(
        direction_coords,
        anchor,
        np.array(best["rough_direction"], dtype=float),
        wire_width,
    )
    if direction is None:
        fitted = fit_diagonal_axis_from_coords(direction_coords)
        if fitted is None:
            direction = np.array(best["rough_direction"], dtype=float)
        else:
            _mean, direction = fitted
            direction_distances = np.linalg.norm(direction_coords - anchor.reshape(1, 2), axis=1)
            far_selector = direction_distances >= float(
                np.percentile(direction_distances, 75.0)
            )
            far_center = (
                np.mean(direction_coords[far_selector], axis=0)
                if np.any(far_selector)
                else np.mean(direction_coords, axis=0)
            )
            if float((far_center - anchor) @ direction) < 0.0:
                direction = -direction
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return None
    direction = direction / direction_norm
    angle = abs(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    angle = min(angle, 180.0 - angle)
    min_angle = float(cfg.diagonal_extension_min_angle_degrees)
    if angle < min_angle or angle > 90.0 - min_angle:
        return None

    selected_local_x = xs[seed_selector]
    selected_local_y = ys[seed_selector]
    bx1 = int(x1 + int(np.min(selected_local_x)))
    by1 = int(y1 + int(np.min(selected_local_y)))
    bx2 = int(x1 + int(np.max(selected_local_x)) + 1)
    by2 = int(y1 + int(np.max(selected_local_y)) + 1)
    cropped = np.zeros((by2 - by1, bx2 - bx1), dtype=bool)
    cropped[selected_local_y + y1 - by1, selected_local_x + x1 - bx1] = True
    runs = mask_to_relative_runs(cropped)
    return {
        "mask": cropped,
        "bbox": [bx1, by1, bx2, by2],
        "pixel_runs": runs,
        "direction": direction,
        "angle_degrees": float(angle),
        "rough_angle_degrees": float(best["rough_angle_degrees"]),
        "pixel_count": int(seed_coords.shape[0]),
        "seed_radius": float(best["radius"]),
        "scan_radius_pixels": float(best["radius"]),
        "scan_recent_width": float(best["recent_width"]),
        "scan_robust_width": float(best["robust_width"]),
        "scan_axis_coverage": float(best["axis_coverage"]),
        "local_axis_aligned_width": float(local_axis_width),
        "stable_seed_width": float(stable_metrics.get("stable_width", 0.0)),
        "stable_seed_bins": float(stable_metrics.get("stable_bins", 0.0)),
        "stable_seed_bin_start": float(stable_metrics.get("stable_bin_start", 0.0)),
        "stable_seed_bin_end": float(stable_metrics.get("stable_bin_end", 0.0)),
        "fan_width_threshold": float(fan_width_threshold),
        "normal_width_limit": float(normal_width_limit),
        "direction_source": "width_scan_before_fan",
    }






def diagonal_candidate_angle_weight(candidate: dict[str, Any]) -> float:
    core_pixels = float(candidate.get("core_pixel_count", 0))
    seed_pixels = float(candidate.get("direction_seed_pixel_count", 0))
    span = float(candidate.get("span", 0.0))
    width = float(candidate.get("width", 0.0))
    axis_coverage = float(candidate.get("axis_coverage", 0.0))
    area_score = max(0.0, span) * max(1.0, width) * max(0.1, axis_coverage)
    return max(core_pixels, seed_pixels * 0.75, area_score * 0.35, 1.0)


def dominant_diagonal_angle_cluster(
    candidates: Sequence[dict[str, Any]],
    cfg: JustWireConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_items = [
        (index, candidate, float(candidate.get("angle_degrees", 0.0)), diagonal_candidate_angle_weight(candidate))
        for index, candidate in enumerate(candidates)
        if math.isfinite(float(candidate.get("angle_degrees", 0.0)))
    ]
    if not valid_items:
        return list(candidates), {"cluster_count": 0, "accepted": True, "reason": "no_valid_angles"}
    valid_items.sort(key=lambda item: item[2])
    for _index, candidate, _angle, weight in valid_items:
        candidate["angle_candidate_weight"] = float(weight)
    cluster_degrees = float(cfg.diagonal_extension_fan_angle_cluster_degrees)
    clusters: list[dict[str, Any]] = []
    for index, candidate, angle, weight in valid_items:
        if not clusters or abs(angle - float(clusters[-1]["center"])) > cluster_degrees:
            clusters.append(
                {
                    "center": float(angle),
                    "weight": float(weight),
                    "items": [(index, candidate, angle, weight)],
                }
            )
            continue
        cluster = clusters[-1]
        previous_weight = float(cluster["weight"])
        new_weight = previous_weight + float(weight)
        cluster["center"] = (float(cluster["center"]) * previous_weight + float(angle) * float(weight)) / max(1.0, new_weight)
        cluster["weight"] = float(new_weight)
        cluster["items"].append((index, candidate, angle, weight))

    if len(clusters) <= 1:
        for _index, candidate, _angle, _weight in valid_items:
            candidate["angle_cluster_index"] = 0
            candidate["angle_cluster_count"] = 1
            candidate["dominant_angle_cluster"] = True
        return list(candidates), {
            "cluster_count": 1,
            "accepted": True,
            "reason": "single_cluster",
            "clusters": [
                {
                    "center": round(float(clusters[0]["center"]), 3),
                    "weight": round(float(clusters[0]["weight"]), 3),
                    "size": len(clusters[0]["items"]),
                }
            ],
        }

    ordered = sorted(enumerate(clusters), key=lambda item: float(item[1]["weight"]), reverse=True)
    dominant_index, dominant = ordered[0]
    second_weight = float(ordered[1][1]["weight"]) if len(ordered) > 1 else 0.0
    total_weight = sum(float(cluster["weight"]) for cluster in clusters)
    dominant_ratio = float(dominant["weight"]) / max(1.0, total_weight)
    margin_ratio = float(dominant["weight"]) / max(1.0, second_weight)
    for cluster_index, cluster in enumerate(clusters):
        for _index, candidate, _angle, _weight in cluster["items"]:
            candidate["angle_cluster_index"] = int(cluster_index)
            candidate["angle_cluster_count"] = int(len(clusters))
            candidate["dominant_angle_cluster"] = cluster_index == int(dominant_index)
    cluster_summary = [
        {
            "index": int(cluster_index),
            "center": round(float(cluster["center"]), 3),
            "weight": round(float(cluster["weight"]), 3),
            "size": len(cluster["items"]),
        }
        for cluster_index, cluster in enumerate(clusters)
    ]
    if (
        dominant_ratio >= float(cfg.diagonal_extension_dominant_cluster_min_weight_ratio)
        and margin_ratio >= float(cfg.diagonal_extension_dominant_cluster_min_margin_ratio)
    ):
        kept_indices = {int(item[0]) for item in dominant["items"]}
        filtered = [candidate for index, candidate in enumerate(candidates) if int(index) in kept_indices]
        for candidate in filtered:
            candidate["dominant_angle_cluster_ratio"] = float(dominant_ratio)
            candidate["dominant_angle_cluster_margin_ratio"] = float(margin_ratio)
        return filtered, {
            "cluster_count": int(len(clusters)),
            "accepted": True,
            "reason": "dominant_cluster_outliers_removed",
            "dominant_cluster_index": int(dominant_index),
            "dominant_weight_ratio": round(float(dominant_ratio), 3),
            "dominant_margin_ratio": round(float(margin_ratio), 3),
            "clusters": cluster_summary,
        }
    fallback_index, fallback_candidate, _fallback_angle, _fallback_weight = max(
        valid_items,
        key=lambda item: (
            float(item[3]),
            float(item[1].get("span", 0.0)),
            float(item[1].get("axis_coverage", 0.0)),
        ),
    )
    fallback_candidate["dominant_angle_cluster_ratio"] = float(dominant_ratio)
    fallback_candidate["dominant_angle_cluster_margin_ratio"] = float(margin_ratio)
    return [fallback_candidate], {
        "cluster_count": int(len(clusters)),
        "accepted": True,
        "reason": "multi_cluster_keep_strongest_candidate",
        "dominant_cluster_index": int(dominant_index),
        "fallback_candidate_index": int(fallback_index),
        "dominant_weight_ratio": round(float(dominant_ratio), 3),
        "dominant_margin_ratio": round(float(margin_ratio), 3),
        "clusters": cluster_summary,
    }


def find_diagonal_endpoint_extension_candidates(
    binary: np.ndarray,
    claimed: np.ndarray,
    segment: dict[str, Any],
    endpoint_index: int,
    text_height: float,
    wire_width: float,
    cfg: JustWireConfig,
    dot_stop_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    endpoint = segment_endpoints(segment)[endpoint_index]
    parent_width = max(1.0, diagonal_parent_effective_width(segment))
    search_distance = int(round(max(float(text_height) * float(cfg.diagonal_extension_search_text_heights), parent_width * 6.0, 16.0)))
    endpoint_x = int(round(float(endpoint[0])))
    endpoint_y = int(round(float(endpoint[1])))
    x1 = max(0, endpoint_x - search_distance)
    y1 = max(0, endpoint_y - search_distance)
    x2 = min(binary.shape[1], endpoint_x + search_distance + 1)
    y2 = min(binary.shape[0], endpoint_y + search_distance + 1)
    if x2 <= x1 or y2 <= y1:
        return []

    local_unclaimed = binary[y1:y2, x1:x2] & ~claimed[y1:y2, x1:x2]
    if not bool(np.any(local_unclaimed)):
        return []
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(local_unclaimed.astype(np.uint8), connectivity=8)
    candidates: list[dict[str, Any]] = []
    endpoint_array = np.array(endpoint, dtype=float)
    min_length = max(
        16.0,
        float(text_height) * float(cfg.diagonal_extension_min_length_text_heights),
        float(wire_width) * float(cfg.diagonal_extension_min_length_wire_widths),
    )
    width_limit = max(
        parent_width * float(cfg.diagonal_extension_max_width_ratio),
        parent_width + float(cfg.diagonal_extension_width_abs_tol_px),
    )
    bridge_tolerance = diagonal_endpoint_bridge_tolerance(wire_width, cfg)
    touching_components: list[tuple[list[int], np.ndarray, int]] = []
    for label in range(1, labels_count):
        lx = int(stats[label, cv2.CC_STAT_LEFT])
        ly = int(stats[label, cv2.CC_STAT_TOP])
        lw = int(stats[label, cv2.CC_STAT_WIDTH])
        lh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        pixel_bbox = [x1 + lx, y1 + ly, x1 + lx + lw, y1 + ly + lh]
        component_mask = labels[ly : ly + lh, lx : lx + lw] == label
        if not diagonal_candidate_touches_endpoint(
            component_mask,
            (x1 + lx, y1 + ly),
            segment,
            endpoint_index,
            binary.shape,
            tolerance=bridge_tolerance,
            bridge_mask=binary,
        ):
            continue
        touching_components.append((pixel_bbox, component_mask, area))

    if not touching_components:
        return []
    endpoint_blocked_by_solid_rectangle_like = False
    for pixel_bbox, component_mask, area in touching_components:
        for branch_mask in diagonal_branch_masks_for_endpoint(
            component_mask,
            pixel_bbox,
            endpoint,
            parent_width,
            wire_width,
            text_height,
            cfg,
        ):
            if not diagonal_candidate_touches_endpoint(
                branch_mask,
                (int(pixel_bbox[0]), int(pixel_bbox[1])),
                segment,
                endpoint_index,
                binary.shape,
                tolerance=bridge_tolerance,
                bridge_mask=binary,
            ):
                continue
            seed = diagonal_endpoint_seed_from_width_scan(
                branch_mask,
                pixel_bbox,
                endpoint,
                parent_width,
                wire_width,
                cfg,
            )
            if seed is None:
                seed = diagonal_endpoint_seed_from_local_recovery(
                    branch_mask,
                    pixel_bbox,
                    endpoint,
                    parent_width,
                    wire_width,
                    text_height,
                    cfg,
                )
            if seed is None:
                continue
            metrics = diagonal_component_metrics(branch_mask, pixel_bbox, endpoint, cfg)
            if not metrics.get("ok"):
                continue
            angle = float(seed["angle_degrees"])
            min_angle = float(cfg.diagonal_extension_min_angle_degrees)
            if angle < min_angle or angle > 90.0 - min_angle:
                continue
            length = float(metrics["length"])
            measured_width = float(metrics["width"])
            seed_from_local_recovery = str(seed.get("direction_source", "")) == "local_no_seed_recovery"
            seed_from_width_scan = str(seed.get("direction_source", "")) == "width_scan_before_fan"
            allow_short_width_scan_seed = (
                seed_from_width_scan
                and length >= max(8.0, parent_width * 2.5, float(wire_width) * 4.0)
                and float(seed.get("scan_axis_coverage", 0.0)) >= 0.70
            )
            if length < min_length * 0.75 and not seed_from_local_recovery and not allow_short_width_scan_seed:
                continue
            if measured_width < max(1.0, parent_width * 0.40) or measured_width > width_limit:
                continue
            if float(metrics["axis_coverage"]) < float(cfg.diagonal_extension_min_axis_coverage):
                continue
            seed_width = max(
                parent_width,
                float(seed.get("local_axis_aligned_width", 0.0)),
                float(seed.get("scan_robust_width", 0.0)),
                float(seed.get("scan_recent_width", 0.0)) * 0.85,
                measured_width,
            )
            width = min(width_limit, max(parent_width, seed_width))
            direction_vector = np.array(seed["direction"], dtype=float)
            direction_norm = float(np.linalg.norm(direction_vector))
            if direction_norm <= 1e-6:
                continue
            direction_vector = direction_vector / direction_norm
            far = endpoint_array + direction_vector * max(length, min_length)
            completion_source = binary
            forward_extra = max(
                float(text_height) * 3.5,
                float(length) * 0.75,
                float(wire_width) * 10.0,
            )
            if angle <= float(cfg.diagonal_extension_low_angle_continuation_degrees):
                forward_extra = max(
                    forward_extra,
                    float(text_height) * float(cfg.diagonal_extension_low_angle_continuation_text_heights),
                    float(length) * float(cfg.diagonal_extension_low_angle_continuation_length_ratio),
                    float(wire_width) * float(cfg.diagonal_extension_low_angle_continuation_wire_widths),
                )
            completion = directional_track_pixels(
                completion_source,
                seed["mask"],
                seed["bbox"],
                endpoint,
                direction_vector,
                width,
                wire_width,
                forward_extra=forward_extra,
                backward_extra=max(float(wire_width) * 3.0, 5.0),
                normal_padding=float(cfg.diagonal_extension_core_width_padding_px),
                cfg=cfg,
            )
            completion = trim_diagonal_completion_far_dot(
                completion,
                endpoint,
                direction_vector,
                width,
                wire_width,
            )
            completion = backfill_diagonal_junction_pixels(
                binary,
                completion,
                endpoint,
                direction_vector,
                width,
                wire_width,
                cfg,
            )
            final_region_metrics = diagonal_completion_region_metrics(
                completion,
                endpoint,
                direction_vector,
                width,
                wire_width,
                cfg,
            )
            completion["projection_continuity"] = dict(final_region_metrics.get("projection_continuity", {}))
            completion["oriented_rect_fit"] = dict(final_region_metrics.get("oriented_rect_fit", {}))
            rectangle_like_metrics = diagonal_solid_rectangle_like_metrics(
                completion,
                width,
                seed,
            )
            if bool(rectangle_like_metrics.get("trim", False)):
                completion = trim_diagonal_completion_solid_rectangle_tail(
                    completion,
                    endpoint,
                    direction_vector,
                    width,
                    wire_width,
                    min_wide_bins=2
                    if str(rectangle_like_metrics.get("reason", "")) == "local_recovery_block_edge_rectangle_like"
                    else None,
                    cut_before_wide_cluster=True,
                )
                retrim_metrics = diagonal_solid_rectangle_like_metrics(
                    completion,
                    width,
                    seed,
                )
                final_region_metrics = diagonal_completion_region_metrics(
                    completion,
                    endpoint,
                    direction_vector,
                    width,
                    wire_width,
                    cfg,
                )
                completion["projection_continuity"] = dict(final_region_metrics.get("projection_continuity", {}))
                completion["oriented_rect_fit"] = dict(final_region_metrics.get("oriented_rect_fit", {}))
                retrim_metrics["pre_trim_metrics"] = dict(rectangle_like_metrics)
                rectangle_like_metrics = retrim_metrics
            if bool(rectangle_like_metrics.get("reject", False)):
                if (
                    str(rectangle_like_metrics.get("seed_source", "")) == "width_scan_before_fan"
                    and angle >= 35.0
                ):
                    endpoint_blocked_by_solid_rectangle_like = True
                continue
            if not bool(final_region_metrics.get("ok", False)):
                continue
            parent_start, parent_end = segment_endpoints(segment)
            if not completion["runs"]:
                continue
            if not completion["runs"]:
                continue
            completed_area = int(run_pixel_count(list(completion["runs"])))
            if completed_area <= 0:
                continue
            if not diagonal_completion_touches_parent_endpoint(binary, completion, segment, endpoint_index):
                continue
            far_point = list(completion.get("far_point", [float(far[0]), float(far[1])]))
            span = float(np.linalg.norm(np.array(far_point, dtype=float) - endpoint_array))
            solid_rectangle_tail_trimmed = bool(completion.get("solid_rectangle_tail_trimmed", False))
            allow_trimmed_short_diagonal = (
                solid_rectangle_tail_trimmed
                and seed_from_local_recovery
                and span >= min_length * 0.55
            )
            allow_short_width_scan_diagonal = (
                allow_short_width_scan_seed
                and span >= max(8.0, parent_width * 2.5, float(wire_width) * 4.0)
                and float(metrics["axis_coverage"]) >= 0.70
            )
            if span < min_length and not allow_trimmed_short_diagonal and not allow_short_width_scan_diagonal:
                continue
            parent_distance = point_segment_distance((float(far_point[0]), float(far_point[1])), parent_start, parent_end)
            trimmed_parent_distance_limit = max(float(width) * 0.8, float(wire_width) * 2.0, 4.0)
            if parent_distance < float(text_height) and not (
                allow_trimmed_short_diagonal and parent_distance >= trimmed_parent_distance_limit
            ):
                continue
            provisional_segment = {
                "orientation": "diagonal",
                "points": [
                    [int(round(float(endpoint[0]))), int(round(float(endpoint[1])))],
                    [int(round(float(far_point[0]))), int(round(float(far_point[1])))],
                ],
                "bbox": [int(v) for v in completion["bbox"]],
                "wire_pixel_bbox": [int(v) for v in completion["bbox"]],
                "wire_pixel_runs": list(completion["runs"]),
            }
            candidates.append(
                {
                    "bbox": [int(v) for v in completion["bbox"]],
                    "points": provisional_segment["points"],
                    "orientation": "diagonal",
                    "span": span,
                    "width": float(width),
                    "area": int(completed_area),
                    "projected_fill_ratio": float(completed_area) / max(1.0, length * width),
                    "angle_degrees": float(angle),
                    "axis_coverage": float(metrics["axis_coverage"]),
                    "direction_vector": [float(direction_vector[0]), float(direction_vector[1])],
                    "diagonal_forward_extra_pixels": float(forward_extra),
                    "low_angle_continuation_applied": bool(
                        angle <= float(cfg.diagonal_extension_low_angle_continuation_degrees)
                    ),
                    "wire_pixel_runs": list(completion["runs"]),
                    "completion_pixels": int(completion["completion_pixels"]),
                    "junction_dot_boundary_trimmed": bool(completion.get("junction_dot_boundary_trimmed", False)),
                    "junction_dot_boundary_trimmed_pixels": int(completion.get("junction_dot_boundary_trimmed_pixels", 0)),
                    "far_endpoint_dot_trimmed": bool(completion.get("far_endpoint_dot_trimmed", False)),
                    "far_endpoint_dot_trimmed_pixels": int(completion.get("far_endpoint_dot_trimmed_pixels", 0)),
                    "solid_rectangle_tail_trimmed": bool(completion.get("solid_rectangle_tail_trimmed", False)),
                    "solid_rectangle_tail_trimmed_pixels": int(completion.get("solid_rectangle_tail_trimmed_pixels", 0)),
                    "track_cross_center": float(completion.get("track_cross_center", 0.0)),
                    "track_cross_half": float(completion.get("track_cross_half", 0.0)),
                    "track_middle_cross_half": float(completion.get("track_middle_cross_half", 0.0)),
                    "track_radial_connected": bool(completion.get("track_radial_connected", False)),
                    "projection_continuity": dict(completion.get("projection_continuity", {})),
                    "oriented_rect_fit": dict(completion.get("oriented_rect_fit", {})),
                    "final_region_metrics": dict(final_region_metrics),
                    "solid_rectangle_like_metrics": dict(rectangle_like_metrics),
                    "direction_seed_bbox": [int(v) for v in seed.get("bbox", [])],
                    "direction_seed_pixel_runs": list(seed.get("pixel_runs", mask_to_relative_runs(seed["mask"]))),
                    "direction_seed_angle_degrees": float(seed.get("angle_degrees", angle)),
                    "direction_seed_source": str(seed.get("direction_source", "")),
                    "direction_seed_pixel_count": int(seed.get("pixel_count", 0)),
                    "direction_seed_scan_radius_pixels": float(seed.get("scan_radius_pixels", seed.get("seed_radius", 0.0))),
                    "direction_seed_scan_recent_width": float(seed.get("scan_recent_width", 0.0)),
                    "direction_seed_scan_robust_width": float(seed.get("scan_robust_width", 0.0)),
                    "direction_seed_scan_axis_coverage": float(seed.get("scan_axis_coverage", 0.0)),
                    "direction_seed_fan_width_threshold": float(seed.get("fan_width_threshold", 0.0)),
                    "direction_seed_normal_width_limit": float(seed.get("normal_width_limit", 0.0)),
                    "track_truncated_at_projection": completion.get("track_truncated_at_projection"),
                    "track_truncated_pixels": int(completion.get("track_truncated_pixels", 0)),
                    "junction_backfill_pixels": int(completion.get("junction_backfill_pixels", 0)),
                    "junction_backfill_radius_pixels": float(completion.get("junction_backfill_radius_pixels", 0.0)),
                    "core_pixel_count": int(metrics.get("core_pixel_count", 0)),
                    "core_width_padding_px": float(metrics.get("core_width_padding_px", 0.0)),
                    "topology_metrics": {},
                    "new_endpoint_parent_distance": float(parent_distance),
                    "search_distance_pixels": int(search_distance),
                    "unclaimed_pixel_ratio": 1.0,
                }
            )
    if endpoint_blocked_by_solid_rectangle_like and not candidates:
        return []
    if not candidates:
        return []
    candidates, angle_cluster_summary = dominant_diagonal_angle_cluster(candidates, cfg)
    if not candidates:
        return []
    for candidate in candidates:
        candidate["angle_cluster_summary"] = dict(angle_cluster_summary)
    candidates.sort(key=lambda item: (
        -float(item.get("angle_candidate_weight", 0.0)),
        -float(item["span"]),
        -float(item["axis_coverage"]),
    ))
    return candidates


def diagonal_candidate_matches_existing(
    item: dict[str, Any],
    existing: dict[str, Any],
    wire_width: float,
    cfg: JustWireConfig,
) -> bool:
    if int(existing.get("segment_index", -1)) != int(item.get("segment_index", -2)):
        return False
    if int(existing.get("endpoint_index", -1)) != int(item.get("endpoint_index", -2)):
        return False
    candidate = item.get("candidate", {})
    first = existing.get("candidate", {})
    angle_delta = abs(
        ((float(candidate.get("angle_degrees", 0.0)) - float(first.get("angle_degrees", 0.0)) + 90.0) % 180.0)
        - 90.0
    )
    angle_limit = float(np.clip(cfg.diagonal_extension_angle_cluster_degrees, 4.0, 14.0))
    if angle_delta > angle_limit:
        return False
    points = candidate.get("points", [])
    first_points = first.get("points", [])
    if len(points) < 2 or len(first_points) < 2:
        return False
    far = np.array(points[1], dtype=float)
    first_far = np.array(first_points[1], dtype=float)
    far_distance = float(np.linalg.norm(far - first_far))
    far_limit = max(8.0, float(wire_width) * 4.0)
    if far_distance <= far_limit:
        return True
    bbox_overlap = bbox_overlap_fraction(
        [int(v) for v in candidate.get("bbox", [0, 0, 0, 0])],
        [int(v) for v in first.get("bbox", [0, 0, 0, 0])],
    )
    return bbox_overlap >= 0.85


def diagonal_item_anchor_point(item: dict[str, Any]) -> np.ndarray:
    points = item.get("candidate", {}).get("points", [])
    if len(points) < 1:
        return np.zeros(2, dtype=float)
    return np.array(points[0], dtype=float)


def diagonal_item_far_point(item: dict[str, Any]) -> np.ndarray:
    points = item.get("candidate", {}).get("points", [])
    if len(points) < 2:
        return diagonal_item_anchor_point(item)
    return np.array(points[1], dtype=float)


def diagonal_items_share_line(
    dense_item: dict[str, Any],
    sparse_item: dict[str, Any],
    wire_width: float,
    cfg: JustWireConfig,
) -> bool:
    dense_candidate = dense_item.get("candidate", {})
    sparse_candidate = sparse_item.get("candidate", {})
    dense_direction = np.array(dense_candidate.get("direction_vector", [0.0, 0.0]), dtype=float)
    sparse_direction = np.array(sparse_candidate.get("direction_vector", [0.0, 0.0]), dtype=float)
    dense_norm = float(np.linalg.norm(dense_direction))
    sparse_norm = float(np.linalg.norm(sparse_direction))
    if dense_norm <= 1e-6 or sparse_norm <= 1e-6:
        return False
    dense_direction /= dense_norm
    sparse_direction /= sparse_norm
    if float(dense_direction @ sparse_direction) < 0.0:
        sparse_direction = -sparse_direction
    angle_delta = abs(math.degrees(math.atan2(
        float(dense_direction[0] * sparse_direction[1] - dense_direction[1] * sparse_direction[0]),
        float(dense_direction @ sparse_direction),
    )))
    if angle_delta > float(np.clip(cfg.diagonal_extension_angle_cluster_degrees, 4.0, 14.0)):
        return False

    anchor = diagonal_item_anchor_point(sparse_item)
    normal = np.array([-sparse_direction[1], sparse_direction[0]], dtype=float)
    dense_points = [diagonal_item_anchor_point(dense_item), diagonal_item_far_point(dense_item)]
    sparse_points = [anchor, diagonal_item_far_point(sparse_item)]
    distance_limit = max(4.0, float(wire_width) * 3.0)
    if max(abs(float((point - anchor) @ normal)) for point in dense_points) > distance_limit:
        return False

    sparse_projection = [float((point - anchor) @ sparse_direction) for point in sparse_points]
    dense_projection = [float((point - anchor) @ sparse_direction) for point in dense_points]
    sparse_min, sparse_max = min(sparse_projection), max(sparse_projection)
    dense_min, dense_max = min(dense_projection), max(dense_projection)
    gap = max(0.0, max(sparse_min, dense_min) - min(sparse_max, dense_max))
    if gap > max(10.0, float(wire_width) * 6.0):
        return False

    sparse_length = float(sparse_candidate.get("span", 0.0))
    dense_length = float(dense_candidate.get("span", 0.0))
    if sparse_length + max(4.0, float(wire_width) * 2.0) < dense_length:
        return False
    return True


def diagonal_item_quality_score(item: dict[str, Any]) -> float:
    candidate = item.get("candidate", {})
    span = float(candidate.get("span", 0.0))
    axis_coverage = float(candidate.get("axis_coverage", 0.0))
    fill_ratio = float(candidate.get("projected_fill_ratio", 0.0))
    track_half = float(candidate.get("track_middle_cross_half", candidate.get("track_cross_half", 0.0)))
    width = float(candidate.get("width", 0.0))
    return (
        span * 0.65
        + axis_coverage * 24.0
        + fill_ratio * 12.0
        - track_half * 2.5
        - width * 0.75
    )


def diagonal_item_angle_delta_degrees(first_item: dict[str, Any], second_item: dict[str, Any]) -> float:
    first_direction = np.array(first_item.get("candidate", {}).get("direction_vector", [0.0, 0.0]), dtype=float)
    second_direction = np.array(second_item.get("candidate", {}).get("direction_vector", [0.0, 0.0]), dtype=float)
    first_norm = float(np.linalg.norm(first_direction))
    second_norm = float(np.linalg.norm(second_direction))
    if first_norm <= 1e-6 or second_norm <= 1e-6:
        return 180.0
    first_direction /= first_norm
    second_direction /= second_norm
    if float(first_direction @ second_direction) < 0.0:
        second_direction = -second_direction
    return abs(math.degrees(math.atan2(
        float(first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0]),
        float(first_direction @ second_direction),
    )))


def diagonal_item_width_ratio(first_item: dict[str, Any], second_item: dict[str, Any]) -> float:
    first_width = max(1.0, float(first_item.get("candidate", {}).get("width", 0.0)))
    second_width = max(1.0, float(second_item.get("candidate", {}).get("width", 0.0)))
    return max(first_width, second_width) / max(1.0, min(first_width, second_width))


def bbox_center_distance(first_bbox: list[int], second_bbox: list[int]) -> float:
    if len(first_bbox) != 4 or len(second_bbox) != 4:
        return float("inf")
    first_center = np.array(
        [
            (float(first_bbox[0]) + float(first_bbox[2])) * 0.5,
            (float(first_bbox[1]) + float(first_bbox[3])) * 0.5,
        ],
        dtype=float,
    )
    second_center = np.array(
        [
            (float(second_bbox[0]) + float(second_bbox[2])) * 0.5,
            (float(second_bbox[1]) + float(second_bbox[3])) * 0.5,
        ],
        dtype=float,
    )
    return float(np.linalg.norm(first_center - second_center))


def same_parent_diagonal_candidates_are_duplicate_estimates(
    first_item: dict[str, Any],
    second_item: dict[str, Any],
    wire_width: float,
    cfg: JustWireConfig,
) -> bool:
    angle_delta = diagonal_item_angle_delta_degrees(first_item, second_item)
    if angle_delta > 12.75:
        return False
    if diagonal_item_width_ratio(first_item, second_item) > 2.5:
        return False

    first_bbox = [int(v) for v in first_item.get("candidate", {}).get("bbox", [0, 0, 0, 0])]
    second_bbox = [int(v) for v in second_item.get("candidate", {}).get("bbox", [0, 0, 0, 0])]
    overlap_fraction = bbox_overlap_fraction(first_bbox, second_bbox)
    if overlap_fraction < 0.50:
        return False

    far_distance = float(np.linalg.norm(diagonal_item_far_point(first_item) - diagonal_item_far_point(second_item)))
    far_limit = max(12.0, float(wire_width) * 6.0)
    center_distance = bbox_center_distance(first_bbox, second_bbox)
    center_limit = max(16.0, float(wire_width) * 8.0)
    if far_distance > far_limit and center_distance > center_limit:
        return False
    return True


def diagonal_items_compete_for_same_pixels(
    first_item: dict[str, Any],
    second_item: dict[str, Any],
    wire_width: float,
    cfg: JustWireConfig,
) -> bool:
    first_candidate = first_item.get("candidate", {})
    second_candidate = second_item.get("candidate", {})
    first_bbox = [int(v) for v in first_candidate.get("bbox", [0, 0, 0, 0])]
    second_bbox = [int(v) for v in second_candidate.get("bbox", [0, 0, 0, 0])]
    overlap_fraction = bbox_overlap_fraction(first_bbox, second_bbox)
    if overlap_fraction < 0.55:
        return False

    first_direction = np.array(first_candidate.get("direction_vector", [0.0, 0.0]), dtype=float)
    second_direction = np.array(second_candidate.get("direction_vector", [0.0, 0.0]), dtype=float)
    first_norm = float(np.linalg.norm(first_direction))
    second_norm = float(np.linalg.norm(second_direction))
    if first_norm <= 1e-6 or second_norm <= 1e-6:
        return False
    first_direction /= first_norm
    second_direction /= second_norm
    if float(first_direction @ second_direction) < 0.0:
        second_direction = -second_direction
    angle_delta = abs(math.degrees(math.atan2(
        float(first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0]),
        float(first_direction @ second_direction),
    )))
    angle_limit = 14.0
    if angle_delta > angle_limit:
        return False

    first_anchor = diagonal_item_anchor_point(first_item)
    second_anchor = diagonal_item_anchor_point(second_item)
    first_far = diagonal_item_far_point(first_item)
    second_far = diagonal_item_far_point(second_item)
    center_distance = bbox_center_distance(first_bbox, second_bbox)
    if overlap_fraction >= 0.80 and center_distance <= max(16.0, float(wire_width) * 8.0):
        return True

    endpoint_limit = max(12.0, float(wire_width) * 6.0)
    far_limit = max(18.0, float(wire_width) * 9.0)
    if (
        float(np.linalg.norm(first_anchor - second_anchor)) > endpoint_limit
        and float(np.linalg.norm(first_far - second_far)) > far_limit
    ):
        return False
    return True


def suppress_dense_endpoint_diagonal_duplicates(
    collected: list[dict[str, Any]],
    wire_width: float,
    cfg: JustWireConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(collected) <= 1:
        return collected, []
    density_radius = max(10.0, float(wire_width) * 6.0)
    anchors = [diagonal_item_far_point(item) for item in collected]
    densities: list[int] = []
    for anchor in anchors:
        densities.append(int(sum(float(np.linalg.norm(anchor - other)) <= density_radius for other in anchors)))
    for item, density in zip(collected, densities):
        item["parent_endpoint_density"] = int(density)

    suppressed_indices: set[int] = set()
    events: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(collected):
        grouped[(int(item.get("segment_index", -1)), int(item.get("endpoint_index", -1)))].append(index)
    for group_indices in grouped.values():
        if len(group_indices) <= 1:
            continue
        ordered = sorted(group_indices, key=lambda index: -diagonal_item_quality_score(collected[index]))
        for first_offset, first_index in enumerate(ordered):
            if first_index in suppressed_indices:
                continue
            duplicate_cluster = [int(first_index)]
            for second_index in ordered[first_offset + 1:]:
                if second_index in suppressed_indices:
                    continue
                if not same_parent_diagonal_candidates_are_duplicate_estimates(
                    collected[first_index],
                    collected[second_index],
                    wire_width,
                    cfg,
                ):
                    continue
                duplicate_cluster.append(int(second_index))
            if len(duplicate_cluster) <= 1:
                continue
            keep_index = max(
                duplicate_cluster,
                key=lambda index: (
                    float(collected[index].get("candidate", {}).get("span", 0.0)),
                    diagonal_item_quality_score(collected[index]),
                    float(collected[index].get("candidate", {}).get("axis_coverage", 0.0)),
                ),
            )
            for dense_index in duplicate_cluster:
                if dense_index == keep_index:
                    continue
                suppressed_indices.add(int(dense_index))
                angle_delta = diagonal_item_angle_delta_degrees(collected[keep_index], collected[dense_index])
                far_distance = float(
                    np.linalg.norm(diagonal_item_far_point(collected[keep_index]) - diagonal_item_far_point(collected[dense_index]))
                )
                kept_bbox = [int(v) for v in collected[keep_index].get("candidate", {}).get("bbox", [])]
                suppressed_bbox = [int(v) for v in collected[dense_index].get("candidate", {}).get("bbox", [])]
                events.append(
                    {
                        "reason": "same_parent_endpoint_similar_branch_keep_longest",
                        "suppressed_parent_segment_id": str(collected[dense_index].get("parent_segment_id", "")),
                        "suppressed_segment_index": int(collected[dense_index].get("segment_index", -1)),
                        "suppressed_endpoint_index": int(collected[dense_index].get("endpoint_index", -1)),
                        "kept_parent_segment_id": str(collected[keep_index].get("parent_segment_id", "")),
                        "kept_segment_index": int(collected[keep_index].get("segment_index", -1)),
                        "kept_endpoint_index": int(collected[keep_index].get("endpoint_index", -1)),
                        "angle_delta_degrees": round(float(angle_delta), 3),
                        "far_endpoint_distance": round(float(far_distance), 3),
                        "bbox_overlap_fraction": round(float(bbox_overlap_fraction(suppressed_bbox, kept_bbox)), 3),
                        "suppressed_score": round(float(diagonal_item_quality_score(collected[dense_index])), 3),
                        "kept_score": round(float(diagonal_item_quality_score(collected[keep_index])), 3),
                        "suppressed_density": int(collected[dense_index].get("parent_endpoint_density", 0)),
                        "kept_density": int(collected[keep_index].get("parent_endpoint_density", 0)),
                        "suppressed_bbox": suppressed_bbox,
                        "kept_bbox": kept_bbox,
                    }
                )
    return [item for index, item in enumerate(collected) if index not in suppressed_indices], events


def make_diagonal_extension_segment(
    candidate: dict[str, Any],
    parent_segment: dict[str, Any],
    parent_segment_index: int,
    parent_endpoint_index: int,
    next_index: int,
) -> dict[str, Any]:
    return {
        "segment_id": f"diag_ext_{next_index:04d}",
        "orientation": "diagonal",
        "points": candidate["points"],
        "bbox": [int(v) for v in candidate["bbox"]],
        "span": float(candidate["span"]),
        "width": float(candidate["width"]),
        "area": int(candidate["area"]),
        "centerline": 0.0,
        "projected_fill_ratio": float(candidate.get("projected_fill_ratio", 0.0)),
        "num_members": 1,
        "source_segment_type": "diagonal_endpoint_extension",
        "source": "diagonal_endpoint_extension",
        "parent_segment_index": int(parent_segment_index),
        "parent_segment_id": str(parent_segment.get("segment_id", "")),
        "parent_endpoint_index": int(parent_endpoint_index),
        "extension_angle_degrees": float(candidate["angle_degrees"]),
        "extension_direction_vector": list(candidate["direction_vector"]),
        "extension_axis_coverage": float(candidate["axis_coverage"]),
        "diagonal_forward_extra_pixels": float(candidate.get("diagonal_forward_extra_pixels", 0.0)),
        "low_angle_continuation_applied": bool(candidate.get("low_angle_continuation_applied", False)),
        "extension_search_distance_pixels": int(candidate.get("search_distance_pixels", 0)),
        "track_cross_center": float(candidate.get("track_cross_center", 0.0)),
        "track_cross_half": float(candidate.get("track_cross_half", 0.0)),
        "track_middle_cross_half": float(candidate.get("track_middle_cross_half", 0.0)),
        "track_radial_connected": bool(candidate.get("track_radial_connected", False)),
        "projection_continuity": dict(candidate.get("projection_continuity", {})),
        "near_endpoint_axis": dict(candidate.get("near_endpoint_axis", {})),
        "oriented_rect_fit": dict(candidate.get("oriented_rect_fit", {})),
        "final_region_metrics": dict(candidate.get("final_region_metrics", {})),
        "direction_seed_bbox": [int(v) for v in candidate.get("direction_seed_bbox", [])],
        "direction_seed_pixel_runs": list(candidate.get("direction_seed_pixel_runs", [])),
        "direction_seed_angle_degrees": float(candidate.get("direction_seed_angle_degrees", 0.0)),
        "direction_seed_source": str(candidate.get("direction_seed_source", "")),
        "direction_seed_pixel_count": int(candidate.get("direction_seed_pixel_count", 0)),
        "direction_seed_scan_radius_pixels": float(candidate.get("direction_seed_scan_radius_pixels", 0.0)),
        "direction_seed_scan_recent_width": float(candidate.get("direction_seed_scan_recent_width", 0.0)),
        "direction_seed_scan_robust_width": float(candidate.get("direction_seed_scan_robust_width", 0.0)),
        "direction_seed_scan_axis_coverage": float(candidate.get("direction_seed_scan_axis_coverage", 0.0)),
        "direction_seed_fan_width_threshold": float(candidate.get("direction_seed_fan_width_threshold", 0.0)),
        "direction_seed_normal_width_limit": float(candidate.get("direction_seed_normal_width_limit", 0.0)),
        "track_truncated_at_projection": candidate.get("track_truncated_at_projection"),
        "track_truncated_pixels": int(candidate.get("track_truncated_pixels", 0)),
        "angle_cluster_index": int(candidate.get("angle_cluster_index", -1)),
        "angle_cluster_count": int(candidate.get("angle_cluster_count", 0)),
        "angle_candidate_weight": float(candidate.get("angle_candidate_weight", 0.0)),
        "dominant_angle_cluster": bool(candidate.get("dominant_angle_cluster", False)),
        "dominant_angle_cluster_ratio": float(candidate.get("dominant_angle_cluster_ratio", 0.0)),
        "dominant_angle_cluster_margin_ratio": float(candidate.get("dominant_angle_cluster_margin_ratio", 0.0)),
        "angle_cluster_summary": dict(candidate.get("angle_cluster_summary", {})),
        "junction_backfill_pixels": int(candidate.get("junction_backfill_pixels", 0)),
        "junction_backfill_radius_pixels": float(candidate.get("junction_backfill_radius_pixels", 0.0)),
        "diagonal_core_pixel_count": int(candidate.get("core_pixel_count", 0)),
        "diagonal_core_width_padding_px": float(candidate.get("core_width_padding_px", 0.0)),
        "new_endpoint_parent_distance": float(candidate.get("new_endpoint_parent_distance", 0.0)),
        "completion_pixels": int(candidate.get("completion_pixels", 0)),
        "junction_dot_boundary_trimmed": bool(candidate.get("junction_dot_boundary_trimmed", False)),
        "junction_dot_boundary_trimmed_pixels": int(candidate.get("junction_dot_boundary_trimmed_pixels", 0)),
        "far_endpoint_dot_trimmed": bool(candidate.get("far_endpoint_dot_trimmed", False)),
        "far_endpoint_dot_trimmed_pixels": int(candidate.get("far_endpoint_dot_trimmed_pixels", 0)),
        "solid_rectangle_tail_trimmed": bool(candidate.get("solid_rectangle_tail_trimmed", False)),
        "solid_rectangle_tail_trimmed_pixels": int(candidate.get("solid_rectangle_tail_trimmed_pixels", 0)),
        "solid_rectangle_tail_trim_cut_projection": float(
            candidate.get("solid_rectangle_tail_trim_cut_projection", 0.0)
        ),
        "solid_rectangle_like_metrics": dict(candidate.get("solid_rectangle_like_metrics", {})),
        "unclaimed_pixel_ratio": float(candidate.get("unclaimed_pixel_ratio", 0.0)),
        "wire_pixel_bbox": [int(v) for v in candidate["bbox"]],
        "wire_pixel_runs": list(candidate["wire_pixel_runs"]),
        "wire_pixel_count": int(candidate["area"]),
        "body_width": float(candidate["width"]),
        "bbox_width": float(candidate["width"]),
    }


def endpoint_has_unclaimed_diagonal_support(
    binary: np.ndarray,
    claimed: np.ndarray,
    segment: dict[str, Any],
    endpoint_index: int,
    wire_width: float,
    dot_stop_mask: np.ndarray | None = None,
) -> bool:
    tolerance = max(2, int(round(float(wire_width) * 2.0)))
    x1, y1, x2, y2 = segment_endpoint_caps(segment, binary.shape, tolerance=tolerance)[endpoint_index]
    if x2 <= x1 or y2 <= y1:
        return False
    local = binary[y1:y2, x1:x2] & ~claimed[y1:y2, x1:x2]
    return bool(np.any(local))


def extend_solid_wires_diagonal_once(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    page: int,
    text_height: float,
    wire_width: float,
    cfg: JustWireConfig,
    dot_stop_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    classify_segment_endpoints(binary, segments)
    claimed, trim_events = build_diagonal_extension_claimed_mask(binary, segments, wire_width, cfg)
    existing_union_mask = build_segment_union_mask(binary, segments)
    queue = [
        (segment_index, endpoint_index)
        for segment_index, segment in enumerate(segments)
        for endpoint_index, role in enumerate(segment.get("endpoint_roles", [])[:2])
        if role == "external_end"
        and endpoint_has_unclaimed_diagonal_support(binary, claimed, segment, endpoint_index, wire_width, dot_stop_mask)
    ]
    initial_queue_size = len(queue)
    accepted: list[dict[str, Any]] = []
    collected: list[dict[str, Any]] = []
    existing_signatures = {
        (str(segment.get("orientation", "")), tuple(int(v) for v in segment.get("bbox", [])))
        for segment in segments
    }
    seen_candidate_signatures: set[tuple[str, tuple[int, ...], int, int]] = set()
    for segment_index, endpoint_index in queue:
        if len(collected) >= int(cfg.diagonal_extension_max_total):
            break
        if segment_index >= len(segments):
            continue
        segment = segments[segment_index]
        if ensure_endpoint_roles(segment)[endpoint_index] != "external_end":
            continue
        candidates = find_diagonal_endpoint_extension_candidates(
            binary,
            claimed,
            segment,
            endpoint_index,
            text_height,
            wire_width,
            cfg,
            dot_stop_mask,
        )
        if not candidates:
            continue
        for candidate in candidates:
            if len(collected) >= int(cfg.diagonal_extension_max_total):
                break
            existing_signature = ("diagonal", tuple(int(v) for v in candidate["bbox"]))
            if existing_signature in existing_signatures:
                continue
            signature = (
                "diagonal",
                tuple(int(v) for v in candidate["bbox"]),
                int(segment_index),
                int(endpoint_index),
            )
            if signature in seen_candidate_signatures:
                continue
            item = {
                "segment_index": int(segment_index),
                "parent_segment_id": str(segment.get("segment_id", "")),
                "parent_source_segment_type": str(segment.get("source_segment_type", "")),
                "endpoint_index": int(endpoint_index),
                "candidate": candidate,
            }
            if any(diagonal_candidate_matches_existing(item, existing, wire_width, cfg) for existing in collected):
                continue
            seen_candidate_signatures.add(signature)
            collected.append(item)

    collected, dense_suppression_events = suppress_dense_endpoint_diagonal_duplicates(collected, wire_width, cfg)

    for item in collected:
        if len(accepted) >= int(cfg.diagonal_extension_max_total):
            break
        segment_index = int(item["segment_index"])
        endpoint_index = int(item["endpoint_index"])
        if segment_index >= len(segments):
            continue
        segment = segments[segment_index]
        candidate = item["candidate"]
        new_segment = make_diagonal_extension_segment(
            candidate,
            segment,
            segment_index,
            endpoint_index,
            len(segments) + 1,
        )
        far_endpoint_index = 1
        far_endpoint_touches_existing = segment_endpoint_touches_mask(
            binary,
            new_segment,
            far_endpoint_index,
            existing_union_mask,
        )
        new_index = len(segments)
        roles = ensure_endpoint_roles(new_segment)
        roles[0] = "connected_end"
        roles[far_endpoint_index] = "connected_end" if far_endpoint_touches_existing else "external_end"
        segments.append(new_segment)
        ensure_endpoint_roles(segment)[endpoint_index] = "external_end_extended_diagonal"
        paint_segment_mask(existing_union_mask, binary, new_segment)
        accepted.append(
            {
                "segment_id": str(new_segment["segment_id"]),
                "parent_segment_id": str(segment.get("segment_id", "")),
                "parent_endpoint_index": int(endpoint_index),
                "bbox": [int(v) for v in new_segment["bbox"]],
                "length_pixels": round(float(new_segment["span"]), 3),
                "width_pixels": round(float(new_segment["width"]), 3),
                "angle_degrees": round(float(new_segment["extension_angle_degrees"]), 3),
                "direction_vector": list(new_segment["extension_direction_vector"]),
                "diagonal_forward_extra_pixels": round(
                    float(new_segment.get("diagonal_forward_extra_pixels", 0.0)), 3
                ),
                "low_angle_continuation_applied": bool(new_segment.get("low_angle_continuation_applied", False)),
                "axis_coverage": round(float(new_segment["extension_axis_coverage"]), 3),
                "completion_pixels": int(new_segment.get("completion_pixels", 0)),
                "far_endpoint_dot_trimmed": bool(new_segment.get("far_endpoint_dot_trimmed", False)),
                "far_endpoint_dot_trimmed_pixels": int(new_segment.get("far_endpoint_dot_trimmed_pixels", 0)),
                "solid_rectangle_tail_trimmed": bool(new_segment.get("solid_rectangle_tail_trimmed", False)),
                "solid_rectangle_tail_trimmed_pixels": int(
                    new_segment.get("solid_rectangle_tail_trimmed_pixels", 0)
                ),
                "solid_rectangle_tail_trim_cut_projection": round(
                    float(new_segment.get("solid_rectangle_tail_trim_cut_projection", 0.0)),
                    3,
                ),
                "solid_rectangle_like_metrics": dict(new_segment.get("solid_rectangle_like_metrics", {})),
                "core_pixel_count": int(new_segment.get("diagonal_core_pixel_count", 0)),
                "core_width_padding_px": round(float(new_segment.get("diagonal_core_width_padding_px", 0.0)), 3),
                "track_cross_half": round(float(new_segment.get("track_cross_half", 0.0)), 3),
                "track_middle_cross_half": round(float(new_segment.get("track_middle_cross_half", 0.0)), 3),
                "track_radial_connected": bool(new_segment.get("track_radial_connected", False)),
                "projection_continuity": dict(new_segment.get("projection_continuity", {})),
                "near_endpoint_axis": dict(new_segment.get("near_endpoint_axis", {})),
                "oriented_rect_fit": dict(new_segment.get("oriented_rect_fit", {})),
                "direction_seed_bbox": list(new_segment.get("direction_seed_bbox", [])),
                "direction_seed_angle_degrees": round(float(new_segment.get("direction_seed_angle_degrees", 0.0)), 3),
                "direction_seed_source": str(new_segment.get("direction_seed_source", "")),
                "direction_seed_pixel_count": int(new_segment.get("direction_seed_pixel_count", 0)),
                "direction_seed_scan_radius_pixels": round(
                    float(new_segment.get("direction_seed_scan_radius_pixels", 0.0)), 3
                ),
                "direction_seed_scan_recent_width": round(
                    float(new_segment.get("direction_seed_scan_recent_width", 0.0)), 3
                ),
                "direction_seed_scan_robust_width": round(
                    float(new_segment.get("direction_seed_scan_robust_width", 0.0)), 3
                ),
                "direction_seed_scan_axis_coverage": round(
                    float(new_segment.get("direction_seed_scan_axis_coverage", 0.0)), 3
                ),
                "track_truncated_at_projection": new_segment.get("track_truncated_at_projection"),
                "track_truncated_pixels": int(new_segment.get("track_truncated_pixels", 0)),
                "angle_cluster_index": int(new_segment.get("angle_cluster_index", -1)),
                "angle_cluster_count": int(new_segment.get("angle_cluster_count", 0)),
                "angle_candidate_weight": round(float(new_segment.get("angle_candidate_weight", 0.0)), 3),
                "dominant_angle_cluster": bool(new_segment.get("dominant_angle_cluster", False)),
                "dominant_angle_cluster_ratio": round(
                    float(new_segment.get("dominant_angle_cluster_ratio", 0.0)),
                    3,
                ),
                "dominant_angle_cluster_margin_ratio": round(
                    float(new_segment.get("dominant_angle_cluster_margin_ratio", 0.0)),
                    3,
                ),
                "angle_cluster_summary": dict(new_segment.get("angle_cluster_summary", {})),
                "junction_backfill_pixels": int(new_segment.get("junction_backfill_pixels", 0)),
                "junction_backfill_radius_pixels": round(
                    float(new_segment.get("junction_backfill_radius_pixels", 0.0)),
                    3,
                ),
                "far_endpoint_touches_existing": bool(far_endpoint_touches_existing),
                "far_endpoint_role": str(ensure_endpoint_roles(new_segment)[far_endpoint_index]),
                "segment_index": int(new_index),
            }
        )
    classify_segment_endpoints(binary, segments)
    net_summary = assign_pixel_net_ids(binary, segments, page)
    return {
        "initial_diagonal_external_endpoint_queue_size": int(initial_queue_size),
        "num_diagonal_extensions": int(len(accepted)),
        "diagonal_extension_history": accepted,
        "diagonal_extension_external_end_trim_wire_widths": float(
            cfg.diagonal_extension_external_end_trim_wire_widths
        ),
        "diagonal_extension_claim_trim_events": trim_events,
        "diagonal_extension_dense_suppression_events": dense_suppression_events,
        "pixel_net_summary": net_summary,
    }


def extend_solid_wires_perpendicular_iterative(
    binary: np.ndarray,
    segments: list[dict[str, Any]],
    page: int,
    text_height: float,
    wire_width: float,
    cfg: JustWireConfig,
    dot_stop_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    for segment in segments:
        ensure_endpoint_roles(segment)
        set_segment_wire_pixels(binary, segment, cfg, dot_stop_mask)
        set_segment_body_width(binary, segment, wire_width, cfg)
    classify_segment_endpoints(binary, segments)
    claimed, trim_events = build_extension_claimed_mask(binary, segments, wire_width, cfg)
    existing_union_mask = build_segment_union_mask(binary, segments)
    queue: list[tuple[int, int, int]] = [
        (segment_index, endpoint_index, 1)
        for segment_index, segment in enumerate(segments)
        for endpoint_index, role in enumerate(segment.get("endpoint_roles", [])[:2])
        if role == "external_end"
    ]
    initial_queue_size = len(queue)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_signatures = {
        (str(segment.get("orientation", "")), tuple(int(v) for v in segment.get("bbox", [])))
        for segment in segments
    }
    while queue and len(accepted) < int(cfg.perpendicular_extension_max_total):
        segment_index, endpoint_index, depth = queue.pop(0)
        if depth > int(cfg.perpendicular_extension_max_depth):
            continue
        if segment_index >= len(segments):
            continue
        segment = segments[segment_index]
        if ensure_endpoint_roles(segment)[endpoint_index] != "external_end":
            continue
        direction_candidates: list[dict[str, Any]] = []
        per_direction_counts: dict[str, int] = {}
        for direction in endpoint_perpendicular_directions(segment):
            candidates = extract_perpendicular_extension_candidates(
                binary,
                claimed,
                segment,
                endpoint_index,
                direction,
                text_height,
                wire_width,
                cfg,
                dot_stop_mask,
            )
            per_direction_counts[f"{direction[0]},{direction[1]}"] = len(candidates)
            if candidates:
                direction_candidates.append(candidates[0])
        if not direction_candidates:
            continue
        if len(direction_candidates) > 1:
            rejected.append(
                {
                    "parent_segment_id": str(segment.get("segment_id", "")),
                    "parent_endpoint_index": int(endpoint_index),
                    "extension_depth": int(depth),
                    "reject_reason": "multiple_perpendicular_directions",
                    "candidate_counts": per_direction_counts,
                }
            )
            continue
        candidate = direction_candidates[0]
        signature = (str(candidate["orientation"]), tuple(int(v) for v in candidate["bbox"]))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        new_segment = make_perpendicular_extension_segment(
            candidate,
            segment,
            segment_index,
            endpoint_index,
            depth,
            len(segments) + 1,
        )
        set_segment_wire_pixels(binary, new_segment, cfg, dot_stop_mask)
        set_segment_body_width(binary, new_segment, wire_width, cfg)
        far_endpoint_index = opposite_endpoint_index_for_direction(new_segment, tuple(candidate["direction"]))
        far_endpoint_touches_existing = segment_endpoint_touches_mask(
            binary,
            new_segment,
            far_endpoint_index,
            existing_union_mask,
        )
        far_endpoint_touches_dot = (
            dot_stop_mask is not None
            and bool(np.any(dot_stop_mask))
            and segment_endpoint_touches_mask(binary, new_segment, far_endpoint_index, dot_stop_mask)
        )
        short_requires_connection = float(new_segment["span"]) < max(
            float(cfg.perpendicular_extension_short_requires_connection_px),
            float(wire_width) * float(cfg.perpendicular_extension_short_requires_connection_wire_widths),
        )
        unconnected_min_length = max(
            float(cfg.perpendicular_extension_unconnected_min_px),
            float(wire_width) * float(cfg.perpendicular_extension_unconnected_min_wire_widths),
        )
        too_short_unconnected = (
            short_requires_connection
            and not far_endpoint_touches_existing
            and not far_endpoint_touches_dot
            and float(new_segment["span"]) < unconnected_min_length
        )
        if too_short_unconnected:
            rejected.append(
                {
                    "parent_segment_id": str(segment.get("segment_id", "")),
                    "parent_endpoint_index": int(endpoint_index),
                    "extension_depth": int(depth),
                    "reject_reason": "short_extension_far_endpoint_not_connected",
                    "bbox": [int(v) for v in new_segment["bbox"]],
                    "orientation": str(new_segment["orientation"]),
                    "direction": list(new_segment["extension_direction"]),
                    "length_pixels": round(float(new_segment["span"]), 3),
                    "width_pixels": round(float(new_segment["width"]), 3),
                    "short_requires_connection": bool(short_requires_connection),
                    "unconnected_min_length_pixels": round(float(unconnected_min_length), 3),
                    "far_endpoint_touches_existing": bool(far_endpoint_touches_existing),
                    "far_endpoint_touches_dot": bool(far_endpoint_touches_dot),
                }
            )
            continue
        new_index = len(segments)
        roles = ensure_endpoint_roles(new_segment)
        roles[1 - far_endpoint_index] = "connected_end"
        if far_endpoint_touches_dot:
            roles[far_endpoint_index] = "junction_dot_end"
            new_segment["dot_blocked_endpoint_indices"] = sorted(
                {
                    int(value)
                    for value in new_segment.get("dot_blocked_endpoint_indices", [])
                    if isinstance(value, (int, np.integer, float))
                }
                | {int(far_endpoint_index)}
            )
        else:
            roles[far_endpoint_index] = "connected_end" if far_endpoint_touches_existing else "external_end"
        segments.append(new_segment)
        paint_segment_mask(existing_union_mask, binary, new_segment)
        ensure_endpoint_roles(segment)[endpoint_index] = "external_end_extended_perpendicular"
        paint_segment_mask(claimed, binary, segment)
        paint_segment_mask(claimed, binary, new_segment)
        if roles[far_endpoint_index] == "external_end":
            trim_length = int(math.ceil(
                max(float(wire_width), segment_effective_width(new_segment))
                * float(cfg.perpendicular_extension_external_end_trim_wire_widths)
            ))
            tx1, ty1, tx2, ty2 = external_endpoint_trim_bbox(new_segment, far_endpoint_index, trim_length, binary.shape)
            if tx2 > tx1 and ty2 > ty1:
                before = int(np.count_nonzero(claimed[ty1:ty2, tx1:tx2]))
                claimed[ty1:ty2, tx1:tx2] = False
                if before > 0:
                    trim_events.append(
                        {
                            "segment_index": int(new_index),
                            "segment_id": str(new_segment.get("segment_id", "")),
                            "endpoint_index": int(far_endpoint_index),
                            "trim_bbox": [int(tx1), int(ty1), int(tx2), int(ty2)],
                            "trim_length_pixels": int(trim_length),
                            "trimmed_claimed_pixels": int(before),
                        }
                    )
        if ensure_endpoint_roles(new_segment)[far_endpoint_index] == "external_end":
            queue.append((new_index, far_endpoint_index, depth + 1))
        accepted.append(
            {
                "segment_id": str(new_segment["segment_id"]),
                "parent_segment_id": str(segment.get("segment_id", "")),
                "parent_endpoint_index": int(endpoint_index),
                "extension_depth": int(depth),
                "bbox": [int(v) for v in new_segment["bbox"]],
                "orientation": str(new_segment["orientation"]),
                "direction": list(new_segment["extension_direction"]),
                "length_pixels": round(float(new_segment["span"]), 3),
                "width_pixels": round(float(new_segment["width"]), 3),
                "body_width_pixels": round(float(new_segment.get("body_width", new_segment["width"])), 3),
                "projected_fill_ratio": round(float(new_segment["projected_fill_ratio"]), 3),
                "unclaimed_pixel_ratio": round(float(new_segment.get("unclaimed_pixel_ratio", 0.0)), 3),
                "search_distance_pixels": int(new_segment.get("extension_search_distance_pixels", 0)),
                "touches_search_far_edge": bool(new_segment.get("extension_touches_search_far_edge", False)),
                "pre_width_expand_bbox": list(new_segment.get("pre_extension_width_expand_bbox", [])),
                "width_expanded_pixels": int(new_segment.get("extension_width_expanded_pixels", 0)),
                "short_requires_connection": bool(short_requires_connection),
                "far_endpoint_touches_existing": bool(far_endpoint_touches_existing),
                "far_endpoint_touches_dot": bool(far_endpoint_touches_dot),
                "far_endpoint_index": int(far_endpoint_index),
                "far_endpoint_role": str(ensure_endpoint_roles(new_segment)[far_endpoint_index]),
            }
        )
    classify_segment_endpoints(binary, segments)
    net_summary = assign_pixel_net_ids(binary, segments, page)
    return {
        "initial_external_endpoint_queue_size": int(initial_queue_size),
        "num_perpendicular_extensions": int(len(accepted)),
        "perpendicular_extension_history": accepted,
        "perpendicular_extension_rejections": rejected,
        "perpendicular_extension_external_end_trim_wire_widths": float(
            cfg.perpendicular_extension_external_end_trim_wire_widths
        ),
        "perpendicular_extension_claim_trim_events": trim_events,
        "pixel_net_summary": net_summary,
    }














def objects_from_segments(
    long_segments: list[dict[str, Any]],
    short_segments: list[dict[str, Any]],
    ultrashort_segments: list[dict[str, Any]],
    short_group_members: dict[str, dict[str, Any]],
    page: int,
    wire_width: float,
) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    for index, segment in enumerate(long_segments, start=1):
        bbox = [int(v) for v in segment["bbox"]]
        segment_id = str(segment["segment_id"])
        group_info = short_group_members.get(segment_id)
        object_type = "dash_member" if group_info else "solid_wire"
        objects.append(
            DetectedObject(
                f"p{page:03d}_{'dash_member' if group_info else 'solid_wire'}_{index:04d}",
                object_type,
                0.84 if group_info else 0.82,
                bbox,
                {"kind": "rectangle", "rect": bbox, "centerline_points": segment["points"]},
                {
                    "segment_id": segment_id,
                    "source_segment_type": str(segment.get("source_segment_type", "long_axis_segment")),
                    "orientation": segment["orientation"],
                    "length_pixels": round(float(segment["span"]), 3),
                    "median_width": round(float(segment["width"]), 3),
                    "estimated_wire_width": round(float(wire_width), 3),
                    "projected_fill_ratio": round(float(segment.get("projected_fill_ratio", 0.0)), 3),
                    "source": str(segment.get("source", "projected_axis_wire_segment")),
                    "horizontal_run_recovery": bool(segment.get("horizontal_run_recovery", False)),
                    "horizontal_run_recovery_raw_axis_bbox": list(segment.get("horizontal_run_recovery_raw_axis_bbox", [])),
                    "axis_gap_bridged_global": bool(segment.get("axis_gap_bridged_global", False)),
                    "source_component_bbox": list(segment.get("source_component_bbox", [])),
                    "source_component_area": int(segment.get("source_component_area", 0)),
                    "net_id": str(segment.get("net_id", "")),
                    "wire_net_id": str(segment.get("wire_net_id", "")),
                    "endpoint_roles": list(segment.get("endpoint_roles", [])),
                    "endpoints": list(segment.get("endpoints", [])),
                    **extension_output_attributes(segment),
                    **merged_segment_attributes(segment),
                    **wire_pixel_attributes(segment),
                    **edge_expansion_attributes(segment),
                    **(group_info or {}),
                },
                source_phase=STAGE_NAME,
            )
        )
    for index, segment in enumerate(short_segments, start=1):
        bbox = [int(v) for v in segment["bbox"]]
        segment_id = str(segment["segment_id"])
        group_info = short_group_members.get(segment_id)
        object_type = "dash_member" if group_info else "solid_wire"
        objects.append(
            DetectedObject(
                f"p{page:03d}_{'dash_member' if group_info else 'solid_wire'}_{len(objects) + 1:04d}",
                object_type,
                0.84 if group_info else 0.78,
                bbox,
                {"kind": "rectangle", "rect": bbox, "centerline_points": segment["points"]},
                {
                    "segment_id": segment_id,
                    "source_segment_type": "short_axis_segment",
                    "orientation": segment["orientation"],
                    "length_pixels": round(float(segment["span"]), 3),
                    "median_width": round(float(segment["width"]), 3),
                    "estimated_wire_width": round(float(wire_width), 3),
                    "projected_fill_ratio": round(float(segment.get("projected_fill_ratio", 0.0)), 3),
                    "source": "projected_axis_wire_segment",
                    "source_component_bbox": list(segment.get("source_component_bbox", [])),
                    "source_component_area": int(segment.get("source_component_area", 0)),
                    **merged_segment_attributes(segment),
                    **wire_pixel_attributes(segment),
                    **edge_expansion_attributes(segment),
                    **(group_info or {}),
                },
                source_phase=STAGE_NAME,
            )
        )
    for index, segment in enumerate(ultrashort_segments, start=1):
        segment_id = str(segment["segment_id"])
        group_info = short_group_members.get(segment_id)
        if not group_info:
            continue
        bbox = [int(v) for v in segment["bbox"]]
        objects.append(
            DetectedObject(
                f"p{page:03d}_dash_member_{len(objects) + 1:04d}",
                "dash_member",
                0.80,
                bbox,
                {"kind": "rectangle", "rect": bbox, "centerline_points": segment["points"]},
                {
                    "segment_id": segment_id,
                    "source_segment_type": "ultrashort_axis_segment",
                    "orientation": segment["orientation"],
                    "length_pixels": round(float(segment["span"]), 3),
                    "median_width": round(float(segment["width"]), 3),
                    "estimated_wire_width": round(float(wire_width), 3),
                    "projected_fill_ratio": round(float(segment.get("projected_fill_ratio", 0.0)), 3),
                    "source": "projected_axis_wire_segment",
                    "source_component_bbox": list(segment.get("source_component_bbox", [])),
                    "source_component_area": int(segment.get("source_component_area", 0)),
                    **merged_segment_attributes(segment),
                    **wire_pixel_attributes(segment),
                    **edge_expansion_attributes(segment),
                    **group_info,
                },
                source_phase=STAGE_NAME,
            )
        )
    return objects


def objects_from_junction_dots(junction_dots: list[dict[str, Any]], page: int) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    for index, dot in enumerate(junction_dots, start=1):
        if bool(dot.get("line_width_rejected", False)):
            continue
        bbox = [int(v) for v in dot.get("bbox", [])]
        if len(bbox) != 4:
            continue
        objects.append(
            DetectedObject(
                f"p{page:03d}_junction_dot_{index:04d}",
                "junction_dot",
                0.80,
                bbox,
                {
                    "kind": "circle_like_component",
                    "bbox": bbox,
                    "centroid": list(dot.get("centroid", [])),
                    "radius_pixels": float(dot.get("radius_pixels", 0.0)),
                },
                {
                    "dot_id": str(dot.get("dot_id", f"junction_dot_{index:04d}")),
                    "pixel_bbox": bbox,
                    "pixel_runs": list(dot.get("pixel_runs", [])),
                    "pixel_count": int(dot.get("pixel_count", 0)),
                    "fill": float(dot.get("fill", 0.0)),
                    "aspect": float(dot.get("aspect", 0.0)),
                    "core_area": int(dot.get("core_area", 0)),
                    "core_radius_threshold_pixels": float(dot.get("core_radius_threshold_pixels", 0.0)),
                    "detection_source": str(dot.get("detection_source", "")),
                    "fanout_score": float(dot.get("fanout_score", 0.0)),
                    "prominence_metrics": dict(dot.get("prominence_metrics", {})),
                    "non_axis_extension_present": bool(dot.get("non_axis_extension_present", False)),
                    "non_axis_line_width_pixels": float(dot.get("non_axis_line_width_pixels", 0.0)),
                    "non_axis_min_diameter_pixels": float(dot.get("non_axis_min_diameter_pixels", 0.0)),
                    "hard_boundary": bool(dot.get("hard_boundary", False)),
                },
                source_phase=STAGE_NAME,
            )
        )
    return objects


def clipped_bbox(bbox: list[int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (
        max(0, min(shape[1], x1)),
        max(0, min(shape[0], y1)),
        max(0, min(shape[1], x2)),
        max(0, min(shape[0], y2)),
    )


def parse_debug_bbox(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--debug-bbox must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--debug-bbox values must be integers") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--debug-bbox requires x2>x1 and y2>y1")
    return (x1, y1, x2, y2)


def overlay(rgb: np.ndarray, objects: list[DetectedObject]) -> np.ndarray:
    out = rgb.copy()
    for obj in objects:
        x1, y1, x2, y2 = clipped_bbox(obj.bbox, rgb.shape[:2])
        if x2 <= x1 or y2 <= y1:
            continue
        color = VISUAL_COLORS.get(obj.type)
        if color is not None:
            runs = obj.attributes.get("pixel_runs", [])
            run_bbox = obj.attributes.get("pixel_bbox", obj.bbox)
            if not runs:
                runs = obj.attributes.get("wire_pixel_runs", [])
                run_bbox = obj.attributes.get("wire_pixel_bbox", obj.bbox)
            if not runs:
                runs = obj.attributes.get("edge_expansion_pixel_runs", [])
                run_bbox = obj.attributes.get("edge_expansion_bbox", obj.bbox)
            if runs:
                ex1, ey1, ex2, ey2 = clipped_bbox(run_bbox, rgb.shape[:2])
                if ex2 <= ex1 or ey2 <= ey1:
                    continue
                run_mask = relative_runs_to_mask(runs, (ey2 - ey1, ex2 - ex1))
                out[ey1:ey2, ex1:ex2][run_mask] = color
            else:
                out[y1:y2, x1:x2] = color
    return out


def write_visual_legend(path: Path) -> None:
    lines = [f"{STAGE_NAME} visual legend", ""]
    for key, description, rgb in VISUAL_LEGEND:
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
        lines.append(f"{key}: {hex_color} RGB{rgb} - {description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_page(
    rgb: np.ndarray,
    page: int,
    wire_width: float,
    cfg: JustWireConfig,
    debug_short_groups: bool = False,
    debug_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[list[DetectedObject], dict[str, Any]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    binary = foreground(rgb)
    timings["binarize"] = time.perf_counter() - t0

    timings["foreground_components"] = 0.0

    t0 = time.perf_counter()
    text_height = estimate_text_height(binary)
    timings["text_height"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    junction_dots = detect_junction_dots(binary, wire_width, cfg)
    hard_boundary_dots = collect_junction_dot_boundaries(
        junction_dots,
        hard_only=True,
    )
    timings["junction_dot_detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    long_min_len = int(round(max(60.0, text_height * 3.0, wire_width * 10.0)))
    local_width_anomaly_min_len = int(round(max(
        float(cfg.local_width_anomaly_min_px),
        float(text_height) * float(cfg.local_width_anomaly_min_text_heights),
        float(wire_width) * float(cfg.local_width_anomaly_min_wire_widths),
    )))
    short_min_len = int(np.floor(max(text_height * cfg.short_min_text_height_ratio, wire_width * cfg.short_min_wire_widths)))
    short_max_len = int(round(max(short_min_len, min(text_height * cfg.short_max_text_height_ratio, long_min_len - 1))))
    long_segments = (
        extract_axis_rectangles(binary, "horizontal", long_min_len, None, wire_width, cfg, local_width_anomaly_min_len)
        + extract_axis_rectangles(binary, "vertical", long_min_len, None, wire_width, cfg, local_width_anomaly_min_len)
    )
    long_segments, junction_dot_axis_split_events = split_axis_segments_at_junction_dots(
        binary,
        long_segments,
        junction_dots,
        wire_width,
        long_min_len,
        cfg,
    )
    long_segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0], s["orientation"]))
    short_segments: list[dict[str, Any]] = []
    assign_segment_ids(long_segments, "long")
    assign_segment_ids(short_segments, "short")
    short_group_members: dict[str, dict[str, Any]] = {}
    short_line_groups: list[dict[str, Any]] = []
    ultrashort_segments: list[dict[str, Any]] = []
    short_group_debug_events: list[dict[str, Any]] = []
    edge_expansion_history = apply_axis_line_edge_expansion(
        rgb,
        binary,
        long_segments + short_segments + ultrashort_segments,
        gray_threshold=175,
        normal_padding=2,
    )
    pre_merge_long_segment_count = len(long_segments)
    long_segments, parallel_merge_history = merge_parallel_adjacent_segments(
        binary,
        long_segments,
        wire_width,
        cfg,
    )
    assign_segment_ids(long_segments, "long")
    dot_stop_mask, dot_stop_dots = build_line_width_filtered_dot_stop_mask(
        binary.shape,
        junction_dots,
        binary,
        long_segments,
        wire_width,
        cfg,
    )
    perpendicular_extension_summary = extend_solid_wires_perpendicular_iterative(
        binary,
        long_segments,
        page,
        text_height,
        wire_width,
        cfg,
        dot_stop_mask,
    )
    diagonal_extension_summary = extend_solid_wires_diagonal_once(
        binary,
        long_segments,
        page,
        text_height,
        wire_width,
        cfg,
        dot_stop_mask,
    )
    external_endpoint_orthogonal_trim_events = trim_external_endpoint_orthogonal_intrusions(
        binary,
        long_segments,
        wire_width,
        cfg,
    )
    connected_endpoint_length_trim_events = trim_connected_endpoint_short_orthogonal_overlaps(
        binary,
        long_segments,
        text_height,
        wire_width,
        cfg,
    )
    junction_dot_trim_events = trim_solid_wires_at_junction_dots(
        binary,
        long_segments,
        junction_dots,
        wire_width,
        cfg,
    )
    classify_segment_endpoints(binary, long_segments)
    post_trim_pixel_net_summary = assign_pixel_net_ids(binary, long_segments, page)
    timings["projected_rectangles"] = time.perf_counter() - t0

    objects = objects_from_segments(
        long_segments,
        short_segments,
        ultrashort_segments,
        short_group_members,
        page,
        wire_width,
    )
    objects.extend(objects_from_junction_dots(junction_dots, page))
    type_counts = Counter(obj.type for obj in objects)
    diagnostics = {
        "estimated_wire_width": float(wire_width),
        "estimated_text_height": float(text_height),
        "long_min_length_pixels": int(long_min_len),
        "short_min_length_pixels": int(short_min_len),
        "short_max_length_pixels": int(short_max_len),
        "ultrashort_search_mode": "short_line_group_endpoints_only",
        "num_solid_wires": int(type_counts.get("solid_wire", 0)),
        "num_dash_members": int(type_counts.get("dash_member", 0)),
        "num_dash_groups": int(len(short_line_groups)),
        "dash_groups": short_line_groups,
        "initial_line_edge_expansion_history": edge_expansion_history,
        "pre_parallel_merge_long_segment_count": int(pre_merge_long_segment_count),
        "post_parallel_merge_long_segment_count": int(len(long_segments)),
        "parallel_merge_count": int(len(parallel_merge_history)),
        "parallel_merge_history": parallel_merge_history,
        "num_perpendicular_extensions": int(perpendicular_extension_summary["num_perpendicular_extensions"]),
        "perpendicular_extension_history": perpendicular_extension_summary["perpendicular_extension_history"],
        "perpendicular_extension_rejections": perpendicular_extension_summary["perpendicular_extension_rejections"],
        "perpendicular_extension_external_end_trim_wire_widths": float(
            perpendicular_extension_summary["perpendicular_extension_external_end_trim_wire_widths"]
        ),
        "perpendicular_extension_claim_trim_events": perpendicular_extension_summary[
            "perpendicular_extension_claim_trim_events"
        ],
        "initial_external_endpoint_queue_size": int(
            perpendicular_extension_summary["initial_external_endpoint_queue_size"]
        ),
        "initial_diagonal_external_endpoint_queue_size": int(
            diagonal_extension_summary["initial_diagonal_external_endpoint_queue_size"]
        ),
        "num_diagonal_extensions": int(diagonal_extension_summary["num_diagonal_extensions"]),
        "diagonal_extension_history": diagonal_extension_summary["diagonal_extension_history"],
        "diagonal_extension_external_end_trim_wire_widths": float(
            diagonal_extension_summary["diagonal_extension_external_end_trim_wire_widths"]
        ),
        "diagonal_extension_claim_trim_events": diagonal_extension_summary[
            "diagonal_extension_claim_trim_events"
        ],
        "diagonal_extension_dense_suppression_events": diagonal_extension_summary[
            "diagonal_extension_dense_suppression_events"
        ],
        "external_endpoint_orthogonal_trim_count": int(len(external_endpoint_orthogonal_trim_events)),
        "external_endpoint_orthogonal_trim_events": external_endpoint_orthogonal_trim_events,
        "connected_endpoint_length_trim_count": int(len(connected_endpoint_length_trim_events)),
        "connected_endpoint_length_trim_events": connected_endpoint_length_trim_events,
        "num_junction_dots": int(len(junction_dots)),
        "num_junction_dot_hard_boundaries": int(len(hard_boundary_dots)),
        "num_junction_dot_stop_boundaries": int(len(dot_stop_dots)),
        "junction_dots": [
            {
                key: value
                for key, value in dot.items()
                if key != "pixel_runs" and not str(key).startswith("_")
            }
            for dot in junction_dots
        ],
        "junction_dot_axis_split_count": int(len(junction_dot_axis_split_events)),
        "junction_dot_axis_split_events": junction_dot_axis_split_events,
        "junction_dot_trim_count": int(len(junction_dot_trim_events)),
        "junction_dot_trim_events": junction_dot_trim_events,
        "post_trim_pixel_net_summary": post_trim_pixel_net_summary,
        **diagonal_extension_summary["pixel_net_summary"],
        "short_group_debug_enabled": bool(debug_short_groups),
        "short_group_debug_bbox": list(debug_bbox) if debug_bbox is not None else None,
        "short_group_debug_events": short_group_debug_events if debug_short_groups else [],
        "timings_seconds": {key: round(value, 4) for key, value in timings.items()},
        "detector_backend": "projected_horizontal_vertical_rectangles_with_short_line_groups",
    }
    return objects, diagnostics




def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="bmw-328i-1997.pdf")
    parser.add_argument("--pages")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--debug-short-groups", action="store_true")
    parser.add_argument("--debug-bbox")
    args = parser.parse_args()
    debug_bbox = parse_debug_bbox(args.debug_bbox)

    cfg = JustWireConfig(dpi=args.dpi)
    pdf = resolve_pdf(args.pdf)
    clean_source_dir = resolve_phase1_clean_dir(pdf)
    available_pages = available_clean_pages(clean_source_dir)
    pages = parse_pages(args.pages, available_pages)
    root, image_dir, json_dir = make_output_dirs(pdf.stem)
    write_visual_legend(root / "legend")

    logging.info("PDF path: %s", pdf)
    logging.info("Phase 2 source images: %s", clean_source_dir)
    logging.info("Pages: %s", pages)
    logging.info("Cleared output: %s", root)

    review_paths: list[Path] = []
    totals: Counter[str] = Counter()
    for page_number in pages:
        page_t0 = time.perf_counter()
        rgb = load_clean_image(clean_source_dir, page_number)
        page_info = PageInfo(pdf.name, page_number, args.dpi, int(rgb.shape[1]), int(rgb.shape[0]))
        binary = foreground(rgb)
        wire_width = estimate_wire_width(binary)
        objects, diagnostics = analyze_page(
            rgb,
            page_number,
            wire_width,
            cfg,
            args.debug_short_groups,
            debug_bbox,
        )
        totals.update(obj.type for obj in objects)

        review = overlay(rgb, objects)
        review_path = image_dir / f"page_{page_number:03d}.png"
        save_rgb(review_path, review, args.dpi)
        review_paths.append(review_path)

        diagnostics["total_page_seconds"] = round(time.perf_counter() - page_t0, 4)
        write_json(
            json_dir / f"page_{page_number:03d}.json",
            page_json(page_info, objects, diagnostics),
        )
        logging.info(
            "Page %03d: %d objects in %.2fs",
            page_number,
            len(objects),
            diagnostics["total_page_seconds"],
        )

    images_to_pdf(review_paths, root / "review.pdf", args.dpi)
    logging.info("Detected: %s", dict(totals))
    logging.info("Output: %s", root)


if __name__ == "__main__":
    main()
'''

_DASH_WIRE_SOURCE = r'''#!/usr/bin/env python3
"""Detect dashed wire members and dashed-wire groups from Phase 1 clean images.

The input contract matches 02_solid_wire.py: pages are loaded from
outputs/01_circuit/<pdf>/clean_image/page_NNN_clean.png.  The clean image is
already a black-on-white product of Phase 1, so this stage only converts black
pixels into a boolean mask; it does not rerender or rebinarize the PDF.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import fitz
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs"
STAGE_NAME = Path(__file__).stem
PHASE1_STAGE = "01_circuit"


@dataclass
class PageInfo:
    pdf: str
    page: int
    dpi: int
    width: int
    height: int

    @property
    def mapping(self) -> dict[str, Any]:
        return {
            "image_origin": "top-left",
            "pixel_units": "pixels",
            "source": f"outputs/{PHASE1_STAGE}/<pdf>/clean_image",
        }


@dataclass
class DetectedObject:
    id: str
    type: str
    confidence: float
    bbox: list[int]
    geometry: dict[str, Any]
    attributes: dict[str, Any] = field(default_factory=dict)
    source_phase: str = STAGE_NAME

    def json(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [], "")}


@dataclass(frozen=True)
class DashWireConfig:
    dpi: int = 300
    black_threshold: int = 128
    min_wire_width_px: int = 1
    max_wire_widths: float = 4.5
    min_projected_fill_ratio: float = 0.35
    min_width_coord_coverage: float = 0.80
    projected_nms_overlap: float = 0.72

    strict_seed_min_text_height_ratio: float = 1.15
    strict_seed_min_wire_widths: float = 3.5
    strict_seed_max_text_height_ratio: float = 4.6
    strict_seed_min_aspect_ratio: float = 3.8
    strict_seed_max_width_ratio: float = 2.0

    extension_min_text_height_ratio: float = 0.75
    extension_min_wire_widths: float = 2.2
    extension_max_text_height_ratio: float = 7.0
    extension_min_aspect_ratio: float = 2.1

    group_centerline_wire_widths: float = 1.5
    group_gap_text_height_ratio: float = 1.35
    group_length_min_ratio: float = 0.45
    group_length_max_ratio: float = 3.0
    group_extension_width_min_ratio: float = 0.55
    group_extension_width_max_ratio: float = 1.75
    group_extension_dot_like_min_cross_ratio: float = 1.55
    group_extension_dot_like_min_fill_ratio: float = 0.45
    group_extension_dot_like_max_fill_ratio: float = 0.96
    group_extension_t_junction_min_run_wire_widths: float = 1.2
    group_extension_t_junction_probe_wire_widths: float = 4.0
    group_extension_letter_like_max_source_fraction: float = 0.55
    group_extension_letter_like_min_source_cross_ratio: float = 3.5
    group_extension_letter_like_min_source_area_ratio: float = 1.8
    group_extension_letter_like_huge_source_area_ratio: float = 15.0
    group_extension_clean_l_leftover_max_wire_widths: float = 2.0
    group_extension_source_runs_max_area: int = 6000
    group_extension_huge_local_axis_pad_ratio: float = 0.35
    group_extension_huge_local_axis_pad_wire_widths: float = 4.0
    group_extension_huge_local_cross_pad_wire_widths: float = 7.0
    group_extension_huge_local_max_residual_components: int = 3
    group_extension_huge_local_max_residual_area_ratio: float = 3.0
    group_extension_huge_local_residual_min_fill_ratio: float = 0.58
    group_extension_huge_local_residual_max_cross_wire_widths: float = 3.0
    group_extension_huge_local_endpoint_t_zone_wire_widths: float = 2.0
    group_extension_limited_side_min_existing_members: int = 2
    group_extension_limited_side_band_wire_widths: float = 1.5
    group_extension_limited_side_min_source_axis_ratio: float = 2.0
    group_extension_limited_side_min_source_area_ratio: float = 4.0
    group_extension_limited_side_max_long_contact_ratio: float = 0.50
    group_extension_limited_side_max_opposite_contact_ratio: float = 0.08
    group_extension_limited_side_max_total_contact_ratio: float = 0.60
    group_extension_limited_side_max_contact_components: int = 4
    group_extension_bracket_endpoint_zone_wire_widths: float = 2.0
    group_extension_bracket_side_band_wire_widths: float = 2.0
    group_extension_bracket_min_endpoint_pixels: int = 2
    group_min_regular_members: int = 2

    terminal_min_wire_widths: float = 1.2
    terminal_max_length_ratio: float = 1.2
    terminal_min_length_ratio: float = 0.05
    terminal_min_aspect_ratio: float = 1.8
    terminal_search_gap_text_height_ratio: float = 1.0
    terminal_source_max_axis_ratio: float = 1.8
    terminal_source_max_area_ratio: float = 4.0
    candidate_source_huge_area_ratio: float = 20.0
    candidate_source_huge_axis_ratio: float = 8.0
    candidate_source_complex_cross_ratio: float = 4.0
    candidate_source_complex_area_ratio: float = 4.0
    group_gap_max_centerline_black_pixels: int = 0
    group_gap_block_min_fill_ratio: float = 0.65
    group_gap_block_min_run_ratio: float = 0.65
    candidate_endpoint_attachment_min_text_height_ratio: float = 1.0
    candidate_endpoint_attachment_zone_text_height_ratio: float = 0.45
    candidate_endpoint_attachment_side_probe_wire_widths: float = 3.0
    candidate_endpoint_attachment_max_mid_pixels: int = 6
    candidate_endpoint_attachment_max_mid_wire_widths: float = 2.0
    terminal_corner_cap_min_perpendicular_length_ratio: float = 0.65
    terminal_corner_cap_max_axis_extra_wire_widths: float = 2.5
    terminal_corner_cap_max_area_ratio: float = 8.0
    terminal_corner_cap_residual_min_fill_ratio: float = 0.68
    terminal_corner_cap_residual_max_components: int = 1
    terminal_corner_cap_orthogonal_min_length_ratio: float = 0.35
    terminal_corner_cap_orthogonal_min_wire_widths: float = 4.0
    terminal_corner_cap_orthogonal_max_length_ratio: float = 2.5
    terminal_corner_cap_residual_max_spur_pixels: int = 6
    terminal_corner_cap_residual_max_spur_wire_widths: float = 2.0
    group_min_representative_text_height_ratio: float = 1.2
    group_min_representative_wire_widths: float = 7.5
    group_min_representative_epsilon_px: float = 0.25
    group_min_strict_seed_members: int = 1
    group_min_independent_regular_members: int = 2
    group_clean_member_min_ratio: float = 0.80
    group_independent_source_component_min_fraction: float = 0.90
    group_width_max_ratio: float = 1.75
    group_text_like_width_max_ratio: float = 2.5
    group_text_like_length_max_ratio: float = 2.35
    group_text_like_gap_max_ratio: float = 4.0
    group_text_like_min_complex_member_ratio: float = 0.70
    group_text_like_max_independent_ratio: float = 0.25
    group_text_like_max_source_fraction: float = 0.55
    group_rectangular_member_min_fill_ratio: float = 0.82
    group_rectangular_member_min_occupancy: float = 0.72
    group_rectangular_exception_max_attachment_text_height_ratio: float = 4.0
    single_seed_terminal_min_length_ratio: float = 0.45
    single_seed_terminal_min_wire_widths: float = 4.0

    dot_core_min_radius_wire_widths: float = 1.55
    dot_weak_core_min_radius_wire_widths: float = 1.15
    dot_weak_min_peak_radius_px: float = 4.0
    dot_clean_min_core_area: int = 3
    dot_weak_min_core_area: int = 1
    dot_min_diameter_wire_widths: float = 1.2
    dot_max_diameter_wire_widths: float = 6.0
    dot_max_diameter_px: int = 28
    dot_max_diameter_extra_px: float = 1.5
    dot_max_aspect_ratio: float = 1.60
    dot_min_fill_ratio: float = 0.35
    dot_max_fill_ratio: float = 0.96
    dot_neighbor_suppression_radius_widths: float = 4.0
    dot_four_way_min_core_area: int = 24
    dot_l_or_t_min_core_area: int = 24
    dot_rectangular_axis_ratio: float = 1.55
    dot_rectangular_projection_min: float = 0.82

    fanout_side_probe_wire_widths: float = 4.0
    fanout_side_min_pixels: int = 14
    fanout_side_min_axis_span_px: int = 8
    fanout_side_min_axis_span_wire_widths: float = 3.0
    fanout_side_inner_touch_px: int = 1

    external_orthogonal_probe_wire_widths: float = 5.0
    external_orthogonal_probe_min_px: int = 6
    external_orthogonal_max_run_wire_widths: float = 1.6
    external_endpoint_retreat_probe_wire_widths: float = 5.0
    external_endpoint_retreat_max_width_ratio: float = 1.75
    external_endpoint_retreat_consecutive_normal: int = 2
    external_endpoint_retreat_min_span_wire_widths: float = 1.5
    external_endpoint_other_group_touch_wire_widths: float = 2.0
    cross_group_single_overrun_endpoint_tolerance_wire_widths: float = 4.0
    cross_group_single_overrun_min_overrun_px: float = 15.0
    cross_group_single_overrun_width_min_ratio: float = 0.85
    cross_group_single_overrun_width_max_ratio: float = 1.25


JUNCTION_DOT_VISUAL_COLOR: tuple[int, int, int] = (0, 0, 255)

# Keep junction dots out of the default overlay while preserving their default color.
VISUAL_COLORS: dict[str, tuple[int, int, int]] = {
    "dash_member": (182, 215, 0),
}

VISUAL_LEGEND: list[tuple[str, str, tuple[int, int, int]]] = [
    ("dash_member", "recognized dashed-wire member pixels", VISUAL_COLORS["dash_member"]),
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def resolve_pdf(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, ROOT_DIR / path, INPUT_DIR / path]
    if path.suffix.lower() != ".pdf":
        candidates.append(INPUT_DIR / f"{value}.pdf")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"PDF not found: {value}. Looked directly and in {INPUT_DIR}")


def resolve_phase1_clean_dir(pdf_path: Path) -> Path:
    clean_dir = OUTPUT_DIR / PHASE1_STAGE / pdf_path.stem / "clean_image"
    if not clean_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Phase 1 clean images: {clean_dir}. "
            f"Run: python scripts/01_circuit.py --pdf {pdf_path.name}"
        )
    return clean_dir


def available_clean_pages(clean_dir: Path) -> list[int]:
    pages: list[int] = []
    for path in sorted(clean_dir.glob("page_*_clean.png")):
        try:
            pages.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not pages:
        raise FileNotFoundError(f"No page_*_clean.png files found under: {clean_dir}")
    return pages


def parse_pages(spec: str | None, available_pages: Sequence[int]) -> list[int]:
    available = set(int(page) for page in available_pages)
    if not spec:
        return sorted(available)
    requested: set[int] = set()
    try:
        for token in spec.split(","):
            token = token.strip()
            if "-" in token:
                start, end = (int(part) for part in token.split("-", 1))
            else:
                start = end = int(token)
            if start < 1 or end < start:
                raise ValueError
            requested.update(range(start, end + 1))
    except ValueError as exc:
        raise ValueError(f"Invalid --pages '{spec}'; use 3 or 1-3,7,10-12") from exc
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Requested page(s) {missing} are missing from Phase 1 clean images")
    return sorted(requested)


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    resolved_output = root.resolve()
    resolved_stage = (OUTPUT_DIR / STAGE_NAME).resolve()
    if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
    shutil.rmtree(root)


def make_output_dirs(stem: str, clear_existing: bool = True) -> tuple[Path, Path, Path]:
    root = OUTPUT_DIR / STAGE_NAME / stem
    if clear_existing:
        clear_output_root(root)
    image_dir = root / "image"
    json_dir = root / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return root, image_dir, json_dir


def load_clean_image(clean_dir: Path, page_number: int) -> np.ndarray:
    path = clean_dir / f"page_{page_number:03d}_clean.png"
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb(path: Path, img: np.ndarray, dpi: int = 300) -> None:
    Image.fromarray(img).save(path, dpi=(dpi, dpi))


def images_to_pdf(paths: Sequence[Path], dest: Path, dpi: int) -> None:
    doc = fitz.open()
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        page = doc.new_page(width=width * 72 / dpi, height=height * 72 / dpi)
        page.insert_image(page.rect, filename=str(path), keep_proportion=True)
    doc.save(dest, deflate=True)
    doc.close()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def page_json(info: PageInfo, objects: Sequence[DetectedObject], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "pdf": info.pdf,
        "page": info.page,
        "dpi": info.dpi,
        "image_size": {"width": info.width, "height": info.height},
        "coordinate_mapping": info.mapping,
        "objects": [obj.json() for obj in objects],
    }
    if extra:
        data.update(extra)
    return data


def clean_black_mask(rgb: np.ndarray, threshold: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < int(threshold)


def mask_to_relative_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for y, row in enumerate(mask.astype(bool)):
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        start = int(xs[0])
        previous = int(xs[0])
        for value in xs[1:]:
            current = int(value)
            if current == previous + 1:
                previous = current
                continue
            runs.append([int(y), start, previous + 1])
            start = current
            previous = current
        runs.append([int(y), start, previous + 1])
    return runs


def relative_runs_to_mask(runs: Sequence[Sequence[int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for run in runs:
        if len(run) != 3:
            continue
        y, x1, x2 = [int(value) for value in run]
        if 0 <= y < shape[0]:
            mask[y, max(0, x1):min(shape[1], x2)] = True
    return mask


def clipped_bbox(bbox: Sequence[int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (
        max(0, min(shape[1], x1)),
        max(0, min(shape[0], y1)),
        max(0, min(shape[1], x2)),
        max(0, min(shape[0], y2)),
    )


def true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[0, values.astype(np.int8), 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def bbox_overlap_fraction(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)))
    return inter / area


def bbox_intersects(a: Sequence[int], b: Sequence[int] | None) -> bool:
    if b is None:
        return True
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def expanded_bboxes_touch(bbox_a: Sequence[int], bbox_b: Sequence[int], gap: float) -> bool:
    ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
    return not (ax2 + gap < bx1 or bx2 + gap < ax1 or ay2 + gap < by1 or by2 + gap < ay1)


def estimate_wire_width(mask: np.ndarray) -> float:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    skel = np.zeros_like(mask, np.uint8)
    work = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(64):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel |= cv2.subtract(work, opened)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    vals = 2 * dist[skel > 0]
    vals = vals[(vals >= 1) & (vals <= max(12, mask.shape[1] * 0.01))]
    return float(np.clip(np.percentile(vals, 35) if vals.size else 2.0, 1.0, 12.0))


def estimate_text_height(binary: np.ndarray) -> float:
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    heights: list[int] = []
    h, w = binary.shape
    max_h = max(12, int(h * 0.035))
    max_w = max(20, int(w * 0.08))
    for label in range(1, num_labels):
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if bh < 5 or bh > max_h or bw < 2 or bw > max_w:
            continue
        fill = area / max(1.0, float(bw * bh))
        if fill < 0.08 or fill > 0.85:
            continue
        heights.append(bh)
    if not heights:
        return 12.0
    return float(np.clip(np.percentile(np.array(heights, dtype=float), 60), 8.0, 28.0))


def detect_junction_dots(binary: np.ndarray, wire_width: float, cfg: DashWireConfig) -> list[dict[str, Any]]:
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    representative_width = max(float(wire_width), float(cfg.min_wire_width_px), 3.0)
    clean_core_radius = max(3.0, representative_width * float(cfg.dot_core_min_radius_wire_widths))
    weak_core_radius = max(3.0, representative_width * float(cfg.dot_weak_core_min_radius_wire_widths))
    min_diameter = max(5.0, representative_width * float(cfg.dot_min_diameter_wire_widths))
    max_diameter = max(
        min_diameter,
        min(float(cfg.dot_max_diameter_px), representative_width * float(cfg.dot_max_diameter_wire_widths)),
    )
    diameter_tolerance = max(
        0.0,
        float(cfg.dot_max_diameter_extra_px),
        representative_width * 0.5,
    )

    def structure_metrics(cx: float, cy: float, radius: float) -> dict[str, Any]:
        outer_radius = max(radius * 3.5, radius + 8.0)
        x1 = max(0, int(math.floor(cx - outer_radius)))
        y1 = max(0, int(math.floor(cy - outer_radius)))
        x2 = min(binary.shape[1], int(math.ceil(cx + outer_radius + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + outer_radius + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return {"annulus_pixels": 0, "angle_bin_count": 0, "fanout_score": 0.0}
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx.astype(float) - float(cx)
        dy = yy.astype(float) - float(cy)
        rr = np.sqrt(dx * dx + dy * dy)
        annulus = (rr >= radius + 1.0) & (rr <= outer_radius) & binary[y1:y2, x1:x2]
        annulus_pixels = int(np.count_nonzero(annulus))
        angles = np.degrees(np.arctan2(dy[annulus], dx[annulus])) if annulus_pixels > 0 else np.array([], dtype=float)
        angle_bins: list[dict[str, int]] = []
        for lower in range(-180, 180, 15):
            count = int(np.count_nonzero((angles >= float(lower)) & (angles < float(lower + 15)))) if angles.size else 0
            if count >= max(6, int(round(representative_width * 2.0))):
                angle_bins.append({"angle_start": int(lower), "pixel_count": int(count)})
        fanout_score = float(annulus_pixels) + float(len(angle_bins)) * 20.0
        return {
            "annulus_pixels": int(annulus_pixels),
            "angle_bin_count": int(len(angle_bins)),
            "angle_bins": angle_bins[:24],
            "fanout_score": round(float(fanout_score), 3),
        }

    def cardinal_extension_metrics(cx: float, cy: float, radius: float) -> dict[str, Any]:
        outer_radius = max(radius * 3.5, radius + 8.0)
        x1 = max(0, int(math.floor(cx - outer_radius)))
        y1 = max(0, int(math.floor(cy - outer_radius)))
        x2 = min(binary.shape[1], int(math.ceil(cx + outer_radius + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + outer_radius + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return {"all_four_cardinal_extensions": False, "cardinal_extension_count": 0}
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx.astype(float) - float(cx)
        dy = yy.astype(float) - float(cy)
        local = binary[y1:y2, x1:x2]
        start = float(radius) + 1.0
        band = max(2.5, representative_width * 0.85)
        min_pixels = max(18, int(round(representative_width * 12.0)))
        masks = {
            "left": (dx <= -start) & (dx >= -outer_radius) & (np.abs(dy) <= band),
            "right": (dx >= start) & (dx <= outer_radius) & (np.abs(dy) <= band),
            "up": (dy <= -start) & (dy >= -outer_radius) & (np.abs(dx) <= band),
            "down": (dy >= start) & (dy <= outer_radius) & (np.abs(dx) <= band),
        }
        counts = {key: int(np.count_nonzero(local & mask)) for key, mask in masks.items()}
        present = {key: int(value) >= int(min_pixels) for key, value in counts.items()}
        extension_count = int(sum(1 for value in present.values() if value))
        return {
            "cardinal_counts": counts,
            "cardinal_present": present,
            "cardinal_min_pixels": int(min_pixels),
            "cardinal_extension_count": int(extension_count),
            "all_four_cardinal_extensions": bool(extension_count == 4),
        }

    def weak_core_radial_balance_ok(local: np.ndarray, cx: float, cy: float, x1: int, y1: int) -> bool:
        yy, xx = np.nonzero(local)
        if yy.size <= 0:
            return False
        dx = xx.astype(float) + float(x1) - float(cx)
        dy = yy.astype(float) + float(y1) - float(cy)
        angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        sector_counts = np.bincount(np.floor(angles / 45.0).astype(int).clip(0, 7), minlength=8)
        min_sector_pixels = max(2, int(round(float(representative_width))))
        occupied = int(np.count_nonzero(sector_counts >= min_sector_pixels))
        if occupied < 5:
            return False
        opposite_pairs = [int(sector_counts[index] + sector_counts[index + 4]) for index in range(4)]
        strong_pairs = int(sum(value >= max(4, int(round(float(representative_width) * 2.0))) for value in opposite_pairs))
        return strong_pairs >= 2

    def candidate_from_core(
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        label: int,
        core_radius: float,
        min_core_area: int,
        source: str,
    ) -> dict[str, Any] | None:
        core_area = int(stats[label, cv2.CC_STAT_AREA])
        if core_area < int(min_core_area):
            return None
        label_mask = labels == label
        max_radius = float(np.max(dist[label_mask]))
        if max_radius < float(core_radius):
            return None
        if source == "weak_core_recovery" and max_radius < float(cfg.dot_weak_min_peak_radius_px):
            return None
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        radius = max(float(max_radius), float(min_diameter) / 2.0)
        pad = max(1.5, float(wire_width) * 0.5)
        x1 = max(0, int(math.floor(cx - radius - pad)))
        y1 = max(0, int(math.floor(cy - radius - pad)))
        x2 = min(binary.shape[1], int(math.ceil(cx + radius + pad + 1.0)))
        y2 = min(binary.shape[0], int(math.ceil(cy + radius + pad + 1.0)))
        if x2 <= x1 or y2 <= y1:
            return None
        width = float(x2 - x1)
        height = float(y2 - y1)
        if width < min_diameter or height < min_diameter:
            return None
        if width > max_diameter + diameter_tolerance or height > max_diameter + diameter_tolerance:
            return None
        aspect = max(width, height) / max(1.0, min(width, height))
        if aspect > float(cfg.dot_max_aspect_ratio):
            return None
        yy, xx = np.mgrid[y1:y2, x1:x2]
        circle = ((xx.astype(float) - cx) ** 2 + (yy.astype(float) - cy) ** 2) <= (radius + pad) ** 2
        local = binary[y1:y2, x1:x2] & circle
        pixel_count = int(np.count_nonzero(local))
        if pixel_count <= 0:
            return None
        fill = float(pixel_count) / max(1.0, float(np.count_nonzero(circle)))
        if fill < float(cfg.dot_min_fill_ratio) or fill > float(cfg.dot_max_fill_ratio):
            return None
        if source == "weak_core_recovery":
            weak_min_core_area = max(
                int(cfg.dot_weak_min_core_area),
                6,
                int(round(representative_width * representative_width * 1.5)),
            )
            if int(core_area) < int(weak_min_core_area):
                return None
            if not weak_core_radial_balance_ok(local, cx, cy, x1, y1):
                return None
        metrics = structure_metrics(cx, cy, radius)
        cardinal_metrics = cardinal_extension_metrics(cx, cy, radius)
        four_way_min_core_area = max(
            int(cfg.dot_four_way_min_core_area),
            int(round(representative_width * representative_width * 2.5)),
        )
        if bool(cardinal_metrics.get("all_four_cardinal_extensions", False)):
            if source != "clean_core" or int(core_area) < int(four_way_min_core_area):
                return None
        l_or_t_min_core_area = max(
            int(cfg.dot_l_or_t_min_core_area),
            int(round(representative_width * representative_width * 2.5)),
        )
        if (
            source == "weak_core_recovery"
            and int(cardinal_metrics.get("cardinal_extension_count", 0)) >= 2
            and int(core_area) < int(l_or_t_min_core_area)
        ):
            return None
        return {
            "dot_id": "",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "centroid": [round(float(cx), 3), round(float(cy), 3)],
            "radius_pixels": round(float(radius), 3),
            "core_radius_threshold_pixels": round(float(core_radius), 3),
            "core_area": int(core_area),
            "pixel_count": int(pixel_count),
            "fill": round(float(fill), 3),
            "aspect": round(float(aspect), 3),
            "detection_source": source,
            "structure_metrics": metrics,
            "cardinal_extension_metrics": cardinal_metrics,
            "fanout_score": float(metrics.get("fanout_score", 0.0)),
            "pixel_runs": mask_to_relative_runs(local),
        }

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for core_radius, min_core_area, source in [
        (clean_core_radius, int(cfg.dot_clean_min_core_area), "clean_core"),
        (weak_core_radius, int(cfg.dot_weak_min_core_area), "weak_core_recovery"),
    ]:
        core = dist >= float(core_radius)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
        for label in range(1, num_labels):
            candidate = candidate_from_core(labels, stats, centroids, label, core_radius, min_core_area, source)
            if candidate is None:
                continue
            key = (
                int(round(float(candidate["centroid"][0]))),
                int(round(float(candidate["centroid"][1]))),
                int(round(float(candidate["radius_pixels"]))),
                int(candidate["core_area"]),
                str(source),
            )
            if key in seen:
                continue
            seen.add(key)
            if source == "weak_core_recovery":
                candidate["fanout_score"] = float(candidate.get("fanout_score", 0.0)) + 10.0
            candidates.append(candidate)
    if not candidates:
        return []

    suppression_radius = max(
        float(min_diameter),
        representative_width * float(cfg.dot_neighbor_suppression_radius_widths),
    )
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(candidates):
        fx, fy = [float(v) for v in first["centroid"]]
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            sx, sy = [float(v) for v in second["centroid"]]
            distance = math.hypot(fx - sx, fy - sy)
            local_limit = max(
                suppression_radius,
                (float(first["radius_pixels"]) + float(second["radius_pixels"])) * 2.05,
            )
            if distance <= local_limit:
                union(first_index, second_index)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        clusters.setdefault(find(index), []).append(candidate)

    kept: list[dict[str, Any]] = []
    for members in clusters.values():
        best = min(
            members,
            key=lambda item: (
                float(item.get("fanout_score", 0.0)),
                0 if str(item.get("detection_source", "")) == "clean_core" else 1,
                -int(item.get("core_area", 0)),
                item["bbox"][1],
                item["bbox"][0],
            ),
        )
        if len(members) > 1:
            best["suppressed_nearby_dot_candidates"] = [
                {
                    "bbox": list(member.get("bbox", [])),
                    "centroid": list(member.get("centroid", [])),
                    "detection_source": str(member.get("detection_source", "")),
                    "fanout_score": round(float(member.get("fanout_score", 0.0)), 3),
                }
                for member in members
                if member is not best
            ]
        kept.append(best)

    dots = sorted(kept, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, dot in enumerate(dots, start=1):
        dot["dot_id"] = f"junction_dot_{index:04d}"
    return dots


def dot_mask_in_bbox(dot: dict[str, Any], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    local = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return local
    db = [int(v) for v in dot.get("bbox", [])]
    if len(db) != 4:
        return local
    dx1, dy1, dx2, dy2 = db
    ix1, iy1 = max(x1, dx1), max(y1, dy1)
    ix2, iy2 = min(x2, dx2), min(y2, dy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return local
    dot_local = relative_runs_to_mask(dot.get("pixel_runs", []), (dy2 - dy1, dx2 - dx1))
    local[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = dot_local[iy1 - dy1:iy2 - dy1, ix1 - dx1:ix2 - dx1]
    return local


def build_dot_mask(shape: tuple[int, int], dots: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for dot in dots:
        bbox = [int(v) for v in dot.get("bbox", [])]
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = clipped_bbox(bbox, shape)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] |= dot_mask_in_bbox(dot, (x1, y1, x2, y2))
    return mask


def refine_projected_width_runs(window: np.ndarray, min_cross: int, cfg: DashWireConfig) -> list[tuple[int, int, float, int]]:
    span = int(window.shape[1])
    if span <= 0:
        return []
    coverage = window.sum(axis=1) / max(1.0, float(span))
    valid = coverage >= cfg.min_width_coord_coverage
    if not bool(np.any(valid)):
        return []
    runs: list[tuple[int, int, float, int]] = []
    for start, end in true_runs(valid):
        width = int(end - start)
        if width < min_cross:
            continue
        refined_window = window[start:end, :]
        if not bool(np.all(np.any(refined_window, axis=0))):
            continue
        ink = int(refined_window.sum())
        fill = ink / max(1.0, float(width * span))
        runs.append((int(start), int(end), float(fill), ink))
    return runs


def make_projected_segment(
    orientation: str,
    x_offset: int,
    y_offset: int,
    axis_start_value: int,
    axis_end_value: int,
    cross_start: int,
    cross_end: int,
    fill: float,
    ink: int,
    source: str = "projected_axis_dash_segment",
) -> dict[str, Any]:
    if orientation == "horizontal":
        bbox = [
            int(x_offset + axis_start_value),
            int(y_offset + cross_start),
            int(x_offset + axis_end_value),
            int(y_offset + cross_end),
        ]
        center = (bbox[1] + bbox[3] - 1) / 2.0
        points = [[bbox[0], int(round(center))], [bbox[2] - 1, int(round(center))]]
    else:
        bbox = [
            int(x_offset + cross_start),
            int(y_offset + axis_start_value),
            int(x_offset + cross_end),
            int(y_offset + axis_end_value),
        ]
        center = (bbox[0] + bbox[2] - 1) / 2.0
        points = [[int(round(center)), bbox[1]], [int(round(center)), bbox[3] - 1]]
    return {
        "orientation": orientation,
        "points": points,
        "bbox": bbox,
        "span": float(axis_end_value - axis_start_value),
        "width": float(cross_end - cross_start),
        "area": int(ink),
        "centerline": float(center),
        "projected_fill_ratio": float(fill),
        "source": source,
    }


def suppress_overlapping_segments(candidates: list[dict[str, Any]], overlap_threshold: float) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item["span"]),
            -float(item["width"]),
            -float(item["projected_fill_ratio"]),
            item["bbox"][1],
            item["bbox"][0],
        ),
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if any(
            candidate["orientation"] == existing["orientation"]
            and bbox_overlap_fraction(candidate["bbox"], existing["bbox"]) >= overlap_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(key=lambda s: (s["bbox"][1], s["bbox"][0], s["orientation"]))
    return kept


def component_projected_rectangles(
    component: np.ndarray,
    orientation: str,
    x_offset: int,
    y_offset: int,
    min_len: int,
    max_len: int | None,
    min_cross: int,
    max_cross: float,
    cfg: DashWireConfig,
) -> list[dict[str, Any]]:
    scan_mask = component if orientation == "horizontal" else component.T
    if not bool(np.any(scan_mask)):
        return []
    max_width = min(int(max(min_cross, round(max_cross))), scan_mask.shape[0])
    integral = np.vstack(
        [
            np.zeros((1, scan_mask.shape[1]), dtype=np.int32),
            np.cumsum(scan_mask.astype(np.int32), axis=0),
        ]
    )
    candidates: list[dict[str, Any]] = []
    for width in range(min_cross, max_width + 1):
        for cross_start in range(0, scan_mask.shape[0] - width + 1):
            cross_end = cross_start + width
            column_counts = integral[cross_end] - integral[cross_start]
            occupied = column_counts > 0
            if not bool(np.any(occupied)):
                continue
            for axis_start_value, axis_end_value in true_runs(occupied):
                span = int(axis_end_value - axis_start_value)
                if span < min_len:
                    continue
                if max_len is not None and span > max_len:
                    continue
                ink = int(column_counts[axis_start_value:axis_end_value].sum())
                fill = ink / max(1.0, float(span * width))
                if fill < cfg.min_projected_fill_ratio:
                    continue
                window = scan_mask[cross_start:cross_end, axis_start_value:axis_end_value]
                for local_cross_start, local_cross_end, refined_fill, refined_ink in refine_projected_width_runs(
                    window,
                    min_cross,
                    cfg,
                ):
                    candidates.append(
                        make_projected_segment(
                            orientation,
                            x_offset,
                            y_offset,
                            axis_start_value,
                            axis_end_value,
                            cross_start + local_cross_start,
                            cross_start + local_cross_end,
                            refined_fill,
                            refined_ink,
                        )
                    )
    return suppress_overlapping_segments(candidates, cfg.projected_nms_overlap)


def attach_segment_pixels(segment: dict[str, Any], mask: np.ndarray) -> None:
    x1, y1, x2, y2 = clipped_bbox(segment["bbox"], mask.shape)
    if x2 <= x1 or y2 <= y1:
        segment["wire_pixel_bbox"] = [x1, y1, x2, y2]
        segment["wire_pixel_runs"] = []
        segment["wire_pixel_count"] = 0
        return
    local = mask[y1:y2, x1:x2].copy()
    segment["wire_pixel_bbox"] = [x1, y1, x2, y2]
    segment["wire_pixel_runs"] = mask_to_relative_runs(local)
    segment["wire_pixel_count"] = int(np.count_nonzero(local))


def extract_axis_rectangles(
    binary: np.ndarray,
    orientation: str,
    min_len: int,
    max_len: int | None,
    wire_width: float,
    cfg: DashWireConfig,
) -> list[dict[str, Any]]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    min_cross = int(max(1, cfg.min_wire_width_px))
    max_cross = max(float(cfg.min_wire_width_px), wire_width * cfg.max_wire_widths)
    segments: list[dict[str, Any]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if orientation == "horizontal":
            if bw < min_len or bh < min_cross:
                continue
        else:
            if bh < min_len or bw < min_cross:
                continue
        component = labels[y:y + bh, x:x + bw] == label
        component_segments = component_projected_rectangles(
            component,
            orientation,
            x,
            y,
            min_len,
            max_len,
            min_cross,
            max_cross,
            cfg,
        )
        component_bbox = [x, y, x + bw, y + bh]
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        component_runs = (
            mask_to_relative_runs(component)
            if component_area <= int(cfg.group_extension_source_runs_max_area)
            else []
        )
        for segment in component_segments:
            segment.update(
                {
                    "source_component_label": int(label),
                    "source_component_bbox": component_bbox,
                    "source_component_area": component_area,
                    "source_component_pixel_runs": component_runs,
                }
            )
            attach_segment_pixels(segment, binary)
        segments.extend(component_segments)
    segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0], s["orientation"]))
    return segments


def axis_start(segment: dict[str, Any]) -> float:
    return float(segment["bbox"][0] if segment["orientation"] == "horizontal" else segment["bbox"][1])


def axis_end(segment: dict[str, Any]) -> float:
    return float(segment["bbox"][2] if segment["orientation"] == "horizontal" else segment["bbox"][3])


def segment_debug_summary(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": str(segment.get("segment_id", "")),
        "candidate_class": str(segment.get("candidate_class", "")),
        "orientation": str(segment.get("orientation", "")),
        "bbox": [int(v) for v in segment.get("bbox", [])],
        "centerline": round(float(segment.get("centerline", 0.0)), 3),
        "span": round(float(segment.get("span", 0.0)), 3),
        "width": round(float(segment.get("width", 0.0)), 3),
        "projected_fill_ratio": round(float(segment.get("projected_fill_ratio", 0.0)), 3),
        "wire_pixel_count": int(segment.get("wire_pixel_count", 0)),
    }


def assign_segment_ids(segments: list[dict[str, Any]], prefix: str) -> None:
    for index, segment in enumerate(segments, start=1):
        segment["segment_id"] = f"{prefix}_{index:04d}"


def annotate_segment_source_component_from_labels(
    segment: dict[str, Any],
    labels: np.ndarray,
    stats: np.ndarray,
    source_name: str,
) -> bool:
    bbox = [int(v) for v in segment.get("wire_pixel_bbox", segment.get("bbox", []))]
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = clipped_bbox(bbox, labels.shape)
    if x2 <= x1 or y2 <= y1:
        return False
    segment_mask = relative_runs_to_mask(segment.get("wire_pixel_runs", []), (y2 - y1, x2 - x1))
    if not bool(np.any(segment_mask)):
        return False
    touched = labels[y1:y2, x1:x2][segment_mask]
    touched = touched[touched > 0]
    if touched.size == 0:
        return False
    label_values, counts = np.unique(touched, return_counts=True)
    label = int(label_values[int(np.argmax(counts))])
    if label <= 0 or label >= stats.shape[0]:
        return False
    if "raw_source_component_bbox" not in segment:
        segment["raw_source_component_bbox"] = list(segment.get("source_component_bbox", []))
        segment["raw_source_component_area"] = int(segment.get("source_component_area", 0))
    lx = int(stats[label, cv2.CC_STAT_LEFT])
    ly = int(stats[label, cv2.CC_STAT_TOP])
    lw = int(stats[label, cv2.CC_STAT_WIDTH])
    lh = int(stats[label, cv2.CC_STAT_HEIGHT])
    segment["source_component_bbox"] = [lx, ly, lx + lw, ly + lh]
    segment["source_component_area"] = int(stats[label, cv2.CC_STAT_AREA])
    segment["source_component_metric_space"] = source_name
    return True


def side_band_fanout_metrics(
    binary: np.ndarray,
    candidate: dict[str, Any],
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    if len(bbox) != 4:
        return {"fanout_like": False, "reason": "missing_bbox"}
    x1, y1, x2, y2 = clipped_bbox(bbox, binary.shape)
    if x2 <= x1 or y2 <= y1:
        return {"fanout_like": False, "reason": "empty_bbox"}
    orientation = str(candidate.get("orientation", ""))
    probe = int(max(2, round(float(wire_width) * float(cfg.fanout_side_probe_wire_widths))))
    min_pixels = int(max(int(cfg.fanout_side_min_pixels), round(float(wire_width) * 6.0)))
    min_axis_span = int(
        max(
            int(cfg.fanout_side_min_axis_span_px),
            round(float(wire_width) * float(cfg.fanout_side_min_axis_span_wire_widths)),
        )
    )
    touch = max(0, int(cfg.fanout_side_inner_touch_px))

    def analyze_side(side_name: str, side_bbox: tuple[int, int, int, int], touch_edge: str) -> dict[str, Any]:
        sx1, sy1, sx2, sy2 = side_bbox
        sx1, sy1, sx2, sy2 = clipped_bbox([sx1, sy1, sx2, sy2], binary.shape)
        if sx2 <= sx1 or sy2 <= sy1:
            return {
                "side": side_name,
                "bbox": [int(sx1), int(sy1), int(sx2), int(sy2)],
                "qualifying_component_count": 0,
                "qualifying_components": [],
            }
        local = binary[sy1:sy2, sx1:sx2].copy()
        if not bool(np.any(local)):
            return {
                "side": side_name,
                "bbox": [int(sx1), int(sy1), int(sx2), int(sy2)],
                "qualifying_component_count": 0,
                "qualifying_components": [],
            }
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
        qualifying: list[dict[str, Any]] = []
        for label in range(1, num_labels):
            lx = int(stats[label, cv2.CC_STAT_LEFT])
            ly = int(stats[label, cv2.CC_STAT_TOP])
            lw = int(stats[label, cv2.CC_STAT_WIDTH])
            lh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if touch_edge == "top":
                touches_inner = ly <= touch
            elif touch_edge == "bottom":
                touches_inner = ly + lh >= local.shape[0] - touch
            elif touch_edge == "left":
                touches_inner = lx <= touch
            else:
                touches_inner = lx + lw >= local.shape[1] - touch
            axis_span = lw if orientation == "horizontal" else lh
            if not touches_inner or area < min_pixels or axis_span < min_axis_span:
                continue
            qualifying.append(
                {
                    "bbox": [int(sx1 + lx), int(sy1 + ly), int(sx1 + lx + lw), int(sy1 + ly + lh)],
                    "pixel_count": int(area),
                    "axis_span_pixels": int(axis_span),
                    "touches_inner_edge": bool(touches_inner),
                }
            )
        qualifying.sort(key=lambda item: (-int(item["pixel_count"]), item["bbox"][1], item["bbox"][0]))
        return {
            "side": side_name,
            "bbox": [int(sx1), int(sy1), int(sx2), int(sy2)],
            "qualifying_component_count": int(len(qualifying)),
            "qualifying_components": qualifying[:8],
        }

    if orientation == "horizontal":
        sides = [
            analyze_side("above", (x1, y1 - probe, x2, y1), "bottom"),
            analyze_side("below", (x1, y2, x2, y2 + probe), "top"),
        ]
    elif orientation == "vertical":
        sides = [
            analyze_side("left", (x1 - probe, y1, x1, y2), "right"),
            analyze_side("right", (x2, y1, x2 + probe, y2), "left"),
        ]
    else:
        sides = []
    fanout_like = len(sides) == 2 and all(int(side["qualifying_component_count"]) > 0 for side in sides)
    return {
        "fanout_like": bool(fanout_like),
        "probe_pixels": int(probe),
        "min_component_pixels": int(min_pixels),
        "min_axis_span_pixels": int(min_axis_span),
        "sides": sides,
    }


def endpoint_only_attachment_metrics(
    binary: np.ndarray,
    candidate: dict[str, Any],
    text_height: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("wire_pixel_bbox", candidate.get("bbox", []))]
    if len(bbox) != 4:
        return {"endpoint_only_attachment": False, "reason": "missing_bbox"}
    orientation = str(candidate.get("orientation", ""))
    span = float(candidate.get("span", 0.0))
    min_span = float(text_height) * float(cfg.candidate_endpoint_attachment_min_text_height_ratio)
    if span < min_span:
        return {
            "endpoint_only_attachment": False,
            "reason": "candidate_span_below_endpoint_attachment_min",
            "candidate_span": round(float(span), 3),
            "min_span": round(float(min_span), 3),
        }
    probe = int(max(1, round(float(wire_width) * float(cfg.candidate_endpoint_attachment_side_probe_wire_widths))))
    endpoint_zone = int(max(2, round(float(text_height) * float(cfg.candidate_endpoint_attachment_zone_text_height_ratio))))
    max_mid_pixels = int(max(
        float(cfg.candidate_endpoint_attachment_max_mid_pixels),
        float(wire_width) * float(cfg.candidate_endpoint_attachment_max_mid_wire_widths),
    ))
    x1, y1, x2, y2 = bbox
    nx1, ny1, nx2, ny2 = clipped_bbox([x1 - probe, y1 - probe, x2 + probe, y2 + probe], binary.shape)
    if nx2 <= nx1 or ny2 <= ny1:
        return {"endpoint_only_attachment": False, "reason": "empty_neighborhood"}
    local = binary[ny1:ny2, nx1:nx2].copy()
    cand_mask = np.zeros_like(local, dtype=bool)
    ix1, iy1 = max(x1, nx1), max(y1, ny1)
    ix2, iy2 = min(x2, nx2), min(y2, ny2)
    if ix2 > ix1 and iy2 > iy1:
        local_candidate = relative_runs_to_mask(
            candidate.get("wire_pixel_runs", []),
            (max(0, y2 - y1), max(0, x2 - x1)),
        )
        cand_mask[iy1 - ny1:iy2 - ny1, ix1 - nx1:ix2 - nx1] = local_candidate[
            iy1 - y1:iy2 - y1,
            ix1 - x1:ix2 - x1,
        ]
    outside = local & ~cand_mask
    if orientation == "vertical":
        axis_coords = np.arange(ny1, ny2)[:, None]
        endpoint_zone_mask = (axis_coords < y1 + endpoint_zone) | (axis_coords >= y2 - endpoint_zone)
    elif orientation == "horizontal":
        axis_coords = np.arange(nx1, nx2)[None, :]
        endpoint_zone_mask = (axis_coords < x1 + endpoint_zone) | (axis_coords >= x2 - endpoint_zone)
    else:
        return {"endpoint_only_attachment": False, "reason": "unknown_orientation"}
    mid_attachment_pixels = int(np.count_nonzero(outside & ~endpoint_zone_mask))
    endpoint_attachment_pixels = int(np.count_nonzero(outside & endpoint_zone_mask))
    ok = mid_attachment_pixels <= max_mid_pixels and endpoint_attachment_pixels > 0
    return {
        "endpoint_only_attachment": bool(ok),
        "reason": "attachments_limited_to_candidate_endpoints" if ok else "midspan_attachment_pixels_exceed_limit",
        "candidate_span": round(float(span), 3),
        "min_span": round(float(min_span), 3),
        "probe_pixels": int(probe),
        "endpoint_zone_pixels": int(endpoint_zone),
        "mid_attachment_pixels": int(mid_attachment_pixels),
        "endpoint_attachment_pixels": int(endpoint_attachment_pixels),
        "mid_attachment_pixel_limit": int(max_mid_pixels),
    }


def group_member_attachment_metrics(
    binary: np.ndarray,
    candidate: dict[str, Any],
    text_height: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("wire_pixel_bbox", candidate.get("bbox", []))]
    if len(bbox) != 4:
        return {"has_attachment_metrics": False, "reason": "missing_bbox"}
    orientation = str(candidate.get("orientation", ""))
    probe = int(max(1, round(float(wire_width) * float(cfg.candidate_endpoint_attachment_side_probe_wire_widths))))
    endpoint_zone = int(max(2, round(float(text_height) * float(cfg.candidate_endpoint_attachment_zone_text_height_ratio))))
    max_mid_pixels = int(max(
        float(cfg.candidate_endpoint_attachment_max_mid_pixels),
        float(wire_width) * float(cfg.candidate_endpoint_attachment_max_mid_wire_widths),
    ))
    x1, y1, x2, y2 = bbox
    nx1, ny1, nx2, ny2 = clipped_bbox([x1 - probe, y1 - probe, x2 + probe, y2 + probe], binary.shape)
    if nx2 <= nx1 or ny2 <= ny1:
        return {"has_attachment_metrics": False, "reason": "empty_neighborhood"}
    local = binary[ny1:ny2, nx1:nx2].copy()
    cand_mask = np.zeros_like(local, dtype=bool)
    ix1, iy1 = max(x1, nx1), max(y1, ny1)
    ix2, iy2 = min(x2, nx2), min(y2, ny2)
    if ix2 > ix1 and iy2 > iy1:
        local_candidate = relative_runs_to_mask(
            candidate.get("wire_pixel_runs", []),
            (max(0, y2 - y1), max(0, x2 - x1)),
        )
        cand_mask[iy1 - ny1:iy2 - ny1, ix1 - nx1:ix2 - nx1] = local_candidate[
            iy1 - y1:iy2 - y1,
            ix1 - x1:ix2 - x1,
        ]
    outside = local & ~cand_mask
    if orientation == "vertical":
        axis_coords = np.arange(ny1, ny2)[:, None]
        endpoint_zone_mask = (axis_coords < y1 + endpoint_zone) | (axis_coords >= y2 - endpoint_zone)
    elif orientation == "horizontal":
        axis_coords = np.arange(nx1, nx2)[None, :]
        endpoint_zone_mask = (axis_coords < x1 + endpoint_zone) | (axis_coords >= x2 - endpoint_zone)
    else:
        return {"has_attachment_metrics": False, "reason": "unknown_orientation"}
    mid_attachment_pixels = int(np.count_nonzero(outside & ~endpoint_zone_mask))
    endpoint_attachment_pixels = int(np.count_nonzero(outside & endpoint_zone_mask))
    total_attachment_pixels = int(mid_attachment_pixels + endpoint_attachment_pixels)
    bbox_area = max(1, int((x2 - x1) * (y2 - y1)))
    wire_pixel_count = int(candidate.get("wire_pixel_count", 0))
    source_component_area = int(candidate.get("source_component_area", 0))
    source_fraction = (
        float(wire_pixel_count) / max(1.0, float(source_component_area))
        if source_component_area > 0
        else 1.0
    )
    independent = (
        total_attachment_pixels == 0
        or source_fraction >= float(cfg.group_independent_source_component_min_fraction)
    )
    clean = independent or mid_attachment_pixels <= max_mid_pixels
    occupancy = float(wire_pixel_count) / max(1.0, float(bbox_area))
    rectangular = (
        float(candidate.get("projected_fill_ratio", 0.0)) >= float(cfg.group_rectangular_member_min_fill_ratio)
        and occupancy >= float(cfg.group_rectangular_member_min_occupancy)
    )
    return {
        "has_attachment_metrics": True,
        "independent_dash": bool(independent),
        "source_component_fraction_independent": bool(
            source_fraction >= float(cfg.group_independent_source_component_min_fraction)
        ),
        "rectangular_dash": bool(rectangular),
        "clean_connection": bool(clean),
        "mid_attachment_pixels": int(mid_attachment_pixels),
        "endpoint_attachment_pixels": int(endpoint_attachment_pixels),
        "total_attachment_pixels": int(total_attachment_pixels),
        "mid_attachment_pixel_limit": int(max_mid_pixels),
        "probe_pixels": int(probe),
        "endpoint_zone_pixels": int(endpoint_zone),
        "source_component_area": int(source_component_area),
        "source_component_fraction": round(float(source_fraction), 3),
        "min_source_component_fraction_for_independent": float(cfg.group_independent_source_component_min_fraction),
        "projected_fill_ratio": round(float(candidate.get("projected_fill_ratio", 0.0)), 3),
        "candidate_occupancy": round(float(occupancy), 3),
    }


def classify_candidates(
    raw_segments: list[dict[str, Any]],
    dash_binary: np.ndarray,
    component_binary: np.ndarray,
    text_height: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strict_max = max(
        text_height * cfg.strict_seed_max_text_height_ratio,
        wire_width * cfg.strict_seed_min_wire_widths,
    )
    extension_max = max(strict_max, text_height * cfg.extension_max_text_height_ratio)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    _num_labels, component_labels, component_stats, _centroids = cv2.connectedComponentsWithStats(
        component_binary.astype(np.uint8),
        connectivity=8,
    )
    for segment in raw_segments:
        span = float(segment.get("span", 0.0))
        width = max(1.0, float(segment.get("width", 0.0)))
        aspect = span / width
        annotate_segment_source_component_from_labels(segment, component_labels, component_stats, "dash_binary_without_junction_dots")
        source_metrics = terminal_source_component_metrics(segment)
        segment.setdefault("candidate_rejection_checks", {})
        if candidate_source_component_is_complex_residual(segment, cfg):
            endpoint_metrics = endpoint_only_attachment_metrics(dash_binary, segment, text_height, wire_width, cfg)
            segment["candidate_rejection_checks"]["candidate_source_component_complex_residual_flagged"] = True
            segment["candidate_rejection_checks"]["endpoint_attachment_metrics"] = endpoint_metrics
            segment["candidate_rejection_checks"]["source_component_metrics"] = source_metrics
        fanout_metrics = side_band_fanout_metrics(dash_binary, segment, wire_width, cfg)
        if bool(fanout_metrics.get("fanout_like", False)):
            segment["candidate_rejection_checks"]["fanout_region_flagged"] = True
            segment["candidate_rejection_checks"]["fanout_side_metrics"] = fanout_metrics
        if aspect < cfg.extension_min_aspect_ratio:
            rejected.append({**segment_debug_summary(segment), "decision": "reject", "reason": "extension_aspect_below_min"})
            continue
        if span > extension_max:
            rejected.append(
                {
                    **segment_debug_summary(segment),
                    "decision": "reject",
                    "reason": "extension_length_above_max",
                    "extension_max_length_pixels": round(float(extension_max), 3),
                }
            )
            continue
        strict_min = max(text_height * cfg.strict_seed_min_text_height_ratio, wire_width * cfg.strict_seed_min_wire_widths)
        strict = (
            span >= strict_min
            and span <= strict_max
            and aspect >= cfg.strict_seed_min_aspect_ratio
        )
        segment["candidate_class"] = "strict_seed" if strict else "extension_only"
        accepted.append(segment)
    strict_seeds = [segment for segment in accepted if segment.get("candidate_class") == "strict_seed"]
    extension_pool = accepted
    return strict_seeds, extension_pool, rejected


def representative_length(members: list[dict[str, Any]]) -> float:
    strict_lengths = [
        float(member["span"])
        for member in members
        if str(member.get("candidate_class", "")) == "strict_seed"
    ]
    lengths = strict_lengths or [float(member["span"]) for member in members]
    return float(np.median(lengths))


def representative_width(members: list[dict[str, Any]], fallback_width: float) -> float:
    widths = [float(member.get("width", fallback_width)) for member in members]
    if not widths:
        return float(fallback_width)
    return float(np.median(widths))


def length_ratio_ok(length: float, base: float, min_ratio: float, max_ratio: float) -> bool:
    ratio = float(length) / max(1.0, float(base))
    return min_ratio <= ratio <= max_ratio


def width_ratio_to_base(width: float, base_width: float) -> float:
    width = max(1.0, float(width))
    base_width = max(1.0, float(base_width))
    return max(width / base_width, base_width / width)


def extension_width_similar_to_group(width: float, base_width: float, cfg: DashWireConfig) -> bool:
    width = max(1.0, float(width))
    base_width = max(1.0, float(base_width))
    ratio = width / base_width
    return float(cfg.group_extension_width_min_ratio) <= ratio <= float(cfg.group_extension_width_max_ratio)


def extension_candidate_source_dot_like_metrics(
    candidate: dict[str, Any],
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    source_bbox = [int(v) for v in candidate.get("source_component_bbox", [])]
    if len(source_bbox) != 4:
        return {"has_source_component": False, "dot_like": False}
    sx1, sy1, sx2, sy2 = source_bbox
    source_w = max(0, sx2 - sx1)
    source_h = max(0, sy2 - sy1)
    source_area = int(candidate.get("source_component_area", 0))
    if source_w <= 0 or source_h <= 0 or source_area <= 0:
        return {"has_source_component": False, "dot_like": False}

    representative_width = max(float(wire_width), float(cfg.min_wire_width_px), 3.0)
    min_diameter = max(5.0, representative_width * float(cfg.dot_min_diameter_wire_widths))
    max_diameter = max(
        min_diameter,
        min(float(cfg.dot_max_diameter_px), representative_width * float(cfg.dot_max_diameter_wire_widths)),
    )
    diameter_tolerance = max(
        float(cfg.dot_max_diameter_extra_px),
        representative_width * 0.75,
    )
    aspect = max(float(source_w), float(source_h)) / max(1.0, min(float(source_w), float(source_h)))
    fill = float(source_area) / max(1.0, float(source_w * source_h))
    orientation = str(candidate.get("orientation", ""))
    candidate_cross = max(1.0, float(candidate.get("width", 0.0)))
    source_cross = float(source_h if orientation == "horizontal" else source_w)
    source_cross_ratio = source_cross / max(1.0, candidate_cross)
    dot_like = (
        min(float(source_w), float(source_h)) >= min_diameter * 0.85
        and max(float(source_w), float(source_h)) <= max_diameter + diameter_tolerance
        and aspect <= float(cfg.dot_max_aspect_ratio)
        and float(cfg.group_extension_dot_like_min_fill_ratio) <= fill <= float(cfg.group_extension_dot_like_max_fill_ratio)
        and source_cross_ratio >= float(cfg.group_extension_dot_like_min_cross_ratio)
    )
    return {
        "has_source_component": True,
        "dot_like": bool(dot_like),
        "source_component_bbox": source_bbox,
        "source_component_area": int(source_area),
        "source_width": int(source_w),
        "source_height": int(source_h),
        "source_aspect": round(float(aspect), 3),
        "source_fill_ratio": round(float(fill), 3),
        "candidate_cross": round(float(candidate_cross), 3),
        "source_cross": round(float(source_cross), 3),
        "source_cross_ratio": round(float(source_cross_ratio), 3),
        "min_diameter": round(float(min_diameter), 3),
        "max_diameter_with_tolerance": round(float(max_diameter + diameter_tolerance), 3),
    }


def local_huge_source_component_shape_metrics(
    binary: np.ndarray | None,
    candidate: dict[str, Any],
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    orientation = str(candidate.get("orientation", ""))
    if binary is None or len(bbox) != 4 or orientation not in {"horizontal", "vertical"}:
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "missing_binary_or_bbox"}
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "empty_bbox"}
    span = max(1.0, float(candidate.get("span", 0.0)))
    candidate_width = max(1.0, float(candidate.get("width", wire_width)))
    axis_pad = int(round(max(
        span * float(cfg.group_extension_huge_local_axis_pad_ratio),
        float(wire_width) * float(cfg.group_extension_huge_local_axis_pad_wire_widths),
        8.0,
    )))
    cross_pad = int(round(max(
        candidate_width * 3.0,
        float(wire_width) * float(cfg.group_extension_huge_local_cross_pad_wire_widths),
        10.0,
    )))
    if orientation == "vertical":
        crop = [x1 - cross_pad, y1 - axis_pad, x2 + cross_pad, y2 + axis_pad]
    else:
        crop = [x1 - axis_pad, y1 - cross_pad, x2 + axis_pad, y2 + cross_pad]
    cx1, cy1, cx2, cy2 = clipped_bbox(crop, binary.shape)
    if cx2 <= cx1 or cy2 <= cy1:
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "empty_crop"}

    local = binary[cy1:cy2, cx1:cx2].astype(bool)
    if not bool(np.any(local)):
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "empty_local_binary"}
    candidate_mask = np.zeros_like(local, dtype=bool)
    ix1, iy1 = max(x1, cx1), max(y1, cy1)
    ix2, iy2 = min(x2, cx2), min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "candidate_outside_crop"}
    wire_mask = relative_runs_to_mask(
        candidate.get("wire_pixel_runs", []),
        (max(0, y2 - y1), max(0, x2 - x1)),
    )
    if bool(np.any(wire_mask)):
        candidate_mask[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = wire_mask[
            iy1 - y1:iy2 - y1,
            ix1 - x1:ix2 - x1,
        ]
    else:
        candidate_mask[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = True
    seed = local & candidate_mask
    if not bool(np.any(seed)):
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "candidate_pixels_not_in_local_component"}

    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
    touched_labels = [int(value) for value in np.unique(labels[seed]) if int(value) > 0]
    if not touched_labels:
        return {"local_fallback_available": False, "local_rectangular_connection_ok": False, "reason": "no_touched_local_component"}
    label = max(touched_labels, key=lambda value: int(np.count_nonzero(labels[seed] == value)))
    component = labels == label
    component_area = int(np.count_nonzero(component))
    candidate_pixels = int(np.count_nonzero(component & candidate_mask))
    residual = component & ~candidate_mask
    residual_area = int(np.count_nonzero(residual))
    if residual_area <= 0:
        return {
            "local_fallback_available": True,
            "local_rectangular_connection_ok": True,
            "reason": "local_component_is_candidate_only",
            "local_crop_bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
            "local_component_area": int(component_area),
            "local_candidate_pixels": int(candidate_pixels),
            "local_candidate_fraction": round(float(candidate_pixels) / max(1.0, float(component_area)), 3),
            "local_residual_area": 0,
        }

    residual_labels_count, residual_labels, residual_stats, _residual_centroids = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8),
        connectivity=8,
    )
    touch_gap = max(1.0, float(wire_width))
    local_candidate_bbox = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
    endpoint_zone = max(1.0, float(wire_width) * float(cfg.group_extension_huge_local_endpoint_t_zone_wire_widths))
    max_residual_cross = max(
        candidate_width * float(cfg.group_extension_width_max_ratio),
        float(wire_width) * float(cfg.group_extension_huge_local_residual_max_cross_wire_widths),
    )
    direct_components: list[dict[str, Any]] = []
    non_direct_area = 0
    endpoint_t_cross_like = False
    for label_index in range(1, residual_labels_count):
        area = int(residual_stats[label_index, cv2.CC_STAT_AREA])
        rx = int(residual_stats[label_index, cv2.CC_STAT_LEFT])
        ry = int(residual_stats[label_index, cv2.CC_STAT_TOP])
        rw = int(residual_stats[label_index, cv2.CC_STAT_WIDTH])
        rh = int(residual_stats[label_index, cv2.CC_STAT_HEIGHT])
        rb = [rx, ry, rx + rw, ry + rh]
        direct = expanded_bboxes_touch(local_candidate_bbox, rb, touch_gap)
        if not direct:
            non_direct_area += area
            continue
        fill = float(area) / max(1.0, float(rw * rh))
        major = float(max(rw, rh))
        minor = float(max(1, min(rw, rh)))
        rectangular = fill >= float(cfg.group_extension_huge_local_residual_min_fill_ratio) and minor <= max_residual_cross
        if orientation == "vertical":
            touches_start = rb[1] <= local_candidate_bbox[1] + endpoint_zone
            touches_end = rb[3] >= local_candidate_bbox[3] - endpoint_zone
            crosses_before = rb[0] < local_candidate_bbox[0]
            crosses_after = rb[2] > local_candidate_bbox[2]
            if (touches_start or touches_end) and crosses_before and crosses_after:
                endpoint_t_cross_like = True
        else:
            touches_start = rb[0] <= local_candidate_bbox[0] + endpoint_zone
            touches_end = rb[2] >= local_candidate_bbox[2] - endpoint_zone
            crosses_before = rb[1] < local_candidate_bbox[1]
            crosses_after = rb[3] > local_candidate_bbox[3]
            if (touches_start or touches_end) and crosses_before and crosses_after:
                endpoint_t_cross_like = True
        direct_components.append(
            {
                "bbox": [int(cx1 + rb[0]), int(cy1 + rb[1]), int(cx1 + rb[2]), int(cy1 + rb[3])],
                "area": int(area),
                "fill_ratio": round(float(fill), 3),
                "major": round(float(major), 3),
                "minor": round(float(minor), 3),
                "rectangular": bool(rectangular),
            }
        )

    direct_area = int(sum(int(component["area"]) for component in direct_components))
    residual_area_limit = max(
        float(candidate_pixels) * float(cfg.group_extension_huge_local_max_residual_area_ratio),
        float(wire_width) * 24.0,
    )
    non_direct_limit = max(2, int(round(float(wire_width) * float(cfg.group_extension_clean_l_leftover_max_wire_widths))))
    ok = (
        bool(direct_components)
        and len(direct_components) <= int(cfg.group_extension_huge_local_max_residual_components)
        and all(bool(component["rectangular"]) for component in direct_components)
        and direct_area <= residual_area_limit
        and non_direct_area <= non_direct_limit
        and not endpoint_t_cross_like
    )
    return {
        "local_fallback_available": True,
        "local_rectangular_connection_ok": bool(ok),
        "reason": "local_direct_residual_rectangular" if ok else "local_direct_residual_not_rectangular_enough",
        "local_crop_bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
        "local_component_area": int(component_area),
        "local_candidate_pixels": int(candidate_pixels),
        "local_candidate_fraction": round(float(candidate_pixels) / max(1.0, float(component_area)), 3),
        "local_residual_area": int(residual_area),
        "direct_residual_area": int(direct_area),
        "non_direct_residual_area": int(non_direct_area),
        "non_direct_residual_area_limit": int(non_direct_limit),
        "direct_residual_area_limit": round(float(residual_area_limit), 3),
        "direct_residual_component_count": int(len(direct_components)),
        "direct_residual_component_limit": int(cfg.group_extension_huge_local_max_residual_components),
        "endpoint_t_cross_like": bool(endpoint_t_cross_like),
        "direct_residual_components": direct_components,
    }


def limited_side_connection_metrics(
    residual: np.ndarray,
    candidate_local_bbox: list[int],
    orientation: str,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    if residual.size == 0 or len(candidate_local_bbox) != 4 or orientation not in {"horizontal", "vertical"}:
        return {"limited_side_connection_ok": False, "reason": "missing_residual_or_bbox"}
    cx1, cy1, cx2, cy2 = [int(v) for v in candidate_local_bbox]
    if cx2 <= cx1 or cy2 <= cy1:
        return {"limited_side_connection_ok": False, "reason": "empty_candidate_bbox"}
    height, width = residual.shape
    band = int(max(1, round(float(wire_width) * float(cfg.group_extension_limited_side_band_wire_widths))))

    def region_count_and_axis_coverage(x1: int, y1: int, x2: int, y2: int, axis_len: int, axis: int) -> tuple[int, float]:
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(width, x2), min(height, y2)
        if rx2 <= rx1 or ry2 <= ry1:
            return 0, 0.0
        region = residual[ry1:ry2, rx1:rx2]
        count = int(np.count_nonzero(region))
        if count <= 0:
            return 0, 0.0
        if axis == 0:
            covered = int(np.count_nonzero(np.any(region, axis=0)))
        else:
            covered = int(np.count_nonzero(np.any(region, axis=1)))
        return count, float(covered) / max(1.0, float(axis_len))

    if orientation == "horizontal":
        axis_len = max(1, cx2 - cx1)
        top_count, top_ratio = region_count_and_axis_coverage(cx1, cy1 - band, cx2, cy1, axis_len, axis=0)
        bottom_count, bottom_ratio = region_count_and_axis_coverage(cx1, cy2, cx2, cy2 + band, axis_len, axis=0)
        left_count, left_ratio = region_count_and_axis_coverage(cx1 - band, cy1, cx1, cy2, max(1, cy2 - cy1), axis=1)
        right_count, right_ratio = region_count_and_axis_coverage(cx2, cy1, cx2 + band, cy2, max(1, cy2 - cy1), axis=1)
        primary = {"top": top_ratio, "bottom": bottom_ratio}
        endpoint = {"left": left_ratio, "right": right_ratio}
        near_mask = np.zeros_like(residual, dtype=bool)
        near_mask[max(0, cy1 - band):cy1, max(0, cx1):min(width, cx2)] = True
        near_mask[cy2:min(height, cy2 + band), max(0, cx1):min(width, cx2)] = True
        near_mask[max(0, cy1):min(height, cy2), max(0, cx1 - band):cx1] = True
        near_mask[max(0, cy1):min(height, cy2), cx2:min(width, cx2 + band)] = True
        contact_pixel_count = top_count + bottom_count + left_count + right_count
    else:
        axis_len = max(1, cy2 - cy1)
        left_count, left_ratio = region_count_and_axis_coverage(cx1 - band, cy1, cx1, cy2, axis_len, axis=1)
        right_count, right_ratio = region_count_and_axis_coverage(cx2, cy1, cx2 + band, cy2, axis_len, axis=1)
        top_count, top_ratio = region_count_and_axis_coverage(cx1, cy1 - band, cx2, cy1, max(1, cx2 - cx1), axis=0)
        bottom_count, bottom_ratio = region_count_and_axis_coverage(cx1, cy2, cx2, cy2 + band, max(1, cx2 - cx1), axis=0)
        primary = {"left": left_ratio, "right": right_ratio}
        endpoint = {"top": top_ratio, "bottom": bottom_ratio}
        near_mask = np.zeros_like(residual, dtype=bool)
        near_mask[max(0, cy1):min(height, cy2), max(0, cx1 - band):cx1] = True
        near_mask[max(0, cy1):min(height, cy2), cx2:min(width, cx2 + band)] = True
        near_mask[max(0, cy1 - band):cy1, max(0, cx1):min(width, cx2)] = True
        near_mask[cy2:min(height, cy2 + band), max(0, cx1):min(width, cx2)] = True
        contact_pixel_count = left_count + right_count + top_count + bottom_count

    primary_values = sorted((float(value) for value in primary.values()), reverse=True)
    max_primary = primary_values[0] if primary_values else 0.0
    second_primary = primary_values[1] if len(primary_values) > 1 else 0.0
    endpoint_max = max((float(value) for value in endpoint.values()), default=0.0)
    total_primary = sum(float(value) for value in primary.values())
    contact_mask = residual & near_mask
    component_count = 0
    if bool(np.any(contact_mask)):
        component_count = int(cv2.connectedComponentsWithStats(contact_mask.astype(np.uint8), connectivity=8)[0] - 1)
    ok = (
        contact_pixel_count > 0
        and max_primary <= float(cfg.group_extension_limited_side_max_long_contact_ratio)
        and second_primary <= float(cfg.group_extension_limited_side_max_opposite_contact_ratio)
        and total_primary <= float(cfg.group_extension_limited_side_max_total_contact_ratio)
        and component_count <= int(cfg.group_extension_limited_side_max_contact_components)
    )
    return {
        "limited_side_connection_ok": bool(ok),
        "reason": "limited_single_side_contact" if ok else "side_contact_too_broad_or_multisided",
        "orientation": orientation,
        "side_band_pixels": int(band),
        "primary_side_contact_ratios": {key: round(float(value), 3) for key, value in primary.items()},
        "endpoint_side_contact_ratios": {key: round(float(value), 3) for key, value in endpoint.items()},
        "max_primary_side_contact_ratio": round(float(max_primary), 3),
        "second_primary_side_contact_ratio": round(float(second_primary), 3),
        "total_primary_side_contact_ratio": round(float(total_primary), 3),
        "max_endpoint_side_contact_ratio": round(float(endpoint_max), 3),
        "contact_pixel_count": int(contact_pixel_count),
        "contact_component_count": int(component_count),
        "limits": {
            "max_long_contact_ratio": float(cfg.group_extension_limited_side_max_long_contact_ratio),
            "max_opposite_contact_ratio": float(cfg.group_extension_limited_side_max_opposite_contact_ratio),
            "max_total_contact_ratio": float(cfg.group_extension_limited_side_max_total_contact_ratio),
            "max_contact_components": int(cfg.group_extension_limited_side_max_contact_components),
        },
    }


def bracket_like_source_component_metrics(
    residual: np.ndarray,
    candidate_local_bbox: list[int],
    orientation: str,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    if residual.size == 0 or len(candidate_local_bbox) != 4 or orientation not in {"horizontal", "vertical"}:
        return {"bracket_like": False, "reason": "missing_residual_or_bbox"}
    cx1, cy1, cx2, cy2 = [int(v) for v in candidate_local_bbox]
    if cx2 <= cx1 or cy2 <= cy1:
        return {"bracket_like": False, "reason": "empty_candidate_bbox"}
    height, width = residual.shape
    span = max(1, (cy2 - cy1) if orientation == "vertical" else (cx2 - cx1))
    endpoint_zone = int(max(
        1,
        min(
            max(1, span // 3),
            round(float(wire_width) * float(cfg.group_extension_bracket_endpoint_zone_wire_widths)),
        ),
    ))
    side_band = int(max(1, round(float(wire_width) * float(cfg.group_extension_bracket_side_band_wire_widths))))
    min_pixels = int(max(1, cfg.group_extension_bracket_min_endpoint_pixels))

    def count_region(x1: int, y1: int, x2: int, y2: int) -> int:
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(width, x2), min(height, y2)
        if rx2 <= rx1 or ry2 <= ry1:
            return 0
        return int(np.count_nonzero(residual[ry1:ry2, rx1:rx2]))

    if orientation == "vertical":
        side_counts = {
            "left": {
                "start": count_region(cx1 - side_band, cy1 - endpoint_zone, cx1, cy1 + endpoint_zone),
                "end": count_region(cx1 - side_band, cy2 - endpoint_zone, cx1, cy2 + endpoint_zone),
            },
            "right": {
                "start": count_region(cx2, cy1 - endpoint_zone, cx2 + side_band, cy1 + endpoint_zone),
                "end": count_region(cx2, cy2 - endpoint_zone, cx2 + side_band, cy2 + endpoint_zone),
            },
        }
    else:
        side_counts = {
            "top": {
                "start": count_region(cx1 - endpoint_zone, cy1 - side_band, cx1 + endpoint_zone, cy1),
                "end": count_region(cx2 - endpoint_zone, cy1 - side_band, cx2 + endpoint_zone, cy1),
            },
            "bottom": {
                "start": count_region(cx1 - endpoint_zone, cy2, cx1 + endpoint_zone, cy2 + side_band),
                "end": count_region(cx2 - endpoint_zone, cy2, cx2 + endpoint_zone, cy2 + side_band),
            },
        }

    bracket_sides = [
        side
        for side, counts in side_counts.items()
        if int(counts.get("start", 0)) >= min_pixels and int(counts.get("end", 0)) >= min_pixels
    ]
    return {
        "bracket_like": bool(bracket_sides),
        "reason": "same_side_endpoint_residuals" if bracket_sides else "no_same_side_endpoint_pair",
        "orientation": orientation,
        "endpoint_zone_pixels": int(endpoint_zone),
        "side_band_pixels": int(side_band),
        "min_endpoint_pixels": int(min_pixels),
        "side_endpoint_pixel_counts": side_counts,
        "bracket_sides": bracket_sides,
    }


def extension_source_component_shape_metrics(
    candidate: dict[str, Any],
    wire_width: float,
    cfg: DashWireConfig,
    local_binary: np.ndarray | None = None,
    allow_limited_side_connection: bool = False,
    allow_bracket_like_reject: bool = False,
) -> dict[str, Any]:
    source_metrics = terminal_source_component_metrics(candidate)
    if not bool(source_metrics.get("has_source_component", False)):
        return {"has_source_component": False, "letter_like": False}
    source_fraction = float(source_metrics.get("candidate_pixel_fraction", 1.0))
    source_cross_ratio = float(source_metrics.get("source_cross_ratio", 1.0))
    source_area_ratio = float(source_metrics.get("source_area_ratio", 1.0))
    source_area = int(source_metrics.get("source_component_area", 0))
    source_runs = candidate.get("source_component_pixel_runs", [])
    huge_source = source_area_ratio >= float(cfg.group_extension_letter_like_huge_source_area_ratio)
    if huge_source and not source_runs:
        local_metrics = local_huge_source_component_shape_metrics(local_binary, candidate, wire_width, cfg)
        if bool(local_metrics.get("local_rectangular_connection_ok", False)):
            return {
                **source_metrics,
                "letter_like": False,
                "clean_l_corner_like": False,
                "huge_source_local_rectangular_connection": True,
                "local_huge_source_component_shape_metrics": local_metrics,
                "reason": "huge_source_component_explained_by_local_rectangular_connection",
                "max_source_area_for_pixel_runs": int(cfg.group_extension_source_runs_max_area),
            }
        return {
            **source_metrics,
            "letter_like": True,
            "clean_l_corner_like": False,
            "huge_source_local_rectangular_connection": False,
            "local_huge_source_component_shape_metrics": local_metrics,
            "reason": "huge_source_component_without_local_rectangle_explanation",
            "max_source_area_for_pixel_runs": int(cfg.group_extension_source_runs_max_area),
        }
    coarse_source_not_letter_like = (
        source_fraction > float(cfg.group_extension_letter_like_max_source_fraction)
        or source_cross_ratio < float(cfg.group_extension_letter_like_min_source_cross_ratio)
        or source_area_ratio < float(cfg.group_extension_letter_like_min_source_area_ratio)
    )
    if coarse_source_not_letter_like and not allow_bracket_like_reject:
        return {
            **source_metrics,
            "letter_like": False,
            "clean_l_corner_like": False,
            "reason": "source_component_not_letter_like_enough",
        }
    if not isinstance(source_runs, list) or not source_runs:
        return {
            **source_metrics,
            "letter_like": True,
            "clean_l_corner_like": False,
            "reason": "letter_like_source_shape_without_pixels",
        }

    bbox = [int(v) for v in candidate.get("bbox", [])]
    source_bbox = [int(v) for v in candidate.get("source_component_bbox", [])]
    if len(bbox) != 4 or len(source_bbox) != 4:
        return {**source_metrics, "letter_like": False, "clean_l_corner_like": False, "reason": "missing_bbox"}
    sx1, sy1, sx2, sy2 = source_bbox
    tx1, ty1, tx2, ty2 = bbox
    source_mask = relative_runs_to_mask(source_runs, (max(0, sy2 - sy1), max(0, sx2 - sx1)))
    if not bool(np.any(source_mask)):
        return {**source_metrics, "letter_like": False, "clean_l_corner_like": False, "reason": "empty_source_component"}
    candidate_mask = relative_runs_to_mask(
        candidate.get("wire_pixel_runs", []),
        (max(0, ty2 - ty1), max(0, tx2 - tx1)),
    )
    residual = source_mask.copy()
    ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
    ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
    if ix2 > ix1 and iy2 > iy1:
        residual[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] &= ~candidate_mask[
            iy1 - ty1:iy2 - ty1,
            ix1 - tx1:ix2 - tx1,
        ]
    residual_area = int(np.count_nonzero(residual))
    if residual_area <= 0:
        return {
            **source_metrics,
            "letter_like": False,
            "clean_l_corner_like": False,
            "residual_area": 0,
            "reason": "no_residual_after_candidate",
        }

    orientation = str(candidate.get("orientation", ""))
    bracket_metrics = bracket_like_source_component_metrics(
        residual,
        [tx1 - sx1, ty1 - sy1, tx2 - sx1, ty2 - sy1],
        orientation,
        wire_width,
        cfg,
    )
    if bool(allow_bracket_like_reject) and bool(bracket_metrics.get("bracket_like", False)):
        return {
            **source_metrics,
            "letter_like": True,
            "clean_l_corner_like": False,
            "limited_side_connection_like": False,
            "bracket_like_source_component": True,
            "bracket_like_source_component_metrics": bracket_metrics,
            "residual_area": int(residual_area),
            "reason": "bracket_like_source_component",
        }
    if coarse_source_not_letter_like:
        return {
            **source_metrics,
            "letter_like": False,
            "clean_l_corner_like": False,
            "bracket_like_source_component": False,
            "bracket_like_source_component_metrics": bracket_metrics,
            "residual_area": int(residual_area),
            "reason": "source_component_not_letter_like_enough",
        }
    orthogonal_orientation = "vertical" if orientation == "horizontal" else "horizontal"
    min_len = int(max(2, round(float(wire_width) * 2.0)))
    max_len = int(max(min_len, round(max(float(candidate.get("span", 0.0)) * 3.0, float(wire_width) * 12.0))))
    orthogonal_segments = extract_axis_rectangles(
        residual,
        orthogonal_orientation,
        min_len,
        max_len,
        wire_width,
        cfg,
    )
    terminal_local_bbox = [tx1 - sx1, ty1 - sy1, tx2 - sx1, ty2 - sy1]
    touch_gap = max(1.0, float(wire_width))
    clean_l_base_width = max(float(candidate.get("width", wire_width)), float(wire_width), 1.0)
    touching_segments = [
        segment
        for segment in orthogonal_segments
        if expanded_bboxes_touch(terminal_local_bbox, segment.get("bbox", []), touch_gap)
        and width_ratio_to_base(float(segment.get("width", 0.0)), clean_l_base_width) <= float(cfg.group_extension_width_max_ratio)
    ]
    clean_l_corner = False
    leftover_pixels = residual_area
    best_orthogonal_bbox: list[int] | None = None
    if touching_segments:
        best = max(
            touching_segments,
            key=lambda segment: (float(segment.get("span", 0.0)) * float(segment.get("width", 0.0)), float(segment.get("span", 0.0))),
        )
        ox1, oy1, ox2, oy2 = [int(v) for v in best["bbox"]]
        orthogonal_mask = relative_runs_to_mask(
            best.get("wire_pixel_runs", []),
            (max(0, oy2 - oy1), max(0, ox2 - ox1)),
        )
        residual_after_l = residual.copy()
        if ox2 > ox1 and oy2 > oy1:
            residual_after_l[oy1:oy2, ox1:ox2] &= ~orthogonal_mask
        leftover_pixels = int(np.count_nonzero(residual_after_l))
        leftover_limit = int(max(2, round(float(wire_width) * float(cfg.group_extension_clean_l_leftover_max_wire_widths))))
        cx1, cy1, cx2, cy2 = terminal_local_bbox
        if orientation == "vertical":
            touches_endpoint = (
                oy1 <= cy1 + touch_gap and oy2 >= cy1 - touch_gap
            ) or (
                oy1 <= cy2 + touch_gap and oy2 >= cy2 - touch_gap
            )
            crosses_both_sides = ox1 < cx1 and ox2 > cx2
        else:
            touches_endpoint = (
                ox1 <= cx1 + touch_gap and ox2 >= cx1 - touch_gap
            ) or (
                ox1 <= cx2 + touch_gap and ox2 >= cx2 - touch_gap
            )
            crosses_both_sides = oy1 < cy1 and oy2 > cy2
        clean_l_corner = leftover_pixels <= leftover_limit and touches_endpoint and not crosses_both_sides
        best_orthogonal_bbox = [int(sx1 + ox1), int(sy1 + oy1), int(sx1 + ox2), int(sy1 + oy2)]

    limited_side_metrics = limited_side_connection_metrics(
        residual,
        terminal_local_bbox,
        orientation,
        wire_width,
        cfg,
    )
    limited_source_shape_ok = (
        source_area_ratio >= float(cfg.group_extension_limited_side_min_source_area_ratio)
        and float(source_metrics.get("source_axis_ratio", 1.0)) >= float(cfg.group_extension_limited_side_min_source_axis_ratio)
    )
    limited_side_connection = (
        bool(allow_limited_side_connection)
        and bool(limited_source_shape_ok)
        and bool(limited_side_metrics.get("limited_side_connection_ok", False))
    )
    letter_like = not clean_l_corner and not limited_side_connection
    if clean_l_corner:
        reason = "clean_l_corner_explains_source_component"
    elif limited_side_connection:
        reason = "limited_side_connection_explains_source_component"
    else:
        reason = "source_residual_not_clean_l_corner"
    return {
        **source_metrics,
        "letter_like": bool(letter_like),
        "clean_l_corner_like": bool(clean_l_corner),
        "limited_side_connection_like": bool(limited_side_connection),
        "limited_side_connection_override_allowed": bool(allow_limited_side_connection),
        "limited_side_connection_source_shape_ok": bool(limited_source_shape_ok),
        "limited_side_connection_metrics": limited_side_metrics,
        "bracket_like_source_component": False,
        "bracket_like_source_component_metrics": bracket_metrics,
        "residual_area": int(residual_area),
        "orthogonal_candidate_count": int(len(orthogonal_segments)),
        "touching_orthogonal_candidate_count": int(len(touching_segments)),
        "best_touching_orthogonal_bbox": best_orthogonal_bbox,
        "leftover_pixels_after_best_l": int(leftover_pixels),
        "clean_l_requires_endpoint_touch": True,
        "clean_l_rejects_t_crossing": True,
        "reason": reason,
    }


def candidate_external_t_junction_metrics(
    binary: np.ndarray,
    candidate: dict[str, Any],
    direction: int,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    orientation = str(candidate.get("orientation", ""))
    if len(bbox) != 4 or orientation not in {"horizontal", "vertical"}:
        return {"t_junction_like": False, "reason": "missing_bbox_or_orientation"}
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return {"t_junction_like": False, "reason": "empty_bbox"}
    probe = int(max(2, round(float(wire_width) * float(cfg.group_extension_t_junction_probe_wire_widths))))
    min_run = int(max(2, round(float(wire_width) * float(cfg.group_extension_t_junction_min_run_wire_widths))))

    def contiguous_run(start_x: int, start_y: int, step_x: int, step_y: int) -> int:
        count = 0
        x, y = int(start_x), int(start_y)
        for _index in range(probe):
            if y < 0 or y >= binary.shape[0] or x < 0 or x >= binary.shape[1]:
                break
            if not bool(binary[y, x]):
                break
            count += 1
            x += int(step_x)
            y += int(step_y)
        return int(count)

    if orientation == "vertical":
        x = int(round(float(candidate.get("centerline", (x1 + x2 - 1) / 2.0))))
        y = int(y1 if direction < 0 else y2 - 1)
        negative_run = contiguous_run(x - 1, y, -1, 0)
        positive_run = contiguous_run(x + 1, y, 1, 0)
        directions = {"left": negative_run, "right": positive_run}
    else:
        x = int(x1 if direction < 0 else x2 - 1)
        y = int(round(float(candidate.get("centerline", (y1 + y2 - 1) / 2.0))))
        negative_run = contiguous_run(x, y - 1, 0, -1)
        positive_run = contiguous_run(x, y + 1, 0, 1)
        directions = {"up": negative_run, "down": positive_run}

    t_like = negative_run >= min_run and positive_run >= min_run
    return {
        "t_junction_like": bool(t_like),
        "endpoint": [int(x), int(y)],
        "orientation": orientation,
        "direction": int(direction),
        "orthogonal_runs": directions,
        "min_run_pixels": int(min_run),
        "probe_pixels": int(probe),
        "reason": "two_sided_orthogonal_runs_at_external_endpoint" if t_like else "not_two_sided",
    }


def longest_true_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    best = 0
    current = 0
    for value in values.astype(bool).ravel():
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def centerline_gap_metrics(
    binary: np.ndarray,
    current_edge: float,
    direction: int,
    orientation: str,
    centerline: float,
    candidate: dict[str, Any],
    cfg: DashWireConfig,
) -> dict[str, Any]:
    candidate_start = axis_start(candidate)
    candidate_end = axis_end(candidate)
    if direction > 0:
        gap_start = int(math.ceil(current_edge))
        gap_end = int(math.floor(candidate_start))
    else:
        gap_start = int(math.ceil(candidate_end))
        gap_end = int(math.floor(current_edge))
    if gap_end <= gap_start:
        return {
            "has_positive_gap": False,
            "gap_start": int(gap_start),
            "gap_end": int(gap_end),
            "gap_length_pixels": 0,
            "centerline_black_pixels": 0,
            "centerline_black_fill_ratio": 0.0,
            "max_consecutive_centerline_black_pixels": 0,
            "max_consecutive_centerline_black_ratio": 0.0,
            "gap_clear": False,
        }

    y_max, x_max = binary.shape
    black_pixels = 0
    centerline_values = np.zeros((0,), dtype=bool)
    if orientation == "horizontal":
        y = int(round(centerline))
        if 0 <= y < y_max:
            x1 = max(0, gap_start)
            x2 = min(x_max, gap_end)
            if x2 > x1:
                centerline_values = binary[y, x1:x2].astype(bool)
                black_pixels = int(np.count_nonzero(centerline_values))
    elif orientation == "vertical":
        x = int(round(centerline))
        if 0 <= x < x_max:
            y1 = max(0, gap_start)
            y2 = min(y_max, gap_end)
            if y2 > y1:
                centerline_values = binary[y1:y2, x].astype(bool)
                black_pixels = int(np.count_nonzero(centerline_values))
    gap_len = int(max(0, gap_end - gap_start))
    max_run = longest_true_run(centerline_values)
    fill_ratio = float(black_pixels) / max(1.0, float(gap_len))
    run_ratio = float(max_run) / max(1.0, float(gap_len))
    sparse_ok = black_pixels <= int(cfg.group_gap_max_centerline_black_pixels)
    bridge_like = (
        not sparse_ok
        and (
            fill_ratio >= float(cfg.group_gap_block_min_fill_ratio)
            or run_ratio >= float(cfg.group_gap_block_min_run_ratio)
        )
    )
    return {
        "has_positive_gap": True,
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
        "gap_length_pixels": int(gap_len),
        "centerline_black_pixels": int(black_pixels),
        "centerline_black_fill_ratio": round(float(fill_ratio), 3),
        "max_consecutive_centerline_black_pixels": int(max_run),
        "max_consecutive_centerline_black_ratio": round(float(run_ratio), 3),
        "gap_clear": bool(not bridge_like),
    }


def find_next_group_segment(
    dash_binary: np.ndarray,
    current_edge: float,
    direction: int,
    orientation: str,
    centerline: float,
    rep_length: float,
    rep_width: float,
    wire_width: float,
    pool: list[dict[str, Any]],
    used_ids: set[str],
    max_gap: float,
    centerline_tolerance: float,
    current_member_count: int,
    cfg: DashWireConfig,
    debug_log: list[dict[str, Any]],
    debug_bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_gap = float("inf")
    evaluated: list[dict[str, Any]] = []
    for candidate in pool:
        candidate_id = str(candidate["segment_id"])
        gap = axis_start(candidate) - current_edge if direction > 0 else current_edge - axis_end(candidate)
        centerline_delta = abs(float(candidate["centerline"]) - centerline)
        length_ratio = float(candidate["span"]) / max(1.0, rep_length)
        width_ratio = width_ratio_to_base(float(candidate.get("width", 0.0)), rep_width)
        gap_metrics: dict[str, Any] = {}
        dot_like_metrics: dict[str, Any] = {}
        t_junction_metrics: dict[str, Any] = {}
        source_shape_metrics: dict[str, Any] = {}
        reason = "eligible"
        if candidate_id in used_ids or candidate["orientation"] != orientation:
            reason = "used_or_wrong_orientation"
        elif centerline_delta > centerline_tolerance:
            reason = "centerline_delta_exceeds_tolerance"
        elif gap < -0.5 * float(centerline_tolerance) or gap >= max_gap:
            reason = "gap_out_of_range"
        elif not length_ratio_ok(float(candidate["span"]), rep_length, cfg.group_length_min_ratio, cfg.group_length_max_ratio):
            reason = "length_ratio_out_of_range"
        elif not extension_width_similar_to_group(float(candidate.get("width", 0.0)), rep_width, cfg):
            reason = "width_ratio_to_group_out_of_range"
        elif str(candidate.get("candidate_class", "")) == "extension_only":
            dot_like_metrics = extension_candidate_source_dot_like_metrics(candidate, wire_width, cfg)
            if bool(dot_like_metrics.get("dot_like", False)):
                reason = "extension_source_component_dot_like"
            else:
                t_junction_metrics = candidate_external_t_junction_metrics(dash_binary, candidate, direction, wire_width, cfg)
                if bool(t_junction_metrics.get("t_junction_like", False)):
                    reason = "extension_external_endpoint_t_junction_like"
        else:
            gap_metrics = centerline_gap_metrics(dash_binary, current_edge, direction, orientation, centerline, candidate, cfg)
            if not bool(gap_metrics.get("gap_clear", False)):
                reason = "centerline_gap_contains_black_pixels"

        if reason == "eligible":
            allow_limited_side_connection = current_member_count >= int(cfg.group_extension_limited_side_min_existing_members)
            source_shape_metrics = extension_source_component_shape_metrics(
                candidate,
                wire_width,
                cfg,
                dash_binary,
                allow_limited_side_connection=allow_limited_side_connection,
                allow_bracket_like_reject=True,
            )
            if bool(source_shape_metrics.get("letter_like", False)):
                reason = "extension_source_component_letter_like"

        if reason == "eligible" and str(candidate.get("candidate_class", "")) == "extension_only":
            gap_metrics = centerline_gap_metrics(dash_binary, current_edge, direction, orientation, centerline, candidate, cfg)
            if not bool(gap_metrics.get("gap_clear", False)):
                reason = "centerline_gap_contains_black_pixels"

        if debug_bbox is not None and bbox_intersects(candidate.get("bbox", []), debug_bbox):
            evaluated.append(
                {
                    **segment_debug_summary(candidate),
                    "gap": round(float(gap), 3),
                    "centerline_delta": round(float(centerline_delta), 3),
                    "length_ratio": round(float(length_ratio), 3),
                    "representative_width": round(float(rep_width), 3),
                    "width_ratio_to_group": round(float(width_ratio), 3),
                    "centerline_gap_metrics": gap_metrics,
                    "extension_source_dot_like_metrics": dot_like_metrics,
                    "extension_external_t_junction_metrics": t_junction_metrics,
                    "extension_source_shape_metrics": source_shape_metrics,
                    "decision": "eligible" if reason == "eligible" else "reject",
                    "reason": reason,
                }
            )
        if reason != "eligible":
            continue
        if gap < best_gap:
            best = candidate
            best_gap = float(gap)
    if evaluated:
        chosen_id = str(best["segment_id"]) if best is not None else None
        for item in evaluated:
            if chosen_id is not None and item["segment_id"] == chosen_id:
                item["decision"] = "accept"
                item["reason"] = "nearest_eligible_candidate"
            elif item["decision"] == "eligible":
                item["decision"] = "reject"
                item["reason"] = "eligible_but_farther_than_chosen"
        debug_log.append(
            {
                "event": "regular_extension_step",
                "direction": "forward" if direction > 0 else "backward",
                "current_edge": round(float(current_edge), 3),
                "orientation": orientation,
                "centerline": round(float(centerline), 3),
                "representative_length": round(float(rep_length), 3),
                "chosen_segment_id": chosen_id,
                "candidate_decisions": evaluated,
            }
        )
    return best


def crop_component_terminal_candidates(
    binary: np.ndarray,
    orientation: str,
    search_bbox: tuple[int, int, int, int],
    centerline: float,
    min_len: int,
    max_len: int,
    target_width: float,
    cfg: DashWireConfig,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = clipped_bbox(search_bbox, binary.shape)
    if x2 <= x1 or y2 <= y1:
        return []
    min_cross = max(1, int(cfg.min_wire_width_px))
    max_cross = max(float(min_cross), float(target_width) * 1.8)
    candidates = (
        extract_axis_rectangles(binary[y1:y2, x1:x2], orientation, min_len, max_len, target_width, cfg)
    )
    for candidate in candidates:
        bbox = [int(v) for v in candidate["bbox"]]
        candidate["bbox"] = [bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1]
        source_bbox = candidate.get("source_component_bbox")
        if isinstance(source_bbox, list) and len(source_bbox) == 4:
            candidate["source_component_bbox"] = [
                int(source_bbox[0]) + x1,
                int(source_bbox[1]) + y1,
                int(source_bbox[2]) + x1,
                int(source_bbox[3]) + y1,
            ]
        if orientation == "horizontal":
            center = (candidate["bbox"][1] + candidate["bbox"][3] - 1) / 2.0
            candidate["points"] = [[candidate["bbox"][0], int(round(center))], [candidate["bbox"][2] - 1, int(round(center))]]
        else:
            center = (candidate["bbox"][0] + candidate["bbox"][2] - 1) / 2.0
            candidate["points"] = [[int(round(center)), candidate["bbox"][1]], [int(round(center)), candidate["bbox"][3] - 1]]
        candidate["centerline"] = float(center)
        attach_segment_pixels(candidate, binary)
        candidate["source"] = "external_endpoint_terminal_search"
        candidate["candidate_class"] = "ultrashort_terminal"
    return [c for c in candidates if float(c.get("width", 0.0)) <= max_cross + 0.5]


def refresh_terminal_source_component(
    candidate: dict[str, Any],
    full_binary: np.ndarray,
    rep_length: float,
    wire_width: float,
) -> None:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    if len(bbox) != 4:
        return
    pad = int(math.ceil(max(float(rep_length) * 1.5, float(wire_width) * 10.0, 12.0)))
    x1, y1, x2, y2 = clipped_bbox([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], full_binary.shape)
    if x2 <= x1 or y2 <= y1:
        return
    local = full_binary[y1:y2, x1:x2]
    if not bool(np.any(local)):
        return
    cx1, cy1, cx2, cy2 = bbox
    seed_local = np.zeros_like(local, dtype=bool)
    ix1, iy1 = max(cx1, x1), max(cy1, y1)
    ix2, iy2 = min(cx2, x2), min(cy2, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return
    seed_local[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = full_binary[iy1:iy2, ix1:ix2]
    if not bool(np.any(seed_local)):
        return
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(local.astype(np.uint8), connectivity=8)
    seed_labels = [int(value) for value in np.unique(labels[seed_local]) if int(value) > 0]
    if not seed_labels:
        return
    label = max(seed_labels, key=lambda value: int(np.count_nonzero(labels[seed_local] == value)))
    if label <= 0 or label >= num_labels:
        return
    lx = int(stats[label, cv2.CC_STAT_LEFT])
    ly = int(stats[label, cv2.CC_STAT_TOP])
    lw = int(stats[label, cv2.CC_STAT_WIDTH])
    lh = int(stats[label, cv2.CC_STAT_HEIGHT])
    candidate["source_component_bbox"] = [int(x1 + lx), int(y1 + ly), int(x1 + lx + lw), int(y1 + ly + lh)]
    candidate["source_component_area"] = int(stats[label, cv2.CC_STAT_AREA])
    component_mask = labels[ly:ly + lh, lx:lx + lw] == label
    candidate["source_component_pixel_runs"] = mask_to_relative_runs(component_mask)


def dot_rectangular_terminal_candidate(
    dot: dict[str, Any],
    orientation: str,
    centerline: float,
    edge: float,
    direction: int,
    rep_length: float,
    cfg: DashWireConfig,
) -> dict[str, Any] | None:
    bbox = [int(v) for v in dot.get("bbox", [])]
    if len(bbox) != 4:
        return None
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return None
    span = float(width if orientation == "horizontal" else height)
    cross = float(height if orientation == "horizontal" else width)
    axis_ratio = span / max(1.0, cross)
    dot_centerline = (bbox[1] + bbox[3] - 1) / 2.0 if orientation == "horizontal" else (bbox[0] + bbox[2] - 1) / 2.0
    centerline_delta = abs(float(dot_centerline) - float(centerline))
    gap = float(bbox[0]) - edge if direction > 0 and orientation == "horizontal" else 0.0
    if orientation == "horizontal" and direction < 0:
        gap = edge - float(bbox[2])
    if orientation == "vertical" and direction > 0:
        gap = float(bbox[1]) - edge
    if orientation == "vertical" and direction < 0:
        gap = edge - float(bbox[3])
    if axis_ratio < float(cfg.dot_rectangular_axis_ratio):
        return None
    if centerline_delta > max(1.0, cross * 1.2):
        return None
    if gap < -cross or gap > max(cross * 2.0, rep_length * cfg.terminal_max_length_ratio):
        return None
    if not length_ratio_ok(span, rep_length, cfg.terminal_min_length_ratio, cfg.terminal_max_length_ratio):
        return None
    if orientation == "horizontal":
        points = [[bbox[0], int(round(dot_centerline))], [bbox[2] - 1, int(round(dot_centerline))]]
    else:
        points = [[int(round(dot_centerline)), bbox[1]], [int(round(dot_centerline)), bbox[3] - 1]]
    return {
        "orientation": orientation,
        "points": points,
        "bbox": bbox,
        "span": float(span),
        "width": float(cross),
        "area": int(dot.get("pixel_count", 0)),
        "centerline": float(dot_centerline),
        "projected_fill_ratio": float(dot.get("fill", 0.0)),
        "source": "junction_dot_rectangular_terminal_review",
        "candidate_class": "ultrashort_terminal",
        "reviewed_dot_id": str(dot.get("dot_id", "")),
        "reviewed_dot_metrics": {
            "axis_ratio": round(float(axis_ratio), 3),
            "centerline_delta": round(float(centerline_delta), 3),
            "gap": round(float(gap), 3),
            "fill": float(dot.get("fill", 0.0)),
            "aspect": float(dot.get("aspect", 0.0)),
        },
        "wire_pixel_bbox": bbox,
        "wire_pixel_runs": list(dot.get("pixel_runs", [])),
        "wire_pixel_count": int(dot.get("pixel_count", 0)),
    }


def find_terminal_near_group_end(
    black_binary: np.ndarray,
    dash_binary: np.ndarray,
    dots: list[dict[str, Any]],
    orientation: str,
    centerline: float,
    edge: float,
    direction: int,
    rep_length: float,
    max_gap: float,
    centerline_tolerance: float,
    wire_width: float,
    cfg: DashWireConfig,
    debug_log: list[dict[str, Any]],
    debug_bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any] | None:
    h, w = dash_binary.shape
    search_len = int(round(max_gap + rep_length * cfg.terminal_max_length_ratio + 2.0 * wire_width))
    cross_pad = int(round(centerline_tolerance + 3.0 * wire_width))
    if orientation == "horizontal":
        x1 = int(np.floor(edge - search_len if direction < 0 else edge - wire_width))
        x2 = int(np.ceil(edge + wire_width if direction < 0 else edge + search_len))
        y1 = int(np.floor(centerline - cross_pad))
        y2 = int(np.ceil(centerline + cross_pad + 1))
    else:
        x1 = int(np.floor(centerline - cross_pad))
        x2 = int(np.ceil(centerline + cross_pad + 1))
        y1 = int(np.floor(edge - search_len if direction < 0 else edge - wire_width))
        y2 = int(np.ceil(edge + wire_width if direction < 0 else edge + search_len))
    x1, y1, x2, y2 = clipped_bbox([x1, y1, x2, y2], (h, w))
    min_len = int(round(max(2.0, float(wire_width) * cfg.terminal_min_wire_widths)))
    max_len = int(round(max(min_len, rep_length * cfg.terminal_max_length_ratio)))
    candidates = crop_component_terminal_candidates(
        dash_binary,
        orientation,
        (x1, y1, x2, y2),
        centerline,
        min_len,
        max_len,
        wire_width,
        cfg,
    )
    for candidate in candidates:
        refresh_terminal_source_component(candidate, dash_binary, rep_length, wire_width)
    dot_review_events: list[dict[str, Any]] = []
    for dot in dots:
        if not bbox_intersects(dot.get("bbox", []), [x1, y1, x2, y2]):
            continue
        reviewed = dot_rectangular_terminal_candidate(dot, orientation, centerline, edge, direction, rep_length, cfg)
        dot_review_events.append(
            {
                "dot_id": str(dot.get("dot_id", "")),
                "dot_bbox": list(dot.get("bbox", [])),
                "decision": "accept_as_terminal_candidate" if reviewed is not None else "reject_as_round_dot",
                "reviewed_candidate": segment_debug_summary(reviewed) if reviewed is not None else {},
                "reviewed_dot_metrics": dict(reviewed.get("reviewed_dot_metrics", {})) if reviewed is not None else {},
            }
        )
        if reviewed is not None:
            candidates.append(reviewed)

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_gap = float("inf")
    for candidate in candidates:
        gap = axis_start(candidate) - edge if direction > 0 else edge - axis_end(candidate)
        aspect = float(candidate["span"]) / max(1.0, float(candidate["width"]))
        centerline_delta = abs(float(candidate["centerline"]) - centerline)
        length_ratio = float(candidate["span"]) / max(1.0, rep_length)
        gap_metrics = centerline_gap_metrics(black_binary, edge, direction, orientation, centerline, candidate, cfg)
        clean_corner_cap = terminal_is_clean_corner_cap(candidate, rep_length, wire_width, cfg)
        source_shape_metrics = extension_source_component_shape_metrics(candidate, wire_width, cfg, black_binary)
        t_junction_metrics = candidate_external_t_junction_metrics(black_binary, candidate, direction, wire_width, cfg)
        reason = "eligible"
        if aspect < float(cfg.terminal_min_aspect_ratio) and not clean_corner_cap:
            reason = "terminal_aspect_below_min"
        elif terminal_source_component_is_complex_residual(candidate, cfg) and not clean_corner_cap:
            reason = "terminal_source_component_not_rectangular"
        elif bool(source_shape_metrics.get("letter_like", False)) and not clean_corner_cap:
            reason = "terminal_source_component_letter_like"
        elif bool(t_junction_metrics.get("t_junction_like", False)):
            reason = "terminal_external_endpoint_t_junction_like"
        elif centerline_delta > centerline_tolerance:
            reason = "centerline_delta_exceeds_tolerance"
        elif gap < -0.5 * float(wire_width) or gap >= max_gap:
            reason = "gap_out_of_range"
        elif not bool(gap_metrics.get("gap_clear", False)):
            reason = "centerline_gap_contains_black_pixels"
        elif not length_ratio_ok(float(candidate["span"]), rep_length, cfg.terminal_min_length_ratio, cfg.terminal_max_length_ratio):
            reason = "terminal_length_ratio_out_of_range"
        evaluated.append(
            {
                **segment_debug_summary(candidate),
                "gap": round(float(gap), 3),
                "aspect_ratio": round(float(aspect), 3),
                "centerline_delta": round(float(centerline_delta), 3),
                "length_ratio": round(float(length_ratio), 3),
                "centerline_gap_metrics": gap_metrics,
                "decision": "eligible" if reason == "eligible" else "reject",
                "reason": reason,
                "clean_corner_cap": bool(clean_corner_cap),
                "terminal_source_component_metrics": terminal_source_component_metrics(candidate),
                "terminal_source_residual_metrics": terminal_source_residual_metrics(candidate),
                "terminal_corner_model_metrics": terminal_corner_model_metrics(candidate, rep_length, wire_width, cfg),
                "terminal_source_shape_metrics": source_shape_metrics,
                "terminal_external_t_junction_metrics": t_junction_metrics,
            }
        )
        if reason == "eligible" and gap < best_gap:
            best = candidate
            best_gap = float(gap)
    if evaluated or dot_review_events:
        chosen_bbox = [int(v) for v in best["bbox"]] if best is not None else None
        for item in evaluated:
            if chosen_bbox is not None and item["bbox"] == chosen_bbox:
                item["decision"] = "accept"
                item["reason"] = "nearest_eligible_terminal_candidate"
            elif item["decision"] == "eligible":
                item["decision"] = "reject"
                item["reason"] = "eligible_terminal_but_farther_than_chosen"
        if debug_bbox is not None and bbox_intersects([x1, y1, x2, y2], debug_bbox):
            debug_log.append(
                {
                    "event": "external_terminal_search",
                    "direction": "forward" if direction > 0 else "backward",
                    "search_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "edge": round(float(edge), 3),
                    "orientation": orientation,
                    "centerline": round(float(centerline), 3),
                    "representative_length": round(float(rep_length), 3),
                    "chosen_bbox": chosen_bbox,
                    "candidate_decisions": evaluated,
                    "dot_review_events": dot_review_events,
                }
            )
    return best


def terminal_source_component_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    source_bbox = [int(v) for v in candidate.get("source_component_bbox", [])]
    if len(bbox) != 4 or len(source_bbox) != 4:
        return {
            "has_source_component": False,
            "source_axis_ratio": 1.0,
            "source_area_ratio": 1.0,
            "candidate_pixel_fraction": 1.0,
        }
    orientation = str(candidate.get("orientation", ""))
    candidate_span = max(1.0, float(candidate.get("span", 0.0)))
    candidate_width = max(1.0, float(candidate.get("width", 0.0)))
    source_span = float(source_bbox[2] - source_bbox[0]) if orientation == "horizontal" else float(source_bbox[3] - source_bbox[1])
    source_cross = float(source_bbox[3] - source_bbox[1]) if orientation == "horizontal" else float(source_bbox[2] - source_bbox[0])
    source_area = max(1, int(candidate.get("source_component_area", 0)))
    candidate_pixels = max(1, int(candidate.get("wire_pixel_count", candidate.get("area", 0))))
    source_axis_ratio = float(source_span) / max(1.0, candidate_span)
    source_area_ratio = float(source_area) / max(1.0, float(candidate_pixels))
    return {
        "has_source_component": True,
        "source_component_bbox": source_bbox,
        "source_component_area": int(source_area),
        "candidate_axis_span_pixels": round(float(candidate_span), 3),
        "candidate_cross_span_pixels": round(float(candidate_width), 3),
        "source_axis_span_pixels": round(float(source_span), 3),
        "source_cross_span_pixels": round(float(source_cross), 3),
        "source_axis_ratio": round(float(source_axis_ratio), 3),
        "source_cross_ratio": round(float(source_cross) / max(1.0, float(candidate_width)), 3),
        "source_area_ratio": round(float(source_area_ratio), 3),
        "candidate_pixel_fraction": round(float(candidate_pixels) / max(1.0, float(source_area)), 3),
    }


def terminal_source_residual_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    source_bbox = [int(v) for v in candidate.get("source_component_bbox", [])]
    source_runs = candidate.get("source_component_pixel_runs", [])
    if len(bbox) != 4 or len(source_bbox) != 4 or not isinstance(source_runs, list) or not source_runs:
        return {
            "has_source_component_pixels": False,
            "residual_area": 0,
            "residual_component_count": 0,
            "residual_fill_ratio": 1.0,
            "residual_rectangular": True,
        }
    sx1, sy1, sx2, sy2 = source_bbox
    source_mask = relative_runs_to_mask(source_runs, (max(0, sy2 - sy1), max(0, sx2 - sx1)))
    if not bool(np.any(source_mask)):
        return {
            "has_source_component_pixels": False,
            "residual_area": 0,
            "residual_component_count": 0,
            "residual_fill_ratio": 1.0,
            "residual_rectangular": True,
        }
    tx1, ty1, tx2, ty2 = bbox
    ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
    ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
    if ix2 > ix1 and iy2 > iy1:
        terminal_mask = relative_runs_to_mask(
            candidate.get("wire_pixel_runs", []),
            (max(0, ty2 - ty1), max(0, tx2 - tx1)),
        )
        source_mask[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] &= ~terminal_mask[
            iy1 - ty1:iy2 - ty1,
            ix1 - tx1:ix2 - tx1,
        ]
    residual_area = int(np.count_nonzero(source_mask))
    if residual_area <= 0:
        return {
            "has_source_component_pixels": True,
            "residual_area": 0,
            "residual_component_count": 0,
            "residual_fill_ratio": 1.0,
            "residual_rectangular": True,
        }
    ys, xs = np.where(source_mask)
    rx1, ry1 = int(xs.min()), int(ys.min())
    rx2, ry2 = int(xs.max() + 1), int(ys.max() + 1)
    residual_bbox_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    fill_ratio = float(residual_area) / float(residual_bbox_area)
    num_labels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(source_mask.astype(np.uint8), connectivity=8)
    component_count = max(0, int(num_labels) - 1)
    return {
        "has_source_component_pixels": True,
        "residual_bbox": [int(sx1 + rx1), int(sy1 + ry1), int(sx1 + rx2), int(sy1 + ry2)],
        "residual_area": int(residual_area),
        "residual_component_count": int(component_count),
        "residual_fill_ratio": round(float(fill_ratio), 3),
        "residual_rectangular": bool(
            component_count <= 1
            and fill_ratio >= 0.68
        ),
    }


def terminal_corner_model_metrics(
    candidate: dict[str, Any],
    rep_length: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in candidate.get("bbox", [])]
    source_bbox = [int(v) for v in candidate.get("source_component_bbox", [])]
    source_runs = candidate.get("source_component_pixel_runs", [])
    if len(bbox) != 4 or len(source_bbox) != 4 or not isinstance(source_runs, list) or not source_runs:
        return {
            "corner_model_ok": False,
            "reason": "missing_source_component_pixels",
        }
    sx1, sy1, sx2, sy2 = source_bbox
    source_mask = relative_runs_to_mask(source_runs, (max(0, sy2 - sy1), max(0, sx2 - sx1)))
    if not bool(np.any(source_mask)):
        return {
            "corner_model_ok": False,
            "reason": "empty_source_component",
        }
    tx1, ty1, tx2, ty2 = bbox
    terminal_local_bbox = [tx1 - sx1, ty1 - sy1, tx2 - sx1, ty2 - sy1]
    terminal_mask = relative_runs_to_mask(
        candidate.get("wire_pixel_runs", []),
        (max(0, ty2 - ty1), max(0, tx2 - tx1)),
    )
    ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
    ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
    residual_mask = source_mask.copy()
    terminal_pixels_removed = 0
    if ix2 > ix1 and iy2 > iy1:
        terminal_slice = terminal_mask[iy1 - ty1:iy2 - ty1, ix1 - tx1:ix2 - tx1]
        terminal_pixels_removed = int(np.count_nonzero(residual_mask[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] & terminal_slice))
        residual_mask[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] &= ~terminal_slice

    orientation = str(candidate.get("orientation", ""))
    orthogonal_orientation = "vertical" if orientation == "horizontal" else "horizontal"
    min_len = int(round(max(
        float(rep_length) * float(cfg.terminal_corner_cap_orthogonal_min_length_ratio),
        float(wire_width) * float(cfg.terminal_corner_cap_orthogonal_min_wire_widths),
    )))
    max_len = int(round(max(float(min_len), float(rep_length) * float(cfg.terminal_corner_cap_orthogonal_max_length_ratio))))
    orthogonal_segments = extract_axis_rectangles(
        residual_mask,
        orthogonal_orientation,
        max(1, min_len),
        max(1, max_len),
        wire_width,
        cfg,
    )
    touch_gap = max(1.0, float(wire_width))
    touching_segments = [
        segment
        for segment in orthogonal_segments
        if expanded_bboxes_touch(terminal_local_bbox, segment.get("bbox", []), touch_gap)
    ]
    if not touching_segments:
        return {
            "corner_model_ok": False,
            "reason": "missing_touching_orthogonal_rectangle",
            "terminal_pixels_removed": int(terminal_pixels_removed),
            "orthogonal_candidate_count": int(len(orthogonal_segments)),
            "orthogonal_min_length": int(min_len),
            "orthogonal_max_length": int(max_len),
        }
    best_orthogonal = max(
        touching_segments,
        key=lambda segment: (float(segment.get("span", 0.0)) * float(segment.get("width", 0.0)), float(segment.get("span", 0.0))),
    )
    ox1, oy1, ox2, oy2 = [int(v) for v in best_orthogonal["bbox"]]
    orthogonal_mask = relative_runs_to_mask(
        best_orthogonal.get("wire_pixel_runs", []),
        (max(0, oy2 - oy1), max(0, ox2 - ox1)),
    )
    residual_after_two = residual_mask.copy()
    if ox2 > ox1 and oy2 > oy1:
        residual_after_two[oy1:oy2, ox1:ox2] &= ~orthogonal_mask
    leftover_pixels = int(np.count_nonzero(residual_after_two))
    max_leftover = int(max(
        float(cfg.terminal_corner_cap_residual_max_spur_pixels),
        float(wire_width) * float(cfg.terminal_corner_cap_residual_max_spur_wire_widths),
    ))
    if leftover_pixels > 0:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(residual_after_two.astype(np.uint8), connectivity=8)
        component_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
        max_component_area = max(component_areas) if component_areas else 0
        component_count = len(component_areas)
    else:
        max_component_area = 0
        component_count = 0
    corner_model_ok = leftover_pixels <= max_leftover and max_component_area <= max_leftover
    return {
        "corner_model_ok": bool(corner_model_ok),
        "reason": "terminal_plus_orthogonal_rectangle_explain_component" if corner_model_ok else "extra_residual_pixels_after_terminal_and_orthogonal_rectangle",
        "terminal_pixels_removed": int(terminal_pixels_removed),
        "orthogonal_bbox": [int(sx1 + ox1), int(sy1 + oy1), int(sx1 + ox2), int(sy1 + oy2)],
        "orthogonal_span": round(float(best_orthogonal.get("span", 0.0)), 3),
        "orthogonal_width": round(float(best_orthogonal.get("width", 0.0)), 3),
        "orthogonal_projected_fill_ratio": round(float(best_orthogonal.get("projected_fill_ratio", 0.0)), 3),
        "orthogonal_candidate_count": int(len(orthogonal_segments)),
        "touching_orthogonal_candidate_count": int(len(touching_segments)),
        "leftover_pixels": int(leftover_pixels),
        "leftover_component_count": int(component_count),
        "leftover_max_component_area": int(max_component_area),
        "leftover_pixel_limit": int(max_leftover),
    }


def terminal_source_component_is_complex_residual(candidate: dict[str, Any], cfg: DashWireConfig) -> bool:
    if str(candidate.get("source", "")) != "external_endpoint_terminal_search":
        return False
    metrics = terminal_source_component_metrics(candidate)
    return bool(metrics.get("has_source_component", False)) and (
        (
            float(metrics.get("source_axis_ratio", 1.0)) > float(cfg.terminal_source_max_axis_ratio)
            and float(metrics.get("source_area_ratio", 1.0)) > float(cfg.terminal_source_max_area_ratio)
        )
        or (
            float(metrics.get("source_cross_ratio", 1.0)) > float(cfg.candidate_source_complex_cross_ratio)
            and float(metrics.get("source_area_ratio", 1.0)) > float(cfg.candidate_source_complex_area_ratio)
        )
    )


def terminal_is_clean_corner_cap(
    candidate: dict[str, Any],
    rep_length: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> bool:
    if str(candidate.get("source", "")) != "external_endpoint_terminal_search":
        return False
    metrics = terminal_source_component_metrics(candidate)
    if not bool(metrics.get("has_source_component", False)):
        return False
    source_axis = float(metrics.get("source_axis_span_pixels", 0.0))
    source_cross = float(metrics.get("source_cross_span_pixels", 0.0))
    candidate_axis = float(metrics.get("candidate_axis_span_pixels", candidate.get("span", 0.0)))
    corner_model = terminal_corner_model_metrics(candidate, rep_length, wire_width, cfg)
    max_axis = max(
        candidate_axis + float(wire_width) * float(cfg.terminal_corner_cap_max_axis_extra_wire_widths),
        float(wire_width) * 3.0,
    )
    min_perpendicular = max(
        float(rep_length) * float(cfg.terminal_corner_cap_min_perpendicular_length_ratio),
        float(wire_width) * 8.0,
    )
    return (
        source_axis <= max_axis
        and source_cross >= min_perpendicular
        and bool(corner_model.get("corner_model_ok", False))
    )


def candidate_source_component_is_complex_residual(candidate: dict[str, Any], cfg: DashWireConfig) -> bool:
    metrics = terminal_source_component_metrics(candidate)
    return (
        bool(metrics.get("has_source_component", False))
        and (
            float(metrics.get("source_area_ratio", 1.0)) > float(cfg.candidate_source_huge_area_ratio)
            or float(metrics.get("source_axis_ratio", 1.0)) > float(cfg.candidate_source_huge_axis_ratio)
            or (
                float(metrics.get("source_cross_ratio", 1.0)) > float(cfg.candidate_source_complex_cross_ratio)
                and float(metrics.get("source_area_ratio", 1.0)) > float(cfg.candidate_source_complex_area_ratio)
            )
        )
    )


def classify_group_endpoints(members: list[dict[str, Any]], group_id: str) -> None:
    ordered = sorted(members, key=lambda item: (axis_start(item), axis_end(item)))
    for index, member in enumerate(ordered):
        start_role = "external_end" if index == 0 else "internal_end"
        end_role = "external_end" if index == len(ordered) - 1 else "internal_end"
        if str(member.get("orientation")) == "horizontal":
            start_point = [float(member["bbox"][0]), float(member["centerline"])]
            end_point = [float(member["bbox"][2] - 1), float(member["centerline"])]
        else:
            start_point = [float(member["centerline"]), float(member["bbox"][1])]
            end_point = [float(member["centerline"]), float(member["bbox"][3] - 1)]
        member["endpoint_roles"] = [start_role, end_role]
        member["endpoints"] = [
            {"point": start_point, "role": start_role, "dash_group_id": group_id, "endpoint_index": 0},
            {"point": end_point, "role": end_role, "dash_group_id": group_id, "endpoint_index": 1},
        ]


def apply_external_orthogonal_fallback(
    binary_without_dots: np.ndarray,
    members: list[dict[str, Any]],
    group: dict[str, Any],
    wire_width: float,
    cfg: DashWireConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    occupied = np.zeros(binary_without_dots.shape, dtype=bool)
    for member in members:
        bbox = [int(v) for v in member.get("wire_pixel_bbox", member.get("bbox", []))]
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = clipped_bbox(bbox, binary_without_dots.shape)
        if x2 <= x1 or y2 <= y1:
            continue
        local = relative_runs_to_mask(member.get("wire_pixel_runs", []), (y2 - y1, x2 - x1))
        occupied[y1:y2, x1:x2] |= local

    probe_len = int(max(cfg.external_orthogonal_probe_min_px, round(float(wire_width) * cfg.external_orthogonal_probe_wire_widths)))
    max_run = int(max(1, round(float(wire_width) * cfg.external_orthogonal_max_run_wire_widths)))
    for member in members:
        for endpoint_index, endpoint in enumerate(member.get("endpoints", [])):
            if endpoint.get("role") != "external_end":
                continue
            px, py = [float(v) for v in endpoint.get("point", [0.0, 0.0])]
            found_runs: list[dict[str, Any]] = []
            orientation = str(member.get("orientation", ""))
            if orientation == "horizontal":
                x = int(round(px))
                y0 = int(round(py))
                for sign in (-1, 1):
                    ys: list[int] = []
                    for offset in range(1, probe_len + 1):
                        y = y0 + sign * offset
                        if y < 0 or y >= binary_without_dots.shape[0] or x < 0 or x >= binary_without_dots.shape[1]:
                            break
                        if binary_without_dots[y, x] and not occupied[y, x]:
                            ys.append(y)
                        elif ys:
                            break
                    if 0 < len(ys) <= max_run:
                        found_runs.append({"direction": [0, sign], "length_pixels": len(ys), "pixels": [[x, y] for y in ys]})
            elif orientation == "vertical":
                x0 = int(round(px))
                y = int(round(py))
                for sign in (-1, 1):
                    xs: list[int] = []
                    for offset in range(1, probe_len + 1):
                        x = x0 + sign * offset
                        if y < 0 or y >= binary_without_dots.shape[0] or x < 0 or x >= binary_without_dots.shape[1]:
                            break
                        if binary_without_dots[y, x] and not occupied[y, x]:
                            xs.append(x)
                        elif xs:
                            break
                    if 0 < len(xs) <= max_run:
                        found_runs.append({"direction": [sign, 0], "length_pixels": len(xs), "pixels": [[x, y] for x in xs]})
            event = {
                "group_id": str(group.get("id", "")),
                "segment_id": str(member.get("segment_id", "")),
                "endpoint_index": int(endpoint_index),
                "point": [round(float(px), 3), round(float(py), 3)],
                "probe_length_pixels": int(probe_len),
                "max_orthogonal_run_pixels": int(max_run),
                "found_orthogonal_runs": found_runs,
                "decision": "orthogonal_pixels_found" if found_runs else "no_orthogonal_pixels_found",
            }
            endpoint["external_orthogonal_fallback"] = event
            events.append(event)
    return events


def member_full_mask(member: dict[str, Any], shape: tuple[int, int]) -> tuple[list[int], np.ndarray] | None:
    bbox = [int(v) for v in member.get("wire_pixel_bbox", member.get("bbox", []))]
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = clipped_bbox(bbox, shape)
    if x2 <= x1 or y2 <= y1:
        return None
    local = relative_runs_to_mask(member.get("wire_pixel_runs", []), (y2 - y1, x2 - x1))
    if not bool(np.any(local)):
        return None
    return [x1, y1, x2, y2], local


def build_group_occupancy(
    group_member_refs: dict[str, list[dict[str, Any]]],
    shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_group: dict[str, np.ndarray] = {}
    all_occupied = np.zeros(shape, dtype=bool)
    for group_id, members in group_member_refs.items():
        occupied = np.zeros(shape, dtype=bool)
        for member in members:
            result = member_full_mask(member, shape)
            if result is None:
                continue
            bbox, local = result
            x1, y1, x2, y2 = bbox
            occupied[y1:y2, x1:x2] |= local
        by_group[str(group_id)] = occupied
        all_occupied |= occupied
    return by_group, all_occupied


def endpoint_touches_other_dash_group(
    endpoint: dict[str, Any],
    group_id: str,
    group_occupancy: dict[str, np.ndarray],
    wire_width: float,
    cfg: DashWireConfig,
) -> tuple[bool, list[str]]:
    point = endpoint.get("point", [0.0, 0.0])
    if len(point) != 2:
        return False, []
    px, py = int(round(float(point[0]))), int(round(float(point[1])))
    radius = int(max(1, round(float(wire_width) * float(cfg.external_endpoint_other_group_touch_wire_widths))))
    hit_group_ids: list[str] = []
    for other_group_id, occupied in group_occupancy.items():
        if str(other_group_id) == str(group_id):
            continue
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(occupied.shape[1], px + radius + 1)
        y2 = min(occupied.shape[0], py + radius + 1)
        if x2 > x1 and y2 > y1 and bool(np.any(occupied[y1:y2, x1:x2])):
            hit_group_ids.append(str(other_group_id))
    return bool(hit_group_ids), hit_group_ids


def endpoint_cross_width(
    binary: np.ndarray,
    orientation: str,
    axis_coord: int,
    centerline: float,
    probe: int,
) -> int:
    if orientation == "horizontal":
        x = int(axis_coord)
        if x < 0 or x >= binary.shape[1]:
            return 0
        y = int(round(centerline))
        y1 = max(0, y - int(probe))
        y2 = min(binary.shape[0], y + int(probe) + 1)
        if y2 <= y1:
            return 0
        return int(np.count_nonzero(binary[y1:y2, x]))
    if orientation == "vertical":
        y = int(axis_coord)
        if y < 0 or y >= binary.shape[0]:
            return 0
        x = int(round(centerline))
        x1 = max(0, x - int(probe))
        x2 = min(binary.shape[1], x + int(probe) + 1)
        if x2 <= x1:
            return 0
        return int(np.count_nonzero(binary[y, x1:x2]))
    return 0


def crop_segment_axis_side_to_original_pixels(
    member: dict[str, Any],
    crop_bbox: Sequence[int],
    image_shape: tuple[int, int],
) -> bool:
    original = member_full_mask(member, image_shape)
    if original is None:
        return False
    old_bbox, old_mask = original
    ox1, oy1, ox2, oy2 = old_bbox
    cx1, cy1, cx2, cy2 = clipped_bbox(crop_bbox, image_shape)
    ix1, iy1 = max(ox1, cx1), max(oy1, cy1)
    ix2, iy2 = min(ox2, cx2), min(oy2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    cropped = old_mask[iy1 - oy1:iy2 - oy1, ix1 - ox1:ix2 - ox1]
    if not bool(np.any(cropped)):
        return False
    ys, xs = np.where(cropped)
    tx1, ty1 = int(xs.min()), int(ys.min())
    tx2, ty2 = int(xs.max() + 1), int(ys.max() + 1)
    tight = cropped[ty1:ty2, tx1:tx2]
    nx1, ny1, nx2, ny2 = ix1 + tx1, iy1 + ty1, ix1 + tx2, iy1 + ty2
    orientation = str(member.get("orientation", ""))
    member["bbox"] = [int(nx1), int(ny1), int(nx2), int(ny2)]
    member["wire_pixel_bbox"] = [int(nx1), int(ny1), int(nx2), int(ny2)]
    member["wire_pixel_runs"] = mask_to_relative_runs(tight)
    member["wire_pixel_count"] = int(np.count_nonzero(tight))
    if orientation == "horizontal":
        member["span"] = float(nx2 - nx1)
        member["width"] = float(ny2 - ny1)
        centerline = (float(ny1) + float(ny2) - 1.0) / 2.0
        member["centerline"] = float(centerline)
        member["points"] = [[int(nx1), int(round(centerline))], [int(nx2 - 1), int(round(centerline))]]
    elif orientation == "vertical":
        member["span"] = float(ny2 - ny1)
        member["width"] = float(nx2 - nx1)
        centerline = (float(nx1) + float(nx2) - 1.0) / 2.0
        member["centerline"] = float(centerline)
        member["points"] = [[int(round(centerline)), int(ny1)], [int(round(centerline)), int(ny2 - 1)]]
    return True


def retreat_member_external_endpoint(
    binary_without_dots: np.ndarray,
    member: dict[str, Any],
    endpoint_index: int,
    representative_width: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    bbox = [int(v) for v in member.get("bbox", [])]
    if len(bbox) != 4:
        return {"retreated": False, "decision": "missing_bbox"}
    orientation = str(member.get("orientation", ""))
    if orientation not in {"horizontal", "vertical"}:
        return {"retreated": False, "decision": "unknown_orientation"}
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return {"retreated": False, "decision": "empty_bbox"}

    centerline = float(member.get("centerline", 0.0))
    probe = int(max(2, round(float(wire_width) * float(cfg.external_endpoint_retreat_probe_wire_widths))))
    normal_limit = max(
        1,
        int(math.ceil(float(representative_width) * float(cfg.external_endpoint_retreat_max_width_ratio))),
    )
    consecutive_needed = int(max(1, cfg.external_endpoint_retreat_consecutive_normal))
    min_span = max(1.0, float(wire_width) * float(cfg.external_endpoint_retreat_min_span_wire_widths))

    if orientation == "horizontal":
        axis_values = range(x1, x2) if endpoint_index == 0 else range(x2 - 1, x1 - 1, -1)
    else:
        axis_values = range(y1, y2) if endpoint_index == 0 else range(y2 - 1, y1 - 1, -1)

    widths: list[dict[str, Any]] = []
    consecutive: list[int] = []
    stable_axis: int | None = None
    first_axis = None
    for axis in axis_values:
        if first_axis is None:
            first_axis = int(axis)
        cross = endpoint_cross_width(binary_without_dots, orientation, int(axis), centerline, probe)
        normal = 0 < cross <= normal_limit
        widths.append({"axis": int(axis), "cross_width": int(cross), "normal": bool(normal)})
        if normal:
            consecutive.append(int(axis))
            if len(consecutive) >= consecutive_needed:
                stable_axis = int(consecutive[0])
                break
        else:
            consecutive = []

    if first_axis is None:
        return {"retreated": False, "decision": "empty_axis_scan"}
    first_width = widths[0]["cross_width"] if widths else 0
    if first_width <= normal_limit:
        return {
            "retreated": False,
            "decision": "external_endpoint_width_already_normal",
            "normal_width_limit": int(normal_limit),
            "first_cross_width": int(first_width),
            "scan_widths": widths[:24],
        }
    if stable_axis is None:
        return {
            "retreated": False,
            "decision": "no_stable_normal_width_found",
            "normal_width_limit": int(normal_limit),
            "first_cross_width": int(first_width),
            "scan_widths": widths[:48],
        }

    old_bbox = list(bbox)
    if orientation == "horizontal":
        if endpoint_index == 0:
            new_crop = [stable_axis, y1, x2, y2]
        else:
            new_crop = [x1, y1, stable_axis + 1, y2]
    else:
        if endpoint_index == 0:
            new_crop = [x1, stable_axis, x2, y2]
        else:
            new_crop = [x1, y1, x2, stable_axis + 1]
    new_span = float(new_crop[2] - new_crop[0]) if orientation == "horizontal" else float(new_crop[3] - new_crop[1])
    if new_span < min_span:
        return {
            "retreated": False,
            "decision": "retreat_would_make_member_too_short",
            "old_bbox": old_bbox,
            "proposed_crop_bbox": [int(v) for v in new_crop],
            "proposed_span": round(float(new_span), 3),
            "min_span": round(float(min_span), 3),
            "normal_width_limit": int(normal_limit),
            "scan_widths": widths[:48],
        }
    changed = crop_segment_axis_side_to_original_pixels(member, new_crop, binary_without_dots.shape)
    return {
        "retreated": bool(changed),
        "decision": "retreated_to_stable_normal_width" if changed else "retreat_crop_removed_all_pixels",
        "old_bbox": old_bbox,
        "new_bbox": [int(v) for v in member.get("bbox", old_bbox)],
        "endpoint_index": int(endpoint_index),
        "stable_axis": int(stable_axis),
        "normal_width_limit": int(normal_limit),
        "first_cross_width": int(first_width),
        "scan_widths": widths[:48],
    }


def apply_external_endpoint_retreat(
    binary_without_dots: np.ndarray,
    groups: list[dict[str, Any]],
    group_member_refs: dict[str, list[dict[str, Any]]],
    wire_width: float,
    cfg: DashWireConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    group_occupancy, _all_occupied = build_group_occupancy(group_member_refs, binary_without_dots.shape)
    for group in groups:
        group_id = str(group.get("id", ""))
        members = group_member_refs.get(group_id, [])
        if not members:
            continue
        classify_group_endpoints(members, group_id)
        rep_width_value = float(group.get("representative_width", 0.0)) or representative_width(members, wire_width)
        for member in sorted(members, key=lambda item: (axis_start(item), axis_end(item))):
            for endpoint_index, endpoint in enumerate(member.get("endpoints", [])):
                if endpoint.get("role") != "external_end":
                    continue
                touches_other, hit_group_ids = endpoint_touches_other_dash_group(endpoint, group_id, group_occupancy, wire_width, cfg)
                if touches_other:
                    event = {
                        "group_id": group_id,
                        "segment_id": str(member.get("segment_id", "")),
                        "endpoint_index": int(endpoint_index),
                        "point": [round(float(v), 3) for v in endpoint.get("point", [0.0, 0.0])],
                        "decision": "skip_connected_to_other_dash_group",
                        "connected_dash_group_ids": hit_group_ids,
                    }
                    endpoint["external_endpoint_retreat"] = event
                    events.append(event)
                    continue
                retreat = retreat_member_external_endpoint(
                    binary_without_dots,
                    member,
                    endpoint_index,
                    rep_width_value,
                    wire_width,
                    cfg,
                )
                event = {
                    "group_id": group_id,
                    "segment_id": str(member.get("segment_id", "")),
                    "endpoint_index": int(endpoint_index),
                    "point": [round(float(v), 3) for v in endpoint.get("point", [0.0, 0.0])],
                    **retreat,
                }
                endpoint["external_endpoint_retreat"] = event
                events.append(event)
        classify_group_endpoints(members, group_id)
    return events


def group_member_axis_index(member: dict[str, Any], members: list[dict[str, Any]]) -> int:
    ordered = sorted(members, key=lambda item: (axis_start(item), axis_end(item)))
    member_id = str(member.get("segment_id", ""))
    for index, item in enumerate(ordered):
        if str(item.get("segment_id", "")) == member_id:
            return int(index)
    return -1


def width_matches_cross_group_reference(width: float, reference_width: float, cfg: DashWireConfig) -> bool:
    reference_width = max(1.0, float(reference_width))
    ratio = max(1.0, float(width)) / reference_width
    return (
        float(cfg.cross_group_single_overrun_width_min_ratio)
        <= ratio
        <= float(cfg.cross_group_single_overrun_width_max_ratio)
    )


def member_overruns_orthogonal_group_band(
    member: dict[str, Any],
    candidate_group: dict[str, Any],
    other_group: dict[str, Any],
    wire_width: float,
    cfg: DashWireConfig,
    member_axis_index: int,
    member_count: int,
) -> dict[str, Any] | None:
    candidate_orientation = str(candidate_group.get("orientation", ""))
    other_orientation = str(other_group.get("orientation", ""))
    if {candidate_orientation, other_orientation} != {"horizontal", "vertical"}:
        return None
    mb = [int(v) for v in member.get("bbox", [])]
    ob = [int(v) for v in other_group.get("bbox", [])]
    if len(mb) != 4 or len(ob) != 4:
        return None
    if member_axis_index not in {0, max(0, int(member_count) - 1)}:
        return None
    endpoint_tol = max(
        2.0,
        float(wire_width) * float(cfg.cross_group_single_overrun_endpoint_tolerance_wire_widths),
    )
    if candidate_orientation == "vertical" and other_orientation == "horizontal":
        cross_axis = float(member.get("centerline", candidate_group.get("centerline", 0.0)))
        if cross_axis < float(ob[0]) or cross_axis > float(ob[2]):
            return None
        if min(abs(cross_axis - float(ob[0])), abs(cross_axis - float(ob[2]))) <= endpoint_tol:
            return None
        overlaps_band = mb[1] < ob[3] and mb[3] > ob[1]
        before_overrun = max(0, ob[1] - mb[1])
        after_overrun = max(0, mb[3] - ob[3])
        outward_overrun = before_overrun if member_axis_index == 0 else after_overrun
        max_overrun = max(before_overrun, after_overrun)
        if not (overlaps_band and outward_overrun > float(cfg.cross_group_single_overrun_min_overrun_px)):
            return None
        return {
            "other_band_bbox": ob,
            "candidate_cross_axis": round(float(cross_axis), 3),
            "min_significant_overrun_pixels": float(cfg.cross_group_single_overrun_min_overrun_px),
            "outward_overrun_pixels": int(outward_overrun),
            "member_axis_index": int(member_axis_index),
            "member_count": int(member_count),
            "overrun_pixels": {
                "before_band": int(before_overrun),
                "after_band": int(after_overrun),
                "max": int(max_overrun),
            },
            "overrun_sides": {
                "before_band": bool(before_overrun > 0),
                "after_band": bool(after_overrun > 0),
            },
        }
    if candidate_orientation == "horizontal" and other_orientation == "vertical":
        cross_axis = float(member.get("centerline", candidate_group.get("centerline", 0.0)))
        if cross_axis < float(ob[1]) or cross_axis > float(ob[3]):
            return None
        if min(abs(cross_axis - float(ob[1])), abs(cross_axis - float(ob[3]))) <= endpoint_tol:
            return None
        overlaps_band = mb[0] < ob[2] and mb[2] > ob[0]
        before_overrun = max(0, ob[0] - mb[0])
        after_overrun = max(0, mb[2] - ob[2])
        outward_overrun = before_overrun if member_axis_index == 0 else after_overrun
        max_overrun = max(before_overrun, after_overrun)
        if not (overlaps_band and outward_overrun > float(cfg.cross_group_single_overrun_min_overrun_px)):
            return None
        return {
            "other_band_bbox": ob,
            "candidate_cross_axis": round(float(cross_axis), 3),
            "min_significant_overrun_pixels": float(cfg.cross_group_single_overrun_min_overrun_px),
            "outward_overrun_pixels": int(outward_overrun),
            "member_axis_index": int(member_axis_index),
            "member_count": int(member_count),
            "overrun_pixels": {
                "before_band": int(before_overrun),
                "after_band": int(after_overrun),
                "max": int(max_overrun),
            },
            "overrun_sides": {
                "before_band": bool(before_overrun > 0),
                "after_band": bool(after_overrun > 0),
            },
        }
    return None


def cross_group_single_overrun_filter(
    black_binary: np.ndarray,
    groups: list[dict[str, Any]],
    group_member_refs: dict[str, list[dict[str, Any]]],
    text_height: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    reject_group_ids: set[str] = set()
    modified_group_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    groups_by_id = {str(group.get("id", "")): group for group in groups}
    for candidate_group in groups:
        candidate_group_id = str(candidate_group.get("id", ""))
        if (
            not candidate_group_id
            or candidate_group_id in reject_group_ids
            or candidate_group_id in modified_group_ids
        ):
            continue
        candidate_members = sorted(
            group_member_refs.get(candidate_group_id, []),
            key=lambda item: (axis_start(item), axis_end(item)),
        )
        if len(candidate_members) < int(cfg.group_min_regular_members):
            continue
        candidate_orientation = str(candidate_group.get("orientation", ""))
        for other_group in groups:
            other_group_id = str(other_group.get("id", ""))
            if (
                not other_group_id
                or other_group_id == candidate_group_id
                or other_group_id in reject_group_ids
                or str(other_group.get("orientation", "")) == candidate_orientation
            ):
                continue
            other_members = group_member_refs.get(other_group_id, [])
            if len(other_members) < int(cfg.group_min_regular_members):
                continue
            cb = [int(v) for v in candidate_group.get("bbox", [])]
            ob = [int(v) for v in other_group.get("bbox", [])]
            if len(cb) != 4 or len(ob) != 4 or not bbox_intersects(cb, ob):
                continue
            overrun_members: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for member in candidate_members:
                member_index = group_member_axis_index(member, candidate_members)
                if member_index not in {0, len(candidate_members) - 1}:
                    continue
                metrics = member_overruns_orthogonal_group_band(
                    member,
                    candidate_group,
                    other_group,
                    wire_width,
                    cfg,
                    member_index,
                    len(candidate_members),
                )
                if metrics is not None:
                    overrun_members.append((member, metrics))
            if len(overrun_members) != 1:
                continue
            overrun_member, metrics = overrun_members[0]
            overrun_index = group_member_axis_index(overrun_member, candidate_members)
            if overrun_index not in {0, len(candidate_members) - 1}:
                continue
            member_width = float(overrun_member.get("width", 0.0))
            other_width = float(other_group.get("representative_width", 0.0)) or representative_width(other_members, wire_width)
            if width_matches_cross_group_reference(member_width, other_width, cfg):
                continue
            overrun_shape_metrics = extension_source_component_shape_metrics(overrun_member, wire_width, cfg, black_binary)
            if bool(overrun_shape_metrics.get("clean_l_corner_like", False)):
                events.append(
                    {
                        "event": "cross_group_single_member_overrun_reviewed",
                        "candidate_group_id": candidate_group_id,
                        "candidate_group_bbox": [int(v) for v in cb],
                        "candidate_group_orientation": candidate_orientation,
                        "reference_group_id": other_group_id,
                        "reference_group_bbox": [int(v) for v in ob],
                        "reference_group_orientation": str(other_group.get("orientation", "")),
                        "overrun_member_id": str(overrun_member.get("segment_id", "")),
                        "overrun_member_bbox": [int(v) for v in overrun_member.get("bbox", [])],
                        "overrun_member_axis_index": int(overrun_index),
                        "overrun_member_count": int(len(overrun_members)),
                        "member_width": round(float(member_width), 3),
                        "reference_group_width": round(float(other_width), 3),
                        "width_ratio": round(float(max(1.0, member_width) / max(1.0, other_width)), 3),
                        **metrics,
                        "overrun_source_shape_metrics": overrun_shape_metrics,
                        "decision": "keep_clean_l_corner_overrun_member",
                        "reason": "overrun_member_is_clean_l_corner_component",
                    }
                )
                continue
            remaining_members = [
                member
                for member in candidate_members
                if str(member.get("segment_id", "")) != str(overrun_member.get("segment_id", ""))
            ]
            regular_ids = set(str(member_id) for member_id in candidate_group.get("regular_member_ids", []))
            terminal_ids = set(str(member_id) for member_id in candidate_group.get("terminal_member_ids", []))
            remaining_regular = [
                member for member in remaining_members if str(member.get("segment_id", "")) in regular_ids
            ]
            removal_ok = False
            removal_connection_quality: dict[str, Any] = {}
            removal_consistency: dict[str, Any] = {}
            removal_failure_reason = ""
            if len(remaining_regular) < int(cfg.group_min_regular_members):
                removal_failure_reason = "remaining_regular_member_count_below_min"
            else:
                removal_connection_quality = dash_group_connection_quality(black_binary, remaining_regular, text_height, wire_width, cfg)
                if not bool(removal_connection_quality.get("connection_quality_ok", False)):
                    removal_failure_reason = "remaining_group_connection_quality_below_min"
                else:
                    removal_consistency = dash_group_consistency_metrics(remaining_regular, removal_connection_quality, cfg)
                    if not bool(removal_consistency.get("group_consistency_ok", False)):
                        removal_failure_reason = "remaining_group_consistency_below_min"
                    else:
                        removal_ok = True
            event = {
                "event": "cross_group_single_member_overrun_reviewed",
                "candidate_group_id": candidate_group_id,
                "candidate_group_bbox": [int(v) for v in cb],
                "candidate_group_orientation": candidate_orientation,
                "reference_group_id": other_group_id,
                "reference_group_bbox": [int(v) for v in ob],
                "reference_group_orientation": str(other_group.get("orientation", "")),
                "overrun_member_id": str(overrun_member.get("segment_id", "")),
                "overrun_member_bbox": [int(v) for v in overrun_member.get("bbox", [])],
                "overrun_member_axis_index": int(overrun_index),
                "overrun_member_count": int(len(overrun_members)),
                "member_width": round(float(member_width), 3),
                "reference_group_width": round(float(other_width), 3),
                "width_ratio": round(float(max(1.0, member_width) / max(1.0, other_width)), 3),
                "width_min_ratio": float(cfg.cross_group_single_overrun_width_min_ratio),
                "width_max_ratio": float(cfg.cross_group_single_overrun_width_max_ratio),
                "remaining_member_count": int(len(remaining_members)),
                "remaining_regular_member_count": int(len(remaining_regular)),
                "remaining_connection_quality": removal_connection_quality,
                "remaining_group_consistency": removal_consistency,
                "overrun_source_shape_metrics": overrun_shape_metrics,
                **metrics,
                "decision": "remove_overrun_member" if removal_ok else "reject_group_after_overrun_member_removal_failed",
                "reason": (
                    "single_external_member_overruns_orthogonal_group_removed"
                    if removal_ok
                    else removal_failure_reason
                ),
            }
            if removal_ok:
                group_member_refs[candidate_group_id] = remaining_members
                candidate_group["member_ids"] = [str(member.get("segment_id", "")) for member in remaining_members]
                candidate_group["regular_member_ids"] = [
                    str(member_id)
                    for member_id in candidate_group.get("regular_member_ids", [])
                    if str(member_id) != str(overrun_member.get("segment_id", ""))
                ]
                candidate_group["terminal_member_ids"] = [
                    str(member_id)
                    for member_id in candidate_group.get("terminal_member_ids", [])
                    if str(member_id) != str(overrun_member.get("segment_id", ""))
                ]
                candidate_group["ultrashort_terminal_member_ids"] = [
                    str(member_id)
                    for member_id in candidate_group.get("ultrashort_terminal_member_ids", [])
                    if str(member_id) != str(overrun_member.get("segment_id", ""))
                ]
                candidate_group["cross_group_single_overrun_removed_member_event"] = event
                modified_group_ids.add(candidate_group_id)
            else:
                reject_group_ids.add(candidate_group_id)
                groups_by_id[candidate_group_id]["cross_group_single_overrun_reject_event"] = event
            events.append(event)
            break
    kept_groups = [group for group in groups if str(group.get("id", "")) not in reject_group_ids]
    kept_refs = {
        group_id: members
        for group_id, members in group_member_refs.items()
        if str(group_id) not in reject_group_ids
    }
    return kept_groups, kept_refs, events


def dash_group_connection_quality(
    black_binary: np.ndarray,
    regular_members: list[dict[str, Any]],
    text_height: float,
    wire_width: float,
    cfg: DashWireConfig,
) -> dict[str, Any]:
    member_metrics: list[dict[str, Any]] = []
    independent_count = 0
    rectangular_count = 0
    strict_count = 0
    clean_count = 0
    dirty_count = 0
    for member in regular_members:
        metrics = group_member_attachment_metrics(black_binary, member, text_height, wire_width, cfg)
        metrics["segment_id"] = str(member.get("segment_id", ""))
        metrics["bbox"] = [int(v) for v in member.get("bbox", [])]
        member_metrics.append(metrics)
        if bool(metrics.get("independent_dash", False)):
            independent_count += 1
        if bool(metrics.get("rectangular_dash", False)):
            rectangular_count += 1
        if str(member.get("candidate_class", "")) == "strict_seed":
            strict_count += 1
        if bool(metrics.get("clean_connection", False)):
            clean_count += 1
        else:
            dirty_count += 1
    regular_count = len(regular_members)
    min_independent = min(int(cfg.group_min_independent_regular_members), regular_count)
    clean_ratio = float(clean_count) / max(1.0, float(regular_count))
    rectangular_ratio = float(rectangular_count) / max(1.0, float(regular_count))
    max_attachment_pixels = max(
        [int(metrics.get("total_attachment_pixels", 0)) for metrics in member_metrics],
        default=0,
    )
    rectangular_attachment_limit = int(round(
        float(text_height) * float(cfg.group_rectangular_exception_max_attachment_text_height_ratio)
    ))
    enough_independent = independent_count >= min_independent
    clean_connected_group = regular_count >= 3 and dirty_count <= 1
    clean_pair_group = regular_count == 2 and dirty_count == 0
    min_rectangular_strict = 2 if regular_count >= 2 else 1
    rectangular_connected_group = (
        regular_count >= 2
        and rectangular_count == regular_count
        and strict_count >= min_rectangular_strict
        and clean_count >= max(1, regular_count // 2)
        and max_attachment_pixels <= rectangular_attachment_limit
    )
    ok = (
        regular_count > 0
        and (enough_independent or clean_connected_group or clean_pair_group or rectangular_connected_group)
    )
    return {
        "connection_quality_ok": bool(ok),
        "regular_member_count": int(regular_count),
        "independent_regular_member_count": int(independent_count),
        "rectangular_regular_member_count": int(rectangular_count),
        "strict_regular_member_count": int(strict_count),
        "min_rectangular_exception_strict_member_count": int(min_rectangular_strict),
        "min_independent_regular_member_count": int(min_independent),
        "enough_independent_regular_members": bool(enough_independent),
        "clean_connected_group_exception": bool(clean_connected_group),
        "clean_pair_group_exception": bool(clean_pair_group),
        "rectangular_connected_group_exception": bool(rectangular_connected_group),
        "clean_regular_member_count": int(clean_count),
        "dirty_regular_member_count": int(dirty_count),
        "max_member_attachment_pixels": int(max_attachment_pixels),
        "rectangular_exception_attachment_pixel_limit": int(rectangular_attachment_limit),
        "clean_regular_member_ratio": round(float(clean_ratio), 3),
        "rectangular_regular_member_ratio": round(float(rectangular_ratio), 3),
        "min_clean_regular_member_ratio": float(cfg.group_clean_member_min_ratio),
        "member_connection_metrics": member_metrics,
    }


def dash_group_consistency_metrics(
    regular_members: list[dict[str, Any]],
    connection_quality: dict[str, Any],
    cfg: DashWireConfig,
) -> dict[str, Any]:
    ordered = sorted(regular_members, key=lambda item: (axis_start(item), axis_end(item)))
    spans = [float(member.get("span", 0.0)) for member in ordered]
    widths = [max(1.0, float(member.get("width", 0.0))) for member in ordered]
    gaps = [
        max(0.0, axis_start(next_member) - axis_end(member))
        for member, next_member in zip(ordered, ordered[1:])
    ]
    span_min = min(spans) if spans else 0.0
    span_max = max(spans) if spans else 0.0
    width_min = min(widths) if widths else 0.0
    width_max = max(widths) if widths else 0.0
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    gap_min = min(positive_gaps) if positive_gaps else 0.0
    gap_max = max(positive_gaps) if positive_gaps else 0.0
    length_ratio = span_max / max(1.0, span_min)
    width_ratio = width_max / max(1.0, width_min)
    gap_ratio = gap_max / max(1.0, gap_min)

    member_metrics = list(connection_quality.get("member_connection_metrics", []))
    regular_count = len(regular_members)
    independent_count = int(connection_quality.get("independent_regular_member_count", 0))
    independent_ratio = float(independent_count) / max(1.0, float(regular_count))
    complex_count = 0
    low_source_fraction_count = 0
    for metrics in member_metrics:
        if int(metrics.get("total_attachment_pixels", 0)) > int(metrics.get("mid_attachment_pixel_limit", 0)) * 3:
            complex_count += 1
        if float(metrics.get("source_component_fraction", 1.0)) < float(cfg.group_text_like_max_source_fraction):
            low_source_fraction_count += 1
    complex_ratio = float(complex_count) / max(1.0, float(regular_count))
    low_source_fraction_ratio = float(low_source_fraction_count) / max(1.0, float(regular_count))

    width_inconsistent = bool(width_ratio > float(cfg.group_width_max_ratio))
    text_like_irregular = bool(
        regular_count >= 3
        and width_ratio > float(cfg.group_text_like_width_max_ratio)
        and length_ratio > float(cfg.group_text_like_length_max_ratio)
        and gap_ratio > float(cfg.group_text_like_gap_max_ratio)
        and complex_ratio >= float(cfg.group_text_like_min_complex_member_ratio)
        and independent_ratio <= float(cfg.group_text_like_max_independent_ratio)
        and low_source_fraction_ratio >= float(cfg.group_text_like_min_complex_member_ratio)
    )
    ok = not (width_inconsistent or text_like_irregular)
    reasons: list[str] = []
    if width_inconsistent:
        reasons.append("member_width_ratio_exceeds_limit")
    if text_like_irregular:
        reasons.append("text_like_length_width_gap_connection_irregularity")
    return {
        "group_consistency_ok": bool(ok),
        "reasons": reasons,
        "regular_member_count": int(regular_count),
        "span_min": round(float(span_min), 3),
        "span_max": round(float(span_max), 3),
        "span_ratio": round(float(length_ratio), 3),
        "width_min": round(float(width_min), 3),
        "width_max": round(float(width_max), 3),
        "width_ratio": round(float(width_ratio), 3),
        "gap_min": round(float(gap_min), 3),
        "gap_max": round(float(gap_max), 3),
        "gap_ratio": round(float(gap_ratio), 3),
        "complex_member_count": int(complex_count),
        "complex_member_ratio": round(float(complex_ratio), 3),
        "independent_member_ratio": round(float(independent_ratio), 3),
        "low_source_fraction_member_count": int(low_source_fraction_count),
        "low_source_fraction_member_ratio": round(float(low_source_fraction_ratio), 3),
        "member_summaries": [segment_debug_summary(member) for member in ordered],
    }


def build_dash_groups(
    black_binary: np.ndarray,
    dash_binary: np.ndarray,
    dots: list[dict[str, Any]],
    strict_seeds: list[dict[str, Any]],
    extension_pool: list[dict[str, Any]],
    text_height: float,
    wire_width: float,
    page: int,
    cfg: DashWireConfig,
    debug_bbox: tuple[int, int, int, int] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    member_info: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    terminal_segments: list[dict[str, Any]] = []
    debug_events: list[dict[str, Any]] = []
    used_regular: set[str] = set()
    group_member_refs: dict[str, list[dict[str, Any]]] = {}
    max_gap = float(text_height) * float(cfg.group_gap_text_height_ratio)
    terminal_gap = float(text_height) * float(cfg.terminal_search_gap_text_height_ratio)
    centerline_tolerance = max(1.0, float(wire_width) * float(cfg.group_centerline_wire_widths))
    seeds = sorted(
        strict_seeds,
        key=lambda item: (
            item["orientation"],
            round(float(item["centerline"]), 1),
            axis_start(item),
            axis_end(item),
        ),
    )
    for seed in seeds:
        seed_id = str(seed["segment_id"])
        if seed_id in used_regular:
            debug_events.append({"event": "seed_rejected", "seed": segment_debug_summary(seed), "reason": "seed_already_used"})
            continue
        members = [seed]
        member_ids = {seed_id}
        orientation = str(seed["orientation"])
        centerline = float(seed["centerline"])
        seed_debug: list[dict[str, Any]] = []
        for direction in (-1, 1):
            while True:
                rep_length = representative_length(members)
                rep_width = representative_width(
                    [member for member in members if str(member.get("candidate_class", "")) == "strict_seed"] or members,
                    wire_width,
                )
                edge = min(axis_start(member) for member in members) if direction < 0 else max(axis_end(member) for member in members)
                next_segment = find_next_group_segment(
                    black_binary,
                    edge,
                    direction,
                    orientation,
                    centerline,
                    rep_length,
                    rep_width,
                    wire_width,
                    extension_pool,
                    used_regular | member_ids,
                    max_gap,
                    centerline_tolerance,
                    len(members),
                    cfg,
                    seed_debug,
                    debug_bbox,
                )
                if next_segment is None:
                    break
                members.append(next_segment)
                member_ids.add(str(next_segment["segment_id"]))

        regular_members = [member for member in members if str(member["segment_id"]).startswith("dash_")]
        if not regular_members:
            debug_events.append(
                {
                    "event": "seed_rejected",
                    "seed": segment_debug_summary(seed),
                    "reason": "no_regular_members_after_extension",
                    "extension_events": seed_debug,
                }
            )
            continue

        rep_length = representative_length(regular_members)
        rep_width = representative_width(regular_members, wire_width)
        terminal_members: list[dict[str, Any]] = []
        terminal_debug: list[dict[str, Any]] = []

        strict_seed_members = [member for member in regular_members if str(member.get("candidate_class", "")) == "strict_seed"]
        single_seed_reference = strict_seed_members[0] if len(strict_seed_members) == 1 else None
        single_seed_reference_length = (
            float(single_seed_reference.get("span", rep_length)) if single_seed_reference is not None else float(rep_length)
        )
        single_seed_terminal_min_len = max(
            single_seed_reference_length * float(cfg.single_seed_terminal_min_length_ratio),
            float(wire_width) * float(cfg.single_seed_terminal_min_wire_widths),
        )
        single_seed_side_supports: list[dict[str, Any]] = []
        if single_seed_reference is not None:
            seed_start = axis_start(single_seed_reference)
            seed_end = axis_end(single_seed_reference)
            support_pool = [member for member in regular_members if member is not single_seed_reference]
            for direction in (-1, 1):
                side_candidates: list[dict[str, Any]] = []
                for member in support_pool:
                    span = float(member.get("span", 0.0))
                    if span < single_seed_terminal_min_len:
                        continue
                    if direction < 0 and axis_end(member) <= seed_start + float(wire_width):
                        side_candidates.append(member)
                    elif direction > 0 and axis_start(member) >= seed_end - float(wire_width):
                        side_candidates.append(member)
                if side_candidates:
                    single_seed_side_supports.append(
                        min(side_candidates, key=lambda member: abs(axis_end(member) - seed_start) if direction < 0 else abs(axis_start(member) - seed_end))
                    )
        single_seed_terminal_ok = (
            single_seed_reference is not None
            and len(single_seed_side_supports) >= 2
        )
        if len(regular_members) < int(cfg.group_min_regular_members) and not single_seed_terminal_ok:
            debug_events.append(
                {
                    "event": "seed_rejected",
                    "seed": segment_debug_summary(seed),
                    "reason": "fewer_than_two_regular_members_without_two_sufficient_terminals",
                    "regular_member_ids": [str(member["segment_id"]) for member in regular_members],
                    "terminal_members": [segment_debug_summary(member) for member in terminal_members],
                    "single_seed_side_supports": [segment_debug_summary(member) for member in single_seed_side_supports],
                    "single_seed_terminal_min_length": round(float(single_seed_terminal_min_len), 3),
                    "extension_events": seed_debug,
                    "terminal_extension_events": terminal_debug,
                }
            )
            continue

        min_group_length = max(
            float(text_height) * float(cfg.group_min_representative_text_height_ratio),
            float(wire_width) * float(cfg.group_min_representative_wire_widths),
        )
        strict_member_count = sum(1 for member in regular_members if str(member.get("candidate_class", "")) == "strict_seed")
        length_below_min = rep_length + float(cfg.group_min_representative_epsilon_px) < min_group_length
        if not single_seed_terminal_ok and (
            length_below_min or strict_member_count < int(cfg.group_min_strict_seed_members)
        ):
            debug_events.append(
                {
                    "event": "seed_rejected",
                    "seed": segment_debug_summary(seed),
                    "reason": "group_representative_length_or_strict_seed_count_below_min",
                    "regular_member_ids": [str(member["segment_id"]) for member in regular_members],
                    "representative_length": round(float(rep_length), 3),
                    "min_group_representative_length": round(float(min_group_length), 3),
                    "strict_seed_member_count": int(strict_member_count),
                    "min_strict_seed_members": int(cfg.group_min_strict_seed_members),
                    "regular_members": [segment_debug_summary(member) for member in regular_members],
                    "extension_events": seed_debug,
                }
            )
            continue

        connection_quality = dash_group_connection_quality(black_binary, regular_members, text_height, wire_width, cfg)
        if not bool(connection_quality.get("connection_quality_ok", False)):
            debug_events.append(
                {
                    "event": "seed_rejected",
                    "seed": segment_debug_summary(seed),
                    "reason": "group_connection_quality_below_min",
                    "regular_member_ids": [str(member["segment_id"]) for member in regular_members],
                    "regular_members": [segment_debug_summary(member) for member in regular_members],
                    "connection_quality": connection_quality,
                    "extension_events": seed_debug,
                }
            )
            continue

        consistency = dash_group_consistency_metrics(regular_members, connection_quality, cfg)
        if not bool(consistency.get("group_consistency_ok", False)):
            debug_events.append(
                {
                    "event": "seed_rejected",
                    "seed": segment_debug_summary(seed),
                    "reason": "group_length_width_gap_consistency_below_min",
                    "regular_member_ids": [str(member["segment_id"]) for member in regular_members],
                    "regular_members": [segment_debug_summary(member) for member in regular_members],
                    "connection_quality": connection_quality,
                    "group_consistency": consistency,
                    "extension_events": seed_debug,
                }
            )
            continue

        for direction in (-1, 1):
            edge = min(axis_start(member) for member in regular_members) if direction < 0 else max(axis_end(member) for member in regular_members)
            terminal = find_terminal_near_group_end(
                black_binary,
                dash_binary,
                dots,
                orientation,
                centerline,
                edge,
                direction,
                rep_length,
                terminal_gap,
                centerline_tolerance,
                wire_width,
                cfg,
                terminal_debug,
                debug_bbox,
            )
            if terminal is not None:
                terminal_members.append(terminal)

        for terminal in terminal_members:
            terminal["segment_id"] = f"ultrashort_{len(terminal_segments) + 1:04d}"
            terminal_segments.append(terminal)

        all_members = sorted(regular_members + terminal_members, key=lambda item: (axis_start(item), axis_end(item)))
        group_id = f"p{page:03d}_dash_group_{len(groups) + 1:04d}"
        classify_group_endpoints(all_members, group_id)
        group_bbox = [
            min(int(member["bbox"][0]) for member in all_members),
            min(int(member["bbox"][1]) for member in all_members),
            max(int(member["bbox"][2]) for member in all_members),
            max(int(member["bbox"][3]) for member in all_members),
        ]
        group = {
            "id": group_id,
            "orientation": orientation,
            "centerline": round(centerline, 3),
            "representative_length": round(rep_length, 3),
            "representative_width": round(rep_width, 3),
            "max_regular_gap_pixels": round(max_gap, 3),
            "max_terminal_gap_pixels": round(terminal_gap, 3),
            "centerline_tolerance_pixels": round(centerline_tolerance, 3),
            "bbox": group_bbox,
            "member_ids": [str(member["segment_id"]) for member in all_members],
            "regular_member_ids": [str(member["segment_id"]) for member in regular_members],
            "connection_quality": connection_quality,
            "group_consistency": consistency,
            "terminal_member_ids": [str(member["segment_id"]) for member in terminal_members],
            "ultrashort_terminal_member_ids": [
                str(member["segment_id"])
                for member in terminal_members
                if str(member["segment_id"]).startswith("ultrashort_")
            ],
            "single_seed_terminal_exception": bool(single_seed_terminal_ok),
            "single_seed_side_support_member_ids": [str(member.get("segment_id", "")) for member in single_seed_side_supports],
            "single_seed_terminal_min_length": round(float(single_seed_terminal_min_len), 3),
        }
        group_member_refs[group_id] = all_members
        group["external_endpoint_orthogonal_fallback_events"] = []
        groups.append(group)
        used_regular.update(str(member["segment_id"]) for member in regular_members)
        debug_events.append(
            {
                "event": "group_accepted",
                "seed": segment_debug_summary(seed),
                "group": group,
                "regular_members": [segment_debug_summary(member) for member in regular_members],
                "terminal_members": [segment_debug_summary(member) for member in terminal_members],
                "single_seed_side_supports": [segment_debug_summary(member) for member in single_seed_side_supports],
                "connection_quality": connection_quality,
                "regular_extension_events": seed_debug,
                "terminal_extension_events": terminal_debug,
                "reason": "single_seed_with_two_sufficient_side_supports" if single_seed_terminal_ok else "at_least_two_regular_members",
            }
        )
    retreat_events = apply_external_endpoint_retreat(dash_binary, groups, group_member_refs, wire_width, cfg)
    if retreat_events:
        debug_events.append(
            {
                "event": "external_endpoint_width_retreat_pass",
                "retreat_event_count": int(len(retreat_events)),
                "retreat_events": retreat_events,
            }
        )
    groups, group_member_refs, cross_group_filter_events = cross_group_single_overrun_filter(
        black_binary,
        groups,
        group_member_refs,
        text_height,
        wire_width,
        cfg,
    )
    if cross_group_filter_events:
        debug_events.append(
            {
                "event": "cross_group_single_member_overrun_filter_pass",
                "reviewed_overrun_count": int(len(cross_group_filter_events)),
                "rejected_group_count": int(sum(1 for event in cross_group_filter_events if str(event.get("decision", "")) == "reject_group_after_overrun_member_removal_failed")),
                "removed_member_count": int(sum(1 for event in cross_group_filter_events if str(event.get("decision", "")) == "remove_overrun_member")),
                "filter_events": cross_group_filter_events,
            }
        )
    for group in groups:
        group_id = str(group.get("id", ""))
        members = sorted(group_member_refs.get(group_id, []), key=lambda item: (axis_start(item), axis_end(item)))
        if not members:
            continue
        classify_group_endpoints(members, group_id)
        group["bbox"] = [
            min(int(member["bbox"][0]) for member in members),
            min(int(member["bbox"][1]) for member in members),
            max(int(member["bbox"][2]) for member in members),
            max(int(member["bbox"][3]) for member in members),
        ]
        group["member_ids"] = [str(member["segment_id"]) for member in members]
        regular_ids = set(str(member_id) for member_id in group.get("regular_member_ids", []))
        terminal_ids = set(str(member_id) for member_id in group.get("terminal_member_ids", []))
        group["external_endpoint_width_retreat_events"] = [
            event for event in retreat_events if str(event.get("group_id", "")) == group_id
        ]
        fallback_events = apply_external_orthogonal_fallback(dash_binary, members, group, wire_width, cfg)
        group["external_endpoint_orthogonal_fallback_events"] = fallback_events
        for order_index, member in enumerate(members):
            member_id = str(member["segment_id"])
            member_info[member_id] = {
                "dash_group_id": group_id,
                "recognized_as": "dash_member",
                "group_role": "terminal" if member_id in terminal_ids else "regular",
                "group_order_index": int(order_index),
                "group_orientation": str(group.get("orientation", "")),
                "group_centerline": round(float(group.get("centerline", 0.0)), 3),
                "group_representative_length": round(float(group.get("representative_length", 0.0)), 3),
                "endpoint_roles": list(member.get("endpoint_roles", [])),
                "endpoints": list(member.get("endpoints", [])),
                **(
                    {"external_endpoint_width_retreat": dict(member.get("external_endpoint_width_retreat", {}))}
                    if member.get("external_endpoint_width_retreat")
                    else {}
                ),
            }
    return member_info, groups, terminal_segments, debug_events


def objects_from_segments(
    regular_segments: list[dict[str, Any]],
    terminal_segments: list[dict[str, Any]],
    member_info: dict[str, dict[str, Any]],
    page: int,
    wire_width: float,
) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    for segment in regular_segments + terminal_segments:
        segment_id = str(segment.get("segment_id", ""))
        group_info = member_info.get(segment_id)
        if not group_info:
            continue
        bbox = [int(v) for v in segment["bbox"]]
        objects.append(
            DetectedObject(
                f"p{page:03d}_dash_member_{len(objects) + 1:04d}",
                "dash_member",
                0.84 if str(segment_id).startswith("dash_") else 0.80,
                bbox,
                {"kind": "rectangle", "rect": bbox, "centerline_points": segment["points"]},
                {
                    "segment_id": segment_id,
                    "source_segment_type": str(segment.get("candidate_class", "dash_axis_segment")),
                    "orientation": segment["orientation"],
                    "length_pixels": round(float(segment["span"]), 3),
                    "median_width": round(float(segment["width"]), 3),
                    "estimated_wire_width": round(float(wire_width), 3),
                    "projected_fill_ratio": round(float(segment.get("projected_fill_ratio", 0.0)), 3),
                    "source": str(segment.get("source", "projected_axis_dash_segment")),
                    "source_component_bbox": list(segment.get("source_component_bbox", [])),
                    "source_component_area": int(segment.get("source_component_area", 0)),
                    "wire_pixel_bbox": list(segment.get("wire_pixel_bbox", segment.get("bbox", []))),
                    "wire_pixel_runs": list(segment.get("wire_pixel_runs", [])),
                    "wire_pixel_count": int(segment.get("wire_pixel_count", 0)),
                    **({"candidate_rejection_checks": dict(segment.get("candidate_rejection_checks", {}))} if segment.get("candidate_rejection_checks") else {}),
                    **({"reviewed_dot_id": str(segment.get("reviewed_dot_id", ""))} if segment.get("reviewed_dot_id") else {}),
                    **({"reviewed_dot_metrics": dict(segment.get("reviewed_dot_metrics", {}))} if segment.get("reviewed_dot_metrics") else {}),
                    **group_info,
                },
                source_phase=STAGE_NAME,
            )
        )
    return objects


def objects_from_groups(groups: list[dict[str, Any]], page: int) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    for index, group in enumerate(groups, start=1):
        bbox = [int(v) for v in group.get("bbox", [])]
        if len(bbox) != 4:
            continue
        objects.append(
            DetectedObject(
                f"p{page:03d}_dash_group_{index:04d}",
                "dash_group",
                0.70,
                bbox,
                {"kind": "group_bbox", "rect": bbox},
                {
                    "dash_group_id": str(group.get("id", "")),
                    "orientation": str(group.get("orientation", "")),
                    "member_ids": list(group.get("member_ids", [])),
                    "regular_member_ids": list(group.get("regular_member_ids", [])),
                    "terminal_member_ids": list(group.get("terminal_member_ids", [])),
                    "centerline": float(group.get("centerline", 0.0)),
                    "representative_length": float(group.get("representative_length", 0.0)),
                    "representative_width": float(group.get("representative_width", 0.0)),
                },
                source_phase=STAGE_NAME,
            )
        )
    return objects


def parse_debug_bbox(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--debug-bbox must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--debug-bbox values must be integers") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--debug-bbox requires x2>x1 and y2>y1")
    return (x1, y1, x2, y2)


def overlay(rgb: np.ndarray, objects: list[DetectedObject]) -> np.ndarray:
    out = rgb.copy()
    for obj in objects:
        x1, y1, x2, y2 = clipped_bbox(obj.bbox, rgb.shape[:2])
        if x2 <= x1 or y2 <= y1:
            continue
        color = VISUAL_COLORS.get(obj.type)
        if color is None:
            continue
        if obj.type == "dash_group":
            continue
        attrs = obj.attributes
        runs = attrs.get("wire_pixel_runs", attrs.get("pixel_runs", []))
        run_bbox = attrs.get("wire_pixel_bbox", attrs.get("pixel_bbox", obj.bbox))
        if runs:
            ex1, ey1, ex2, ey2 = clipped_bbox(run_bbox, rgb.shape[:2])
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            run_mask = relative_runs_to_mask(runs, (ey2 - ey1, ex2 - ex1))
            out[ey1:ey2, ex1:ex2][run_mask] = color
        else:
            out[y1:y2, x1:x2] = color
    return out


def write_visual_legend(path: Path) -> None:
    lines = [f"{STAGE_NAME} visual legend", ""]
    for key, description, rgb in VISUAL_LEGEND:
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
        lines.append(f"{key}: {hex_color} RGB{rgb} - {description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_page(
    rgb: np.ndarray,
    page: int,
    cfg: DashWireConfig,
    debug_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[list[DetectedObject], dict[str, Any]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    black = clean_black_mask(rgb, cfg.black_threshold)
    wire_width = estimate_wire_width(black)
    text_height = estimate_text_height(black)
    timings["black_mask_and_estimates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dots = detect_junction_dots(black, wire_width, cfg)
    dot_mask = build_dot_mask(black.shape, dots)
    dash_binary = black & ~dot_mask
    timings["junction_dot_detection_and_removal"] = time.perf_counter() - t0

    component_binary = dash_binary

    t0 = time.perf_counter()
    min_len = int(np.floor(max(text_height * cfg.extension_min_text_height_ratio, wire_width * cfg.extension_min_wire_widths)))
    max_len = int(round(max(min_len, text_height * cfg.extension_max_text_height_ratio)))
    raw_segments = (
        extract_axis_rectangles(dash_binary, "horizontal", min_len, max_len, wire_width, cfg)
        + extract_axis_rectangles(dash_binary, "vertical", min_len, max_len, wire_width, cfg)
    )
    raw_segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0], s["orientation"]))
    strict_seeds, extension_pool, rejected_candidates = classify_candidates(
        raw_segments,
        dash_binary,
        component_binary,
        text_height,
        wire_width,
        cfg,
    )
    assign_segment_ids(extension_pool, "dash")
    timings["dash_candidate_extraction"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    member_info, dash_groups, ultrashort_segments, group_debug_events = build_dash_groups(
        black,
        dash_binary,
        dots,
        strict_seeds,
        extension_pool,
        text_height,
        wire_width,
        page,
        cfg,
        debug_bbox,
    )
    timings["dash_grouping"] = time.perf_counter() - t0

    objects = objects_from_segments(extension_pool, ultrashort_segments, member_info, page, wire_width)
    objects.extend(objects_from_groups(dash_groups, page))
    type_counts = Counter(obj.type for obj in objects)
    external_fallback_events = [
        event
        for group in dash_groups
        for event in group.get("external_endpoint_orthogonal_fallback_events", [])
    ]
    diagnostics = {
        "estimated_wire_width": float(wire_width),
        "estimated_text_height": float(text_height),
        "black_threshold": int(cfg.black_threshold),
        "candidate_min_length_pixels": int(min_len),
        "candidate_max_length_pixels": int(max_len),
        "num_black_pixels": int(np.count_nonzero(black)),
        "num_dot_pixels_removed": int(np.count_nonzero(dot_mask)),
        "num_junction_dots": int(len(dots)),
        "junction_dots": [{key: value for key, value in dot.items() if key != "pixel_runs"} for dot in dots],
        "num_raw_dash_candidates": int(len(raw_segments)),
        "num_strict_dash_seeds": int(len(strict_seeds)),
        "num_extension_dash_candidates": int(len(extension_pool)),
        "num_rejected_dash_candidates": int(len(rejected_candidates)),
        "rejected_dash_candidates": rejected_candidates,
        "num_dash_members": int(type_counts.get("dash_member", 0)),
        "num_dash_groups": int(len(dash_groups)),
        "dash_groups": dash_groups,
        "external_endpoint_orthogonal_fallback_count": int(len(external_fallback_events)),
        "external_endpoint_orthogonal_fallback_events": external_fallback_events,
        "debug_bbox": list(debug_bbox) if debug_bbox is not None else None,
        "dash_group_debug_events": group_debug_events,
        "timings_seconds": {key: round(value, 4) for key, value in timings.items()},
        "detector_backend": "standalone_dash_wire_projected_rectangles_group_connection_validation",
    }
    return objects, diagnostics


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="bmw-328i-1997.pdf")
    parser.add_argument("--pages")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--debug-bbox")
    parser.add_argument("--preserve-existing-output", action="store_true")
    args = parser.parse_args()
    debug_bbox = parse_debug_bbox(args.debug_bbox)

    cfg = DashWireConfig(dpi=args.dpi)
    pdf = resolve_pdf(args.pdf)
    clean_source_dir = resolve_phase1_clean_dir(pdf)
    available_pages = available_clean_pages(clean_source_dir)
    pages = parse_pages(args.pages, available_pages)
    root, image_dir, json_dir = make_output_dirs(pdf.stem, clear_existing=not args.preserve_existing_output)
    write_visual_legend(root / "legend")

    logging.info("PDF path: %s", pdf)
    logging.info("Phase 1 source images: %s", clean_source_dir)
    logging.info("Pages: %s", pages)
    logging.info("%s output: %s", "Preserving existing" if args.preserve_existing_output else "Cleared", root)

    review_paths: list[Path] = []
    totals: Counter[str] = Counter()
    for page_number in pages:
        page_t0 = time.perf_counter()
        rgb = load_clean_image(clean_source_dir, page_number)
        page_info = PageInfo(pdf.name, page_number, args.dpi, int(rgb.shape[1]), int(rgb.shape[0]))
        objects, diagnostics = analyze_page(
            rgb,
            page_number,
            cfg,
            debug_bbox=debug_bbox,
        )
        totals.update(obj.type for obj in objects)

        review = overlay(rgb, objects)
        review_path = image_dir / f"page_{page_number:03d}.png"
        save_rgb(review_path, review, args.dpi)
        review_paths.append(review_path)

        diagnostics["total_page_seconds"] = round(time.perf_counter() - page_t0, 4)
        write_json(json_dir / f"page_{page_number:03d}.json", page_json(page_info, objects, diagnostics))
        logging.info("Page %03d: %d objects in %.2fs", page_number, len(objects), diagnostics["total_page_seconds"])

    images_to_pdf(review_paths, root / "review.pdf", args.dpi)
    logging.info("Detected: %s", dict(totals))
    logging.info("Output: %s", root)


if __name__ == "__main__":
    main()
'''


def _load_embedded_module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(SCRIPTS_DIR / f"{name}.embedded.py")
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


solid_module = _load_embedded_module("_embedded_04_wire_solid_backup", _SOLID_WIRE_SOURCE)
dash_module = _load_embedded_module("_embedded_04_wire_dash_backup", _DASH_WIRE_SOURCE)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def image_to_pdf(images: Sequence[np.ndarray], dest: Path, dpi: int) -> None:
    doc = fitz.open()
    try:
        for image in images:
            pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            stream = io.BytesIO()
            pil.save(stream, format="PNG", dpi=(dpi, dpi))
            width, height = pil.size
            page = doc.new_page(width=width * 72.0 / dpi, height=height * 72.0 / dpi)
            page.insert_image(page.rect, stream=stream.getvalue(), keep_proportion=True)
        doc.save(dest, deflate=True)
    finally:
        doc.close()


def parse_page_range(spec: str) -> tuple[int, int]:
    token = str(spec).strip()
    if "-" in token:
        left, right = token.split("-", 1)
        start, end = int(left), int(right)
    else:
        start = end = int(token)
    if start < 1 or end < start:
        raise ValueError(f"Invalid page range: {spec}")
    return start, end


def resolve_pdf(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, ROOT_DIR / path, INPUT_DIR / path]
    if path.suffix.lower() != ".pdf":
        candidates.append(INPUT_DIR / f"{value}.pdf")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"PDF not found: {value}. Looked directly and in {INPUT_DIR}")


def resolve_input_dir(pdf_stem: str) -> Path:
    direct = ROOT_DIR / "outputs" / SOURCE_STAGE / pdf_stem / "circuit_images"
    if direct.is_dir():
        return direct
    stage_root = ROOT_DIR / "outputs" / SOURCE_STAGE
    if stage_root.is_dir():
        for candidate in stage_root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == pdf_stem.lower():
                image_dir = candidate / "circuit_images"
                if image_dir.is_dir():
                    return image_dir
    raise FileNotFoundError(
        f"Missing {SOURCE_STAGE} circuit images: {direct}. "
        f"Run: python scripts/03_rect.py --pdf {pdf_stem}.pdf first."
    )


def available_pages(image_dir: Path) -> list[int]:
    pages: list[int] = []
    for path in sorted(image_dir.glob("page_*.png")):
        try:
            pages.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    if not pages:
        raise FileNotFoundError(f"No page_NNN.png files found under: {image_dir}")
    return sorted(pages)


def select_pages(spec: str | None, pages: Sequence[int]) -> list[int]:
    if spec is None:
        return list(sorted(pages))
    start, end = parse_page_range(spec)
    available = set(int(page) for page in pages)
    requested = list(range(start, min(end, max(available)) + 1))
    missing = [page for page in requested if page not in available]
    if missing:
        raise ValueError(f"Requested page(s) {missing} are missing from {SOURCE_STAGE}/circuit_images")
    return requested


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    resolved_output = root.resolve()
    resolved_stage = OUTPUT_DIR.resolve()
    if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
    shutil.rmtree(root)


def make_output_dirs(pdf_stem: str, preserve: bool) -> dict[str, Path]:
    root = OUTPUT_DIR / pdf_stem
    if root.exists() and not preserve:
        clear_output_root(root)
    paths = {"root": root, "images": root / "images", "json": root / "json", "debug": root / "debug"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, image: np.ndarray, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path, dpi=(dpi, dpi))


def _attrs(obj: Any) -> dict[str, Any]:
    return dict(getattr(obj, "attributes", {}) or {})


def _geom(obj: Any) -> dict[str, Any]:
    return dict(getattr(obj, "geometry", {}) or {})


def _bbox(value: Sequence[Any]) -> list[int]:
    return [int(round(float(v))) for v in value[:4]]


def _run_count(runs: Sequence[Sequence[Any]]) -> int:
    return int(sum(max(0, int(run[2]) - int(run[1])) for run in runs if len(run) == 3))


def _runs_to_mask(runs: Sequence[Sequence[Any]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for run in runs:
        if len(run) != 3:
            continue
        y, x1, x2 = [int(v) for v in run]
        if 0 <= y < shape[0]:
            mask[y, max(0, x1):min(shape[1], x2)] = True
    return mask


def _mask_to_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for y, row in enumerate(mask.astype(bool)):
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        start = previous = int(xs[0])
        for value in xs[1:]:
            current = int(value)
            if current == previous + 1:
                previous = current
            else:
                runs.append([int(y), start, previous + 1])
                start = previous = current
        runs.append([int(y), start, previous + 1])
    return runs


def _object_mask(obj: Any) -> tuple[list[int] | None, list[list[int]], int]:
    attrs = _attrs(obj)
    bbox = attrs.get("wire_pixel_bbox") or attrs.get("pixel_bbox") or attrs.get("edge_expansion_bbox")
    runs = attrs.get("wire_pixel_runs") or attrs.get("pixel_runs") or attrs.get("edge_expansion_pixel_runs") or []
    if bbox and runs:
        clean_runs = [[int(r), int(a), int(b)] for r, a, b in runs]
        return _bbox(bbox), clean_runs, int(attrs.get("wire_pixel_count") or attrs.get("pixel_count") or _run_count(clean_runs))
    bbox = _bbox(getattr(obj, "bbox", []))
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None, [], 0
    runs = [[row, 0, bbox[2] - bbox[0]] for row in range(bbox[3] - bbox[1])]
    return bbox, runs, _run_count(runs)


def _points(obj: Any, orientation: str, bbox: Sequence[int]) -> list[list[float]]:
    points = _geom(obj).get("centerline_points") or []
    if len(points) >= 2:
        return [[round(float(pt[0]), 3), round(float(pt[1]), 3)] for pt in points[:2]]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if orientation == "vertical":
        cx = (x1 + x2) / 2.0
        return [[round(cx, 3), round(y1, 3)], [round(cx, 3), round(y2, 3)]]
    cy = (y1 + y2) / 2.0
    return [[round(x1, 3), round(cy, 3)], [round(x2, 3), round(cy, 3)]]


def _length(points: Sequence[Sequence[float]], bbox: Sequence[int], orientation: str) -> int:
    if len(points) >= 2:
        return int(round(math.hypot(float(points[1][0]) - float(points[0][0]), float(points[1][1]) - float(points[0][1]))))
    return int(max(0, int(bbox[3]) - int(bbox[1]) if orientation == "vertical" else int(bbox[2]) - int(bbox[0])))


def _solid_wire(obj: Any) -> dict[str, Any] | None:
    if getattr(obj, "type", "") != "solid_wire":
        return None
    attrs = _attrs(obj)
    bbox = _bbox(getattr(obj, "bbox", []))
    source = str(attrs.get("source_segment_type") or attrs.get("source") or "long_axis_segment")
    orientation = str(attrs.get("orientation") or ("diagonal" if source == "diagonal_endpoint_extension" else "horizontal"))
    points = _points(obj, orientation, bbox)
    mask_bbox, runs, pixels = _object_mask(obj)
    wire_type = "solid_wire_diagonal_extension" if source == "diagonal_endpoint_extension" else ("solid_wire_extension" if source == "perpendicular_endpoint_extension" else "solid_wire_seed")
    extension_match = None
    if source in {"diagonal_endpoint_extension", "perpendicular_endpoint_extension"}:
        extension_match = {
            "base_id": str(attrs.get("parent_segment_id", "")),
            "base_endpoint_index": int(attrs.get("parent_endpoint_index", -1)),
            "angle_deg": round(float(attrs.get("extension_angle_degrees", 0.0)), 3) if source == "diagonal_endpoint_extension" else None,
        }
    return {
        "id": str(getattr(obj, "id", attrs.get("segment_id", ""))),
        "type": wire_type,
        "bbox": bbox,
        "orientation": orientation,
        "confidence": round(float(getattr(obj, "confidence", 0.0)), 4),
        "geometry": {
            "kind": "axis_aligned_segment" if orientation in {"horizontal", "vertical"} else "diagonal_segment",
            "centerline_points": points,
            "length_px": int(round(float(attrs.get("length_pixels") or _length(points, bbox, orientation)))),
            "thickness_px": int(round(max(1.0, float(attrs.get("median_width") or attrs.get("body_width") or attrs.get("bbox_width") or 1.0)))),
        },
        "mask": {"bbox": mask_bbox, "pixel_count": int(pixels), "runs": runs},
        "quality": {"core_band_bbox": None, "shape_metrics": {}, "shape_protection": None, "component_shape_metrics": {}, "component_shape_protection": None},
        "attributes": {"source": source, "extension_round": None, "extension_match": extension_match, "endpoint_backoff": None, "internal_anomaly_trim": None, "dash_marks": None},
        "connections": [],
    }


def _dash_mark(obj: Any) -> dict[str, Any]:
    attrs = _attrs(obj)
    bbox = _bbox(getattr(obj, "bbox", []))
    orientation = str(attrs.get("orientation") or "horizontal")
    points = _points(obj, orientation, bbox)
    mask_bbox, runs, pixels = _object_mask(obj)
    return {
        "id": str(getattr(obj, "id", attrs.get("segment_id", ""))),
        "segment_id": str(attrs.get("segment_id", "")),
        "bbox": bbox,
        "orientation": orientation,
        "centerline_points": points,
        "length_px": int(round(float(attrs.get("length_pixels") or _length(points, bbox, orientation)))),
        "thickness_px": int(round(max(1.0, float(attrs.get("median_width") or 1.0)))),
        "group_role": attrs.get("group_role"),
        "group_order_index": attrs.get("group_order_index"),
        "mask_bbox": mask_bbox,
        "mask_pixel_count": int(pixels),
        "mask_runs": runs,
    }


def _union_marks(group_bbox: list[int], marks: list[dict[str, Any]]) -> tuple[list[int] | None, list[list[int]], int]:
    gx1, gy1, gx2, gy2 = group_bbox
    if gx2 <= gx1 or gy2 <= gy1:
        return None, [], 0
    canvas = np.zeros((gy2 - gy1, gx2 - gx1), dtype=bool)
    for mark in marks:
        mb = mark.get("mask_bbox")
        if not mb:
            continue
        mx1, my1, mx2, my2 = [int(v) for v in mb]
        local = _runs_to_mask(mark.get("mask_runs") or [], (my2 - my1, mx2 - mx1))
        y1, y2 = max(gy1, my1), min(gy2, my2)
        x1, x2 = max(gx1, mx1), min(gx2, mx2)
        if x2 > x1 and y2 > y1:
            canvas[y1 - gy1:y2 - gy1, x1 - gx1:x2 - gx1] |= local[y1 - my1:y2 - my1, x1 - mx1:x2 - mx1]
    return group_bbox, _mask_to_runs(canvas), int(canvas.sum())


def _dash_wire(group_obj: Any, members: list[Any]) -> dict[str, Any] | None:
    if getattr(group_obj, "type", "") != "dash_group":
        return None
    attrs = _attrs(group_obj)
    bbox = _bbox(getattr(group_obj, "bbox", []))
    orientation = str(attrs.get("orientation") or "horizontal")
    centerline = float(attrs.get("centerline", (bbox[1] + bbox[3]) / 2.0 if orientation == "horizontal" else (bbox[0] + bbox[2]) / 2.0))
    if orientation == "vertical":
        points = [[round(centerline, 3), float(bbox[1])], [round(centerline, 3), float(bbox[3])]]
        length = bbox[3] - bbox[1]
    else:
        points = [[float(bbox[0]), round(centerline, 3)], [float(bbox[2]), round(centerline, 3)]]
        length = bbox[2] - bbox[0]
    marks = [_dash_mark(member) for member in members]
    mask_bbox, runs, pixels = _union_marks(bbox, marks)
    return {
        "id": str(getattr(group_obj, "id", attrs.get("dash_group_id", ""))),
        "type": "dashed_wire_group",
        "bbox": bbox,
        "orientation": orientation,
        "confidence": round(float(getattr(group_obj, "confidence", 0.70)), 4),
        "geometry": {"kind": "axis_aligned_segment", "centerline_points": points, "length_px": int(length), "thickness_px": int(round(max(1.0, float(attrs.get("representative_width", 1.0)))))},
        "mask": {"bbox": mask_bbox, "pixel_count": int(pixels), "runs": runs},
        "quality": {"core_band_bbox": None, "shape_metrics": {}, "shape_protection": None, "component_shape_metrics": {}, "component_shape_protection": None},
        "attributes": {"source": "dashed_group", "extension_round": None, "extension_match": None, "endpoint_backoff": None, "internal_anomaly_trim": None, "dash_marks": marks},
        "connections": [],
    }


def backup_objects_to_wires(solid_objects: list[Any], dash_objects: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solid_wires = [wire for wire in (_solid_wire(obj) for obj in solid_objects) if wire is not None]
    members_by_group: dict[str, list[Any]] = {}
    for obj in dash_objects:
        if getattr(obj, "type", "") == "dash_member":
            group_id = str(_attrs(obj).get("dash_group_id", ""))
            if group_id:
                members_by_group.setdefault(group_id, []).append(obj)
    dashed_wires: list[dict[str, Any]] = []
    for obj in dash_objects:
        if getattr(obj, "type", "") == "dash_group":
            group_id = str(_attrs(obj).get("dash_group_id") or getattr(obj, "id", ""))
            wire = _dash_wire(obj, members_by_group.get(group_id, []))
            if wire is not None:
                dashed_wires.append(wire)
    return solid_wires + dashed_wires, {
        "num_backup_solid_objects": len(solid_objects),
        "num_backup_dash_objects": len(dash_objects),
        "num_converted_solid_wires": len(solid_wires),
        "num_converted_dashed_groups": len(dashed_wires),
        "num_dash_member_groups": len(members_by_group),
    }


def _point_box_distance(point: Sequence[float], box: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    return float(math.hypot(max(x1 - px, px - x2, 0.0), max(y1 - py, py - y2, 0.0)))


def _wire_mask(wire: dict[str, Any]) -> tuple[list[int], np.ndarray] | None:
    bbox = wire.get("mask", {}).get("bbox")
    if not bbox:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2], _runs_to_mask(wire.get("mask", {}).get("runs") or [], (y2 - y1, x2 - x1))


def _near_mask(point: Sequence[float], bbox: Sequence[int], arr: np.ndarray, tol: float) -> tuple[bool, float | None]:
    x1, y1, _, _ = bbox
    lx, ly = float(point[0]) - x1, float(point[1]) - y1
    xs0, xs1 = int(max(0, math.floor(lx - tol))), int(min(arr.shape[1], math.ceil(lx + tol) + 1))
    ys0, ys1 = int(max(0, math.floor(ly - tol))), int(min(arr.shape[0], math.ceil(ly + tol) + 1))
    if xs0 >= xs1 or ys0 >= ys1:
        return False, None
    yy, xx = np.where(arr[ys0:ys1, xs0:xs1])
    if xx.size == 0:
        return False, None
    dists = np.hypot((xx + xs0) - lx, (yy + ys0) - ly)
    dist = float(dists[int(np.argmin(dists))])
    return dist <= tol, dist


def compute_connections(wires: list[dict[str, Any]]) -> dict[str, Any]:
    if not wires:
        return {"total_connections": 0, "endpoint_contacts": 0, "body_contacts": 0, "wires_with_no_connections": 0}
    tol = float(max(3.0, np.median([max(1, int(w.get("geometry", {}).get("thickness_px", 1))) for w in wires]) * 2.0))
    masks = {id(wire): _wire_mask(wire) for wire in wires}
    total = endpoint_contacts = body_contacts = 0
    for wire in wires:
        conns: list[dict[str, Any]] = []
        for idx, point in enumerate(wire.get("geometry", {}).get("centerline_points", [])[:2]):
            self_endpoint = "start" if idx == 0 else "end"
            for other in wires:
                if other is wire:
                    continue
                obox = other.get("mask", {}).get("bbox") or other.get("bbox")
                if not obox or _point_box_distance(point, obox) > tol:
                    continue
                other_mask = masks[id(other)]
                if other_mask is None:
                    continue
                touched, dist = _near_mask(point, other_mask[0], other_mask[1], tol)
                if touched:
                    conns.append({"wire_id": str(other.get("id")), "self_endpoint": self_endpoint, "contact": "body", "other_endpoint": None, "distance_px": round(float(dist), 4) if dist is not None else None})
                    total += 1
                    body_contacts += 1
        wire["connections"] = conns
    return {"total_connections": int(total), "endpoint_contacts": int(endpoint_contacts), "body_contacts": int(body_contacts), "wires_with_no_connections": int(sum(1 for wire in wires if not wire.get("connections")))}


def color_for(wire: dict[str, Any]) -> tuple[int, int, int]:
    return ORANGE if wire.get("type") == "dashed_wire_group" else GREEN


def paint_wire_mask(out: np.ndarray, wire: dict[str, Any], color: tuple[int, int, int]) -> None:
    bbox = wire.get("mask", {}).get("bbox")
    if not bbox:
        return
    bx1, by1, _, _ = [int(v) for v in bbox]
    color_arr = np.array(color, dtype=np.uint8)
    h, w = out.shape[:2]
    for row, x_start, x_end in wire.get("mask", {}).get("runs") or []:
        y = by1 + int(row)
        if 0 <= y < h:
            out[y, max(0, bx1 + int(x_start)):min(w, bx1 + int(x_end))] = color_arr


def paint_wire_bool(mask: np.ndarray, wire: dict[str, Any]) -> None:
    bbox = wire.get("mask", {}).get("bbox")
    if not bbox:
        return
    bx1, by1, _, _ = [int(v) for v in bbox]
    h, w = mask.shape[:2]
    for row, x_start, x_end in wire.get("mask", {}).get("runs") or []:
        y = by1 + int(row)
        if 0 <= y < h:
            mask[y, max(0, bx1 + int(x_start)):min(w, bx1 + int(x_end))] = True


def make_unlabeled_review(rgb: np.ndarray, wires: list[dict[str, Any]]) -> np.ndarray:
    out = rgb.copy()
    for wire in wires:
        paint_wire_mask(out, wire, color_for(wire))
    return out


def draw_label(out: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y = max(12, int(y))
    cv2.putText(out, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)


def make_labeled_review(rgb: np.ndarray, wires: list[dict[str, Any]]) -> np.ndarray:
    out = make_unlabeled_review(rgb, wires)
    for wire in wires:
        x1, y1, x2, y2 = [int(v) for v in wire.get("bbox", [0, 0, 0, 0])]
        color = color_for(wire)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    return out


def _object_debug_json(objects: list[Any]) -> list[dict[str, Any]]:
    result = []
    for obj in objects:
        result.append(obj.json() if hasattr(obj, "json") else asdict(obj) if is_dataclass(obj) else dict(obj))
    return result


def process_pdf(pdf_path: Path, args: argparse.Namespace, solid_cfg: Any, dash_cfg: Any) -> None:
    stem = pdf_path.stem
    input_dir = resolve_input_dir(stem)
    pages = select_pages(args.pages, available_pages(input_dir))
    output = make_output_dirs(stem, args.preserve)
    debug_bbox = dash_module.parse_debug_bbox(args.debug_bbox) if args.debug_bbox else None
    print(f"\nProcessing: {stem}")
    print(f"Input circuit_images ({SOURCE_STAGE}): {input_dir}")
    print(f"Output: {output['root']}")
    print(f"Pages: {pages}")
    review_pages: list[np.ndarray] = []
    result_pages: list[np.ndarray] = []
    summary_pages: list[dict[str, Any]] = []
    for page in pages:
        t0 = time.perf_counter()
        print(f"  Page {page}")
        src = input_dir / f"page_{page:03d}.png"
        rgb = load_rgb(src)
        height, width = rgb.shape[:2]
        binary = solid_module.foreground(rgb)
        wire_width = solid_module.estimate_wire_width(binary)
        solid_objects, solid_debug = solid_module.analyze_page(rgb, page, wire_width, solid_cfg, args.debug_short_groups, debug_bbox)
        dash_objects, dash_debug = dash_module.analyze_page(rgb, page, dash_cfg, debug_bbox=debug_bbox)
        wires, adapter_debug = backup_objects_to_wires(solid_objects, dash_objects)
        adapter_debug["dash_junction_dot_suppression"] = {
            "enabled": True,
            "method": "embedded_dash_analyze_page_removes_build_dot_mask_from_dash_binary",
            "num_junction_dots": int(dash_debug.get("num_junction_dots", 0)),
            "num_dot_pixels_removed": int(dash_debug.get("num_dot_pixels_removed", 0)),
            "display_junction_dots": False,
        }
        connections_debug = compute_connections(wires)
        wire_mask = np.zeros((height, width), dtype=bool)
        for wire in wires:
            paint_wire_bool(wire_mask, wire)
        labeled = make_labeled_review(rgb, wires)
        unlabeled = make_unlabeled_review(rgb, wires)
        image_path = output["images"] / f"page_{page:03d}.png"
        json_path = output["json"] / f"page_{page:03d}.json"
        debug_path = output["debug"] / f"page_{page:03d}.json"
        save_rgb(image_path, labeled, args.dpi)
        num_solid = sum(1 for wire in wires if str(wire["type"]).startswith("solid_wire"))
        num_diag = sum(1 for wire in wires if wire["type"] == "solid_wire_diagonal_extension")
        num_dash = sum(1 for wire in wires if wire["type"] == "dashed_wire_group")
        summary = {"num_solid_wires": int(num_solid), "num_diagonal_extensions": int(num_diag), "num_dashed_wires": int(num_dash), "total_wires": int(len(wires)), "wire_mask_pixels": int(wire_mask.sum())}
        save_json(json_path, {"pdf": pdf_path.name, "page": int(page), "stage": STAGE_NAME, "source": {"stage": SOURCE_STAGE, "circuit_image_path": str(src)}, "image_width": int(width), "image_height": int(height), "review_image_path": str(image_path), "summary": summary, "wires": wires})
        save_json(debug_path, {"solid_debug": solid_debug, "dashed_debug": dash_debug, "adapter_debug": adapter_debug, "connections_debug": connections_debug, "backup_objects": {"solid": _object_debug_json(solid_objects), "dash": _object_debug_json(dash_objects)}, "config": {"solid": asdict(solid_cfg), "dash": asdict(dash_cfg)}, "total_page_seconds": round(time.perf_counter() - t0, 4)})
        review_pages.append(labeled)
        result_pages.append(unlabeled)
        summary_pages.append({"page": int(page), **summary, "connections_total": int(connections_debug["total_connections"]), "review_image_path": f"images/page_{page:03d}.png", "json_path": f"json/page_{page:03d}.json", "debug_path": f"debug/page_{page:03d}.json"})
        print(f"    solid={num_solid} diagonal={num_diag} dashed={num_dash} total={len(wires)} mask_pixels={int(wire_mask.sum())} connections={connections_debug['total_connections']}")
    if review_pages:
        image_to_pdf(review_pages, output["root"] / "review.pdf", args.dpi)
        image_to_pdf(result_pages, output["root"] / "result.pdf", args.dpi)
    save_json(output["json"] / "summary.json", {"pdf": pdf_path.name, "stage": STAGE_NAME, "input_stage": SOURCE_STAGE, "input_circuit_images": str(input_dir), "output_root": str(output["root"]), "pages": summary_pages, "config": {"solid": asdict(solid_cfg), "dash": asdict(dash_cfg)}})
    print(f"Review PDF: {output['root'] / 'review.pdf'}")
    print(f"Result PDF: {output['root'] / 'result.pdf'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect solid and dashed wires from 03_rect circuit images using embedded backup algorithms.")
    parser.add_argument("--pdf", type=str, default="bmw-328i-1997.pdf")
    parser.add_argument("--pages", type=str, default="1-5")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("-p", "--preserve", action="store_true", help="Preserve the output folder instead of clearing it first.")
    parser.add_argument("--debug-short-groups", action="store_true")
    parser.add_argument("--debug-bbox")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    print("Page range:", "ALL" if args.pages is None else args.pages)
    solid_cfg = solid_module.JustWireConfig(dpi=args.dpi)
    dash_cfg = dash_module.DashWireConfig(
        dpi=args.dpi,
        group_min_representative_text_height_ratio=0.9,
        group_min_representative_wire_widths=4.5,
    )
    if args.pdf is not None:
        process_pdf(resolve_pdf(args.pdf), args, solid_cfg, dash_cfg)
        return
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF found under:\n{INPUT_DIR}")
    for pdf_path in pdf_files:
        process_pdf(pdf_path, args, solid_cfg, dash_cfg)


if __name__ == "__main__":
    main()
