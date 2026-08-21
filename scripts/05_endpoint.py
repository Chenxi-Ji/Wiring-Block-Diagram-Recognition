#!/usr/bin/env python3
"""Phase 5: endpoint classification and local net stitching.

Inputs:
  * outputs/03_rect/<pdf>/circuit_images/page_NNN.png
  * outputs/04_wire/<pdf>/json/page_NNN.json
  * outputs/03_rect/<pdf>/json/page_NNN.json

Outputs:
  * outputs/05_endpoint/<pdf>/image/page_NNN.png
  * outputs/05_endpoint/<pdf>/circuit_images/page_NNN.png
  * outputs/05_endpoint/<pdf>/json/page_NNN.json
  * outputs/05_endpoint/<pdf>/review.pdf
  * outputs/05_endpoint/<pdf>/result.pdf

This stage is intentionally self-contained except for pipeline_io.  It consumes
the previous phase artifacts, suppresses rectangle-owned wires/dash groups,
classifies external endpoints, records small endpoint-attached residual
structures, and assigns local net ids without merging distinct GND terminals.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
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

SOURCE_STAGE = "03_rect"
WIRE_STAGE = "04_wire"
RECT_STAGE = "03_rect"


VISUAL_COLORS: dict[str, tuple[int, int, int]] = {
    "solid_wire": (0, 190, 0),
    "dash_group": (255, 145, 0),
    "connection_dot": (0, 120, 255),
    "round_terminal_object": (0, 80, 255),
    "ground_terminal": (255, 0, 0),
    "small_connected_structure": (0, 210, 210),
    "arrow_structure": (0, 120, 255),
    "brace_structure": (190, 0, 125),
    "rectangle_frame": (170, 120, 45),
}

VISUAL_LEGEND: list[tuple[str, str, tuple[int, int, int]]] = [
    ("solid_wire", "retained solid wire", VISUAL_COLORS["solid_wire"]),
    ("dash_group", "retained dashed wire group", VISUAL_COLORS["dash_group"]),
    ("connection_dot", "connection dot inferred from disconnected endpoints; actual blob pixels only", VISUAL_COLORS["connection_dot"]),
    ("round_terminal_object", "round terminal inferred from one disconnected endpoint; actual blob pixels only", VISUAL_COLORS["round_terminal_object"]),
    ("ground_terminal", "local GND terminal inferred from disconnected endpoints; actual blob pixels only", VISUAL_COLORS["ground_terminal"]),
    ("small_connected_structure", "small endpoint-attached residual structure", VISUAL_COLORS["small_connected_structure"]),
    ("arrow_structure", "endpoint-attached arrow terminal structure", VISUAL_COLORS["arrow_structure"]),
    ("brace_structure", "flat endpoint-attached brace structure with V-shaped join", VISUAL_COLORS["brace_structure"]),
    ("rectangle_frame", "rectangle frame suppression boundary", VISUAL_COLORS["rectangle_frame"]),
]


@dataclass(frozen=True)
class EndpointConfig:
    dpi: int = 300
    black_threshold: int = 128
    rect_inside_padding_px: int = 1
    rect_endpoint_padding_px: int = 0
    endpoint_touch_radius_widths: float = 2.8
    endpoint_touch_radius_min_px: int = 4
    endpoint_search_radius_widths: float = 12.0
    endpoint_search_radius_min_px: int = 18
    endpoint_retreat_probe_widths: float = 5.0
    endpoint_retreat_max_width_ratio: float = 1.75
    endpoint_retreat_consecutive_normal: int = 2
    endpoint_retreat_min_span_widths: float = 1.5
    small_structure_max_area_widths2: float = 36.0
    small_structure_max_area_px: int = 120
    small_structure_max_bbox_widths: float = 12.0
    small_structure_hole_area_ratio: float = 0.035
    brace_min_length_widths: float = 22.0
    brace_max_thickness_widths: float = 8.0
    brace_min_aspect_ratio: float = 6.0
    brace_min_v_depth_widths: float = 2.0
    brace_min_v_span_widths: float = 3.0
    endpoint_dot_search_widths: float = 10.0
    endpoint_dot_backward_widths: float = 4.0
    endpoint_dot_lateral_tolerance_widths: float = 1.6
    endpoint_dot_min_diameter_widths: float = 1.2
    endpoint_dot_max_diameter_widths: float = 8.0
    endpoint_dot_min_fill_ratio: float = 0.62
    endpoint_dot_max_hole_area_ratio: float = 0.02
    endpoint_dot_min_hits: int = 2
    endpoint_dot_min_center_gap_widths: float = 1.0
    endpoint_dot_gnd_veto_search_widths: float = 7.0
    ground_search_forward_widths: float = 22.0
    ground_search_backward_widths: float = 4.0
    ground_search_cross_widths: float = 9.0
    ground_plate_min_length_widths: float = 2.0
    ground_plate_max_thickness_widths: float = 2.2
    ground_min_plate_count: int = 3
    arrow_min_area_widths2: float = 8.0
    arrow_min_bbox_widths: float = 2.6
    arrow_max_bbox_widths: float = 12.0
    arrow_max_aspect_ratio: float = 3.2
    arrow_min_width_drop_ratio: float = 0.30
    arrow_max_tip_width_ratio: float = 0.55


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def foreground(rgb: np.ndarray, threshold: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < int(threshold)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_rgb(path: Path, img: np.ndarray, dpi: int) -> None:
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


def clip_box(box: Sequence[float | int], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    return [max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2))]


def box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def box_intersection(a: Sequence[int], b: Sequence[int]) -> list[int] | None:
    x1, y1 = max(int(a[0]), int(b[0])), max(int(a[1]), int(b[1]))
    x2, y2 = min(int(a[2]), int(b[2])), min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_intersects(a: Sequence[int], b: Sequence[int]) -> bool:
    return box_intersection(a, b) is not None


def box_inside(inner: Sequence[int], outer: Sequence[int], pad: int = 0) -> bool:
    return (
        int(inner[0]) >= int(outer[0]) - pad
        and int(inner[1]) >= int(outer[1]) - pad
        and int(inner[2]) <= int(outer[2]) + pad
        and int(inner[3]) <= int(outer[3]) + pad
    )


def point_inside_box(point: Sequence[float], box: Sequence[int], pad: int = 0) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        float(box[0] - pad) <= x <= float(box[2] + pad - 1)
        and float(box[1] - pad) <= y <= float(box[3] + pad - 1)
    )


def mask_to_relative_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for y, row in enumerate(mask.astype(bool)):
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        start = int(xs[0])
        prev = int(xs[0])
        for value in xs[1:]:
            value = int(value)
            if value == prev + 1:
                prev = value
            else:
                runs.append([int(y), start, prev + 1])
                start = value
                prev = value
        runs.append([int(y), start, prev + 1])
    return runs


def relative_runs_to_mask(runs: Sequence[Sequence[int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for run in runs or []:
        if len(run) != 3:
            continue
        y, x1, x2 = [int(v) for v in run]
        if 0 <= y < shape[0]:
            mask[y, max(0, x1) : min(shape[1], x2)] = True
    return mask


def object_pixel_mask(obj: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    attrs = obj.get("attributes", {}) or {}
    top_mask = obj.get("mask", {}) or {}
    bbox = top_mask.get("bbox")
    runs = top_mask.get("runs")
    if isinstance(bbox, list) and len(bbox) == 4 and runs:
        full = np.zeros(shape, dtype=bool)
        x1, y1, x2, y2 = clip_box(bbox, shape)
        if x2 > x1 and y2 > y1:
            local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
            full[y1:y2, x1:x2] |= local
        return full
    candidates = [
        ("wire_pixel_bbox", "wire_pixel_runs"),
        ("pixel_bbox", "pixel_runs"),
        ("edge_expansion_bbox", "edge_expansion_pixel_runs"),
    ]
    full = np.zeros(shape, dtype=bool)
    for bbox_key, runs_key in candidates:
        bbox = attrs.get(bbox_key)
        runs = attrs.get(runs_key)
        if isinstance(bbox, list) and len(bbox) == 4 and runs:
            x1, y1, x2, y2 = clip_box(bbox, shape)
            if x2 <= x1 or y2 <= y1:
                continue
            local = relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
            full[y1:y2, x1:x2] |= local
            return full
    bbox = obj.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = clip_box(bbox, shape)
        if x2 > x1 and y2 > y1:
            full[y1:y2, x1:x2] = True
    return full


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def draw_mask(out: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 1.0) -> None:
    if not bool(np.any(mask)):
        return
    if alpha >= 1.0:
        out[mask] = color
        return
    color_arr = np.array(color, dtype=np.float32)
    pixels = out[mask].astype(np.float32)
    out[mask] = np.clip((1.0 - alpha) * pixels + alpha * color_arr, 0, 255).astype(np.uint8)


def endpoint_disk(shape: tuple[int, int], point: Sequence[float], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(round(float(point[0]))), int(round(float(point[1])))), int(radius), 255, -1)
    return mask.astype(bool)


def direction_from_points(points: Sequence[Sequence[float]], endpoint_index: int) -> np.ndarray | None:
    if len(points) < 2:
        return None
    p0 = np.array(points[0], dtype=float)
    p1 = np.array(points[-1], dtype=float)
    vec = p0 - p1 if endpoint_index == 0 else p1 - p0
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-6:
        return None
    return vec / norm


def line_points_from_object(obj: dict[str, Any]) -> list[list[float]]:
    geom = obj.get("geometry", {}) or {}
    points = geom.get("centerline_points") or geom.get("points")
    if isinstance(points, list) and len(points) >= 2:
        return [[float(points[0][0]), float(points[0][1])], [float(points[-1][0]), float(points[-1][1])]]
    bbox = obj.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    orientation = str((obj.get("attributes", {}) or {}).get("orientation", ""))
    if orientation == "vertical":
        cx = (x1 + x2 - 1.0) / 2.0
        return [[cx, y1], [cx, y2 - 1.0]]
    cy = (y1 + y2 - 1.0) / 2.0
    return [[x1, cy], [x2 - 1.0, cy]]


def dash_group_mask(
    group: dict[str, Any],
    member_by_segment_id: dict[str, dict[str, Any]],
    binary: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(binary.shape, dtype=bool)
    attrs = group.get("attributes", {}) or {}
    member_ids = [str(v) for v in attrs.get("member_ids", []) or []]
    for segment_id in member_ids:
        member = member_by_segment_id.get(segment_id)
        if member is not None:
            mask |= object_pixel_mask(member, binary.shape)
    if bool(np.any(mask)):
        return mask
    x1, y1, x2, y2 = clip_box(group.get("bbox", [0, 0, 0, 0]), binary.shape)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
    return mask


def dash_group_points(group: dict[str, Any]) -> list[list[float]]:
    bbox = group.get("bbox", [0, 0, 0, 0])
    attrs = group.get("attributes", {}) or {}
    orientation = str(attrs.get("orientation", "horizontal"))
    center = float(attrs.get("centerline", 0.0))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if orientation == "vertical":
        cx = center if center else (x1 + x2 - 1.0) / 2.0
        return [[cx, y1], [cx, y2 - 1.0]]
    cy = center if center else (y1 + y2 - 1.0) / 2.0
    return [[x1, cy], [x2 - 1.0, cy]]


def available_clean_pages(clean_dir: Path) -> list[int]:
    pages: set[int] = set()
    for path in clean_dir.glob("page_*.png"):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            pages.add(int(digits[-3:]))
    return sorted(pages)


def available_stage_json_pages(stage: str, pdf_stem: str) -> list[int]:
    json_dir = OUTPUT_DIR / stage / pdf_stem / "json"
    pages: set[int] = set()
    if not json_dir.is_dir():
        return []
    for path in json_dir.glob("page_*.json"):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            pages.add(int(digits[-3:]))
    return sorted(pages)


def parse_pages(spec: str | None, available_pages: Sequence[int]) -> list[int]:
    available = set(int(page) for page in available_pages)
    if not spec:
        return sorted(available)
    raw_spec = spec
    requested: set[int] = set()
    for part in raw_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(v.strip()) for v in part.split("-", 1)]
            requested.update(range(start, end + 1))
        else:
            requested.add(int(part))
    pages = sorted(page for page in requested if page in available)
    missing = sorted(requested - available)
    if missing:
        raise FileNotFoundError(f"Requested page(s) missing from one or more required inputs: {missing}")
    if not pages:
        raise ValueError(f"No pages selected from spec: {raw_spec}")
    return pages


def resolve_pdf(name: str) -> Path:
    pdf = Path(name)
    if not pdf.is_absolute():
        pdf = INPUT_DIR / pdf
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    return pdf


def resolve_source_circuit_dir(pdf_stem: str) -> Path:
    circuit_dir = OUTPUT_DIR / SOURCE_STAGE / pdf_stem / "circuit_images"
    if not circuit_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {SOURCE_STAGE} circuit images: {circuit_dir}. "
            f"Run: python scripts/03_rect.py --pdf {pdf_stem}.pdf"
        )
    return circuit_dir


def load_source_image(circuit_dir: Path, page: int) -> np.ndarray:
    candidates = [
        circuit_dir / f"page_{page:03d}.png",
        circuit_dir / f"page_{page:03d}_clean.png",
    ]
    for path in candidates:
        if path.is_file():
            return np.array(Image.open(path).convert("RGB"))
    raise FileNotFoundError(f"Missing source image for page {page:03d} under {circuit_dir}")


def load_stage_page_json(stage: str, pdf_stem: str, page: int) -> dict[str, Any]:
    path = OUTPUT_DIR / stage / pdf_stem / "json" / f"page_{page:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {stage} JSON: {path}")
    return read_json(path)


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    resolved_output = root.resolve()
    resolved_stage = (OUTPUT_DIR / STAGE_NAME).resolve()
    if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
    shutil.rmtree(root)


def make_output_dirs(stem: str) -> tuple[Path, Path, Path, Path]:
    root = OUTPUT_DIR / STAGE_NAME / stem
    clear_output_root(root)
    image_dir = root / "image"
    circuit_image_dir = root / "circuit_images"
    json_dir = root / "json"
    for path in (image_dir, circuit_image_dir, json_dir):
        path.mkdir(parents=True, exist_ok=True)
    return root, image_dir, circuit_image_dir, json_dir


def write_visual_legend(path: Path) -> None:
    lines = [f"{STAGE_NAME} visual legend", ""]
    for key, description, rgb in VISUAL_LEGEND:
        lines.append(f"{key}: #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X} RGB{rgb} - {description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rect_frame_bboxes(rect_json: dict[str, Any]) -> list[list[int]]:
    frames: list[list[int]] = []
    for rect in rect_json.get("rects", []) or []:
        bbox = rect.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            frames.append([int(v) for v in bbox])
    for obj in rect_json.get("objects", []) or []:
        if str(obj.get("type", "")).endswith("_rectangle_frame"):
            bbox = obj.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                frames.append([int(v) for v in bbox])
    return frames


def rect_source_ids(rect_json: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    source_object_ids: set[str] = set()
    source_segment_ids: set[str] = set()
    source_group_ids: set[str] = set()
    for obj in rect_json.get("objects", []) or []:
        obj_type = str(obj.get("type", ""))
        if obj_type not in {"solid_rectangle_edge", "dashed_rectangle_edge_group"}:
            continue
        attrs = obj.get("attributes", {}) or {}
        for key in ("source_object_id",):
            value = attrs.get(key)
            if value:
                source_object_ids.add(str(value))
        for key in ("source_segment_id",):
            value = attrs.get(key)
            if value:
                source_segment_ids.add(str(value))
        for key in ("source_group_id", "dash_group_id"):
            value = attrs.get(key)
            if value:
                source_group_ids.add(str(value))
    return source_object_ids, source_segment_ids, source_group_ids


def rectangle_masks(rect_json: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    edge_mask = np.zeros(shape, dtype=bool)
    fill_mask = np.zeros(shape, dtype=bool)
    for frame in rect_frame_bboxes(rect_json):
        x1, y1, x2, y2 = clip_box(frame, shape)
        if x2 > x1 and y2 > y1:
            fill_mask[y1:y2, x1:x2] = True
    for rect in rect_json.get("rects", []) or []:
        mask_info = rect.get("mask", {}) or {}
        bbox = mask_info.get("bbox")
        runs = mask_info.get("runs") or mask_info.get("pixel_runs")
        if isinstance(bbox, list) and len(bbox) == 4 and runs:
            x1, y1, x2, y2 = clip_box(bbox, shape)
            if x2 > x1 and y2 > y1:
                edge_mask[y1:y2, x1:x2] |= relative_runs_to_mask(runs, (y2 - y1, x2 - x1))
    for obj in rect_json.get("objects", []) or []:
        if str(obj.get("type", "")) in {"solid_rectangle_edge", "dashed_rectangle_edge_group"}:
            edge_mask |= object_pixel_mask(obj, shape)
    return edge_mask, fill_mask


def estimate_wire_width_from_json(*stage_jsons: dict[str, Any]) -> float:
    values: list[float] = []
    for data in stage_jsons:
        for obj in list(data.get("objects", []) or []) + list(data.get("wires", []) or []):
            attrs = obj.get("attributes", {}) or {}
            for key in ("estimated_wire_width", "median_width", "representative_width"):
                try:
                    value = float(attrs.get(key, 0.0))
                except (TypeError, ValueError):
                    continue
                if 0.5 <= value <= 20.0:
                    values.append(value)
            try:
                value = float((obj.get("geometry", {}) or {}).get("thickness_px", 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if 0.5 <= value <= 20.0:
                values.append(value)
    if not values:
        return 2.0
    return float(np.clip(np.percentile(np.array(values, dtype=float), 50), 1.0, 10.0))


def scan_cross_width(binary: np.ndarray, orientation: str, axis: int, cross_center: float, search: int) -> int:
    h, w = binary.shape
    if orientation == "horizontal":
        if axis < 0 or axis >= w:
            return 0
        y1 = max(0, int(round(cross_center)) - search)
        y2 = min(h, int(round(cross_center)) + search + 1)
        ys = np.flatnonzero(binary[y1:y2, axis])
        return 0 if ys.size == 0 else int(ys.max() - ys.min() + 1)
    if axis < 0 or axis >= h:
        return 0
    x1 = max(0, int(round(cross_center)) - search)
    x2 = min(w, int(round(cross_center)) + search + 1)
    xs = np.flatnonzero(binary[axis, x1:x2])
    return 0 if xs.size == 0 else int(xs.max() - xs.min() + 1)


def update_item_from_mask(item: dict[str, Any]) -> None:
    bbox = mask_bbox(item["mask"])
    if bbox is None:
        return
    item["bbox"] = bbox
    obj = item["object"]
    obj["bbox"] = [int(v) for v in bbox]
    attrs = obj.setdefault("attributes", {})
    attrs["wire_pixel_bbox"] = [int(v) for v in bbox]
    attrs["wire_pixel_runs"] = mask_to_relative_runs(item["mask"][bbox[1] : bbox[3], bbox[0] : bbox[2]])
    attrs["wire_pixel_count"] = int(np.count_nonzero(item["mask"]))
    orientation = str(item.get("orientation", ""))
    if orientation == "vertical":
        cx = (bbox[0] + bbox[2] - 1.0) / 2.0
        item["points"] = [[cx, float(bbox[1])], [cx, float(bbox[3] - 1)]]
    else:
        cy = (bbox[1] + bbox[3] - 1.0) / 2.0
        item["points"] = [[float(bbox[0]), cy], [float(bbox[2] - 1), cy]]
    obj["geometry"] = {
        "kind": "rectangle",
        "rect": [int(v) for v in bbox],
        "centerline_points": [[round(float(v), 3) for v in point] for point in item["points"]],
    }


def retreat_endpoint_if_wide(
    item: dict[str, Any],
    endpoint_index: int,
    binary: np.ndarray,
    wire_width: float,
    cfg: EndpointConfig,
) -> dict[str, Any]:
    orientation = str(item.get("orientation", ""))
    if orientation not in {"horizontal", "vertical"}:
        return {"retreated": False, "decision": "skip_non_axis_orientation"}
    points = item.get("points", [])
    if len(points) < 2:
        return {"retreated": False, "decision": "missing_points"}
    bbox = [int(v) for v in item.get("bbox", [])]
    if len(bbox) != 4 or box_area(bbox) <= 0:
        return {"retreated": False, "decision": "empty_bbox"}
    representative_width = float((item.get("object", {}).get("attributes", {}) or {}).get("median_width", wire_width))
    normal_limit = max(1, int(math.ceil(representative_width * cfg.endpoint_retreat_max_width_ratio)))
    search = max(2, int(round(wire_width * cfg.endpoint_retreat_probe_widths)))
    consecutive_needed = max(1, int(cfg.endpoint_retreat_consecutive_normal))
    min_span = max(1.0, wire_width * cfg.endpoint_retreat_min_span_widths)
    old_bbox = [int(v) for v in bbox]
    if orientation == "horizontal":
        cross_center = float(points[endpoint_index][1])
        axis_values = range(bbox[0], bbox[2]) if endpoint_index == 0 else range(bbox[2] - 1, bbox[0] - 1, -1)
    else:
        cross_center = float(points[endpoint_index][0])
        axis_values = range(bbox[1], bbox[3]) if endpoint_index == 0 else range(bbox[3] - 1, bbox[1] - 1, -1)
    widths: list[dict[str, Any]] = []
    stable_axis: int | None = None
    consecutive = 0
    first_width = 0
    for axis in axis_values:
        cross_width = scan_cross_width(binary, orientation, int(axis), cross_center, search)
        widths.append({"axis": int(axis), "cross_width": int(cross_width), "normal": bool(cross_width <= normal_limit)})
        if len(widths) == 1:
            first_width = int(cross_width)
        if cross_width <= normal_limit:
            consecutive += 1
            if consecutive >= consecutive_needed:
                stable_axis = int(axis)
                break
        else:
            consecutive = 0
    if first_width <= normal_limit:
        return {
            "retreated": False,
            "decision": "endpoint_width_already_normal",
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
            "scan_widths": widths[:24],
        }
    new_mask = item["mask"].copy()
    if orientation == "horizontal":
        if endpoint_index == 0:
            new_mask[:, :stable_axis] = False
        else:
            new_mask[:, stable_axis + 1 :] = False
    else:
        if endpoint_index == 0:
            new_mask[:stable_axis, :] = False
        else:
            new_mask[stable_axis + 1 :, :] = False
    new_bbox = mask_bbox(new_mask)
    if new_bbox is None:
        return {"retreated": False, "decision": "retreat_removed_all_pixels", "old_bbox": old_bbox}
    new_span = (new_bbox[2] - new_bbox[0]) if orientation == "horizontal" else (new_bbox[3] - new_bbox[1])
    if float(new_span) < min_span:
        return {
            "retreated": False,
            "decision": "retreat_would_make_wire_too_short",
            "old_bbox": old_bbox,
            "proposed_bbox": [int(v) for v in new_bbox],
            "proposed_span": round(float(new_span), 3),
        }
    old_count = int(np.count_nonzero(item["mask"]))
    item["mask"] = new_mask
    update_item_from_mask(item)
    return {
        "retreated": True,
        "decision": "retreated_to_stable_normal_width",
        "old_bbox": old_bbox,
        "new_bbox": [int(v) for v in item["bbox"]],
        "endpoint_index": int(endpoint_index),
        "stable_axis": int(stable_axis),
        "normal_width_limit": int(normal_limit),
        "first_cross_width": int(first_width),
        "removed_pixels": int(old_count - np.count_nonzero(new_mask)),
        "scan_widths": widths[:24],
    }


def wire_kind_from_04_wire(obj: dict[str, Any]) -> str | None:
    obj_type = str(obj.get("type", ""))
    if obj_type == "dashed_wire_group":
        return "dash_group"
    if obj_type.startswith("solid_wire"):
        return "solid_wire"
    return None


def wire_orientation(obj: dict[str, Any]) -> str:
    attrs = obj.get("attributes", {}) or {}
    value = obj.get("orientation") or attrs.get("orientation")
    if value:
        return str(value)
    points = (obj.get("geometry", {}) or {}).get("centerline_points") or []
    if isinstance(points, list) and len(points) >= 2:
        dx = abs(float(points[-1][0]) - float(points[0][0]))
        dy = abs(float(points[-1][1]) - float(points[0][1]))
        if dx >= dy * 2.0:
            return "horizontal"
        if dy >= dx * 2.0:
            return "vertical"
        return "diagonal"
    return ""


def normalized_04_wire_object(obj: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(obj)
    attrs = out.setdefault("attributes", {})
    orientation = wire_orientation(out)
    if orientation:
        attrs.setdefault("orientation", orientation)
    obj_type = str(out.get("type", ""))
    if obj_type == "dashed_wire_group":
        attrs.setdefault("dash_group_id", str(out.get("id", "")))
        attrs.setdefault("representative_width", (out.get("geometry", {}) or {}).get("thickness_px"))
    else:
        attrs.setdefault("segment_id", str(out.get("id", "")))
        attrs.setdefault("median_width", (out.get("geometry", {}) or {}).get("thickness_px"))
    return out


def make_line_items_from_wire_json(
    binary: np.ndarray,
    wire_json: dict[str, Any],
    rect_json: dict[str, Any],
    cfg: EndpointConfig,
    wire_width: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frames = rect_frame_bboxes(rect_json)
    diagnostics: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    for raw_obj in wire_json.get("wires", []) or []:
        kind = wire_kind_from_04_wire(raw_obj)
        if kind is None:
            diagnostics.append(
                {
                    "object_id": str(raw_obj.get("id", "")),
                    "type": str(raw_obj.get("type", "")),
                    "decision": "ignored",
                    "reason": "unsupported_04_wire_type",
                }
            )
            continue
        obj = normalized_04_wire_object(raw_obj)
        obj_id = str(obj.get("id", ""))
        bbox = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
        reason = ""
        if any(box_inside(bbox, frame, cfg.rect_inside_padding_px) for frame in frames):
            reason = "bbox_inside_rectangle_frame"
        if reason:
            diagnostics.append({"object_id": obj_id, "type": obj.get("type", ""), "bbox": bbox, "decision": "suppressed", "reason": reason})
            continue
        if kind == "dash_group":
            partial_frames = [frame for frame in frames if box_intersects(bbox, frame)]
            if partial_frames:
                diagnostics.append(
                    {
                        "object_id": obj_id,
                        "type": obj.get("type", ""),
                        "bbox": bbox,
                        "decision": "retained_with_partial_rectangle_overlap_warning",
                        "overlap_frame_bboxes": partial_frames[:6],
                    }
                )
        mask = object_pixel_mask(obj, binary.shape)
        item = {
            "id": obj_id,
            "kind": kind,
            "object": obj,
            "mask": mask,
            "bbox": bbox,
            "orientation": wire_orientation(obj),
            "points": line_points_from_object(obj),
            "endpoint_events": [],
            "suppression_status": "retained",
        }
        if kind == "solid_wire":
            for endpoint_index, point in enumerate(list(item["points"])):
                if any(point_inside_box(point, frame, cfg.rect_endpoint_padding_px) for frame in frames):
                    event = retreat_endpoint_if_wide(item, endpoint_index, binary, wire_width, cfg)
                    event.update({"object_id": obj_id, "endpoint_index": int(endpoint_index), "point_before": [round(float(v), 3) for v in point]})
                    item["endpoint_events"].append(event)
                    diagnostics.append({"object_id": obj_id, "type": obj.get("type", ""), "decision": "partial_rectangle_endpoint_retreat", **event})
        retained.append(item)
    return retained, symbols, diagnostics


def make_line_items(
    binary: np.ndarray,
    solid_json: dict[str, Any],
    dash_json: dict[str, Any],
    rect_json: dict[str, Any],
    cfg: EndpointConfig,
    wire_width: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if "wires" in solid_json:
        return make_line_items_from_wire_json(binary, solid_json, rect_json, cfg, wire_width)
    frames = rect_frame_bboxes(rect_json)
    source_object_ids, source_segment_ids, source_group_ids = rect_source_ids(rect_json)
    diagnostics: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []

    for obj in solid_json.get("objects", []) or []:
        obj_type = str(obj.get("type", ""))
        attrs = obj.get("attributes", {}) or {}
        if obj_type == "junction_dot":
            diagnostics.append(
                {
                    "object_id": str(obj.get("id", "")),
                    "type": obj_type,
                    "bbox": [int(v) for v in obj.get("bbox", [0, 0, 0, 0])],
                    "decision": "ignored",
                    "reason": "phase2_junction_dot_is_not_authoritative_for_phase5",
                }
            )
            continue
        if obj_type != "solid_wire":
            continue
        obj_id = str(obj.get("id", ""))
        segment_id = str(attrs.get("segment_id", ""))
        bbox = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
        reason = ""
        if obj_id in source_object_ids or segment_id in source_segment_ids:
            reason = "source_used_as_rectangle_edge"
        elif any(box_inside(bbox, frame, cfg.rect_inside_padding_px) for frame in frames):
            reason = "bbox_inside_rectangle_frame"
        if reason:
            diagnostics.append({"object_id": obj_id, "type": obj_type, "bbox": bbox, "decision": "suppressed", "reason": reason})
            continue
        mask = object_pixel_mask(obj, binary.shape)
        item = {
            "id": obj_id,
            "kind": "solid_wire",
            "object": copy.deepcopy(obj),
            "mask": mask,
            "bbox": bbox,
            "orientation": str(attrs.get("orientation", "")),
            "points": line_points_from_object(obj),
            "endpoint_events": [],
            "suppression_status": "retained",
        }
        for endpoint_index, point in enumerate(list(item["points"])):
            if any(point_inside_box(point, frame, cfg.rect_endpoint_padding_px) for frame in frames):
                event = retreat_endpoint_if_wide(item, endpoint_index, binary, wire_width, cfg)
                event.update({"object_id": obj_id, "endpoint_index": int(endpoint_index), "point_before": [round(float(v), 3) for v in point]})
                item["endpoint_events"].append(event)
                diagnostics.append({"object_id": obj_id, "type": obj_type, "decision": "partial_rectangle_endpoint_retreat", **event})
        retained.append(item)

    dash_members = {
        str((obj.get("attributes", {}) or {}).get("segment_id", "")): obj
        for obj in dash_json.get("objects", []) or []
        if str(obj.get("type", "")) == "dash_member"
    }
    for obj in dash_json.get("objects", []) or []:
        if str(obj.get("type", "")) != "dash_group":
            continue
        attrs = obj.get("attributes", {}) or {}
        obj_id = str(obj.get("id", ""))
        group_id = str(attrs.get("dash_group_id", obj_id))
        bbox = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
        reason = ""
        if obj_id in source_group_ids or group_id in source_group_ids:
            reason = "source_used_as_dashed_rectangle_edge"
        elif any(box_inside(bbox, frame, cfg.rect_inside_padding_px) for frame in frames):
            reason = "bbox_inside_rectangle_frame"
        if reason:
            diagnostics.append({"object_id": obj_id, "type": "dash_group", "bbox": bbox, "decision": "suppressed", "reason": reason})
            continue
        partial_frames = [frame for frame in frames if box_intersects(bbox, frame)]
        if partial_frames:
            diagnostics.append(
                {
                    "object_id": obj_id,
                    "type": "dash_group",
                    "bbox": bbox,
                    "decision": "retained_with_partial_rectangle_overlap_warning",
                    "overlap_frame_bboxes": partial_frames[:6],
                }
            )
        mask = dash_group_mask(obj, dash_members, binary)
        retained.append(
            {
                "id": obj_id,
                "kind": "dash_group",
                "object": copy.deepcopy(obj),
                "mask": mask,
                "bbox": bbox,
                "orientation": str(attrs.get("orientation", "")),
                "points": dash_group_points(obj),
                "endpoint_events": [],
                "suppression_status": "retained",
            }
        )
    return retained, symbols, diagnostics


def build_owner_label_map(nodes: Sequence[dict[str, Any]], shape: tuple[int, int]) -> tuple[np.ndarray, dict[int, str]]:
    labels = np.zeros(shape, dtype=np.int32)
    label_to_id: dict[int, str] = {}
    for index, node in enumerate(nodes, start=1):
        labels[node["mask"]] = int(index)
        label_to_id[int(index)] = str(node["id"])
    return labels, label_to_id


def point_to_box_distance(point: Sequence[float], box: Sequence[int]) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    dx = max(x1 - px, px - (x2 - 1.0), 0.0)
    dy = max(y1 - py, py - (y2 - 1.0), 0.0)
    return float(math.hypot(dx, dy))


def owner_ids_connected_to_endpoint_pixels(
    nodes: Sequence[dict[str, Any]],
    current_id: str,
    touch_mask: np.ndarray,
    point: Sequence[float],
    radius: int,
    binary: np.ndarray,
) -> list[str]:
    current_node = next((node for node in nodes if str(node["id"]) == current_id), None)
    if current_node is None:
        return []
    seed_mask = current_node["mask"] & touch_mask
    if not bool(np.any(seed_mask)):
        return []
    h, w = binary.shape
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x1 = max(0, x - int(radius))
    x2 = min(w, x + int(radius) + 1)
    y1 = max(0, y - int(radius))
    y2 = min(h, y + int(radius) + 1)
    if x2 <= x1 or y2 <= y1:
        return []
    owner_ids: list[str] = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id == current_id:
            continue
        bbox = node.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4 and point_to_box_distance(point, [int(v) for v in bbox]) > float(radius):
            continue
        candidate_mask = node["mask"] & touch_mask
        if not bool(np.any(candidate_mask)):
            continue
        pair_mask = (current_node["mask"] | node["mask"])[y1:y2, x1:x2]
        if not bool(np.any(pair_mask)):
            continue
        _count, local_labels = cv2.connectedComponents(pair_mask.astype(np.uint8), connectivity=8)
        local_seed = seed_mask[y1:y2, x1:x2]
        seed_component_labels = {int(v) for v in np.unique(local_labels[local_seed]) if int(v) > 0}
        if not seed_component_labels:
            continue
        candidate_component_labels = {int(v) for v in np.unique(local_labels[candidate_mask[y1:y2, x1:x2]]) if int(v) > 0}
        if seed_component_labels & candidate_component_labels:
            owner_ids.append(node_id)
    return sorted(set(owner_ids))


def union_nodes_by_connected_pixels(nodes: Sequence[dict[str, Any]], uf: UnionFind) -> list[dict[str, Any]]:
    union_mask = np.zeros(nodes[0]["mask"].shape, dtype=bool) if nodes else np.zeros((0, 0), dtype=bool)
    for node in nodes:
        uf.add(str(node["id"]))
        union_mask |= node["mask"]
    if not nodes or not bool(np.any(union_mask)):
        return []
    _count, labels = cv2.connectedComponents(union_mask.astype(np.uint8), connectivity=8)
    component_to_nodes: dict[int, set[str]] = defaultdict(set)
    for node in nodes:
        touched = labels[node["mask"]]
        for label in np.unique(touched[touched > 0]):
            component_to_nodes[int(label)].add(str(node["id"]))
    events: list[dict[str, Any]] = []
    for label, node_ids in component_to_nodes.items():
        ordered = sorted(node_ids)
        if len(ordered) < 2:
            continue
        first = ordered[0]
        for other in ordered[1:]:
            uf.union(first, other)
        events.append({"component_label": int(label), "node_ids": ordered})
    return events


def compact_dot_candidates(foreground_mask: np.ndarray, wire_width: float, cfg: EndpointConfig) -> list[dict[str, Any]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground_mask.astype(np.uint8),
        connectivity=8,
    )
    min_d = max(2.0, wire_width * cfg.endpoint_dot_min_diameter_widths)
    max_d = max(min_d + 1.0, wire_width * cfg.endpoint_dot_max_diameter_widths)
    candidates: list[dict[str, Any]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        diameter = max(float(bw), float(bh))
        if diameter < min_d or diameter > max_d:
            continue
        aspect = max(float(bw), float(bh)) / max(1.0, min(float(bw), float(bh)))
        if aspect > 1.8:
            continue
        fill = area / max(1.0, float(bw * bh))
        if fill < cfg.endpoint_dot_min_fill_ratio:
            continue
        local_mask = labels[y : y + bh, x : x + bw] == label
        hole_area = internal_hole_area(local_mask)
        hole_ratio = float(hole_area) / max(1.0, float(area))
        if hole_ratio > cfg.endpoint_dot_max_hole_area_ratio:
            continue
        cx, cy = map(float, centroids[label])
        candidates.append(
            {
                "label": int(label),
                "bbox": [x, y, x + bw, y + bh],
                "center": [cx, cy],
                "area": int(area),
                "diameter": round(float(diameter), 3),
                "fill_ratio": round(float(fill), 3),
                "hole_area": int(hole_area),
                "hole_area_ratio": round(float(hole_ratio), 4),
                "pixel_mask": labels == label,
            }
        )
    return candidates


def candidate_has_ground_icon_context(
    foreground_mask: np.ndarray,
    candidate: dict[str, Any],
    wire_width: float,
    cfg: EndpointConfig,
) -> bool:
    h, w = foreground_mask.shape
    x1, y1, x2, y2 = map(int, candidate["bbox"])
    cx, cy = map(float, candidate["center"])
    diameter = float(candidate.get("diameter", max(x2 - x1, y2 - y1)))
    pad_x = max(8, int(round(wire_width * cfg.endpoint_dot_gnd_veto_search_widths)))
    pad_y = max(8, int(round(wire_width * cfg.endpoint_dot_gnd_veto_search_widths)))
    rx1, rx2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    ry1, ry2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return False
    roi = foreground_mask[ry1:ry2, rx1:rx2].astype(np.uint8).copy()
    bx1 = max(0, x1 - rx1 - 1)
    bx2 = min(roi.shape[1], x2 - rx1 + 1)
    by1 = max(0, y1 - ry1 - 1)
    by2 = min(roi.shape[0], y2 - ry1 + 1)
    roi[by1:by2, bx1:bx2] = 0

    def has_parallel_plates(mask: np.ndarray, local_center: tuple[float, float]) -> bool:
        row_threshold = max(2, int(round(wire_width * 0.8)))
        row_runs = projection_runs(np.sum(mask.astype(bool), axis=1) >= row_threshold)
        row_groups: list[list[dict[str, Any]]] = []
        min_width = max(3.0, diameter * 0.28, wire_width * 0.8)
        max_width = max(min_width + 1.0, diameter * 2.2, wire_width * 5.0)
        for a, b in row_runs:
            col_runs = projection_runs(np.any(mask[a:b, :].astype(bool), axis=0))
            row_bars: list[dict[str, Any]] = []
            for ca, cb in col_runs:
                width = int(cb - ca)
                if width < min_width or width > max_width:
                    continue
                row_bars.append({"run": [int(a), int(b)], "x1": int(ca), "x2": int(cb), "width": width})
            if row_bars:
                row_groups.append(row_bars)
        for idx in range(0, max(0, len(row_groups) - cfg.ground_min_plate_count + 1)):
            for combo in itertools.product(*row_groups[idx : idx + cfg.ground_min_plate_count]):
                group = [dict(v) for v in combo]
                widths = [int(v["width"]) for v in group]
                if widths[0] < diameter * 0.55 or widths[0] > diameter * 2.2:
                    continue
                min_drop = max(2, int(round(wire_width * 0.7)))
                if any((left - right) < min_drop for left, right in zip(widths, widths[1:])):
                    continue
                centers = [(float(v["x1"]) + float(v["x2"]) - 1.0) / 2.0 for v in group]
                if max(centers) - min(centers) > max(5.0, wire_width * 1.8):
                    continue
                if abs(float(np.median(np.array(centers, dtype=float))) - local_center[0]) > max(diameter * 2.5, wire_width * 5.0):
                    continue
                gaps_ok = all((group[i + 1]["run"][0] - group[i]["run"][1]) <= max(8.0, wire_width * 2.5) for i in range(len(group) - 1))
                if gaps_ok:
                    return True
        return False

    center = (float(cx - rx1), float(cy - ry1))
    if has_parallel_plates(roi, center):
        return True
    return has_parallel_plates(roi.T, (center[1], center[0]))


def projection_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active.astype(bool).tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(active.shape[0])))
    return runs


def orient_forward_crop(
    binary: np.ndarray,
    point: Sequence[float],
    outward: Sequence[float],
    forward: int,
    backward: int,
    cross: int,
) -> tuple[np.ndarray, list[int], str] | None:
    h, w = binary.shape
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    unit = np.array(outward, dtype=float)
    norm = float(np.linalg.norm(unit))
    if norm <= 1.0e-6:
        return None
    unit = unit / norm
    if abs(unit[1]) >= abs(unit[0]):
        x1 = max(0, x - cross)
        x2 = min(w, x + cross + 1)
        if unit[1] >= 0:
            y1 = max(0, y - backward)
            y2 = min(h, y + forward + 1)
            orientation = "vertical_down"
            crop = binary[y1:y2, x1:x2]
        else:
            y1 = max(0, y - forward)
            y2 = min(h, y + backward + 1)
            orientation = "vertical_up"
            crop = np.flipud(binary[y1:y2, x1:x2])
        return crop.astype(bool), [x1, y1, x2, y2], orientation
    y1 = max(0, y - cross)
    y2 = min(h, y + cross + 1)
    if unit[0] >= 0:
        x1 = max(0, x - backward)
        x2 = min(w, x + forward + 1)
        orientation = "horizontal_right"
        crop = np.rot90(binary[y1:y2, x1:x2], k=3)
    else:
        x1 = max(0, x - forward)
        x2 = min(w, x + backward + 1)
        orientation = "horizontal_left"
        crop = np.rot90(binary[y1:y2, x1:x2], k=1)
    return crop.astype(bool), [x1, y1, x2, y2], orientation


def oriented_bbox_to_page_bbox(local_bbox: Sequence[int], crop_bbox: Sequence[int], orientation: str) -> list[int]:
    lx1, ly1, lx2, ly2 = [int(v) for v in local_bbox]
    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    crop_w = x2 - x1
    crop_h = y2 - y1
    if orientation == "vertical_down":
        return [x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2]
    if orientation == "vertical_up":
        return [x1 + lx1, y1 + (crop_h - ly2), x1 + lx2, y1 + (crop_h - ly1)]
    if orientation == "horizontal_right":
        return [x1 + ly1, y1 + (crop_h - lx2), x1 + ly2, y1 + (crop_h - lx1)]
    return [x1 + (crop_w - ly2), y1 + lx1, x1 + (crop_w - ly1), y1 + lx2]


def oriented_mask_to_page_mask(oriented_mask: np.ndarray, crop_bbox: Sequence[int], orientation: str, shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    crop_w = x2 - x1
    crop_h = y2 - y1
    out = np.zeros(shape, dtype=bool)
    ys, xs = np.nonzero(oriented_mask)
    if ys.size == 0:
        return out
    if orientation == "vertical_down":
        page_xs = x1 + xs
        page_ys = y1 + ys
    elif orientation == "vertical_up":
        page_xs = x1 + xs
        page_ys = y1 + (crop_h - 1 - ys)
    elif orientation == "horizontal_right":
        page_xs = x1 + ys
        page_ys = y1 + (crop_h - 1 - xs)
    else:
        page_xs = x1 + (crop_w - 1 - ys)
        page_ys = y1 + xs
    valid = (0 <= page_xs) & (page_xs < shape[1]) & (0 <= page_ys) & (page_ys < shape[0])
    out[page_ys[valid], page_xs[valid]] = True
    return out


def _component_excess_pixels(
    labels: np.ndarray,
    component_labels: set[int],
    allowed_box: Sequence[int],
) -> int:
    if not component_labels:
        return 0
    x1, y1, x2, y2 = [int(v) for v in allowed_box]
    component_mask = np.isin(labels, list(component_labels))
    allowed = np.zeros(labels.shape, dtype=bool)
    allowed[max(0, y1) : min(labels.shape[0], y2), max(0, x1) : min(labels.shape[1], x2)] = True
    return int(np.count_nonzero(component_mask & ~allowed))


def _dot_candidate_near_plate(
    fg: np.ndarray,
    line_box: np.ndarray,
    connector_box: np.ndarray,
    plate_center_x: float,
    plate_center_y: float,
    wire_width: float,
    cfg: EndpointConfig,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(connector_box)
    if xs.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    connector_local = fg[y1:y2, x1:x2] & ~line_box[y1:y2, x1:x2]
    candidates = compact_dot_candidates(connector_local, wire_width, cfg)
    if not candidates:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for cand in candidates:
        cx, cy = [float(v) for v in cand["center"]]
        page_cx = x1 + cx
        page_cy = y1 + cy
        diameter = float(cand.get("diameter", max(cand["bbox"][2] - cand["bbox"][0], cand["bbox"][3] - cand["bbox"][1])))
        center_tolerance = max(wire_width * 1.2, diameter * 0.75)
        if abs(page_cx - plate_center_x) > center_tolerance:
            continue
        if abs(page_cy - plate_center_y) > max(wire_width * 5.0, diameter * 2.5):
            continue
        score = abs(page_cx - plate_center_x) + abs(page_cy - plate_center_y) * 0.25
        candidate = {
            **cand,
            "center": [float(page_cx), float(page_cy)],
            "bbox": [int(cand["bbox"][0] + x1), int(cand["bbox"][1] + y1), int(cand["bbox"][2] + x1), int(cand["bbox"][3] + y1)],
        }
        if best is None or score < best[0]:
            best = (score, candidate)
    return None if best is None else best[1]


def _ground_symbol_like_for_line_axis(
    fg: np.ndarray,
    wire_width: float,
    cfg: EndpointConfig,
    axis_name: str,
) -> tuple[bool, list[int] | None, np.ndarray | None, dict[str, Any]]:
    h, w = fg.shape
    if h < 8 or w < 8:
        return False, None, None, {"ground_like": False, "reason": "crop_too_small", "line_axis": axis_name}
    row_counts = np.sum(fg, axis=1)
    bar_threshold = max(2, min(int(round(max(2.0, w * 0.28))), int(round(max(3.0, wire_width * 1.0)))))
    row_runs = projection_runs(row_counts >= bar_threshold)
    bar_rows: list[list[dict[str, Any]]] = []
    min_bar_width = max(3.0, wire_width * 0.8)
    for a, b in row_runs:
        active_cols = np.any(fg[a:b, :], axis=0)
        col_runs = projection_runs(active_cols)
        row_bars: list[dict[str, Any]] = []
        for ca, cb in col_runs:
            width = int(cb - ca)
            if width < min_bar_width:
                continue
            row_bars.append({"run": [int(a), int(b)], "x1": int(ca), "x2": int(cb), "width": width})
        if not row_bars:
            continue
        bar_rows.append(row_bars)
    best: tuple[float, list[dict[str, Any]], int, dict[str, Any]] | None = None
    _count, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    for idx in range(0, max(0, len(bar_rows) - cfg.ground_min_plate_count + 1)):
        row_group = bar_rows[idx : idx + cfg.ground_min_plate_count]
        for combo in itertools.product(*row_group):
            group = [dict(v) for v in combo]
            widths = [int(v["width"]) for v in group]
            if len(widths) < int(cfg.ground_min_plate_count):
                continue
            longest_idx = int(np.argmax(np.array(widths, dtype=int)))
            sorted_widths = sorted(widths, reverse=True)
            min_drop = max(2, int(round(wire_width * 0.7)))
            if sorted_widths[0] - sorted_widths[1] < min_drop:
                continue
            if sorted_widths[0] < max(5.0, wire_width * cfg.ground_plate_min_length_widths):
                continue
            centers = [(float(v["x1"]) + float(v["x2"]) - 1.0) / 2.0 for v in group]
            if max(centers) - min(centers) > max(6.0, wire_width * 2.0):
                continue
            row_centers = [(float(v["run"][0]) + float(v["run"][1]) - 1.0) / 2.0 for v in group]
            vertical_gap_ok = all((group[i + 1]["run"][0] - group[i]["run"][1]) <= max(10.0, wire_width * 3.0) for i in range(len(group) - 1))
            if not vertical_gap_ok:
                continue
            line_pad = max(1, int(round(wire_width * 0.55)))
            connector_pad_x = max(3, int(round(wire_width * 3.0)))
            connector_pad_y = max(3, int(round(wire_width * 8.0)))
            longest_excess = 0
            dot_candidate: dict[str, Any] | None = None
            longest_component_labels: set[int] = set()
            longest_component_mask = np.zeros(labels.shape, dtype=bool)
            non_longest_excess: list[int] = []
            for bar_index, bar in enumerate(group):
                bx1 = max(0, int(bar["x1"]) - line_pad)
                bx2 = min(w, int(bar["x2"]) + line_pad)
                by1 = max(0, int(bar["run"][0]) - line_pad)
                by2 = min(h, int(bar["run"][1]) + line_pad)
                comp_labels = {int(v) for v in np.unique(labels[by1:by2, bx1:bx2]) if int(v) > 0}
                if bar_index == longest_idx:
                    longest_component_labels = set(comp_labels)
                    cx1 = max(0, int(bar["x1"]) - connector_pad_x)
                    cx2 = min(w, int(bar["x2"]) + connector_pad_x)
                    cy1 = max(0, int(bar["run"][0]) - connector_pad_y)
                    cy2 = min(h, int(bar["run"][1]) + connector_pad_y)
                    longest_component_mask = np.isin(labels, list(comp_labels))
                    line_box = np.zeros(labels.shape, dtype=bool)
                    line_box[by1:by2, bx1:bx2] = True
                    connector_box = np.zeros(labels.shape, dtype=bool)
                    connector_box[cy1:cy2, cx1:cx2] = True
                    longest_excess = int(np.count_nonzero(longest_component_mask & connector_box & ~line_box))
                    dot_candidate = _dot_candidate_near_plate(
                        fg,
                        line_box,
                        connector_box,
                        plate_center_x=(float(bar["x1"]) + float(bar["x2"]) - 1.0) / 2.0,
                        plate_center_y=(float(bar["run"][0]) + float(bar["run"][1]) - 1.0) / 2.0,
                        wire_width=wire_width,
                        cfg=cfg,
                    )
                else:
                    excess = _component_excess_pixels(labels, comp_labels, [bx1, by1, bx2, by2])
                    non_longest_excess.append(excess)
            if any(excess > max(2, int(round(wire_width * 0.7))) for excess in non_longest_excess):
                continue
            if dot_candidate is None:
                continue
            dot_diameter = float(dot_candidate.get("diameter", 0.0))
            if dot_diameter <= 0.0:
                continue
            dot_bbox = [int(v) for v in dot_candidate["bbox"]]
            dot_component_labels = {
                int(v)
                for v in np.unique(labels[dot_bbox[1] : dot_bbox[3], dot_bbox[0] : dot_bbox[2]])
                if int(v) > 0
            }
            if not (dot_component_labels & longest_component_labels):
                continue
            if sorted_widths[0] > dot_diameter * 4.0:
                continue
            if any((right - left) > dot_diameter * 2.0 for left, right in zip(sorted(row_centers), sorted(row_centers)[1:])):
                continue
            dot_pad = max(1, int(round(wire_width * 0.8)))
            allowed = np.zeros(labels.shape, dtype=bool)
            allowed[
                max(0, dot_bbox[1] - dot_pad) : min(h, dot_bbox[3] + dot_pad),
                max(0, dot_bbox[0] - dot_pad) : min(w, dot_bbox[2] + dot_pad),
            ] = True
            for bar in group:
                bx1 = max(0, int(bar["x1"]) - line_pad)
                bx2 = min(w, int(bar["x2"]) + line_pad)
                by1 = max(0, int(bar["run"][0]) - line_pad)
                by2 = min(h, int(bar["run"][1]) + line_pad)
                allowed[by1:by2, bx1:bx2] = True
            longest_extra_outside_dot = int(np.count_nonzero(longest_component_mask & connector_box & ~allowed))
            if longest_extra_outside_dot > max(2, int(round(wire_width * 0.7))):
                continue
            first_y = min(int(v["run"][0]) for v in group)
            center_penalty = max(centers) - min(centers)
            score = float(first_y) + center_penalty * 2.0 - float(sum(widths)) * 0.1 - float(longest_excess) * 0.02
            if best is None or score < best[0]:
                best = (score, group, int(longest_excess), dot_candidate)
    if best is None:
        return False, None, None, {"ground_like": False, "line_axis": axis_name, "bar_row_count": len(bar_rows), "bars": [v for row in bar_rows[:8] for v in row[:4]]}
    group = best[1]
    best_longest_excess = int(best[2])
    best_dot_candidate = best[3]
    best_dot_diameter = float(best_dot_candidate.get("diameter", 0.0))
    widths = [int(v["width"]) for v in group]
    longest_idx = int(np.argmax(np.array(widths, dtype=int)))
    x1 = min(int(v["x1"]) for v in group)
    x2 = max(int(v["x2"]) for v in group)
    y1 = min(int(v["run"][0]) for v in group)
    y2 = max(int(v["run"][1]) for v in group)
    pad = max(2, int(round(wire_width * 1.0)))
    longest = group[longest_idx]
    bbox = [
        max(0, min(x1, int(longest["x1"]) - int(round(wire_width * 3.0))) - pad),
        max(0, min(y1, int(longest["run"][0]) - int(round(wire_width * 8.0))) - pad),
        min(w, max(x2, int(longest["x2"]) + int(round(wire_width * 3.0))) + pad),
        min(h, max(y2, int(longest["run"][1]) + int(round(wire_width * 3.0))) + pad),
    ]
    bar_mask = np.zeros_like(fg, dtype=bool)
    x_pad = max(1, int(round(wire_width * 0.8)))
    y_pad = max(2, int(round(wire_width * 0.8)))
    for bar in group:
        bx1 = max(0, int(bar["x1"]) - x_pad)
        bx2 = min(w, int(bar["x2"]) + x_pad)
        by1 = max(0, int(bar["run"][0]) - y_pad)
        by2 = min(h, int(bar["run"][1]) + y_pad)
        bar_mask[by1:by2, bx1:bx2] |= fg[by1:by2, bx1:bx2]
    _count, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    lx1 = max(0, int(longest["x1"]) - x_pad)
    lx2 = min(w, int(longest["x2"]) + x_pad)
    ly1 = max(0, int(longest["run"][0]) - y_pad)
    ly2 = min(h, int(longest["run"][1]) + y_pad)
    longest_component_labels = {int(v) for v in np.unique(labels[ly1:ly2, lx1:lx2]) if int(v) > 0}
    if longest_component_labels:
        bar_mask |= np.isin(labels, list(longest_component_labels))
    dot_bbox = [int(v) for v in best_dot_candidate["bbox"]]
    bar_mask[dot_bbox[1] : dot_bbox[3], dot_bbox[0] : dot_bbox[2]] |= fg[dot_bbox[1] : dot_bbox[3], dot_bbox[0] : dot_bbox[2]]
    bar_mask[: bbox[1], :] = False
    bar_mask[bbox[3] :, :] = False
    bar_mask[:, : bbox[0]] = False
    bar_mask[:, bbox[2] :] = False
    tight_bbox = mask_bbox(bar_mask)
    if tight_bbox is not None:
        bbox = [int(v) for v in tight_bbox]
    return True, bbox, bar_mask, {
        "ground_like": True,
        "line_axis": axis_name,
        "bar_count": len(group),
        "bar_widths": widths,
        "bar_runs": [v["run"] for v in group],
        "longest_bar_index": int(longest_idx),
        "longest_connected_extra_pixels": int(best_longest_excess),
        "dot_diameter": round(float(best_dot_diameter), 3),
        "dot_bbox_oriented": [int(v) for v in best_dot_candidate["bbox"]],
        "dot_center_oriented": [round(float(v), 3) for v in best_dot_candidate["center"]],
        "max_plate_spacing": round(float(max(np.diff(sorted([(float(v["run"][0]) + float(v["run"][1]) - 1.0) / 2.0 for v in group])))), 3),
        "first_bar_start": int(group[0]["run"][0]),
        "candidate_bbox_oriented": [int(v) for v in bbox],
    }


def ground_symbol_like_in_forward_crop(fg: np.ndarray, wire_width: float, cfg: EndpointConfig) -> tuple[bool, list[int] | None, np.ndarray | None, dict[str, Any]]:
    candidates: list[tuple[float, list[int], np.ndarray, dict[str, Any]]] = []
    ok, bbox, mask, metrics = _ground_symbol_like_for_line_axis(fg, wire_width, cfg, "horizontal_parallel_lines")
    if ok and bbox is not None and mask is not None:
        score = float(metrics.get("first_bar_start", bbox[1]))
        candidates.append((score, bbox, mask, metrics))
    ok_t, bbox_t, mask_t, metrics_t = _ground_symbol_like_for_line_axis(fg.T, wire_width, cfg, "vertical_parallel_lines")
    if ok_t and bbox_t is not None and mask_t is not None:
        transposed_mask = mask_t.T
        transposed_bbox = mask_bbox(transposed_mask)
        if transposed_bbox is not None:
            metrics_t = {
                **metrics_t,
                "candidate_bbox_transposed": [int(v) for v in bbox_t],
                "candidate_bbox_oriented": [int(v) for v in transposed_bbox],
                "first_bar_start": int(transposed_bbox[1]),
            }
            score = float(transposed_bbox[1])
            candidates.append((score, [int(v) for v in transposed_bbox], transposed_mask, metrics_t))
    if not candidates:
        return False, None, None, {
            "ground_like": False,
            "horizontal_debug": metrics,
            "vertical_debug": metrics_t,
        }
    _score, best_bbox, best_mask, best_metrics = sorted(candidates, key=lambda v: v[0])[0]
    return True, best_bbox, best_mask, best_metrics


def find_ground_symbol_candidate_for_endpoint(
    binary: np.ndarray,
    endpoint: dict[str, Any],
    wire_width: float,
    cfg: EndpointConfig,
) -> dict[str, Any] | None:
    outward = endpoint.get("outward")
    if not outward:
        return None
    forward = max(24, int(round(wire_width * cfg.ground_search_forward_widths)))
    backward = max(2, int(round(wire_width * cfg.ground_search_backward_widths)))
    cross = max(14, int(round(wire_width * cfg.ground_search_cross_widths)))
    oriented = orient_forward_crop(binary, endpoint["point"], outward, forward, backward, cross)
    if oriented is None:
        return None
    crop, crop_bbox, orientation = oriented
    ok, oriented_bbox, oriented_mask, metrics = ground_symbol_like_in_forward_crop(crop, wire_width, cfg)
    if not ok or oriented_bbox is None or oriented_mask is None:
        return None
    first_bar_start = int(metrics.get("first_bar_start", 0))
    min_forward_start = max(0, int(round(float(backward) - wire_width * 1.5)))
    if first_bar_start < min_forward_start:
        return None
    page_bbox = clip_box(oriented_bbox_to_page_bbox(oriented_bbox, crop_bbox, orientation), binary.shape)
    if box_area(page_bbox) <= 0:
        return None
    mask = oriented_mask_to_page_mask(oriented_mask, crop_bbox, orientation, binary.shape)
    if not bool(np.any(mask)):
        return None
    mask_bbox_value = mask_bbox(mask)
    if mask_bbox_value is not None:
        page_bbox = [int(v) for v in mask_bbox_value]
    metrics.update({"orientation": orientation, "search_bbox": [int(v) for v in crop_bbox], "source": "direct_ground_symbol_topology"})
    return {"bbox": page_bbox, "mask": mask, "metrics": metrics}


def best_compact_candidate_for_endpoint(
    endpoint_point: Sequence[float],
    outward: Sequence[float],
    candidates: Sequence[dict[str, Any]],
    forward_search: float,
    backward_search: float,
    lateral_tol: float,
) -> tuple[dict[str, Any], float, float] | None:
    p = np.array(endpoint_point, dtype=float)
    unit = np.array(outward, dtype=float)
    norm = float(np.linalg.norm(unit))
    if norm <= 1.0e-6:
        return None
    unit = unit / norm
    min_along = -max(0.0, float(backward_search))
    max_along = max(float(forward_search), 0.0)
    best: tuple[float, dict[str, Any], float, float] | None = None
    for cand in candidates:
        center = np.array(cand["center"], dtype=float)
        rel = center - p
        along = float(rel @ unit)
        if along < min_along or along > max_along:
            continue
        perp = float(np.linalg.norm(rel - unit * along))
        if perp > lateral_tol:
            continue
        score = abs(along) + perp * 2.0 if min_along < 0.0 else along + perp * 2.0
        if best is None or score < best[0]:
            best = (score, cand, along, perp)
    if best is None:
        return None
    _score, cand, along, perp = best
    return cand, along, perp


def connected_residual_labels_near_endpoint(
    item_mask: np.ndarray,
    residual_labels: np.ndarray,
    endpoint_point: Sequence[float],
    search_radius: int,
    touch_radius: int,
) -> list[int]:
    x = int(round(float(endpoint_point[0])))
    y = int(round(float(endpoint_point[1])))
    h, w = residual_labels.shape
    x1 = max(0, x - int(search_radius))
    x2 = min(w, x + int(search_radius) + 1)
    y1 = max(0, y - int(search_radius))
    y2 = min(h, y + int(search_radius) + 1)
    if x2 <= x1 or y2 <= y1:
        return []
    local_item = item_mask[y1:y2, x1:x2].astype(bool)
    if not bool(np.any(local_item)):
        return []
    seed = endpoint_disk(local_item.shape, [float(x - x1), float(y - y1)], max(1, touch_radius // 2))
    seed &= local_item
    if not bool(np.any(seed)):
        return []
    endpoint_anchor = seed
    local_residual = residual_labels[y1:y2, x1:x2]
    connected_labels: set[int] = set()
    for residual_label in np.unique(local_residual):
        residual_label = int(residual_label)
        if residual_label <= 0:
            continue
        local_component = local_residual == residual_label
        if not bool(np.any(local_component)):
            continue
        pair_mask = endpoint_anchor | local_component
        _count, pair_labels = cv2.connectedComponents(pair_mask.astype(np.uint8), connectivity=8)
        seed_component_labels = {int(v) for v in np.unique(pair_labels[endpoint_anchor]) if int(v) > 0}
        residual_component_labels = {int(v) for v in np.unique(pair_labels[local_component]) if int(v) > 0}
        if seed_component_labels & residual_component_labels:
            connected_labels.add(residual_label)
    return sorted(connected_labels)


def component_summary(labels: np.ndarray, stats: np.ndarray, label: int) -> dict[str, Any]:
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    area = int(stats[label, cv2.CC_STAT_AREA])
    return {"label": int(label), "bbox": [x, y, x + w, y + h], "area": area, "width": w, "height": h}


def internal_hole_area(mask: np.ndarray) -> int:
    local = mask.astype(bool)
    if not bool(np.any(local)):
        return 0
    padded = np.pad(local, 1, mode="constant", constant_values=False)
    background = (~padded).astype(np.uint8)
    flood = np.zeros((background.shape[0] + 2, background.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(background, flood, (0, 0), 0)
    holes = background.astype(bool)
    return int(np.count_nonzero(holes[1:-1, 1:-1]))


def solid_wire_has_declared_connection(item: dict[str, Any]) -> bool:
    attrs = (item.get("object", {}) or {}).get("attributes", {}) or {}
    for endpoint in attrs.get("endpoints", []) or []:
        if endpoint.get("connected_segment_ids"):
            return True
    return False


def endpoint_role_requires_local_detection(kind: str, role: str) -> bool:
    if kind in {"solid_wire", "dash_group"}:
        return role not in {"connected_end", "internal_end"}
    return False


def source_component_mask_for_item(item: dict[str, Any], binary: np.ndarray) -> tuple[np.ndarray, list[int], dict[str, Any]] | None:
    attrs = (item.get("object", {}) or {}).get("attributes", {}) or {}
    bbox = attrs.get("source_component_bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x1, y1, x2, y2 = clip_box(bbox, binary.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = binary[y1:y2, x1:x2].astype(np.uint8)
    if not bool(np.any(crop)):
        return None
    count, labels = cv2.connectedComponents(crop, connectivity=8)
    item_crop = item["mask"][y1:y2, x1:x2]
    touched = labels[item_crop & (labels > 0)]
    if touched.size == 0:
        return None
    values, counts = np.unique(touched, return_counts=True)
    label = int(values[int(np.argmax(counts))])
    if label <= 0 or label >= count:
        return None
    mask = labels == label
    comp_bbox_local = mask_bbox(mask)
    if comp_bbox_local is None:
        return None
    bx1, by1, bx2, by2 = comp_bbox_local
    full_bbox = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
    metrics = {"source_component_bbox": [x1, y1, x2, y2], "component_label": int(label)}
    return mask, full_bbox, metrics


def brace_structure_candidate(
    item: dict[str, Any],
    binary: np.ndarray,
    wire_width: float,
    cfg: EndpointConfig,
) -> tuple[np.ndarray, list[int], dict[str, Any]] | None:
    if str(item.get("kind", "")) != "solid_wire":
        return None
    if solid_wire_has_declared_connection(item):
        return None
    orientation = str(item.get("orientation", ""))
    if orientation not in {"horizontal", "vertical"}:
        return None
    source = source_component_mask_for_item(item, binary)
    if source is None:
        return None
    comp_mask, bbox, metrics = source
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    along = width if orientation == "horizontal" else height
    across = height if orientation == "horizontal" else width
    if along < max(24.0, wire_width * cfg.brace_min_length_widths):
        return None
    if across > max(18.0, wire_width * cfg.brace_max_thickness_widths):
        return None
    if float(along) / max(1.0, float(across)) < cfg.brace_min_aspect_ratio:
        return None
    x1, y1, x2, y2 = bbox
    item_local = item["mask"][y1:y2, x1:x2]
    comp_local = comp_mask[(y1 - metrics["source_component_bbox"][1]) : (y2 - metrics["source_component_bbox"][1]), (x1 - metrics["source_component_bbox"][0]) : (x2 - metrics["source_component_bbox"][0])]
    residual_local = comp_local & ~item_local
    if not bool(np.any(residual_local)):
        return None
    points = item.get("points", [])
    if len(points) < 2:
        return None
    p0 = np.array(points[0], dtype=float)
    p1 = np.array(points[-1], dtype=float)
    centerline = (p0[1] + p1[1]) / 2.0 if orientation == "horizontal" else (p0[0] + p1[0]) / 2.0
    ys, xs = np.nonzero(residual_local)
    abs_xs = xs.astype(float) + x1
    abs_ys = ys.astype(float) + y1
    offsets = abs_ys - centerline if orientation == "horizontal" else abs_xs - centerline
    off_axis = np.abs(offsets) >= max(2.0, wire_width * 0.8)
    if int(np.count_nonzero(off_axis)) < max(6, int(round(wire_width * 2.0))):
        return None
    off_along = abs_xs[off_axis] if orientation == "horizontal" else abs_ys[off_axis]
    off_offsets = offsets[off_axis]
    v_depth = float(np.max(np.abs(off_offsets)))
    v_span = float(np.max(off_along) - np.min(off_along)) if off_along.size else 0.0
    if v_depth < max(4.0, wire_width * cfg.brace_min_v_depth_widths):
        return None
    if v_span < max(6.0, wire_width * cfg.brace_min_v_span_widths):
        return None
    metrics.update(
        {
            "orientation": orientation,
            "brace_bbox": [int(v) for v in bbox],
            "along_span": round(float(along), 3),
            "across_span": round(float(across), 3),
            "aspect_ratio": round(float(along) / max(1.0, float(across)), 3),
            "v_depth": round(v_depth, 3),
            "v_span": round(v_span, 3),
            "source_item_id": str(item["id"]),
        }
    )
    full_mask = np.zeros(binary.shape, dtype=bool)
    full_mask[y1:y2, x1:x2] = comp_local
    return full_mask, [int(v) for v in bbox], metrics


def small_connected_structure_candidate(local_mask: np.ndarray, wire_width: float, cfg: EndpointConfig) -> tuple[bool, dict[str, Any]]:
    area = int(np.count_nonzero(local_mask))
    bbox = mask_bbox(local_mask)
    if bbox is None:
        return False, {"decision": "empty"}
    x1, y1, x2, y2 = bbox
    width = int(x2 - x1)
    height = int(y2 - y1)
    hole_area = internal_hole_area(local_mask[y1:y2, x1:x2])
    hole_ratio = float(hole_area) / max(1.0, float(area))
    if hole_ratio >= cfg.small_structure_hole_area_ratio:
        return False, {
            "decision": "reject_internal_hole_text_like",
            "hole_area": int(hole_area),
            "hole_area_ratio": round(hole_ratio, 4),
            "width": width,
            "height": height,
            "area": area,
        }
    return True, {
        "decision": "accept_small_connected_structure",
        "hole_area": int(hole_area),
        "hole_area_ratio": round(hole_ratio, 4),
        "width": width,
        "height": height,
        "area": area,
    }


def arrow_structure_candidate(
    comp_mask: np.ndarray,
    endpoint_point: Sequence[float],
    outward: Sequence[float] | None,
    wire_width: float,
    cfg: EndpointConfig,
) -> tuple[bool, dict[str, Any]]:
    bbox = mask_bbox(comp_mask)
    if bbox is None:
        return False, {"decision": "empty"}
    if outward is None:
        return False, {"decision": "missing_outward"}
    unit = np.array(outward, dtype=float)
    norm = float(np.linalg.norm(unit))
    if norm <= 1.0e-6:
        return False, {"decision": "bad_outward"}
    unit = unit / norm
    perp = np.array([-unit[1], unit[0]], dtype=float)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = int(x2 - x1)
    height = int(y2 - y1)
    area = int(np.count_nonzero(comp_mask))
    min_side = min(width, height)
    max_side = max(width, height)
    min_bbox = max(5.0, wire_width * cfg.arrow_min_bbox_widths)
    max_bbox = max(min_bbox + 2.0, wire_width * cfg.arrow_max_bbox_widths)
    if min_side < min_bbox or max_side > max_bbox:
        return False, {"decision": "reject_arrow_bbox_size", "width": width, "height": height, "area": area}
    aspect = float(max_side) / max(1.0, float(min_side))
    if aspect > cfg.arrow_max_aspect_ratio:
        return False, {"decision": "reject_arrow_aspect", "width": width, "height": height, "aspect_ratio": round(aspect, 3), "area": area}
    if area < max(12, int(round(wire_width * wire_width * cfg.arrow_min_area_widths2))):
        return False, {"decision": "reject_arrow_area", "width": width, "height": height, "area": area}
    hole_area = internal_hole_area(comp_mask[y1:y2, x1:x2])
    if float(hole_area) / max(1.0, float(area)) >= cfg.small_structure_hole_area_ratio:
        return False, {"decision": "reject_arrow_hole", "hole_area": int(hole_area), "area": area}
    ys, xs = np.nonzero(comp_mask)
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    origin = np.array(endpoint_point, dtype=float)
    rel = pts - origin
    along = rel @ unit
    across = rel @ perp
    if along.size == 0:
        return False, {"decision": "empty_projection"}
    outward_tip = float(np.max(along))
    if outward_tip < max(1.0, wire_width * 0.4):
        return False, {"decision": "reject_arrow_no_outward_tip", "area": area, "outward_tip": round(outward_tip, 3)}
    along_min = float(np.percentile(along, 5))
    along_max = float(np.max(along))
    along_span = along_max - along_min
    if along_span < max(5.0, wire_width * cfg.arrow_min_bbox_widths):
        return False, {"decision": "reject_arrow_along_span", "along_span": round(along_span, 3), "area": area}
    bins = np.linspace(along_min, along_max + 1.0e-6, 5)
    widths: list[float] = []
    counts: list[int] = []
    for i in range(4):
        if i == 3:
            sel = (along >= bins[i]) & (along <= bins[i + 1])
        else:
            sel = (along >= bins[i]) & (along < bins[i + 1])
        counts.append(int(np.count_nonzero(sel)))
        if np.count_nonzero(sel) < max(2, int(round(wire_width))):
            widths.append(0.0)
            continue
        values = across[sel]
        widths.append(float(np.max(values) - np.min(values) + 1.0))
    start_width = max(widths[:2])
    end_width = max(widths[2:])
    tip_width = widths[-1]
    if start_width <= 0.0 or end_width <= 0.0:
        return False, {"decision": "reject_arrow_sparse_profile", "profile_widths": [round(v, 3) for v in widths], "profile_counts": counts}
    taper_deltas = [left - right for left, right in zip(widths, widths[1:]) if left > 0.0 and right > 0.0]
    taper_step_min = max(1.0, wire_width * 0.35)
    if len(taper_deltas) < 3 or any(delta < taper_step_min for delta in taper_deltas):
        return False, {
            "decision": "reject_arrow_plateau_taper",
            "profile_widths": [round(v, 3) for v in widths],
            "profile_counts": counts,
            "taper_deltas": [round(v, 3) for v in taper_deltas],
            "min_taper_step": round(float(taper_step_min), 3),
        }
    linearity_tolerance = max(3.0, start_width * 0.35)
    if max(taper_deltas) - min(taper_deltas) > linearity_tolerance:
        return False, {
            "decision": "reject_arrow_nonlinear_taper",
            "profile_widths": [round(v, 3) for v in widths],
            "profile_counts": counts,
            "taper_deltas": [round(v, 3) for v in taper_deltas],
            "linearity_tolerance": round(float(linearity_tolerance), 3),
        }
    monotonic_tol = max(0.5, wire_width * 0.2)
    if any(right > left + monotonic_tol for left, right in zip(widths, widths[1:]) if left > 0.0 and right > 0.0):
        return False, {
            "decision": "reject_arrow_nonmonotonic_taper",
            "profile_widths": [round(v, 3) for v in widths],
            "profile_counts": counts,
        }
    if end_width > start_width * (1.0 - cfg.arrow_min_width_drop_ratio) + max(1.0, wire_width * 0.5):
        return False, {
            "decision": "reject_arrow_no_taper",
            "profile_widths": [round(v, 3) for v in widths],
            "profile_counts": counts,
            "start_width": round(start_width, 3),
            "end_width": round(end_width, 3),
        }
    tip_limit = min(start_width * cfg.arrow_max_tip_width_ratio, start_width * 0.40, max(8.0, wire_width * 2.0))
    if tip_width > tip_limit:
        return False, {
            "decision": "reject_arrow_tip_too_wide",
            "profile_widths": [round(v, 3) for v in widths],
            "profile_counts": counts,
            "start_width": round(start_width, 3),
            "end_width": round(end_width, 3),
            "tip_width": round(tip_width, 3),
            "tip_limit": round(tip_limit, 3),
        }
    return True, {
        "decision": "accept_arrow_structure",
        "width": width,
        "height": height,
        "area": area,
        "aspect_ratio": round(aspect, 3),
        "along_span": round(along_span, 3),
        "profile_widths": [round(v, 3) for v in widths],
        "profile_counts": counts,
        "start_width": round(start_width, 3),
        "end_width": round(end_width, 3),
        "tip_width": round(tip_width, 3),
        "taper_deltas": [round(v, 3) for v in taper_deltas],
    }


def arrow_structure_candidate_for_endpoint(
    binary: np.ndarray,
    item_mask: np.ndarray,
    endpoint_point: Sequence[float],
    outward: Sequence[float] | None,
    wire_width: float,
    cfg: EndpointConfig,
) -> tuple[np.ndarray, list[int], dict[str, Any]] | None:
    if outward is None:
        return None
    forward = max(18, int(round(wire_width * 9.0)))
    backward = max(2, int(round(wire_width * 2.0)))
    cross = max(10, int(round(wire_width * 6.0)))
    oriented_binary = orient_forward_crop(binary, endpoint_point, outward, forward, backward, cross)
    oriented_item = orient_forward_crop(item_mask, endpoint_point, outward, forward, backward, cross)
    if oriented_binary is None or oriented_item is None:
        return None
    crop, crop_bbox, orientation = oriented_binary
    item_crop, _item_bbox, _item_orientation = oriented_item
    local = crop.astype(bool) & ~item_crop.astype(bool)
    h, w = local.shape
    if h < 4 or w < 4:
        return None
    start = min(h - 1, max(0, int(round(backward - wire_width * 0.5))))
    max_len = max(8, int(round(wire_width * 7.0)))
    end = min(h, start + max_len)
    if end <= start:
        return None
    near = local[start:end, :]
    if not bool(np.any(near)):
        return None
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(near.astype(np.uint8), connectivity=8)
    center_x = w // 2
    seed_radius = max(2, int(round(wire_width * 1.5)))
    sx1 = max(0, center_x - seed_radius)
    sx2 = min(w, center_x + seed_radius + 1)
    sy2 = min(near.shape[0], max(2, int(round(wire_width * 2.5))))
    seed_labels = {int(v) for v in np.unique(labels[:sy2, sx1:sx2]) if int(v) > 0}
    if not seed_labels:
        return None
    best_mask: np.ndarray | None = None
    best_area = 0
    for label in seed_labels:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_mask = labels == int(label)
    if best_mask is None:
        return None
    local_arrow = np.zeros_like(local, dtype=bool)
    local_arrow[start:end, :] = best_mask
    page_mask = oriented_mask_to_page_mask(local_arrow, crop_bbox, orientation, binary.shape)
    bbox = mask_bbox(page_mask)
    if bbox is None:
        return None
    accepted, debug = arrow_structure_candidate(page_mask, endpoint_point, outward, wire_width, cfg)
    if not accepted:
        return None
    debug = {**debug, "source": "endpoint_forward_local_arrow_window", "orientation": orientation, "search_bbox": [int(v) for v in crop_bbox]}
    return page_mask, [int(v) for v in bbox], debug


def output_object(
    object_id: str,
    obj_type: str,
    bbox: Sequence[int],
    geometry: dict[str, Any],
    attributes: dict[str, Any],
    connected_to: Sequence[str] | None = None,
    source_phase: str = STAGE_NAME,
    confidence: float = 0.80,
) -> dict[str, Any]:
    return {
        "id": str(object_id),
        "type": str(obj_type),
        "confidence": float(confidence),
        "bbox": [int(v) for v in bbox],
        "geometry": geometry,
        "attributes": attributes,
        "connected_to": list(connected_to or []),
        "source_phase": source_phase,
    }


def generated_kind_overlap_pixels(mask: np.ndarray, generated_nodes: Sequence[dict[str, Any]], kind: str) -> int:
    overlap = 0
    for node in generated_nodes:
        if str(node.get("kind", "")) != kind:
            continue
        overlap += int(np.count_nonzero(mask & node["mask"]))
    return overlap


def classify_endpoints(
    binary: np.ndarray,
    line_items: list[dict[str, Any]],
    symbol_nodes: list[dict[str, Any]],
    rect_edge_mask: np.ndarray,
    rect_fill_mask: np.ndarray,
    wire_width: float,
    cfg: EndpointConfig,
    page: int,
    uf: UnionFind,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    generated_nodes: list[dict[str, Any]] = []
    endpoint_objects: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    base_nodes = line_items + symbol_nodes
    recognized_mask = rect_edge_mask | rect_fill_mask
    for node in base_nodes:
        recognized_mask |= node["mask"]
    owner_kind_by_id = {str(node["id"]): str(node.get("kind", "")) for node in base_nodes}
    unrecognized = binary & ~recognized_mask
    labels_count, residual_labels, residual_stats, _centroids = cv2.connectedComponentsWithStats(
        unrecognized.astype(np.uint8),
        connectivity=8,
    )
    touch_radius = max(int(cfg.endpoint_touch_radius_min_px), int(round(wire_width * cfg.endpoint_touch_radius_widths)))
    search_radius = max(int(cfg.endpoint_search_radius_min_px), int(round(wire_width * cfg.endpoint_search_radius_widths)))
    small_area_limit = max(int(cfg.small_structure_max_area_px), int(round(wire_width * wire_width * cfg.small_structure_max_area_widths2)))
    small_bbox_limit = max(8, int(round(wire_width * cfg.small_structure_max_bbox_widths)))
    disconnected_endpoint_records: list[dict[str, Any]] = []
    small_labels_used: set[int] = set()
    brace_id_by_bbox: dict[tuple[int, int, int, int], str] = {}
    arrow_id_by_bbox: dict[tuple[int, int, int, int], str] = {}

    for item in line_items:
        item["endpoint_classifications"] = []
        points = item.get("points", [])
        endpoint_roles = [
            str(ep.get("role", "")) for ep in ((item.get("object", {}).get("attributes", {}) or {}).get("endpoints", []) or [])
        ]
        for endpoint_index, point in enumerate(points[:2]):
            original_role = endpoint_roles[endpoint_index] if endpoint_index < len(endpoint_roles) else "external_end"
            if not endpoint_role_requires_local_detection(str(item.get("kind", "")), original_role):
                continue
            if np.any(rect_fill_mask & endpoint_disk(binary.shape, point, max(1, touch_radius // 2))):
                classification = {
                    "endpoint_index": int(endpoint_index),
                    "point": [round(float(v), 3) for v in point],
                    "original_role": original_role,
                    "classification": "inside_rectangle_frame",
                }
                item["endpoint_classifications"].append(classification)
                events.append({"object_id": item["id"], **classification})
                continue
            touch = endpoint_disk(binary.shape, point, touch_radius)
            search = endpoint_disk(binary.shape, point, search_radius)
            touched_owner_ids = owner_ids_connected_to_endpoint_pixels(
                base_nodes,
                str(item["id"]),
                touch,
                point,
                touch_radius,
                binary,
            )
            nearby_residual_values = np.unique(residual_labels[touch])
            nearby_residual_values = [int(v) for v in nearby_residual_values if int(v) > 0]
            outward = direction_from_points(points, endpoint_index)
            connected_residual_values = connected_residual_labels_near_endpoint(
                item_mask=item["mask"],
                residual_labels=residual_labels,
                endpoint_point=point,
                search_radius=search_radius,
                touch_radius=touch_radius,
            )
            for other_id in touched_owner_ids:
                uf.union(str(item["id"]), other_id)
            if touched_owner_ids:
                touched_dash_group_ids = [
                    owner_id
                    for owner_id in touched_owner_ids
                    if owner_kind_by_id.get(owner_id) == "dash_group"
                ]
                classification = {
                    "endpoint_index": int(endpoint_index),
                    "point": [round(float(v), 3) for v in point],
                    "original_role": original_role,
                    "classification": "connected_end",
                    "connected_owner_ids": touched_owner_ids,
                    "connected_dash_group_ids": touched_dash_group_ids,
                    "residual_component_labels": [],
                    "nearby_residual_component_labels": [int(v) for v in nearby_residual_values],
                    "small_structure_ids": [],
                    "brace_structure_ids": [],
                    "arrow_structure_ids": [],
                    "local_extension_skipped": True,
                    "local_extension_skip_reason": "endpoint_touches_existing_wire",
                }
                item["endpoint_classifications"].append(classification)
                endpoint_id = f"p{page:03d}_connected_endpoint_{len(endpoint_objects) + 1:04d}"
                ep_bbox = clip_box(
                    [
                        float(point[0]) - touch_radius,
                        float(point[1]) - touch_radius,
                        float(point[0]) + touch_radius + 1,
                        float(point[1]) + touch_radius + 1,
                    ],
                    binary.shape,
                )
                endpoint_objects.append(
                    output_object(
                        endpoint_id,
                        "connected_endpoint",
                        ep_bbox,
                        {"kind": "point", "point": [round(float(v), 3) for v in point]},
                        {
                            "owner_id": str(item["id"]),
                            "owner_kind": str(item["kind"]),
                            "endpoint_index": int(endpoint_index),
                            "classification": "connected_end",
                            "connected_owner_ids": touched_owner_ids,
                            "connected_dash_group_ids": touched_dash_group_ids,
                            "local_extension_skipped": True,
                            "local_extension_skip_reason": "endpoint_touches_existing_wire",
                            "residual_components": [],
                        },
                        connected_to=[str(item["id"])] + touched_owner_ids,
                        confidence=0.76,
                    )
                )
                events.append({"object_id": item["id"], **classification})
                continue
            accepted_residual_values: list[int] = []
            small_structure_ids: list[str] = []
            brace_structure_ids: list[str] = []
            arrow_structure_ids: list[str] = []
            residual_summaries: list[dict[str, Any]] = []
            direct_arrow = arrow_structure_candidate_for_endpoint(unrecognized, item["mask"], point, outward, wire_width, cfg)
            if direct_arrow is not None:
                arrow_mask, arrow_bbox, arrow_metrics = direct_arrow
                brace_overlap = generated_kind_overlap_pixels(arrow_mask, generated_nodes, "brace_structure")
                if brace_overlap > 0:
                    events.append(
                        {
                            "event": "arrow_structure_rejected",
                            "reason": "overlaps_existing_brace_structure",
                            "connected_owner_id": str(item["id"]),
                            "bbox": arrow_bbox,
                            "overlap_pixels": int(brace_overlap),
                            "source": "endpoint_forward_local_arrow_window",
                        }
                    )
                    direct_arrow = None
            if direct_arrow is not None:
                arrow_mask, arrow_bbox, arrow_metrics = direct_arrow
                arrow_key = tuple(int(v) for v in arrow_bbox)
                arrow_id = arrow_id_by_bbox.get(arrow_key)
                if arrow_id is None:
                    arrow_id = f"p{page:03d}_arrow_structure_{len(generated_nodes) + 1:04d}"
                    local_arrow = arrow_mask[arrow_bbox[1] : arrow_bbox[3], arrow_bbox[0] : arrow_bbox[2]]
                    obj = output_object(
                        arrow_id,
                        "arrow_structure",
                        arrow_bbox,
                        {"kind": "pixel_region", "pixel_bbox": [int(v) for v in arrow_bbox]},
                        {
                            "source": "endpoint_forward_local_arrow_window",
                            "connected_endpoint_owner_id": str(item["id"]),
                            "connected_endpoint_index": int(endpoint_index),
                            "pixel_bbox": [int(v) for v in arrow_bbox],
                            "pixel_runs": mask_to_relative_runs(local_arrow),
                            "pixel_count": int(np.count_nonzero(local_arrow)),
                            "structure_decision": arrow_metrics,
                        },
                        connected_to=[str(item["id"])],
                        confidence=0.78,
                    )
                    generated_nodes.append({"id": arrow_id, "kind": "arrow_structure", "object": obj, "mask": arrow_mask, "bbox": arrow_bbox})
                    uf.add(arrow_id)
                    uf.union(str(item["id"]), arrow_id)
                    arrow_id_by_bbox[arrow_key] = arrow_id
                    events.append({"event": "arrow_structure_inferred", "id": arrow_id, "connected_owner_id": str(item["id"]), "bbox": arrow_bbox, "source": "endpoint_forward_local_arrow_window"})
                arrow_structure_ids.append(arrow_id)
            for label in connected_residual_values:
                summary = component_summary(residual_labels, residual_stats, label)
                residual_summaries.append(summary)
                arrow_mask = (residual_labels == label) & search
                arrow_bbox = mask_bbox(arrow_mask)
                if arrow_bbox is not None:
                    accepted_arrow, arrow_debug = arrow_structure_candidate(arrow_mask, point, outward, wire_width, cfg)
                    summary["arrow_decision"] = arrow_debug
                    if accepted_arrow:
                        brace_overlap = generated_kind_overlap_pixels(arrow_mask, generated_nodes, "brace_structure")
                        if brace_overlap > 0:
                            accepted_arrow = False
                            summary["arrow_decision"] = {
                                **arrow_debug,
                                "decision": "reject_arrow_overlaps_existing_brace_structure",
                                "overlap_pixels": int(brace_overlap),
                            }
                    if accepted_arrow:
                        arrow_key = tuple(int(v) for v in arrow_bbox)
                        arrow_id = arrow_id_by_bbox.get(arrow_key)
                        if arrow_id is None:
                            arrow_id = f"p{page:03d}_arrow_structure_{len(generated_nodes) + 1:04d}"
                            local_arrow = arrow_mask[arrow_bbox[1] : arrow_bbox[3], arrow_bbox[0] : arrow_bbox[2]]
                            obj = output_object(
                                arrow_id,
                                "arrow_structure",
                                arrow_bbox,
                                {"kind": "pixel_region", "pixel_bbox": [int(v) for v in arrow_bbox]},
                                {
                                    "source": "endpoint_connected_arrow_residual_component",
                                    "connected_endpoint_owner_id": str(item["id"]),
                                    "connected_endpoint_index": int(endpoint_index),
                                    "residual_component_label": int(label),
                                    "pixel_bbox": [int(v) for v in arrow_bbox],
                                    "pixel_runs": mask_to_relative_runs(local_arrow),
                                    "pixel_count": int(np.count_nonzero(local_arrow)),
                                    "global_component_area": int(summary["area"]),
                                    "structure_decision": arrow_debug,
                                },
                                connected_to=[str(item["id"])],
                                confidence=0.78,
                            )
                            generated_nodes.append({"id": arrow_id, "kind": "arrow_structure", "object": obj, "mask": arrow_mask, "bbox": arrow_bbox})
                            uf.add(arrow_id)
                            uf.union(str(item["id"]), arrow_id)
                            arrow_id_by_bbox[arrow_key] = arrow_id
                            events.append({"event": "arrow_structure_inferred", "id": arrow_id, "connected_owner_id": str(item["id"]), "bbox": arrow_bbox})
                        accepted_residual_values.append(int(label))
                        arrow_structure_ids.append(arrow_id)
                        small_labels_used.add(label)
                        summary["structure_decision"] = "accepted_as_arrow_structure"
                        summary["arrow_structure_id"] = arrow_id
                        continue
                brace = brace_structure_candidate(item, binary, wire_width, cfg)
                if brace is not None:
                    brace_mask, brace_bbox, brace_metrics = brace
                    brace_key = tuple(int(v) for v in brace_bbox)
                    brace_id = brace_id_by_bbox.get(brace_key)
                    if brace_id is None:
                        brace_id = f"p{page:03d}_brace_structure_{len(generated_nodes) + 1:04d}"
                        local_brace = brace_mask[brace_bbox[1] : brace_bbox[3], brace_bbox[0] : brace_bbox[2]]
                        obj = output_object(
                            brace_id,
                            "brace_structure",
                            brace_bbox,
                            {"kind": "pixel_region", "pixel_bbox": [int(v) for v in brace_bbox]},
                            {
                                "source": "flat_solid_wire_component_with_v_join",
                                "connected_endpoint_owner_id": str(item["id"]),
                                "connected_endpoint_index": int(endpoint_index),
                                "residual_component_label": int(label),
                                "pixel_bbox": [int(v) for v in brace_bbox],
                                "pixel_runs": mask_to_relative_runs(local_brace),
                                "pixel_count": int(np.count_nonzero(local_brace)),
                                **brace_metrics,
                            },
                            connected_to=[str(item["id"])],
                            confidence=0.78,
                        )
                        generated_nodes.append({"id": brace_id, "kind": "brace_structure", "object": obj, "mask": brace_mask, "bbox": brace_bbox})
                        uf.add(brace_id)
                        uf.union(str(item["id"]), brace_id)
                        brace_id_by_bbox[brace_key] = brace_id
                        events.append({"event": "brace_structure_inferred", "id": brace_id, "connected_owner_id": str(item["id"]), "bbox": brace_bbox})
                    accepted_residual_values.append(int(label))
                    brace_structure_ids.append(brace_id)
                    small_labels_used.add(label)
                    summary["structure_decision"] = "accepted_as_brace_structure"
                    summary["brace_structure_id"] = brace_id
                    continue
                if label in small_labels_used:
                    accepted_residual_values.append(int(label))
                    continue
                if summary["area"] <= small_area_limit and summary["width"] <= small_bbox_limit and summary["height"] <= small_bbox_limit:
                    comp_mask = residual_labels == label
                    # The search disk limits the visual bbox; returned labels
                    # have already passed line-removed endpoint connectivity.
                    comp_mask &= search
                    bbox = mask_bbox(comp_mask)
                    if bbox is None:
                        continue
                    local = comp_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                    accepted_small, small_debug = small_connected_structure_candidate(local, wire_width, cfg)
                    summary["structure_decision"] = small_debug
                    if not accepted_small:
                        events.append(
                            {
                                "event": "small_connected_structure_rejected",
                                "object_id": str(item["id"]),
                                "endpoint_index": int(endpoint_index),
                                "residual_component_label": int(label),
                                "bbox": [int(v) for v in bbox],
                                **small_debug,
                            }
                        )
                        continue
                    accepted_residual_values.append(int(label))
                    small_id = f"p{page:03d}_small_connected_structure_{len(generated_nodes) + 1:04d}"
                    obj = output_object(
                        small_id,
                        "small_connected_structure",
                        bbox,
                        {"kind": "pixel_region", "pixel_bbox": [int(v) for v in bbox]},
                        {
                            "source": "endpoint_connected_residual_component",
                            "connected_endpoint_owner_id": str(item["id"]),
                            "connected_endpoint_index": int(endpoint_index),
                            "residual_component_label": int(label),
                            "pixel_bbox": [int(v) for v in bbox],
                            "pixel_runs": mask_to_relative_runs(local),
                            "pixel_count": int(np.count_nonzero(local)),
                            "global_component_area": int(summary["area"]),
                            "structure_decision": small_debug,
                        },
                        connected_to=[str(item["id"])],
                        confidence=0.72,
                    )
                    generated_nodes.append({"id": small_id, "kind": "small_connected_structure", "object": obj, "mask": comp_mask, "bbox": bbox})
                    uf.add(small_id)
                    uf.union(str(item["id"]), small_id)
                    small_labels_used.add(label)
                    small_structure_ids.append(small_id)
                else:
                    accepted_residual_values.append(int(label))
                    summary["structure_decision"] = "connected_residual_too_large_for_small_structure"
            connected = bool(touched_owner_ids or accepted_residual_values or arrow_structure_ids)
            classification = {
                "endpoint_index": int(endpoint_index),
                "point": [round(float(v), 3) for v in point],
                "original_role": original_role,
                "classification": "connected_end" if connected else "disconnected_end",
                "connected_owner_ids": touched_owner_ids,
                "residual_component_labels": [int(v) for v in accepted_residual_values],
                "nearby_residual_component_labels": [int(v) for v in nearby_residual_values],
                "small_structure_ids": small_structure_ids,
                "brace_structure_ids": brace_structure_ids,
                "arrow_structure_ids": arrow_structure_ids,
            }
            item["endpoint_classifications"].append(classification)
            endpoint_type = "connected_endpoint" if connected else "disconnected_endpoint"
            endpoint_id = f"p{page:03d}_{endpoint_type}_{len(endpoint_objects) + 1:04d}"
            ep_bbox = clip_box(
                [float(point[0]) - touch_radius, float(point[1]) - touch_radius, float(point[0]) + touch_radius + 1, float(point[1]) + touch_radius + 1],
                binary.shape,
            )
            endpoint_objects.append(
                output_object(
                    endpoint_id,
                    endpoint_type,
                    ep_bbox,
                    {"kind": "point", "point": [round(float(v), 3) for v in point]},
                    {
                        "owner_id": str(item["id"]),
                        "owner_kind": str(item["kind"]),
                        "endpoint_index": int(endpoint_index),
                        "classification": classification["classification"],
                        "connected_owner_ids": touched_owner_ids,
                        "residual_components": residual_summaries[:12],
                    },
                    connected_to=[str(item["id"])] + touched_owner_ids,
                    confidence=0.70,
                )
            )
            events.append({"object_id": item["id"], **classification})
            if not connected:
                disconnected_endpoint_records.append(
                    {
                        "owner_id": str(item["id"]),
                        "owner_kind": str(item["kind"]),
                        "endpoint_index": int(endpoint_index),
                        "point": [float(point[0]), float(point[1])],
                        "outward": None if outward is None else [float(outward[0]), float(outward[1])],
                    }
                )

    dot_nodes, dot_events = infer_connection_dots_and_ground(
        binary=binary,
        unrecognized=unrecognized,
        endpoint_records=disconnected_endpoint_records,
        wire_width=wire_width,
        cfg=cfg,
        page=page,
        uf=uf,
    )
    generated_nodes.extend(dot_nodes)
    events.extend(dot_events)
    generated_mask = np.zeros(binary.shape, dtype=bool)
    for node in generated_nodes:
        generated_mask |= node["mask"]
    return generated_nodes, endpoint_objects, events, generated_mask


def infer_connection_dots_and_ground(
    binary: np.ndarray,
    unrecognized: np.ndarray,
    endpoint_records: list[dict[str, Any]],
    wire_width: float,
    cfg: EndpointConfig,
    page: int,
    uf: UnionFind,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = compact_dot_candidates(unrecognized, wire_width, cfg)
    forward_search = max(2.0, wire_width * cfg.endpoint_dot_search_widths)
    backward_search = max(0.0, wire_width * cfg.endpoint_dot_backward_widths)
    lateral_tol = max(1.5, wire_width * cfg.endpoint_dot_lateral_tolerance_widths)
    hits_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gnd_hits: dict[int, list[dict[str, Any]]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    label_to_candidate = {int(c["label"]): c for c in candidates}
    ground_context_by_label: dict[int, bool] = {}
    for endpoint in endpoint_records:
        outward = endpoint.get("outward")
        if not outward:
            continue
        direct_ground = (
            find_ground_symbol_candidate_for_endpoint(unrecognized, endpoint, wire_width, cfg)
            if str(endpoint.get("owner_kind", "")) in {"solid_wire", "dash_group"}
            else None
        )
        if direct_ground is not None:
            owner_id = str(endpoint["owner_id"])
            bbox = [int(v) for v in direct_ground["bbox"]]
            gnd_id = f"p{page:03d}_ground_terminal_{len(nodes) + 1:04d}"
            obj = output_object(
                gnd_id,
                "ground_terminal",
                bbox,
                {"kind": "pixel_region", "pixel_bbox": [int(v) for v in bbox]},
                {
                    "source": "disconnected_endpoint_direct_ground_topology",
                    "estimated_wire_width": round(float(wire_width), 3),
                    "extension_hit": {
                        "owner_id": owner_id,
                        "owner_kind": str(endpoint["owner_kind"]),
                        "endpoint_index": int(endpoint["endpoint_index"]),
                    },
                    "global_ground_merge": False,
                    "pixel_bbox": [int(v) for v in bbox],
                    "pixel_runs": mask_to_relative_runs(direct_ground["mask"][bbox[1] : bbox[3], bbox[0] : bbox[2]]),
                    "pixel_count": int(np.count_nonzero(direct_ground["mask"])),
                    **direct_ground["metrics"],
                },
                connected_to=[owner_id],
                confidence=0.84,
            )
            nodes.append({"id": gnd_id, "kind": "ground_terminal", "object": obj, "mask": direct_ground["mask"], "bbox": bbox})
            uf.add(gnd_id)
            uf.union(gnd_id, owner_id)
            events.append({"event": "ground_terminal_inferred", "id": gnd_id, "connected_owner_id": owner_id, "bbox": bbox, "source": "direct_ground_topology"})
            continue
        if not candidates:
            continue
        best = best_compact_candidate_for_endpoint(
            endpoint_point=endpoint["point"],
            outward=outward,
            candidates=candidates,
            forward_search=forward_search,
            backward_search=backward_search,
            lateral_tol=lateral_tol,
        )
        if best is None:
            continue
        cand, along, perp = best
        hit = {
            "owner_id": str(endpoint["owner_id"]),
            "owner_kind": str(endpoint["owner_kind"]),
            "endpoint_index": int(endpoint["endpoint_index"]),
            "along": round(float(along), 3),
            "perpendicular_distance": round(float(perp), 3),
        }
        label = int(cand["label"])
        if label not in ground_context_by_label:
            ground_context_by_label[label] = candidate_has_ground_icon_context(binary, cand, wire_width, cfg)
        if ground_context_by_label[label]:
            gnd_hits[label].append(hit)
        else:
            hits_by_label[label].append(hit)

    for label, raw_hits in sorted(hits_by_label.items()):
        unique_owner_ids = sorted({str(hit["owner_id"]) for hit in raw_hits})
        cand = label_to_candidate[label]
        bbox = [int(v) for v in cand["bbox"]]
        if len(unique_owner_ids) >= 2 and len(raw_hits) >= int(cfg.endpoint_dot_min_hits):
            dot_id = f"p{page:03d}_connection_dot_{len(nodes) + 1:04d}"
            obj = output_object(
                dot_id,
                "connection_dot",
                bbox,
                {"kind": "point", "point": [round(float(cand["center"][0]), 3), round(float(cand["center"][1]), 3)]},
                {
                    "source": "disconnected_endpoint_dot_candidate",
                    "estimated_wire_width": round(float(wire_width), 3),
                    "area": int(cand["area"]),
                    "diameter": cand["diameter"],
                    "fill_ratio": cand["fill_ratio"],
                    "extension_hit_count": int(len(raw_hits)),
                    "unique_owner_hit_count": int(len(unique_owner_ids)),
                    "extension_hits": raw_hits[:20],
                    "gnd_context": False,
                },
                connected_to=unique_owner_ids,
                confidence=0.88,
            )
            nodes.append({"id": dot_id, "kind": "connection_dot", "object": obj, "mask": cand["pixel_mask"], "bbox": bbox})
            uf.add(dot_id)
            for owner_id in unique_owner_ids:
                uf.union(dot_id, owner_id)
            events.append({"event": "connection_dot_inferred", "id": dot_id, "connected_owner_ids": unique_owner_ids, "bbox": bbox})
            continue
        if len(unique_owner_ids) == 1:
            owner_id = unique_owner_ids[0]
            round_id = f"p{page:03d}_round_terminal_object_{len(nodes) + 1:04d}"
            obj = output_object(
                round_id,
                "round_terminal_object",
                bbox,
                {"kind": "point", "point": [round(float(cand["center"][0]), 3), round(float(cand["center"][1]), 3)]},
                {
                    "source": "disconnected_endpoint_round_object_candidate",
                    "estimated_wire_width": round(float(wire_width), 3),
                    "area": int(cand["area"]),
                    "diameter": cand["diameter"],
                    "fill_ratio": cand["fill_ratio"],
                    "extension_hit_count": int(len(raw_hits)),
                    "unique_owner_hit_count": int(len(unique_owner_ids)),
                    "extension_hits": raw_hits[:20],
                    "gnd_context": False,
                },
                connected_to=[owner_id],
                confidence=0.82,
            )
            nodes.append({"id": round_id, "kind": "round_terminal_object", "object": obj, "mask": cand["pixel_mask"], "bbox": bbox})
            uf.add(round_id)
            uf.union(round_id, owner_id)
            events.append({"event": "round_terminal_object_inferred", "id": round_id, "connected_owner_id": owner_id, "bbox": bbox})

    for label, raw_hits in sorted(gnd_hits.items()):
        cand = label_to_candidate[label]
        for hit in raw_hits:
            bbox = [int(v) for v in cand["bbox"]]
            gnd_id = f"p{page:03d}_ground_terminal_{len(nodes) + 1:04d}"
            owner_id = str(hit["owner_id"])
            obj = output_object(
                gnd_id,
                "ground_terminal",
                bbox,
                {"kind": "point", "point": [round(float(cand["center"][0]), 3), round(float(cand["center"][1]), 3)]},
                {
                    "source": "disconnected_endpoint_ground_candidate",
                    "estimated_wire_width": round(float(wire_width), 3),
                    "area": int(cand["area"]),
                    "diameter": cand["diameter"],
                    "fill_ratio": cand["fill_ratio"],
                    "extension_hit": hit,
                    "global_ground_merge": False,
                },
                connected_to=[owner_id],
                confidence=0.76,
            )
            nodes.append({"id": gnd_id, "kind": "ground_terminal", "object": obj, "mask": cand["pixel_mask"], "bbox": bbox})
            uf.add(gnd_id)
            uf.union(gnd_id, owner_id)
            events.append({"event": "ground_terminal_inferred", "id": gnd_id, "connected_owner_id": owner_id, "bbox": bbox})
    return nodes, events


def apply_existing_connections(line_items: Sequence[dict[str, Any]], uf: UnionFind) -> list[dict[str, Any]]:
    by_segment_id: dict[str, str] = {}
    for item in line_items:
        attrs = item.get("object", {}).get("attributes", {}) or {}
        segment_id = str(attrs.get("segment_id", ""))
        if segment_id:
            by_segment_id[segment_id] = str(item["id"])
        by_segment_id[str(item["id"])] = str(item["id"])
    events: list[dict[str, Any]] = []
    for item in line_items:
        attrs = item.get("object", {}).get("attributes", {}) or {}
        for endpoint in attrs.get("endpoints", []) or []:
            connected_ids = endpoint.get("connected_segment_ids", []) or []
            for connected_id in connected_ids:
                other = by_segment_id.get(str(connected_id))
                if other and other != item["id"]:
                    uf.union(str(item["id"]), other)
                    events.append({"source_id": str(item["id"]), "connected_segment_id": str(connected_id), "target_id": other})
        for connection in item.get("object", {}).get("connections", []) or []:
            connected_id = str(connection.get("wire_id", ""))
            other = by_segment_id.get(connected_id)
            if other and other != item["id"]:
                uf.union(str(item["id"]), other)
                events.append(
                    {
                        "source_id": str(item["id"]),
                        "connected_wire_id": connected_id,
                        "target_id": other,
                        "contact": connection.get("contact"),
                        "self_endpoint": connection.get("self_endpoint"),
                    }
                )
    return events


def assign_net_ids(nodes: Sequence[dict[str, Any]], uf: UnionFind, page: int) -> tuple[dict[str, str], list[dict[str, Any]]]:
    roots: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        node_id = str(node["id"])
        roots[uf.find(node_id)].append(node_id)
    ordered_roots = sorted(roots, key=lambda root: (min(roots[root]), len(roots[root])))
    root_to_net = {root: f"p{page:03d}_net_{idx:04d}" for idx, root in enumerate(ordered_roots, start=1)}
    node_to_net: dict[str, str] = {}
    nets: list[dict[str, Any]] = []
    kind_by_id = {str(node["id"]): str(node.get("kind", "")) for node in nodes}
    for root in ordered_roots:
        member_ids = sorted(roots[root])
        net_id = root_to_net[root]
        for member_id in member_ids:
            node_to_net[member_id] = net_id
        nets.append(
            {
                "id": net_id,
                "member_ids": member_ids,
                "member_kinds": {member_id: kind_by_id.get(member_id, "") for member_id in member_ids},
                "contains_ground_terminal": any(kind_by_id.get(member_id) == "ground_terminal" for member_id in member_ids),
                "global_ground_merge": False,
            }
        )
    return node_to_net, nets


def attach_net_to_object(obj: dict[str, Any], net_id: str | None) -> None:
    if not net_id:
        return
    attrs = obj.setdefault("attributes", {})
    attrs["net_id"] = str(net_id)
    attrs["endpoint_stage_net_id"] = str(net_id)


def page_json(
    pdf_name: str,
    page: int,
    dpi: int,
    width: int,
    height: int,
    objects: Sequence[dict[str, Any]],
    nets: Sequence[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pdf": pdf_name,
        "page": int(page),
        "dpi": int(dpi),
        "image_size": {"width": int(width), "height": int(height)},
        "coordinate_mapping": {
            "image_origin": "top-left",
            "pixel_units": "pixels",
            "source": f"outputs/{SOURCE_STAGE}/<pdf>/circuit_images",
        },
        "objects": list(objects),
        "nets": list(nets),
        "diagnostics": diagnostics,
    }


def overlay(
    rgb: np.ndarray,
    line_items: Sequence[dict[str, Any]],
    generated_nodes: Sequence[dict[str, Any]],
) -> np.ndarray:
    out = rgb.copy()
    for item in line_items:
        color = VISUAL_COLORS.get(str(item["kind"]), (90, 90, 90))
        draw_mask(out, item["mask"], color)
    for node in generated_nodes:
        color = VISUAL_COLORS.get(str(node["kind"]), (0, 210, 210))
        draw_mask(out, node["mask"], color)
    return out


def make_remaining_circuit_image(
    binary: np.ndarray,
    line_items: Sequence[dict[str, Any]],
    generated_nodes: Sequence[dict[str, Any]],
) -> np.ndarray:
    remove_mask = np.zeros(binary.shape, dtype=bool)
    for item in line_items:
        if str(item.get("kind", "")) in {"solid_wire", "dash_group"}:
            remove_mask |= item["mask"]
    for node in generated_nodes:
        remove_mask |= node["mask"]
    remaining = binary & ~remove_mask
    gray = np.where(remaining, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def process_page(
    rgb: np.ndarray,
    pdf_name: str,
    page: int,
    wire_json: dict[str, Any],
    rect_json: dict[str, Any],
    cfg: EndpointConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    t0 = time.perf_counter()
    binary = foreground(rgb, cfg.black_threshold)
    wire_width = estimate_wire_width_from_json(wire_json, rect_json)
    rect_edge_mask, rect_fill_mask = rectangle_masks(rect_json, binary.shape)
    line_items, symbol_nodes, suppression_events = make_line_items(binary, wire_json, {}, rect_json, cfg, wire_width)
    uf = UnionFind()
    base_nodes = line_items + symbol_nodes
    physical_union_events = union_nodes_by_connected_pixels(base_nodes, uf)
    existing_connection_events = apply_existing_connections(line_items, uf)
    generated_nodes, endpoint_objects, endpoint_events, generated_mask = classify_endpoints(
        binary,
        line_items,
        symbol_nodes,
        rect_edge_mask,
        rect_fill_mask,
        wire_width,
        cfg,
        page,
        uf,
    )
    all_net_nodes = line_items + symbol_nodes + generated_nodes
    for node in all_net_nodes:
        uf.add(str(node["id"]))
    node_to_net, nets = assign_net_ids(all_net_nodes, uf, page)

    output_objects: list[dict[str, Any]] = []
    for item in line_items:
        obj = item["object"]
        attrs = obj.setdefault("attributes", {})
        attrs["endpoint_stage_status"] = "retained"
        attrs["endpoint_stage_kind"] = str(item["kind"])
        attrs["endpoint_stage_events"] = item.get("endpoint_events", [])
        attrs["endpoint_classifications"] = item.get("endpoint_classifications", [])
        attach_net_to_object(obj, node_to_net.get(str(item["id"])))
        output_objects.append(obj)
    for node in symbol_nodes:
        obj = node["object"]
        attrs = obj.setdefault("attributes", {})
        attrs["endpoint_stage_status"] = "retained"
        attach_net_to_object(obj, node_to_net.get(str(node["id"])))
        output_objects.append(obj)
    for node in generated_nodes:
        obj = node["object"]
        attach_net_to_object(obj, node_to_net.get(str(node["id"])))
        output_objects.append(obj)
    output_objects.extend(endpoint_objects)

    review = overlay(rgb, line_items, generated_nodes)
    remaining_circuit = make_remaining_circuit_image(binary, line_items, generated_nodes)

    type_counts = Counter(str(obj.get("type", "")) for obj in output_objects)
    diagnostics = {
        "estimated_wire_width": round(float(wire_width), 3),
        "num_retained_line_items": int(len(line_items)),
        "num_retained_symbol_nodes": int(len(symbol_nodes)),
        "num_generated_nodes": int(len(generated_nodes)),
        "num_endpoint_objects": int(len(endpoint_objects)),
        "num_nets": int(len(nets)),
        "object_type_counts": dict(type_counts),
        "suppression_events": suppression_events,
        "physical_union_events": physical_union_events[:500],
        "existing_connection_events": existing_connection_events[:500],
        "endpoint_events": endpoint_events[:1000],
        "total_page_seconds": round(time.perf_counter() - t0, 4),
        "detector_backend": "endpoint_suppression_residual_connectivity_and_union_find",
    }
    data = page_json(pdf_name, page, cfg.dpi, rgb.shape[1], rgb.shape[0], output_objects, nets, diagnostics)
    return review, remaining_circuit, data


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="bmw-328i-1997.pdf")
    parser.add_argument("--pages", default=None, help=f"Default is all pages found in {SOURCE_STAGE}/circuit_images.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--wire-stage", default=WIRE_STAGE)
    parser.add_argument("--rect-stage", default=RECT_STAGE)
    parser.add_argument("--black-threshold", type=int, default=128)
    parser.add_argument("--write-legend", action="store_true", help="Write the optional visual legend file.")
    args = parser.parse_args()

    pdf = resolve_pdf(args.pdf)
    source_image_dir = resolve_source_circuit_dir(pdf.stem)
    source_pages = set(available_clean_pages(source_image_dir))
    wire_pages = set(available_stage_json_pages(args.wire_stage, pdf.stem))
    rect_pages = set(available_stage_json_pages(args.rect_stage, pdf.stem))
    available_pages = sorted(source_pages & wire_pages & rect_pages)
    if not available_pages:
        raise FileNotFoundError(
            "No common pages found across "
            f"{SOURCE_STAGE}/circuit_images, {args.wire_stage}/json, and {args.rect_stage}/json"
        )
    pages = parse_pages(args.pages, available_pages)
    cfg = EndpointConfig(dpi=args.dpi, black_threshold=args.black_threshold)
    root, image_dir, circuit_image_dir, json_dir = make_output_dirs(pdf.stem)
    if args.write_legend:
        write_visual_legend(root / "legend")

    logging.info("PDF path: %s", pdf)
    logging.info("Source circuit images: %s", source_image_dir)
    logging.info("Wire stage: %s", args.wire_stage)
    logging.info("Rect stage: %s", args.rect_stage)
    logging.info(
        "Available pages: source=%s wire=%s rect=%s common=%s",
        sorted(source_pages),
        sorted(wire_pages),
        sorted(rect_pages),
        available_pages,
    )
    logging.info("Pages: %s", pages)
    logging.info("Cleared output: %s", root)
    logging.info("Legend: %s", "written" if args.write_legend else "disabled")

    review_paths: list[Path] = []
    result_paths: list[Path] = []
    totals: Counter[str] = Counter()
    for page in pages:
        rgb = load_source_image(source_image_dir, page)
        wire_json = load_stage_page_json(args.wire_stage, pdf.stem, page)
        rect_json = load_stage_page_json(args.rect_stage, pdf.stem, page)
        review, remaining_circuit, data = process_page(rgb, pdf.name, page, wire_json, rect_json, cfg)
        review_path = image_dir / f"page_{page:03d}.png"
        circuit_image_path = circuit_image_dir / f"page_{page:03d}.png"
        save_rgb(review_path, review, args.dpi)
        save_rgb(circuit_image_path, remaining_circuit, args.dpi)
        write_json(json_dir / f"page_{page:03d}.json", data)
        review_paths.append(review_path)
        result_paths.append(circuit_image_path)
        totals.update(str(obj.get("type", "")) for obj in data.get("objects", []))
        logging.info(
            "Page %03d: objects=%d nets=%d generated=%d in %.2fs",
            page,
            len(data.get("objects", [])),
            len(data.get("nets", [])),
            int(data.get("diagnostics", {}).get("num_generated_nodes", 0)),
            float(data.get("diagnostics", {}).get("total_page_seconds", 0.0)),
        )

    images_to_pdf(review_paths, root / "review.pdf", args.dpi)
    images_to_pdf(result_paths, root / "result.pdf", args.dpi)
    logging.info("Detected: %s", dict(totals))
    logging.info("Output: %s", root)


if __name__ == "__main__":
    main()
