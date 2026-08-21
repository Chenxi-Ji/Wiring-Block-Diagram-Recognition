#!/usr/bin/env python3
"""Detect closed rectangular component frames from 01_circuit outputs.

Inputs:
  * outputs/01_circuit/<pdf_stem>/circuit_images/page_NNN.png

Outputs:
  * outputs/03_rect/<pdf_stem>/images/page_NNN.png
  * outputs/03_rect/<pdf_stem>/circuit_images/page_NNN.png (frames + their interior erased)
  * outputs/03_rect/<pdf_stem>/module_images/page_NNN/module_NNNN.png (one crop per frame, border erased)
  * outputs/03_rect/<pdf_stem>/masks/page_NNN.png
  * outputs/03_rect/<pdf_stem>/json/page_NNN.json
  * outputs/03_rect/<pdf_stem>/debug/page_NNN.json
  * outputs/03_rect/<pdf_stem>/review.pdf
  * outputs/03_rect/<pdf_stem>/result.pdf

This stage is intentionally separate from wire extraction. It finds component
or switch border frames that are made from four connected or near-connected
axis-aligned sides, then colors those frame pixels red in the review output.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from pipeline_io import image_to_pdf, parse_page_range, save_json


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / Path(__file__).stem
PHASE1_STAGE = "01_circuit"
STAGE_NAME = Path(__file__).stem

RED = (220, 0, 0)


@dataclass(frozen=True)
class RectConfig:
    ink_threshold: int = 210
    min_side_length_px: int = 32
    max_side_thickness_px: int = 12
    corner_tolerance_px: int = 8
    edge_band_px: int = 5
    min_edge_coverage: float = 0.58
    max_edge_gap_px: int = 20
    bridge_gap_px: int = 17
    composite_bridge_gap_px: int = 21
    dedupe_iou: float = 0.82
    dedupe_inside_ratio: float = 0.88
    min_frame_area_px: int = 1200
    corner_probe_px: int = 16
    corner_required_coverage: float = 0.45
    corner_forbidden_coverage: float = 0.65
    dashed_corner_forbidden_coverage: float = 0.78
    corner_forbidden_min_active_px: int = 7
    edge_endpoint_tolerance_px: int = 10
    solid_edge_min_coverage: float = 0.9
    solid_edge_max_gap_px: int = 3
    dashed_edge_min_gaps: int = 1
    dashed_gap_tolerance_px: int = 6
    dashed_mask_min_side_coverage: float = 0.62
    dashed_mask_coverage_max_area_px: int = 20000
    dashed_terminal_stub_max_px: int = 2
    dashed_terminal_stub_max_count: int = 2
    dashed_terminal_stub_max_area_px: int = 5000
    dashed_edge_max_thick_candidates: int = 1
    edge_refine_search_px: int = 10
    refined_edge_min_span_ratio: float = 0.95
    corner_mask_extension_px: int = 1
    corner_supplement_normal_tolerance_px: int = 3
    corner_supplement_min_density: float = 0.45
    small_dashed_min_side_length_px: int = 33
    small_dashed_max_area_px: int = 5000
    local_adjust_min_area_px: int = 1200
    local_adjust_max_area_px: int = 16000
    local_adjust_max_side_px: int = 180
    local_adjust_min_coverage: float = 0.55
    local_adjust_min_span_ratio: float = 0.78
    local_adjust_vertical_seed: bool = False
    min_candidate_span_ratio: float = 0.55
    side_search_min_tolerance_px: int = 60
    medium_recovery_min_area_px: int = 8000
    medium_recovery_max_area_px: int = 180000
    medium_recovery_min_side_px: int = 50
    medium_recovery_max_side_px: int = 900
    medium_recovery_min_coverage: float = 0.48
    medium_recovery_min_span_ratio: float = 0.72
    medium_recovery_max_gap_px: int = 34
    medium_recovery_endpoint_slack_px: int = 28
    enable_local_adjust: bool = False
    enable_medium_recovery: bool = False
    dashed_edge_min_fill_coverage: float = 0.92
    dashed_gap_reconstruction_tolerance: float = 1.3
    solid_edge_endpoint_trim_min_coverage: float = 0.8


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
    direct = ROOT_DIR / "outputs" / PHASE1_STAGE / pdf_stem / "circuit_images"
    if direct.is_dir():
        return direct
    stage_root = ROOT_DIR / "outputs" / PHASE1_STAGE
    if stage_root.is_dir():
        for candidate in stage_root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == pdf_stem.lower():
                image_dir = candidate / "circuit_images"
                if image_dir.is_dir():
                    return image_dir
    raise FileNotFoundError(
        f"Missing {PHASE1_STAGE} circuit images: {direct}. "
        f"Run: python scripts/01_circuit.py --pdf {pdf_stem}.pdf"
    )


def available_pages(image_dir: Path) -> list[int]:
    pages: list[int] = []
    for path in sorted(image_dir.glob("page_*.png")):
        try:
            pages.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
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
        raise ValueError(f"Requested page(s) {missing} are missing from {PHASE1_STAGE}/circuit_images")
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
    paths = {
        "root": root,
        "images": root / "images",
        "circuit_images": root / "circuit_images",
        "module_images": root / "module_images",
        "masks": root / "masks",
        "json": root / "json",
        "debug": root / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def binarize(rgb: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < int(threshold)


def clip_box(box: Sequence[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def overlap_len(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(float(a2), float(b2)) - max(float(a1), float(b1)))


def overlap_area(a: Sequence[int], b: Sequence[int]) -> float:
    return overlap_len(a[0], a[2], b[0], b[2]) * overlap_len(a[1], a[3], b[1], b[3])


def inside_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    area = box_area(a)
    return 0.0 if area <= 0 else float(overlap_area(a, b) / area)


def box_center(box: Sequence[float]) -> list[float]:
    return [0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3]))]


def contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.astype(bool)):
        if value and start is None:
            start = int(index)
        elif not value and start is not None:
            runs.append((start, int(index)))
            start = None
    if start is not None:
        runs.append((start, int(len(values))))
    return runs


def dashed_reconstructed_coverage(
    run_lengths: Sequence[int],
    gap_lengths: Sequence[int],
    total_len: int,
    gap_tolerance: float = 1.3,
) -> float:
    """Fill-rate for a dashed edge that treats normal periodic gaps as "covered".

    Each gap is credited up to ``gap_tolerance * median(gap_lengths)`` — a gap close to
    the typical dash spacing counts as fully filled, while the excess of an abnormally
    large gap (a real break in the line, not just dash spacing) is left uncovered.
    """
    if total_len <= 0:
        return 0.0
    if not gap_lengths:
        return float(sum(run_lengths) / max(1, total_len))
    typical_gap = float(np.median(np.asarray(gap_lengths, dtype=float)))
    cap = max(1.0, typical_gap * float(gap_tolerance))
    reconstructed = float(sum(run_lengths)) + sum(min(float(gap), cap) for gap in gap_lengths)
    return float(min(1.0, reconstructed / max(1, total_len)))


def axis_open_mask(binary: np.ndarray, orientation: str, min_length: int, bridge_gap: int) -> np.ndarray:
    src = binary.astype(np.uint8)
    if bridge_gap > 1:
        close_kernel = (bridge_gap, 1) if orientation == "horizontal" else (1, bridge_gap)
        src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel))
    open_kernel = (min_length, 1) if orientation == "horizontal" else (1, min_length)
    opened = cv2.morphologyEx(src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel))
    return opened.astype(bool)


def split_contaminated_component(
    binary: np.ndarray,
    orientation: str,
    bbox: Sequence[int],
    cfg: RectConfig,
) -> list[tuple[int, int, int, int]]:
    """Recover clean sub-segments from a bridged candidate whose gap-bridging closure
    fused a thin dashed/solid line with nearby unrelated ink (typically a text label).

    Scans RAW (unbridged) pixel thickness along the axis; wherever it jumps to an
    anomalous value the line is cut there (the two sides were never really pixel-connected
    — only the closure operation joined them), and each surviving thin, long-enough
    sub-range is re-measured on raw pixels and returned as its own candidate box.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    local = binary[y1:y2, x1:x2]
    if local.size == 0:
        return []
    col_thickness = local.sum(axis=0) if orientation == "horizontal" else local.sum(axis=1)
    nonzero = col_thickness[col_thickness > 0]
    if nonzero.size == 0:
        return []
    typical = float(np.median(nonzero))
    limit = max(float(cfg.max_side_thickness_px), typical * 2.5)
    clean = col_thickness <= limit
    clean_runs = contiguous_true_runs(clean)
    # Text glyphs produce many short, irregular thin-column runs; a real line's own
    # segments (a solid run, or one dash) are each substantially longer. Drop the short
    # ones as noise before merging, so text doesn't get stitched back onto the real edge.
    anchor_min_len = max(15, int(cfg.min_side_length_px) // 2)
    anchor_runs = [(a, b) for a, b in clean_runs if int(b - a) >= anchor_min_len]
    # A short anomalous stretch (a crossing wire/arrow stub) shouldn't split an
    # otherwise-clean line into two short pieces; bridge anchor runs separated only
    # by a narrow spike back together.
    spike_tol = max(2, int(cfg.corner_tolerance_px))
    merged_runs: list[tuple[int, int]] = []
    for a, b in anchor_runs:
        if merged_runs and a - merged_runs[-1][1] <= spike_tol:
            merged_runs[-1] = (merged_runs[-1][0], b)
        else:
            merged_runs.append((a, b))
    out: list[tuple[int, int, int, int]] = []
    for a, b in merged_runs:
        length = int(b - a)
        if length < int(cfg.min_side_length_px):
            continue
        clean_cols = np.where(clean[a:b])[0] + a
        if clean_cols.size == 0:
            continue
        strip = local[:, clean_cols] if orientation == "horizontal" else local[clean_cols, :]
        # Don't take the union of ink across every clean column: an unrelated mark in
        # just one or two columns (e.g. a pin label sitting under the line) would drag
        # the derived band wide again. Instead find the row(s) most columns agree on —
        # the actual line — and grow outward only while that agreement stays high.
        # Two genuinely distinct real lines (a border and a nearby wire) can land in the
        # same axis-range and get flattened into one blob by the bridging closure; find
        # each distinct peak in turn (masking out what's already been claimed) rather
        # than only the single strongest one, so neither line is silently dropped.
        row_counts = strip.sum(axis=1 if orientation == "horizontal" else 0).astype(float)
        claimed = np.zeros_like(row_counts, dtype=bool)
        first_peak_strength: float | None = None
        for _ in range(2):
            available = row_counts.copy()
            available[claimed] = 0.0
            if available.max(initial=0) <= 0:
                break
            peak = int(np.argmax(available))
            if first_peak_strength is None:
                first_peak_strength = float(available[peak])
            elif float(available[peak]) < 0.3 * first_peak_strength:
                # Too weak relative to the primary line to be a second real line --
                # more likely leftover noise (a symbol stroke, an antialiasing sliver).
                break
            row_threshold = max(1.0, float(available[peak]) * 0.4)
            band = (row_counts >= row_threshold) & ~claimed
            n0 = peak
            while n0 - 1 >= 0 and band[n0 - 1]:
                n0 -= 1
            n1 = peak + 1
            while n1 < band.size and band[n1]:
                n1 += 1
            claimed[n0:n1] = True
            # A distinct real line may only occupy part of this (possibly much wider)
            # merged axis-range -- e.g. a short border sharing the range with a longer
            # wire. Don't require it to span most of the range; instead find its own
            # longest contiguous run within the range and use that as the emitted extent.
            sub_strip = (
                local[n0:n1, a:b] if orientation == "horizontal" else local[a:b, n0:n1]
            )
            presence = sub_strip.any(axis=0 if orientation == "horizontal" else 1)
            presence_runs = contiguous_true_runs(presence)
            if not presence_runs:
                continue
            # A dashed line's presence is naturally interrupted by dash-to-dash gaps --
            # bridge those (same tolerance as candidate extraction) before picking the
            # longest run, so one dashed edge doesn't get truncated to its first dash.
            bridged_runs: list[tuple[int, int]] = []
            for pr_a, pr_b in presence_runs:
                if bridged_runs and pr_a - bridged_runs[-1][1] <= int(cfg.bridge_gap_px):
                    bridged_runs[-1] = (bridged_runs[-1][0], pr_b)
                else:
                    bridged_runs.append((pr_a, pr_b))
            pa, pb = max(bridged_runs, key=lambda run: run[1] - run[0])
            if int(pb - pa) < int(cfg.min_side_length_px):
                continue
            if orientation == "horizontal":
                out.append((int(x1 + a + pa), int(x1 + a + pb), int(y1 + n0), int(y1 + n1)))
            else:
                out.append((int(x1 + n0), int(x1 + n1), int(y1 + a + pa), int(y1 + a + pb)))
    return out


def extract_edge_candidates(
    binary: np.ndarray,
    orientation: str,
    cfg: RectConfig,
    min_side_length_px: int | None = None,
) -> list[dict[str, Any]]:
    min_length = int(cfg.min_side_length_px if min_side_length_px is None else min_side_length_px)
    opened = axis_open_mask(binary, orientation, min_length, cfg.bridge_gap_px)
    count, _, stats, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), 8)
    edges: list[dict[str, Any]] = []

    def add_edge(x: int, y: int, w: int, h: int, area: int, recovered: bool) -> None:
        length = w if orientation == "horizontal" else h
        thickness = h if orientation == "horizontal" else w
        if length < int(min_length):
            return
        if thickness > cfg.max_side_thickness_px:
            return
        if float(length / max(1, thickness)) < 3.0:
            return
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        edge = {
            "bbox": bbox,
            "orientation": orientation,
            "length_px": int(length),
            "thickness_px": int(thickness),
            "center": box_center(bbox),
            "area_px": int(area),
        }
        if recovered:
            edge["recovered_from_contaminated_component"] = True
        edges.append(edge)

    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        length = w if orientation == "horizontal" else h
        thickness = h if orientation == "horizontal" else w
        if length < int(min_length):
            continue
        if thickness > cfg.max_side_thickness_px:
            for rx1, rx2, ry1, ry2 in split_contaminated_component(
                binary, orientation, [x, y, x + w, y + h], cfg
            ):
                add_edge(rx1, ry1, rx2 - rx1, ry2 - ry1, int(rx2 - rx1) * int(ry2 - ry1), recovered=True)
            continue
        add_edge(x, y, w, h, area, recovered=False)
    return edges


def edge_is_curved(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    band_px: int,
    margin_px: int = 3,
) -> bool:
    """Detect a curved border (e.g. a circular connector ring) masquerading as a
    straight edge. A real straight line jumps directly between two offset levels
    (jagged/antialiased at worst); a circular arc drifts smoothly through several
    intermediate offsets like a shallow bowl. We tell them apart by comparing the
    total offset range to the largest single step between adjacent samples: a
    real edge's range is achieved in one big jump, a curved edge's range is spread
    across many small steps.
    """
    h, w = binary.shape
    band = max(1, int(band_px))
    span = int(axis_end - axis_start)
    if span <= 2 * margin_px + 4:
        return False
    a0 = int(axis_start) + margin_px
    a1 = int(axis_end) - margin_px
    if orientation == "horizontal":
        a0, a1 = max(0, a0), min(w, a1)
        if a1 <= a0:
            return False
        y0 = max(0, normal_center - band)
        y1 = min(h, normal_center + band + 1)
        if y1 <= y0:
            return False
        local = binary[y0:y1, a0:a1]
        has_ink = local.any(axis=0)
        if int(np.count_nonzero(has_ink)) < 6:
            return False
        first_idx = np.argmax(local, axis=0)
        offsets_arr = first_idx[has_ink]
    else:
        a0, a1 = max(0, a0), min(h, a1)
        if a1 <= a0:
            return False
        x0 = max(0, normal_center - band)
        x1 = min(w, normal_center + band + 1)
        if x1 <= x0:
            return False
        local = binary[a0:a1, x0:x1]
        has_ink = local.any(axis=1)
        if int(np.count_nonzero(has_ink)) < 6:
            return False
        first_idx = np.argmax(local, axis=1)
        offsets_arr = first_idx[has_ink]
    offset_range = int(offsets_arr.max() - offsets_arr.min())
    if offset_range < 3:
        return False
    max_step = int(np.abs(np.diff(offsets_arr)).max())
    return bool(max_step <= max(1, offset_range // 2))


def axis_projection_coverage(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    band_px: int,
) -> dict[str, Any]:
    h, w = binary.shape
    if axis_end <= axis_start:
        return {"coverage": 0.0, "max_gap_px": 0, "active_pixels": 0}
    band = max(1, int(band_px))
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_center - band, axis_end, normal_center + band + 1], w, h)
    else:
        box = clip_box([normal_center - band, axis_start, normal_center + band + 1, axis_end], w, h)
    if box is None:
        return {"coverage": 0.0, "max_gap_px": 0, "active_pixels": 0}
    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2]
    projection = local.sum(axis=0) > 0 if orientation == "horizontal" else local.sum(axis=1) > 0
    active = int(np.count_nonzero(projection))
    gaps = contiguous_true_runs(~projection)
    inner_gaps = [(int(a), int(b)) for a, b in gaps if a > 0 and b < projection.size]
    ink_runs = contiguous_true_runs(projection)
    max_gap = max((int(b - a) for a, b in gaps), default=0)
    dashed_effective_coverage = dashed_reconstructed_coverage(
        [int(b - a) for a, b in ink_runs],
        [int(b - a) for a, b in inner_gaps],
        int(projection.size),
    )
    if ink_runs:
        span_ratio = float((ink_runs[-1][1] - ink_runs[0][0]) / max(1, projection.size))
    else:
        span_ratio = 0.0
    return {
        "coverage": round(float(active / max(1, projection.size)), 4),
        "dashed_effective_coverage": round(float(dashed_effective_coverage), 4),
        "span_ratio": round(float(span_ratio), 4),
        "max_gap_px": int(max_gap),
        "inner_gap_lengths_px": [int(b - a) for a, b in inner_gaps],
        "ink_run_lengths_px": [int(b - a) for a, b in ink_runs],
        "active_pixels": int(active),
    }


def oriented_side_pixels(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    max_band_px: int,
    style: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=bool)
    band = max(1, int(max_band_px))
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_center - band, axis_end, normal_center + band + 1], w, h)
    else:
        box = clip_box([normal_center - band, axis_start, normal_center + band + 1, axis_end], w, h)
    if box is None:
        return out, {"bbox": None, "width_px": 0, "offset_px": 0}

    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2].astype(np.uint8)
    side_len = local.shape[1] if orientation == "horizontal" else local.shape[0]
    if str(style) == "dashed":
        kernel_len = max(2, min(7, int(side_len) // 10))
    else:
        kernel_len = max(5, min(21, int(side_len) // 8))
    kernel = (kernel_len, 1) if orientation == "horizontal" else (1, kernel_len)
    opened = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel)).astype(bool)
    if int(opened.sum()) == 0:
        opened = local.astype(bool)

    out[y1:y2, x1:x2] = opened
    ys, xs = np.where(opened)
    if xs.size == 0:
        return out, {"bbox": None, "width_px": 0, "offset_px": 0}
    if orientation == "horizontal":
        width_px = int(ys.max() - ys.min() + 1)
        offset_px = int(round(float(y1 + np.median(ys) - int(normal_center))))
    else:
        width_px = int(xs.max() - xs.min() + 1)
        offset_px = int(round(float(x1 + np.median(xs) - int(normal_center))))
    return out, {
        "bbox": [int(x1 + xs.min()), int(y1 + ys.min()), int(x1 + xs.max()) + 1, int(y1 + ys.max()) + 1],
        "width_px": int(width_px),
        "offset_px": int(offset_px),
        "kernel_len_px": int(kernel_len),
    }


def oriented_side_pixels_in_band(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_start: int,
    normal_end: int,
    style: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=bool)
    if axis_end <= axis_start or normal_end <= normal_start:
        return out, {"bbox": None, "width_px": 0, "offset_px": 0}
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_start, axis_end, normal_end], w, h)
    else:
        box = clip_box([normal_start, axis_start, normal_end, axis_end], w, h)
    if box is None:
        return out, {"bbox": None, "width_px": 0, "offset_px": 0}

    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2].astype(np.uint8)
    side_len = local.shape[1] if orientation == "horizontal" else local.shape[0]
    if str(style) == "dashed":
        kernel_len = max(2, min(7, int(side_len) // 10))
    else:
        kernel_len = max(5, min(21, int(side_len) // 8))
    kernel = (kernel_len, 1) if orientation == "horizontal" else (1, kernel_len)
    opened = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel)).astype(bool)
    if int(opened.sum()) == 0:
        opened = local.astype(bool)

    out[y1:y2, x1:x2] = opened
    ys, xs = np.where(opened)
    width_px = int(normal_end - normal_start)
    normal_center = 0.5 * float(normal_start + normal_end)
    if xs.size == 0:
        return out, {
            "bbox": None,
            "width_px": int(width_px),
            "offset_px": 0,
            "band_bbox": [int(v) for v in box],
            "kernel_len_px": int(kernel_len),
        }
    if orientation == "horizontal":
        offset_px = int(round(float(y1 + np.median(ys) - normal_center)))
    else:
        offset_px = int(round(float(x1 + np.median(xs) - normal_center)))
    return out, {
        "bbox": [int(x1 + xs.min()), int(y1 + ys.min()), int(x1 + xs.max()) + 1, int(y1 + ys.max()) + 1],
        "width_px": int(width_px),
        "offset_px": int(offset_px),
        "band_bbox": [int(v) for v in box],
        "kernel_len_px": int(kernel_len),
    }


def refined_side_ref(edge: dict[str, Any]) -> dict[str, Any] | None:
    refinement = edge.get("refinement", {})
    required = {"axis_start", "axis_end", "normal_start", "normal_end"}
    if not isinstance(refinement, dict) or not required.issubset(refinement):
        return None
    return refinement


def corner_endpoint_supplement_pixels(
    binary: np.ndarray,
    base_mask: np.ndarray,
    side_specs: Sequence[tuple[str, str, int, int, int, int]],
    cfg: RectConfig,
) -> tuple[np.ndarray, dict[str, list[dict[str, Any]]]]:
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=bool)
    touch_mask = cv2.dilate(
        base_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    ).astype(bool)
    probe = max(int(cfg.corner_probe_px), int(cfg.edge_endpoint_tolerance_px), 12)
    endpoint_tol = max(2, int(cfg.corner_tolerance_px))
    info: dict[str, list[dict[str, Any]]] = {}

    for name, orientation, axis_start, axis_end, normal_start, normal_end in side_specs:
        axis_start = int(axis_start)
        axis_end = int(axis_end)
        normal_start = int(normal_start)
        normal_end = int(normal_end)
        if axis_end <= axis_start or normal_end <= normal_start:
            continue
        width = max(1, int(normal_end - normal_start))
        normal_tol = max(
            1,
            min(
                int(cfg.corner_supplement_normal_tolerance_px),
                int(width) + 2,
            ),
        )
        side_items: list[dict[str, Any]] = []
        endpoint_specs = [
            ("start", int(axis_start), int(axis_start), int(min(axis_end, axis_start + probe))),
            ("end", int(axis_end), int(max(axis_start, axis_end - probe)), int(axis_end)),
        ]
        for endpoint_name, endpoint_axis, search_start, search_end in endpoint_specs:
            if search_end <= search_start:
                continue
            if orientation == "horizontal":
                box = clip_box(
                    [
                        search_start - endpoint_tol,
                        normal_start - normal_tol,
                        search_end + endpoint_tol,
                        normal_end + normal_tol,
                    ],
                    w,
                    h,
                )
            else:
                box = clip_box(
                    [
                        normal_start - normal_tol,
                        search_start - endpoint_tol,
                        normal_end + normal_tol,
                        search_end + endpoint_tol,
                    ],
                    w,
                    h,
                )
            if box is None:
                continue
            bx1, by1, bx2, by2 = box
            local = binary[by1:by2, bx1:bx2].astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
            for label in range(1, count):
                lx, ly, lw, lh, area = [int(value) for value in stats[label]]
                if area <= 0:
                    continue
                gx1, gy1, gx2, gy2 = bx1 + lx, by1 + ly, bx1 + lx + lw, by1 + ly + lh
                if orientation == "horizontal":
                    comp_axis_start, comp_axis_end = gx1, gx2
                    comp_normal_start, comp_normal_end = gy1, gy2
                else:
                    comp_axis_start, comp_axis_end = gy1, gy2
                    comp_normal_start, comp_normal_end = gx1, gx2
                comp_axis_len = int(comp_axis_end - comp_axis_start)
                comp_normal_len = int(comp_normal_end - comp_normal_start)
                if comp_axis_len <= 0 or comp_normal_len <= 0:
                    continue
                if comp_axis_len > probe + endpoint_tol * 2:
                    continue
                if comp_normal_len > width + normal_tol * 2:
                    continue
                comp_normal_center = 0.5 * float(comp_normal_start + comp_normal_end)
                side_normal_center = 0.5 * float(normal_start + normal_end)
                if abs(comp_normal_center - side_normal_center) > float(normal_tol) + max(1.0, width / 2.0):
                    continue
                normal_overlap = overlap_len(comp_normal_start, comp_normal_end, normal_start, normal_end)
                if normal_overlap <= 0 and abs(comp_normal_center - side_normal_center) > float(normal_tol):
                    continue
                if endpoint_name == "start":
                    endpoint_distance = max(0, int(comp_axis_start - endpoint_axis))
                else:
                    endpoint_distance = max(0, int(endpoint_axis - comp_axis_end))
                if endpoint_distance > endpoint_tol:
                    continue
                comp = labels[ly : ly + lh, lx : lx + lw] == label
                if orientation == "horizontal":
                    strip_normal_start = max(gy1, int(normal_start) - int(normal_tol))
                    strip_normal_end = min(gy2, int(normal_end) + int(normal_tol))
                    if strip_normal_end <= strip_normal_start:
                        continue
                    oriented_comp = np.zeros_like(comp, dtype=bool)
                    oriented_comp[
                        strip_normal_start - gy1 : strip_normal_end - gy1,
                        :,
                    ] = comp[
                        strip_normal_start - gy1 : strip_normal_end - gy1,
                        :,
                    ]
                    axis_projection = oriented_comp.sum(axis=0) > 0
                    strip_area = oriented_comp.shape[1] * max(1, strip_normal_end - strip_normal_start)
                else:
                    strip_normal_start = max(gx1, int(normal_start) - int(normal_tol))
                    strip_normal_end = min(gx2, int(normal_end) + int(normal_tol))
                    if strip_normal_end <= strip_normal_start:
                        continue
                    oriented_comp = np.zeros_like(comp, dtype=bool)
                    oriented_comp[
                        :,
                        strip_normal_start - gx1 : strip_normal_end - gx1,
                    ] = comp[
                        :,
                        strip_normal_start - gx1 : strip_normal_end - gx1,
                    ]
                    axis_projection = oriented_comp.sum(axis=1) > 0
                    strip_area = oriented_comp.shape[0] * max(1, strip_normal_end - strip_normal_start)
                bbox_density = float(area / max(1, lw * lh))
                strip_density = float(oriented_comp.sum() / max(1, strip_area))
                axis_density = float(np.count_nonzero(axis_projection) / max(1, axis_projection.size))
                density = max(bbox_density, axis_density)
                if density < float(cfg.corner_supplement_min_density):
                    continue
                touch = touch_mask[gy1:gy2, gx1:gx2]
                if not bool(np.any(oriented_comp & touch)):
                    continue
                target = out[gy1:gy2, gx1:gx2]
                current = base_mask[gy1:gy2, gx1:gx2] | target
                new_pixels = oriented_comp & ~current
                if int(new_pixels.sum()) <= 0:
                    continue
                target |= oriented_comp
                side_items.append(
                    {
                        "endpoint": endpoint_name,
                        "bbox": [int(gx1), int(gy1), int(gx2), int(gy2)],
                        "axis_len_px": int(comp_axis_len),
                        "normal_len_px": int(comp_normal_len),
                        "new_pixels": int(new_pixels.sum()),
                        "density": round(float(density), 4),
                        "bbox_density": round(float(bbox_density), 4),
                        "strip_density": round(float(strip_density), 4),
                        "axis_density": round(float(axis_density), 4),
                        "normal_center_distance_px": round(float(abs(comp_normal_center - side_normal_center)), 4),
                        "endpoint_distance_px": int(endpoint_distance),
                    }
                )
        if side_items:
            info[str(name)] = side_items
    return out, info


def side_mask_gap_metrics(
    mask: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    band_px: int,
    cfg: RectConfig,
) -> dict[str, Any]:
    h, w = mask.shape
    band = max(1, int(band_px))
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_center - band, axis_end, normal_center + band + 1], w, h)
    else:
        box = clip_box([normal_center - band, axis_start, normal_center + band + 1, axis_end], w, h)
    if box is None:
        return {"coverage": 0.0, "gap_lengths_px": [], "run_lengths_px": [], "max_gap_px": 0}
    x1, y1, x2, y2 = box
    local = mask[y1:y2, x1:x2]
    projection = local.sum(axis=0) > 0 if orientation == "horizontal" else local.sum(axis=1) > 0
    gaps = [
        int(b - a)
        for a, b in contiguous_true_runs(~projection)
        if a > 0 and b < projection.size and int(b - a) > int(cfg.solid_edge_max_gap_px)
    ]
    runs = [int(b - a) for a, b in contiguous_true_runs(projection)]
    return {
        "coverage": round(float(np.count_nonzero(projection) / max(1, projection.size)), 4),
        "gap_lengths_px": gaps,
        "run_lengths_px": runs,
        "max_gap_px": max(gaps, default=0),
        "median_gap_px": round(float(np.median(gaps)), 4) if gaps else 0.0,
        "median_run_px": round(float(np.median(runs)), 4) if runs else 0.0,
    }


def validate_colored_dashed_gaps(mask: np.ndarray, frame: dict[str, Any], cfg: RectConfig) -> dict[str, Any]:
    if str(frame.get("geometry", {}).get("line_style", "solid")) != "dashed":
        return {"accepted": True, "reason": None}
    seed_axis = str(frame.get("geometry", {}).get("seed_axis", ""))
    local_adjusted = seed_axis == "local_adjusted"
    medium_large_recovery = seed_axis == "medium_large_recovery"
    local_narrow_symbol = bool(frame.get("geometry", {}).get("local_narrow_symbol_frame", False))
    x1, y1, x2, y2 = [int(v) for v in frame["bbox"]]
    side_widths = frame.get("mask_side_widths_px", {})
    side_offsets = frame.get("mask_side_offsets_px", {})
    edges = frame.get("edges", {})
    refined_refs = {
        name: refined_side_ref(edges.get(name, {}))
        for name in ("top", "bottom", "left", "right")
    }
    if all(item is not None for item in refined_refs.values()):
        refs = {name: item for name, item in refined_refs.items() if item is not None}
        specs = {
            "top": (
                "horizontal",
                int(refs["top"]["axis_start"]),
                int(refs["top"]["axis_end"]),
                int(round(0.5 * float(int(refs["top"]["normal_start"]) + int(refs["top"]["normal_end"])))),
            ),
            "bottom": (
                "horizontal",
                int(refs["bottom"]["axis_start"]),
                int(refs["bottom"]["axis_end"]),
                int(round(0.5 * float(int(refs["bottom"]["normal_start"]) + int(refs["bottom"]["normal_end"])))),
            ),
            "left": (
                "vertical",
                int(refs["left"]["axis_start"]),
                int(refs["left"]["axis_end"]),
                int(round(0.5 * float(int(refs["left"]["normal_start"]) + int(refs["left"]["normal_end"])))),
            ),
            "right": (
                "vertical",
                int(refs["right"]["axis_start"]),
                int(refs["right"]["axis_end"]),
                int(round(0.5 * float(int(refs["right"]["normal_start"]) + int(refs["right"]["normal_end"])))),
            ),
        }
    else:
        specs = {
            "top": ("horizontal", x1, x2, y1),
            "bottom": ("horizontal", x1, x2, y2),
            "left": ("vertical", y1, y2, x1),
            "right": ("vertical", y1, y2, x2),
        }
    metrics = {}
    gap_medians: list[float] = []
    run_medians: list[float] = []
    coverages: list[float] = []
    for name, (orientation, axis_start, axis_end, normal_center) in specs.items():
        width = max(1, int(side_widths.get(name, 1)))
        band = max(1, int(np.ceil(float(width) / 2.0)) + 1)
        normal = int(normal_center) + int(side_offsets.get(name, 0))
        item = side_mask_gap_metrics(mask, orientation, axis_start, axis_end, normal, band, cfg)
        metrics[name] = item
        coverages.append(float(item.get("coverage", 0.0)))
        if item["gap_lengths_px"]:
            gap_medians.append(float(item["median_gap_px"]))
        if item["run_lengths_px"]:
            run_medians.append(float(item["median_run_px"]))

    gap_range = max(gap_medians) - min(gap_medians) if len(gap_medians) >= 2 else 0.0
    run_range = max(run_medians) - min(run_medians) if len(run_medians) >= 2 else 0.0
    max_allowed_gap_range = max(float(cfg.dashed_gap_tolerance_px) * 2.0, 10.0)
    max_allowed_run_range = max(float(cfg.dashed_gap_tolerance_px) * 3.0, 18.0)
    if medium_large_recovery:
        max_allowed_gap_range = max(max_allowed_gap_range, float(cfg.medium_recovery_max_gap_px))
        max_allowed_run_range = max(max_allowed_run_range, 36.0)
    min_side_coverage = min(coverages) if coverages else 0.0
    area = box_area(frame["bbox"])
    terminal_stub_count = 0
    for item in metrics.values():
        runs = [int(value) for value in item.get("run_lengths_px", [])]
        if runs and runs[0] <= int(cfg.dashed_terminal_stub_max_px):
            terminal_stub_count += 1
        if len(runs) > 1 and runs[-1] <= int(cfg.dashed_terminal_stub_max_px):
            terminal_stub_count += 1
    if (
        area <= int(cfg.dashed_mask_coverage_max_area_px)
        and min_side_coverage < float(cfg.dashed_mask_min_side_coverage)
        and not local_narrow_symbol
        and not medium_large_recovery
    ):
        return {
            "accepted": False,
            "reason": "colored_dashed_side_undercovered",
            "min_side_coverage": round(float(min_side_coverage), 4),
            "side_metrics": metrics,
        }
    if (
        area <= int(cfg.dashed_terminal_stub_max_area_px)
        and terminal_stub_count > int(cfg.dashed_terminal_stub_max_count)
        and not bool(local_adjusted)
        and not medium_large_recovery
    ):
        return {
            "accepted": False,
            "reason": "colored_dashed_terminal_stubs",
            "terminal_stub_count": int(terminal_stub_count),
            "side_metrics": metrics,
        }
    if len(gap_medians) >= 2 and gap_range > max_allowed_gap_range and not local_narrow_symbol:
        return {
            "accepted": False,
            "reason": "colored_dashed_gap_inconsistent",
            "gap_median_range_px": round(float(gap_range), 4),
            "side_metrics": metrics,
        }
    if (
        len(run_medians) >= 3
        and run_range > max_allowed_run_range
        and not bool(local_adjusted)
        and not medium_large_recovery
    ):
        return {
            "accepted": False,
            "reason": "colored_dashed_run_inconsistent",
            "run_median_range_px": round(float(run_range), 4),
            "side_metrics": metrics,
        }
    return {
        "accepted": True,
        "reason": None,
        "gap_median_range_px": round(float(gap_range), 4),
        "run_median_range_px": round(float(run_range), 4),
        "min_side_coverage": round(float(min_side_coverage), 4),
        "terminal_stub_count": int(terminal_stub_count),
        "side_metrics": metrics,
    }


def canonical_rect_mask(binary: np.ndarray, frame: dict[str, Any], cfg: RectConfig) -> np.ndarray:
    h, w = int(binary.shape[0]), int(binary.shape[1])
    out = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in frame["bbox"]]
    style = str(frame.get("geometry", {}).get("line_style", "solid"))
    edges = frame.get("edges", {})
    refined_refs = {
        name: refined_side_ref(edges.get(name, {}))
        for name in ("top", "bottom", "left", "right")
    }
    if all(item is not None for item in refined_refs.values()):
        refs = {name: item for name, item in refined_refs.items() if item is not None}
        edge_pixels = np.zeros_like(out, dtype=bool)
        corner_ext = max(0, int(frame.get("geometry", {}).get("corner_mask_extension_px", cfg.corner_mask_extension_px)))
        side_specs = [
            (
                "top",
                "horizontal",
                min(int(refs["top"]["axis_start"]), int(refs["left"]["normal_start"])) - corner_ext,
                max(int(refs["top"]["axis_end"]), int(refs["right"]["normal_end"])) + corner_ext,
                int(refs["top"]["normal_start"]),
                int(refs["top"]["normal_end"]),
            ),
            (
                "bottom",
                "horizontal",
                min(int(refs["bottom"]["axis_start"]), int(refs["left"]["normal_start"])) - corner_ext,
                max(int(refs["bottom"]["axis_end"]), int(refs["right"]["normal_end"])) + corner_ext,
                int(refs["bottom"]["normal_start"]),
                int(refs["bottom"]["normal_end"]),
            ),
            (
                "left",
                "vertical",
                min(int(refs["left"]["axis_start"]), int(refs["top"]["normal_start"])) - corner_ext,
                max(int(refs["left"]["axis_end"]), int(refs["bottom"]["normal_end"])) + corner_ext,
                int(refs["left"]["normal_start"]),
                int(refs["left"]["normal_end"]),
            ),
            (
                "right",
                "vertical",
                min(int(refs["right"]["axis_start"]), int(refs["top"]["normal_start"])) - corner_ext,
                max(int(refs["right"]["axis_end"]), int(refs["bottom"]["normal_end"])) + corner_ext,
                int(refs["right"]["normal_start"]),
                int(refs["right"]["normal_end"]),
            ),
        ]
        side_bands: dict[str, int] = {}
        side_widths: dict[str, int] = {}
        side_offsets: dict[str, int] = {}
        side_filters: dict[str, dict[str, Any]] = {}
        for name, orientation, axis_start, axis_end, normal_start, normal_end in side_specs:
            side_mask, side_info = oriented_side_pixels_in_band(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                int(normal_start),
                int(normal_end),
                style,
            )
            edge_pixels |= side_mask
            side_width = int(side_info.get("width_px", 0))
            side_bands[name] = max(0, int((side_width - 1) // 2)) if side_width > 0 else 0
            side_widths[name] = int(side_width)
            side_offsets[name] = int(side_info.get("offset_px", 0))
            side_filters[name] = side_info
        if bool(frame.get("geometry", {}).get("local_narrow_symbol_frame", False)):
            corner_supplement_info = {}
        else:
            corner_supplements, corner_supplement_info = corner_endpoint_supplement_pixels(
                binary,
                edge_pixels,
                side_specs,
                cfg,
            )
            if corner_supplement_info:
                edge_pixels |= corner_supplements
        if edge_pixels.any():
            frame["mask_side_bands_px"] = side_bands
            frame["mask_side_widths_px"] = side_widths
            frame["mask_side_offsets_px"] = side_offsets
            frame["mask_side_filters"] = side_filters
            frame["mask_corner_supplements"] = corner_supplement_info
            return edge_pixels

    half = max(1, int(round(float(frame.get("line_thickness_px", 1)) / 2.0)))
    edge_pixels = np.zeros_like(out, dtype=bool)
    max_pixel_band = max(2, int(half) + 2)
    side_specs = [
        ("top", "horizontal", x1, x2, y1),
        ("bottom", "horizontal", x1, x2, y2),
        ("left", "vertical", y1, y2, x1),
        ("right", "vertical", y1, y2, x2),
    ]
    side_bands: dict[str, int] = {}
    side_widths: dict[str, int] = {}
    side_offsets: dict[str, int] = {}
    side_filters: dict[str, dict[str, Any]] = {}
    for name, orientation, axis_start, axis_end, normal_center in side_specs:
        side_mask, side_info = oriented_side_pixels(
            binary,
            orientation,
            int(axis_start),
            int(axis_end),
            int(normal_center),
            max_pixel_band,
            style,
        )
        edge_pixels |= side_mask
        side_width = int(side_info.get("width_px", 0))
        side_bands[name] = max(0, int((side_width - 1) // 2)) if side_width > 0 else 0
        side_widths[name] = int(side_width)
        side_offsets[name] = int(side_info.get("offset_px", 0))
        side_filters[name] = side_info
    if edge_pixels.any():
        frame["mask_side_bands_px"] = side_bands
        frame["mask_side_widths_px"] = side_widths
        frame["mask_side_offsets_px"] = side_offsets
        frame["mask_side_filters"] = side_filters
        return edge_pixels
    if style == "dashed":
        draw_dashed_canonical_side(out, binary, "horizontal", x1, x2, y1, half)
        draw_dashed_canonical_side(out, binary, "horizontal", x1, x2, y2, half)
        draw_dashed_canonical_side(out, binary, "vertical", y1, y2, x1, half)
        draw_dashed_canonical_side(out, binary, "vertical", y1, y2, x2, half)
    else:
        boxes = [
            [x1, y1 - half, x2, y1 + half + 1],
            [x1, y2 - half, x2, y2 + half + 1],
            [x1 - half, y1, x1 + half + 1, y2],
            [x2 - half, y1, x2 + half + 1, y2],
        ]
        for raw_box in boxes:
            box = clip_box(raw_box, w, h)
            if box is None:
                continue
            bx1, by1, bx2, by2 = box
            out[by1:by2, bx1:bx2] = True
    return out


def draw_dashed_canonical_side(
    out: np.ndarray,
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    half: int,
) -> None:
    h, w = binary.shape
    band = max(2, int(half) + 2)
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_center - band, axis_end, normal_center + band + 1], w, h)
    else:
        box = clip_box([normal_center - band, axis_start, normal_center + band + 1, axis_end], w, h)
    if box is None:
        return
    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2]
    projection = local.sum(axis=0) > 0 if orientation == "horizontal" else local.sum(axis=1) > 0
    runs = contiguous_true_runs(projection)
    for start, end in runs:
        if orientation == "horizontal":
            raw_box = [x1 + start, normal_center - half, x1 + end, normal_center + half + 1]
        else:
            raw_box = [normal_center - half, y1 + start, normal_center + half + 1, y1 + end]
        run_box = clip_box(raw_box, w, h)
        if run_box is None:
            continue
        rx1, ry1, rx2, ry2 = run_box
        out[ry1:ry2, rx1:rx2] = True


def edge_axis_range(edge: dict[str, Any]) -> tuple[int, int]:
    x1, y1, x2, y2 = [int(v) for v in edge["bbox"]]
    if edge["orientation"] == "horizontal":
        return x1, x2
    return y1, y2


def edge_normal_center(edge: dict[str, Any]) -> float:
    x1, y1, x2, y2 = [int(v) for v in edge["bbox"]]
    if edge["orientation"] == "horizontal":
        return 0.5 * float(y1 + y2)
    return 0.5 * float(x1 + x2)


def horizontal_pair_alignment(top: dict[str, Any], bottom: dict[str, Any], cfg: RectConfig) -> dict[str, Any]:
    tx1, _, tx2, _ = [int(v) for v in top["bbox"]]
    bx1, _, bx2, _ = [int(v) for v in bottom["bbox"]]
    top_len = max(1, tx2 - tx1)
    bottom_len = max(1, bx2 - bx1)
    overlap = float(overlap_len(tx1, tx2, bx1, bx2))
    shorter = float(min(top_len, bottom_len))
    endpoint_spread = int(max(abs(tx1 - bx1), abs(tx2 - bx2)))
    weak_endpoint_tol = max(
        int(cfg.min_side_length_px),
        int(cfg.edge_endpoint_tolerance_px) * 4,
        int(cfg.corner_tolerance_px) * 5,
    )
    overlap_ratio = float(overlap / max(1.0, shorter))
    accepted = overlap >= float(cfg.min_side_length_px) and (
        overlap_ratio >= 0.35 or endpoint_spread <= weak_endpoint_tol
    )
    reason = None
    if overlap < float(cfg.min_side_length_px):
        reason = "horizontal_overlap_below_min"
    elif not accepted:
        reason = "horizontal_pair_weak_alignment_failed"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "overlap_px": round(float(overlap), 4),
        "overlap_ratio": round(float(overlap_ratio), 4),
        "endpoint_spread_px": int(endpoint_spread),
        "weak_endpoint_tolerance_px": int(weak_endpoint_tol),
    }


def vertical_pair_alignment(left: dict[str, Any], right: dict[str, Any], cfg: RectConfig) -> dict[str, Any]:
    _, ly1, _, ly2 = [int(v) for v in left["bbox"]]
    _, ry1, _, ry2 = [int(v) for v in right["bbox"]]
    left_len = max(1, ly2 - ly1)
    right_len = max(1, ry2 - ry1)
    overlap = float(overlap_len(ly1, ly2, ry1, ry2))
    shorter = float(min(left_len, right_len))
    endpoint_spread = int(max(abs(ly1 - ry1), abs(ly2 - ry2)))
    weak_endpoint_tol = max(
        int(cfg.min_side_length_px),
        int(cfg.edge_endpoint_tolerance_px) * 4,
        int(cfg.corner_tolerance_px) * 5,
    )
    overlap_ratio = float(overlap / max(1.0, shorter))
    accepted = overlap >= float(cfg.min_side_length_px) and (
        overlap_ratio >= 0.35 or endpoint_spread <= weak_endpoint_tol
    )
    reason = None
    if overlap < float(cfg.min_side_length_px):
        reason = "vertical_overlap_below_min"
    elif not accepted:
        reason = "vertical_pair_weak_alignment_failed"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "overlap_px": round(float(overlap), 4),
        "overlap_ratio": round(float(overlap_ratio), 4),
        "endpoint_spread_px": int(endpoint_spread),
        "weak_endpoint_tolerance_px": int(weak_endpoint_tol),
    }


def best_side_edge(
    edges: list[dict[str, Any]],
    top_y: int,
    bottom_y: int,
    x_min: int,
    x_max: int,
    cfg: RectConfig,
) -> dict[str, Any] | None:
    if bottom_y <= top_y:
        return None
    desired_height = int(bottom_y - top_y)
    min_overlap = max(float(cfg.min_side_length_px) * 0.5, float(desired_height) * 0.35)
    ranked: list[tuple[float, float, float, float, dict[str, Any]]] = []
    window_center = 0.5 * float(x_min + x_max)
    half_window = max(1.0, 0.5 * float(x_max - x_min))
    endpoint_tol = max(int(cfg.edge_endpoint_tolerance_px), int(cfg.corner_tolerance_px) * 2)
    for edge in edges:
        if edge["orientation"] != "vertical":
            continue
        x = float(edge_normal_center(edge))
        if x < float(x_min) or x > float(x_max):
            continue
        y1, y2 = edge_axis_range(edge)
        overlap = float(overlap_len(y1, y2, top_y, bottom_y))
        if overlap < min_overlap:
            continue
        overlap_ratio = overlap / max(1.0, float(desired_height))
        top_gap = max(0, int(y1) - int(top_y))
        bottom_gap = max(0, int(bottom_y) - int(y2))
        endpoint_penalty = float(max(0, top_gap - endpoint_tol) + max(0, bottom_gap - endpoint_tol))
        distance_px = abs(x - window_center)
        distance_penalty = float(distance_px / half_window)
        endpoint_penalty_norm = float(endpoint_penalty / max(1.0, float(desired_height)))
        length_ratio = float(edge.get("length_px", 0)) / max(1.0, float(desired_height))
        excess_length_penalty = max(0.0, length_ratio - 1.5)
        score = (
            1.8 * float(overlap_ratio)
            - 0.85 * distance_penalty
            - 0.65 * endpoint_penalty_norm
            - 0.04 * excess_length_penalty
        )
        ranked.append(
            (
                float(score),
                float(overlap_ratio),
                -distance_px,
                -endpoint_penalty,
                edge,
            )
        )
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[:4])[4]


def best_cap_edge(
    edges: list[dict[str, Any]],
    left_x: int,
    right_x: int,
    y_min: int,
    y_max: int,
    cfg: RectConfig,
) -> dict[str, Any] | None:
    if right_x <= left_x:
        return None
    desired_width = int(right_x - left_x)
    min_overlap = max(float(cfg.min_side_length_px) * 0.5, float(desired_width) * 0.35)
    ranked: list[tuple[float, float, float, float, dict[str, Any]]] = []
    window_center = 0.5 * float(y_min + y_max)
    half_window = max(1.0, 0.5 * float(y_max - y_min))
    endpoint_tol = max(int(cfg.edge_endpoint_tolerance_px), int(cfg.corner_tolerance_px) * 2)
    for edge in edges:
        if edge["orientation"] != "horizontal":
            continue
        y = float(edge_normal_center(edge))
        if y < float(y_min) or y > float(y_max):
            continue
        x1, x2 = edge_axis_range(edge)
        overlap = float(overlap_len(x1, x2, left_x, right_x))
        if overlap < min_overlap:
            continue
        overlap_ratio = overlap / max(1.0, float(desired_width))
        left_gap = max(0, int(x1) - int(left_x))
        right_gap = max(0, int(right_x) - int(x2))
        endpoint_penalty = float(max(0, left_gap - endpoint_tol) + max(0, right_gap - endpoint_tol))
        distance_px = abs(y - window_center)
        distance_penalty = float(distance_px / half_window)
        endpoint_penalty_norm = float(endpoint_penalty / max(1.0, float(desired_width)))
        length_ratio = float(edge.get("length_px", 0)) / max(1.0, float(desired_width))
        excess_length_penalty = max(0.0, length_ratio - 1.5)
        score = (
            1.8 * float(overlap_ratio)
            - 0.85 * distance_penalty
            - 0.65 * endpoint_penalty_norm
            - 0.04 * excess_length_penalty
        )
        ranked.append(
            (
                float(score),
                float(overlap_ratio),
                -distance_px,
                -endpoint_penalty,
                edge,
            )
        )
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[:4])[4]


def edge_candidate_span_ratio(edge: dict[str, Any], start: int, end: int) -> float:
    desired = max(1.0, float(end - start))
    a1, a2 = edge_axis_range(edge)
    return float(overlap_len(a1, a2, start, end) / desired)


def edge_band_projection_metrics(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_start: int,
    normal_end: int,
    cfg: RectConfig,
) -> dict[str, Any]:
    h, w = binary.shape
    if axis_end <= axis_start or normal_end <= normal_start:
        return {"accepted": False, "coverage": 0.0, "span_ratio": 0.0, "max_gap_px": 0}
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_start, axis_end, normal_end], w, h)
    else:
        box = clip_box([normal_start, axis_start, normal_end, axis_end], w, h)
    if box is None:
        return {"accepted": False, "coverage": 0.0, "span_ratio": 0.0, "max_gap_px": 0}
    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2]
    projection = local.sum(axis=0) > 0 if orientation == "horizontal" else local.sum(axis=1) > 0
    runs = contiguous_true_runs(projection)
    gaps = contiguous_true_runs(~projection)
    inner_gaps = [(int(a), int(b)) for a, b in gaps if a > 0 and b < projection.size]
    active = int(np.count_nonzero(projection))
    coverage = float(active / max(1, projection.size))
    if runs:
        active_start, active_end = int(runs[0][0]), int(runs[-1][1])
        span_ratio = float((active_end - active_start) / max(1, projection.size))
    else:
        active_start, active_end = 0, 0
        span_ratio = 0.0
    inner_gap_lengths = [int(b - a) for a, b in inner_gaps]
    max_gap = max(inner_gap_lengths, default=0)
    dashed_effective_coverage = dashed_reconstructed_coverage(
        [int(b - a) for a, b in runs],
        inner_gap_lengths,
        int(projection.size),
        float(cfg.dashed_gap_reconstruction_tolerance),
    )
    dash_gap_count = sum(1 for value in inner_gap_lengths if int(value) > int(cfg.solid_edge_max_gap_px))
    min_dash_gaps = int(cfg.dashed_edge_min_gaps)
    if int(projection.size) >= int(cfg.min_side_length_px) * 2:
        min_dash_gaps = max(2, min_dash_gaps)
    endpoint_gap_px = int(active_start + max(0, projection.size - active_end))
    endpoint_slack_px = max(int(cfg.edge_endpoint_tolerance_px), int(cfg.corner_tolerance_px) * 2)
    endpoint_trimmed_solid_like = (
        coverage >= float(cfg.solid_edge_endpoint_trim_min_coverage)
        and span_ratio >= max(0.0, float(cfg.refined_edge_min_span_ratio) - 0.03)
        and endpoint_gap_px <= int(endpoint_slack_px)
    )
    solid_like = (
        (coverage >= float(cfg.solid_edge_min_coverage) or endpoint_trimmed_solid_like)
        and max_gap <= int(cfg.solid_edge_max_gap_px)
    )
    dashed_like = (
        dashed_effective_coverage >= float(cfg.dashed_edge_min_fill_coverage)
        and span_ratio >= float(cfg.refined_edge_min_span_ratio)
        and max_gap <= int(cfg.max_edge_gap_px)
        and dash_gap_count >= min_dash_gaps
    )
    normal_profiles: list[dict[str, Any]] = []
    normal_count = local.shape[0] if orientation == "horizontal" else local.shape[1]
    for normal_index in range(int(normal_count)):
        normal_projection = local[normal_index, :] if orientation == "horizontal" else local[:, normal_index]
        normal_runs = contiguous_true_runs(normal_projection)
        normal_gaps = contiguous_true_runs(~normal_projection)
        normal_inner_gaps = [
            (int(a), int(b))
            for a, b in normal_gaps
            if a > 0 and b < normal_projection.size
        ]
        normal_active = int(np.count_nonzero(normal_projection))
        if normal_runs:
            normal_active_start, normal_active_end = int(normal_runs[0][0]), int(normal_runs[-1][1])
            normal_span_ratio = float((normal_active_end - normal_active_start) / max(1, normal_projection.size))
        else:
            normal_active_start, normal_active_end = 0, 0
            normal_span_ratio = 0.0
        normal_inner_gap_lengths = [int(b - a) for a, b in normal_inner_gaps]
        normal_max_gap = max(normal_inner_gap_lengths, default=0)
        normal_gap_count = sum(
            1 for value in normal_inner_gap_lengths if int(value) > int(cfg.solid_edge_max_gap_px)
        )
        normal_endpoint_gap_px = int(
            normal_active_start + max(0, normal_projection.size - normal_active_end)
        )
        normal_endpoint_trimmed_solid_like = (
            float(normal_active / max(1, normal_projection.size)) >= float(cfg.solid_edge_endpoint_trim_min_coverage)
            and normal_span_ratio >= max(0.0, float(cfg.refined_edge_min_span_ratio) - 0.03)
            and normal_endpoint_gap_px <= int(endpoint_slack_px)
        )
        normal_solid_like = (
            (
                float(normal_active / max(1, normal_projection.size)) >= float(cfg.solid_edge_min_coverage)
                or normal_endpoint_trimmed_solid_like
            )
            and normal_max_gap <= int(cfg.solid_edge_max_gap_px)
        )
        normal_dashed_like = (
            float(normal_active / max(1, normal_projection.size)) >= float(cfg.min_edge_coverage) * 0.75
            and normal_span_ratio >= float(cfg.refined_edge_min_span_ratio)
            and normal_max_gap <= int(cfg.max_edge_gap_px)
            and normal_gap_count >= min_dash_gaps
        )
        normal_profiles.append(
            {
                "index": int(normal_index),
                "accepted": bool(normal_solid_like or normal_dashed_like),
                "style_hint": "solid" if normal_solid_like else "dashed" if normal_dashed_like else "unknown",
                "coverage": round(float(normal_active / max(1, normal_projection.size)), 4),
                "span_ratio": round(float(normal_span_ratio), 4),
                "max_gap_px": int(normal_max_gap),
                "endpoint_gap_px": int(normal_endpoint_gap_px),
                "active_axis_start": int(axis_start + normal_active_start),
                "active_axis_end": int(axis_start + normal_active_end),
            }
        )
    accepted_normal_profiles = [item for item in normal_profiles if bool(item["accepted"])]
    normal_valid_ratio = float(len(accepted_normal_profiles) / max(1, len(normal_profiles)))
    min_normal_span_ratio = min(
        (float(item["span_ratio"]) for item in accepted_normal_profiles),
        default=0.0,
    )
    invalid_normal_count = int(len(normal_profiles) - len(accepted_normal_profiles))
    # A real line that's 1px jagged/antialiased across its height can shift which exact
    # column carries the ink at different points, so no single column independently
    # covers the whole span even though the combined band does perfectly. Don't reject
    # a near-perfect combined match just because the per-column sub-check is strict.
    combined_strong = coverage >= 0.95 and span_ratio >= 0.95 and max_gap <= int(cfg.solid_edge_max_gap_px)
    normals_valid_enough = bool(normal_profiles) and (normal_valid_ratio >= 0.75 or combined_strong)
    return {
        "accepted": bool((solid_like or dashed_like) and normals_valid_enough),
        "style_hint": "solid" if solid_like else "dashed" if dashed_like else "unknown",
        "coverage": round(float(coverage), 4),
        "dashed_effective_coverage": round(float(dashed_effective_coverage), 4),
        "span_ratio": round(float(span_ratio), 4),
        "max_gap_px": int(max_gap),
        "inner_gap_lengths_px": inner_gap_lengths,
        "ink_run_lengths_px": [int(b - a) for a, b in runs],
        "active_pixels": int(active),
        "active_axis_start": int(axis_start + active_start),
        "active_axis_end": int(axis_start + active_end),
        "endpoint_gap_px": int(endpoint_gap_px),
        "normal_valid_ratio": round(float(normal_valid_ratio), 4),
        "min_normal_span_ratio": round(float(min_normal_span_ratio), 4),
        "invalid_normal_count": int(invalid_normal_count),
        "normal_profiles": normal_profiles,
        "bbox": [int(v) for v in box],
    }


def refine_side_edge(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_guess: int,
    cfg: RectConfig,
    expected_style: str | None = None,
) -> dict[str, Any] | None:
    search = max(0, int(cfg.edge_refine_search_px))
    max_width = max(1, int(cfg.max_side_thickness_px))

    def layer_matches_edge(layer: dict[str, Any], reference: dict[str, Any]) -> bool:
        if not bool(layer.get("accepted", False)):
            return False
        layer_style = str(layer.get("style_hint", "unknown"))
        ref_style = str(reference.get("style_hint", "unknown"))
        if expected_style in {"solid", "dashed"} and layer_style != expected_style:
            return False
        if expected_style not in {"solid", "dashed"} and ref_style in {"solid", "dashed"} and layer_style != ref_style:
            return False
        ref_coverage = float(reference.get("coverage", 0.0))
        layer_coverage = float(layer.get("coverage", 0.0))
        if layer_coverage < max(float(cfg.min_edge_coverage) * 0.75, ref_coverage * 0.68):
            return False
        if float(layer.get("span_ratio", 0.0)) < max(
            float(cfg.refined_edge_min_span_ratio),
            float(reference.get("span_ratio", 0.0)) - 0.08,
        ):
            return False
        if int(layer.get("max_gap_px", 0)) > max(
            int(cfg.max_edge_gap_px),
            int(reference.get("max_gap_px", 0)) + 2,
        ):
            return False
        if int(layer.get("endpoint_gap_px", 0)) > max(
            int(cfg.edge_endpoint_tolerance_px),
            int(reference.get("endpoint_gap_px", 0)) + int(cfg.corner_tolerance_px),
        ):
            return False
        return True

    def expand_to_matching_width(candidate: dict[str, Any]) -> dict[str, Any]:
        normal_start = int(candidate["normal_start"])
        normal_end = int(candidate["normal_end"])
        reference = edge_band_projection_metrics(
            binary,
            orientation,
            int(axis_start),
            int(axis_end),
            normal_start,
            normal_end,
            cfg,
        )
        if not bool(reference.get("accepted", False)):
            return candidate
        while normal_end - normal_start < max_width:
            before = normal_start - 1
            layer = edge_band_projection_metrics(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                before,
                before + 1,
                cfg,
            )
            if not layer_matches_edge(layer, reference):
                break
            normal_start = before
            reference = edge_band_projection_metrics(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                normal_start,
                normal_end,
                cfg,
            )
        while normal_end - normal_start < max_width:
            after = normal_end
            layer = edge_band_projection_metrics(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                after,
                after + 1,
                cfg,
            )
            if not layer_matches_edge(layer, reference):
                break
            normal_end = after + 1
            reference = edge_band_projection_metrics(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                normal_start,
                normal_end,
                cfg,
            )
        if normal_start == int(candidate["normal_start"]) and normal_end == int(candidate["normal_end"]):
            return candidate
        expanded = edge_band_projection_metrics(
            binary,
            orientation,
            int(axis_start),
            int(axis_end),
            normal_start,
            normal_end,
            cfg,
        )
        if not bool(expanded.get("accepted", False)):
            return candidate
        normal_center = 0.5 * float(normal_start + normal_end)
        return {
            **candidate,
            **expanded,
            "orientation": str(orientation),
            "axis_start": int(axis_start),
            "axis_end": int(axis_end),
            "normal_start": int(normal_start),
            "normal_end": int(normal_end),
            "normal_center": round(float(normal_center), 4),
            "width_px": int(normal_end - normal_start),
            "normal_distance_px": round(abs(normal_center - float(normal_guess)), 4),
            "score": round(float(candidate.get("score", 0.0)), 6),
            "expanded_from_centerline": True,
        }

    best: dict[str, Any] | None = None
    for width in range(1, max_width + 1):
        for normal_start in range(int(normal_guess) - search, int(normal_guess) + search + 1):
            normal_end = int(normal_start) + int(width)
            metrics = edge_band_projection_metrics(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                int(normal_start),
                int(normal_end),
                cfg,
            )
            if not bool(metrics.get("accepted", False)):
                continue
            normal_center = 0.5 * float(normal_start + normal_end)
            distance = abs(normal_center - float(normal_guess))
            style_hint = str(metrics.get("style_hint", "unknown"))
            style_bonus = 0.03 if style_hint == "solid" else 0.0
            if expected_style in {"solid", "dashed"}:
                style_bonus = 0.08 if style_hint == expected_style else -0.35
            score = (
                float(metrics.get("span_ratio", 0.0))
                + float(metrics.get("coverage", 0.0))
                + 0.18 * float(metrics.get("normal_valid_ratio", 0.0))
                + style_bonus
                - 0.005 * float(metrics.get("max_gap_px", 0))
                - 0.035 * float(metrics.get("invalid_normal_count", 0))
                - 0.012 * float(max(0, int(width) - 1))
                - 0.002 * float(distance)
            )
            candidate = {
                **metrics,
                "orientation": str(orientation),
                "axis_start": int(axis_start),
                "axis_end": int(axis_end),
                "normal_start": int(normal_start),
                "normal_end": int(normal_end),
                "normal_center": round(float(normal_center), 4),
                "width_px": int(width),
                "normal_distance_px": round(float(distance), 4),
                "score": round(float(score), 6),
            }
            if best is None or (
                float(candidate["score"]),
                -int(candidate["width_px"]),
                -float(candidate["normal_distance_px"]),
            ) > (
                float(best["score"]),
                -int(best["width_px"]),
                -float(best["normal_distance_px"]),
            ):
                best = candidate
    if best is None:
        return None
    return expand_to_matching_width(best)


def refined_edge(edge: dict[str, Any], refinement: dict[str, Any]) -> dict[str, Any]:
    orientation = str(edge["orientation"])
    axis_start = int(refinement["axis_start"])
    axis_end = int(refinement["axis_end"])
    normal_start = int(refinement["normal_start"])
    normal_end = int(refinement["normal_end"])
    if orientation == "horizontal":
        bbox = [axis_start, normal_start, axis_end, normal_end]
        length = int(axis_end - axis_start)
    else:
        bbox = [normal_start, axis_start, normal_end, axis_end]
        length = int(axis_end - axis_start)
    out = {**edge}
    out["source_bbox"] = [int(v) for v in edge["bbox"]]
    out["bbox"] = [int(v) for v in bbox]
    out["length_px"] = int(length)
    out["thickness_px"] = int(normal_end - normal_start)
    out["center"] = box_center(bbox)
    out["refinement"] = {
        key: value
        for key, value in refinement.items()
        if key not in {"bbox"}
    }
    return out


def synthetic_edge(
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    thickness_px: int = 2,
) -> dict[str, Any]:
    half = max(1, int(thickness_px) // 2)
    if orientation == "horizontal":
        bbox = [int(axis_start), int(normal_center) - half, int(axis_end), int(normal_center) + half]
        length = int(axis_end - axis_start)
        thickness = int(bbox[3] - bbox[1])
    else:
        bbox = [int(normal_center) - half, int(axis_start), int(normal_center) + half, int(axis_end)]
        length = int(axis_end - axis_start)
        thickness = int(bbox[2] - bbox[0])
    return {
        "bbox": [int(v) for v in bbox],
        "orientation": str(orientation),
        "length_px": int(length),
        "thickness_px": int(thickness),
        "center": box_center(bbox),
        "area_px": int(max(1, length) * max(1, thickness)),
        "synthetic": True,
    }


def refine_frame_edges(
    binary: np.ndarray,
    top: dict[str, Any],
    bottom: dict[str, Any],
    left_edge: dict[str, Any],
    right_edge: dict[str, Any],
    x1: int,
    top_y: int,
    x2: int,
    bottom_y: int,
    cfg: RectConfig,
    expected_styles: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    expected_styles = expected_styles or {}
    top_ref = refine_side_edge(binary, "horizontal", x1, x2, top_y, cfg, expected_styles.get("top"))
    bottom_ref = refine_side_edge(binary, "horizontal", x1, x2, bottom_y, cfg, expected_styles.get("bottom"))
    if top_ref is None or bottom_ref is None:
        return None
    top_y = int(round(float(top_ref["normal_center"])))
    bottom_y = int(round(float(bottom_ref["normal_center"])))
    if bottom_y <= top_y:
        return None

    left_ref = refine_side_edge(binary, "vertical", top_y, bottom_y, x1, cfg, expected_styles.get("left"))
    right_ref = refine_side_edge(binary, "vertical", top_y, bottom_y, x2, cfg, expected_styles.get("right"))
    if left_ref is None or right_ref is None:
        return None
    x1 = int(round(float(left_ref["normal_center"])))
    x2 = int(round(float(right_ref["normal_center"])))
    if x2 <= x1:
        return None

    top_ref = refine_side_edge(binary, "horizontal", x1, x2, top_y, cfg, expected_styles.get("top"))
    bottom_ref = refine_side_edge(binary, "horizontal", x1, x2, bottom_y, cfg, expected_styles.get("bottom"))
    if top_ref is None or bottom_ref is None:
        return None
    top_y = int(round(float(top_ref["normal_center"])))
    bottom_y = int(round(float(bottom_ref["normal_center"])))
    if bottom_y <= top_y:
        return None

    left_ref = refine_side_edge(binary, "vertical", top_y, bottom_y, x1, cfg, expected_styles.get("left"))
    right_ref = refine_side_edge(binary, "vertical", top_y, bottom_y, x2, cfg, expected_styles.get("right"))
    if left_ref is None or right_ref is None:
        return None
    x1 = int(round(float(left_ref["normal_center"])))
    x2 = int(round(float(right_ref["normal_center"])))
    if x2 <= x1:
        return None

    return {
        "x1": int(x1),
        "x2": int(x2),
        "top_y": int(top_y),
        "bottom_y": int(bottom_y),
        "top": refined_edge(top, top_ref),
        "bottom": refined_edge(bottom, bottom_ref),
        "left": refined_edge(left_edge, left_ref),
        "right": refined_edge(right_edge, right_ref),
        "metrics": {
            "top": top_ref,
            "bottom": bottom_ref,
            "left": left_ref,
            "right": right_ref,
        },
    }


def direction_coverage(
    binary: np.ndarray,
    x: int,
    y: int,
    direction: str,
    cfg: RectConfig,
    probe_px: int | None = None,
) -> dict[str, Any]:
    h, w = binary.shape
    probe = max(4, int(cfg.corner_probe_px if probe_px is None else probe_px))
    band = max(1, int(cfg.edge_band_px))
    if direction == "right":
        box = clip_box([x + 1, y - band, x + probe + 1, y + band + 1], w, h)
        axis = 0
    elif direction == "left":
        box = clip_box([x - probe, y - band, x, y + band + 1], w, h)
        axis = 0
    elif direction == "down":
        box = clip_box([x - band, y + 1, x + band + 1, y + probe + 1], w, h)
        axis = 1
    elif direction == "up":
        box = clip_box([x - band, y - probe, x + band + 1, y], w, h)
        axis = 1
    else:
        raise ValueError(f"Unknown direction: {direction}")
    if box is None:
        return {"coverage": 0.0, "active_pixels": 0, "bbox": None}
    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2]
    projection = local.sum(axis=axis) > 0
    active = int(np.count_nonzero(projection))
    return {
        "coverage": round(float(active / max(1, projection.size)), 4),
        "active_pixels": int(active),
        "bbox": [int(v) for v in box],
    }


def corner_connected_component(
    binary: np.ndarray,
    x: int,
    y: int,
    expected_directions: set[str],
    cfg: RectConfig,
) -> dict[str, Any]:
    h, w = binary.shape
    probe = max(4, int(cfg.corner_probe_px))
    box = clip_box([x - probe, y - probe, x + probe + 1, y + probe + 1], w, h)
    if box is None:
        return {"component": None, "bbox": None, "origin": [0, 0], "seed_labels": []}
    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2].astype(np.uint8)
    count, labels = cv2.connectedComponents(local, 8)
    cx = int(x - x1)
    cy = int(y - y1)
    seed_probe = max(4, min(int(cfg.corner_tolerance_px), int(cfg.corner_probe_px)))
    band = max(1, int(cfg.edge_band_px))
    seed_labels_set: set[int] = set()

    def add_seed(direction: str) -> None:
        if direction == "right":
            raw_box = [cx + 1, cy - band, cx + seed_probe + 1, cy + band + 1]
        elif direction == "left":
            raw_box = [cx - seed_probe, cy - band, cx, cy + band + 1]
        elif direction == "down":
            raw_box = [cx - band, cy + 1, cx + band + 1, cy + seed_probe + 1]
        elif direction == "up":
            raw_box = [cx - band, cy - seed_probe, cx + band + 1, cy]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        seed_box = clip_box(raw_box, local.shape[1], local.shape[0])
        if seed_box is None:
            return
        sx1, sy1, sx2, sy2 = seed_box
        seed = labels[sy1:sy2, sx1:sx2]
        seed_labels_set.update(int(label) for label in np.unique(seed) if int(label) != 0)

    for direction in expected_directions:
        add_seed(direction)

    seed_labels = sorted(seed_labels_set)
    if not seed_labels:
        return {
            "component": np.zeros_like(local, dtype=bool),
            "bbox": [int(v) for v in box],
            "origin": [int(cx), int(cy)],
            "seed_labels": [],
        }
    component = np.isin(labels, np.asarray(seed_labels, dtype=labels.dtype))
    return {
        "component": component,
        "bbox": [int(v) for v in box],
        "origin": [int(cx), int(cy)],
        "seed_labels": seed_labels,
    }


def connected_direction_coverage(
    component: np.ndarray,
    origin: Sequence[int],
    direction: str,
    cfg: RectConfig,
    probe_px: int | None = None,
    band_px: int | None = None,
) -> dict[str, Any]:
    if component.size == 0:
        return {"coverage": 0.0, "active_pixels": 0, "bbox": None}
    h, w = component.shape
    x, y = [int(v) for v in origin]
    probe = max(4, int(cfg.corner_probe_px if probe_px is None else probe_px))
    band = max(1, int(cfg.edge_band_px if band_px is None else band_px))
    if direction == "right":
        box = clip_box([x + 1, y - band, x + probe + 1, y + band + 1], w, h)
        axis = 0
    elif direction == "left":
        box = clip_box([x - probe, y - band, x, y + band + 1], w, h)
        axis = 0
    elif direction == "down":
        box = clip_box([x - band, y + 1, x + band + 1, y + probe + 1], w, h)
        axis = 1
    elif direction == "up":
        box = clip_box([x - band, y - probe, x + band + 1, y], w, h)
        axis = 1
    else:
        raise ValueError(f"Unknown direction: {direction}")
    if box is None:
        return {"coverage": 0.0, "active_pixels": 0, "bbox": None}
    x1, y1, x2, y2 = box
    local = component[y1:y2, x1:x2]
    projection = local.sum(axis=axis) > 0
    active = int(np.count_nonzero(projection))
    return {
        "coverage": round(float(active / max(1, projection.size)), 4),
        "active_pixels": int(active),
        "bbox": [int(v) for v in box],
    }


def corner_l_shape_metrics(binary: np.ndarray, x: int, y: int, corner: str, cfg: RectConfig) -> dict[str, Any]:
    expected_by_corner = {
        "top_left": {"right", "down"},
        "top_right": {"left", "down"},
        "bottom_left": {"right", "up"},
        "bottom_right": {"left", "up"},
    }
    expected = expected_by_corner[corner]
    component_data = corner_connected_component(binary, x, y, expected, cfg)
    component = component_data["component"]
    origin = component_data["origin"]
    expected_directions = {
        name: connected_direction_coverage(component, origin, name, cfg)
        for name in ("left", "right", "up", "down")
    }
    forbidden_probe = max(4, min(int(cfg.corner_probe_px), int(cfg.corner_tolerance_px)))
    forbidden_band = max(1, min(2, int(cfg.edge_band_px)))
    forbidden_directions = {
        name: connected_direction_coverage(
            component,
            origin,
            name,
            cfg,
            probe_px=forbidden_probe,
            band_px=forbidden_band,
        )
        for name in ("left", "right", "up", "down")
    }
    missing = [
        name
        for name in expected
        if float(expected_directions[name]["coverage"]) < float(cfg.corner_required_coverage)
    ]
    forbidden = [
        name
        for name in forbidden_directions
        if name not in expected
        and float(forbidden_directions[name]["coverage"]) > float(cfg.corner_forbidden_coverage)
        and int(forbidden_directions[name]["active_pixels"]) >= int(cfg.corner_forbidden_min_active_px)
    ]
    if len(forbidden) >= 2:
        reason = "corner_has_cross_junction"
    elif forbidden:
        reason = "corner_has_t_junction"
    elif missing:
        reason = "corner_missing_required_leg"
    else:
        reason = None
    return {
        "corner": corner,
        "point": [int(x), int(y)],
        "component_bbox": component_data["bbox"],
        "component_seed_labels": component_data["seed_labels"],
        "expected_directions": sorted(expected),
        "direction_metrics": expected_directions,
        "forbidden_direction_metrics": forbidden_directions,
        "forbidden_probe_px": int(forbidden_probe),
        "forbidden_band_px": int(forbidden_band),
        "missing_expected_directions": missing,
        "forbidden_directions": forbidden,
        "accepted": reason is None,
        "reason": reason,
    }


def validate_l_corners(binary: np.ndarray, bbox: Sequence[int], cfg: RectConfig) -> tuple[bool, list[dict[str, Any]]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    binary = frame_axis_corridor_mask(binary, bbox, cfg)
    corners = [
        corner_l_shape_metrics(binary, x1, y1, "top_left", cfg),
        corner_l_shape_metrics(binary, x2, y1, "top_right", cfg),
        corner_l_shape_metrics(binary, x1, y2, "bottom_left", cfg),
        corner_l_shape_metrics(binary, x2, y2, "bottom_right", cfg),
    ]
    return all(bool(item["accepted"]) for item in corners), corners


def validate_dashed_corners(binary: np.ndarray, bbox: Sequence[int], cfg: RectConfig) -> tuple[bool, list[dict[str, Any]]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = int(x2 - x1)
    height = int(y2 - y1)
    allow_single_t_junction = width <= int(cfg.min_side_length_px) * 5 and height <= int(cfg.min_side_length_px) * 4
    axis_binary = frame_axis_corridor_mask(binary, bbox, cfg)
    expected_by_corner = {
        "top_left": (x1, y1, {"right", "down"}),
        "top_right": (x2, y1, {"left", "down"}),
        "bottom_left": (x1, y2, {"right", "up"}),
        "bottom_right": (x2, y2, {"left", "up"}),
    }
    corners: list[dict[str, Any]] = []
    min_coverage = max(0.1, float(cfg.corner_required_coverage) * 0.2)
    for name, (cx, cy, expected) in expected_by_corner.items():
        directions = {
            direction: direction_coverage(axis_binary, cx, cy, direction, cfg)
            for direction in ("left", "right", "up", "down")
        }
        missing = [
            direction
            for direction in expected
            if float(directions[direction]["coverage"]) < min_coverage
        ]
        forbidden = [
            direction
            for direction in directions
            if direction not in expected
            and float(directions[direction]["coverage"]) > float(cfg.dashed_corner_forbidden_coverage)
            and int(directions[direction]["active_pixels"]) >= int(cfg.corner_forbidden_min_active_px)
        ]
        if len(forbidden) >= 2:
            reason = "dashed_corner_has_cross_junction"
        elif forbidden and not allow_single_t_junction:
            reason = "dashed_corner_has_t_junction"
        elif missing:
            reason = "dashed_corner_missing_expected_dash"
        else:
            reason = None
        corners.append(
            {
                "corner": name,
                "point": [int(cx), int(cy)],
                "expected_directions": sorted(expected),
                "direction_metrics": directions,
                "missing_expected_directions": missing,
                "forbidden_directions": forbidden,
                "accepted": reason is None,
                "reason": reason,
                "dashed_corner_min_coverage": round(float(min_coverage), 4),
            }
        )
    return all(bool(item["accepted"]) for item in corners), corners


def frame_axis_corridor_mask(binary: np.ndarray, bbox: Sequence[int], cfg: RectConfig) -> np.ndarray:
    h, w = binary.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    band = max(1, int(cfg.edge_band_px))
    probe = max(4, int(cfg.corner_probe_px))
    out = np.zeros_like(binary, dtype=bool)
    corridors = [
        [x1 - probe, y1 - band, x2 + probe + 1, y1 + band + 1],
        [x1 - probe, y2 - band, x2 + probe + 1, y2 + band + 1],
        [x1 - band, y1 - probe, x1 + band + 1, y2 + probe + 1],
        [x2 - band, y1 - probe, x2 + band + 1, y2 + probe + 1],
    ]
    for raw_box in corridors:
        box = clip_box(raw_box, w, h)
        if box is None:
            continue
        bx1, by1, bx2, by2 = box
        out[by1:by2, bx1:bx2] |= binary[by1:by2, bx1:bx2]
    return out


def classify_edge_style(metric: dict[str, Any], cfg: RectConfig) -> dict[str, Any]:
    coverage = float(metric.get("coverage", 0.0))
    max_gap = int(metric.get("max_gap_px", 0))
    span_ratio = float(metric.get("span_ratio", 0.0))
    endpoint_gap_px = int(metric.get("endpoint_gap_px", 0))
    gaps = [
        int(gap)
        for gap in metric.get("inner_gap_lengths_px", [])
        if int(gap) > int(cfg.solid_edge_max_gap_px)
    ]
    # coarse-check callers pass a metric without this key; fall back to raw coverage there.
    dashed_effective_coverage = float(metric.get("dashed_effective_coverage", coverage))
    endpoint_slack_px = max(int(cfg.edge_endpoint_tolerance_px), int(cfg.corner_tolerance_px) * 2)
    endpoint_trimmed_solid_like = (
        coverage >= float(cfg.solid_edge_endpoint_trim_min_coverage)
        and span_ratio >= max(0.0, float(cfg.refined_edge_min_span_ratio) - 0.03)
        and endpoint_gap_px <= int(endpoint_slack_px)
    )
    if (
        (coverage >= float(cfg.solid_edge_min_coverage) or endpoint_trimmed_solid_like)
        and max_gap <= int(cfg.solid_edge_max_gap_px)
    ):
        return {
            "style": "solid",
            "gap_lengths_px": gaps,
            "median_gap_px": 0.0,
            "reason": None,
        }
    if (
        len(gaps) >= int(cfg.dashed_edge_min_gaps)
        and max_gap <= int(cfg.max_edge_gap_px)
        and dashed_effective_coverage >= float(cfg.dashed_edge_min_fill_coverage)
        and span_ratio >= float(cfg.refined_edge_min_span_ratio)
    ):
        return {
            "style": "dashed",
            "gap_lengths_px": gaps,
            "median_gap_px": round(float(np.median(gaps)), 4),
            "reason": None,
        }
    return {
        "style": "unknown",
        "gap_lengths_px": gaps,
        "median_gap_px": round(float(np.median(gaps)), 4) if gaps else 0.0,
        "reason": "edge_style_unknown",
    }


def classify_frame_style(edge_metrics: dict[str, dict[str, Any]], cfg: RectConfig) -> dict[str, Any]:
    edges = {name: classify_edge_style(metric, cfg) for name, metric in edge_metrics.items()}
    styles = [str(item["style"]) for item in edges.values()]
    if all(style == "solid" for style in styles):
        return {"accepted": True, "style": "solid", "edges": edges, "reason": None}
    if not all(style == "dashed" for style in styles):
        return {"accepted": False, "style": "mixed", "edges": edges, "reason": "mixed_edge_styles"}
    medians = [float(item["median_gap_px"]) for item in edges.values()]
    if max(medians) - min(medians) > float(cfg.dashed_gap_tolerance_px):
        return {
            "accepted": False,
            "style": "dashed",
            "edges": edges,
            "reason": "dashed_gap_inconsistent",
            "gap_median_range_px": round(float(max(medians) - min(medians)), 4),
        }
    return {
        "accepted": True,
        "style": "dashed",
        "edges": edges,
        "reason": None,
        "gap_median_range_px": round(float(max(medians) - min(medians)), 4),
    }


def local_adjust_edge_metrics(
    binary: np.ndarray,
    bbox: Sequence[int],
    cfg: RectConfig,
) -> dict[str, dict[str, Any]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    band = int(cfg.edge_band_px)
    return {
        "top": edge_band_projection_metrics(
            binary, "horizontal", x1, x2, y1 - band, y1 + band + 1, cfg
        ),
        "bottom": edge_band_projection_metrics(
            binary, "horizontal", x1, x2, y2 - band, y2 + band + 1, cfg
        ),
        "left": edge_band_projection_metrics(
            binary, "vertical", y1, y2, x1 - band, x1 + band + 1, cfg
        ),
        "right": edge_band_projection_metrics(
            binary, "vertical", y1, y2, x2 - band, x2 + band + 1, cfg
        ),
    }


def local_adjust_horizontal_external_score(
    binary: np.ndarray,
    bbox: Sequence[int],
    cfg: RectConfig,
) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, int(x2 - x1))
    probe = max(18, min(40, width // 2))
    values: list[float] = []
    for y in (y1, y2):
        values.append(
            float(axis_projection_coverage(binary, "horizontal", x1 - probe, x1, y, cfg.edge_band_px)["coverage"])
        )
        values.append(
            float(axis_projection_coverage(binary, "horizontal", x2, x2 + probe, y, cfg.edge_band_px)["coverage"])
        )
    return float(sum(values))


def local_adjust_interior_density(binary: np.ndarray, bbox: Sequence[int], pad_px: int = 6) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = max(1, int(pad_px))
    if x2 - x1 <= pad * 2 or y2 - y1 <= pad * 2:
        return 0.0
    interior = binary[y1 + pad : y2 - pad, x1 + pad : x2 - pad]
    return float(interior.mean()) if interior.size else 0.0


def local_adjust_side_style(item: dict[str, Any], cfg: RectConfig) -> str:
    coverage = float(item.get("coverage", 0.0))
    max_gap = int(item.get("max_gap_px", 0))
    gaps = [int(value) for value in item.get("inner_gap_lengths_px", [])]
    if coverage >= float(cfg.solid_edge_min_coverage) and max_gap <= int(cfg.solid_edge_max_gap_px):
        return "solid"
    if coverage >= float(cfg.local_adjust_min_coverage) and max_gap <= int(cfg.max_edge_gap_px) + 4:
        if gaps:
            return "dashed"
        if coverage >= float(cfg.min_edge_coverage) and float(item.get("endpoint_gap_px", 0)) <= max(
            int(cfg.edge_endpoint_tolerance_px),
            int(cfg.corner_tolerance_px) * 2,
        ):
            return "short_dash"
    return "unknown"


def local_adjusted_frame_from_bbox(
    binary: np.ndarray,
    bbox: Sequence[int],
    cfg: RectConfig,
    source: str,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = int(x2 - x1)
    height = int(y2 - y1)
    if width < int(cfg.small_dashed_min_side_length_px) or height < int(cfg.small_dashed_min_side_length_px):
        return None
    if width > int(cfg.local_adjust_max_side_px) or height > int(cfg.local_adjust_max_side_px):
        return None
    if int(width * height) < int(cfg.local_adjust_min_area_px):
        return None
    if int(width * height) > int(cfg.local_adjust_max_area_px):
        return None
    aspect = float(width / max(1, height))
    if aspect < 0.48 or aspect > 5.2:
        return None
    interior_density = local_adjust_interior_density(binary, bbox)
    if interior_density > 0.17 and aspect >= 0.85:
        return None
    if local_adjust_horizontal_external_score(binary, bbox, cfg) > 1.4 and interior_density > 0.02:
        return None

    edge_metrics = local_adjust_edge_metrics(binary, bbox, cfg)
    max_allowed_gap = int(cfg.max_edge_gap_px) + 4
    endpoint_slack = max(int(cfg.edge_endpoint_tolerance_px), int(cfg.corner_tolerance_px) * 2)
    for item in edge_metrics.values():
        if float(item.get("coverage", 0.0)) < float(cfg.local_adjust_min_coverage):
            return None
        if float(item.get("span_ratio", 0.0)) < float(cfg.local_adjust_min_span_ratio):
            return None
        if int(item.get("max_gap_px", 0)) > max_allowed_gap:
            return None
        if int(item.get("max_gap_px", 0)) > 14:
            return None
        if int(item.get("endpoint_gap_px", 0)) > endpoint_slack:
            return None

    frame_style = classify_frame_style(edge_metrics, cfg)
    side_styles = {name: local_adjust_side_style(item, cfg) for name, item in edge_metrics.items()}
    if bool(frame_style.get("accepted", False)):
        line_style = str(frame_style["style"])
    else:
        known_styles = [value for value in side_styles.values() if value != "unknown"]
        if len(known_styles) < 4:
            return None
        dashed_like = sum(1 for value in known_styles if value in {"dashed", "short_dash"})
        solid_like = sum(1 for value in known_styles if value == "solid")
        if dashed_like >= 2:
            line_style = "dashed"
        elif solid_like == 4:
            line_style = "solid"
        else:
            return None
        frame_style = {
            **frame_style,
            "accepted": True,
            "style": line_style,
            "reason": "local_adjusted_mixed_edges_accepted",
            "local_side_styles": side_styles,
        }
    narrow_symbol_frame = (
        line_style == "dashed"
        and aspect < 0.85
        and interior_density <= 0.2
        and height >= 45
        and width >= 45
        and side_styles.get("top") == "dashed"
        and side_styles.get("bottom") == "dashed"
        and side_styles.get("left") == "solid"
        and side_styles.get("right") == "dashed"
        and float(edge_metrics["top"].get("span_ratio", 0.0)) >= 0.9
        and float(edge_metrics["bottom"].get("span_ratio", 0.0)) >= 0.9
    )
    compact_mixed_symbol_frame = (
        width >= 45
        and width <= 90
        and height <= 95
        and aspect <= 1.9
        and interior_density <= 0.42
        and side_styles.get("top") in {"solid", "dashed", "short_dash"}
        and side_styles.get("bottom") in {"solid", "dashed", "short_dash"}
        and side_styles.get("left") in {"dashed", "short_dash"}
        and side_styles.get("right") in {"dashed", "short_dash"}
        and min(float(edge_metrics[name].get("coverage", 0.0)) for name in ("top", "bottom", "left", "right"))
        >= 0.64
        and min(float(edge_metrics[name].get("span_ratio", 0.0)) for name in ("top", "bottom", "left", "right"))
        >= 0.82
        and max(int(edge_metrics[name].get("endpoint_gap_px", 0)) for name in ("top", "bottom", "left", "right"))
        <= max(endpoint_slack, 12)
    )
    if compact_mixed_symbol_frame and line_style == "dashed":
        frame_style = {
            **frame_style,
            "accepted": True,
            "style": "dashed",
            "reason": "local_adjusted_compact_mixed_symbol_accepted",
            "local_side_styles": side_styles,
        }
    if line_style == "dashed" and (
        side_styles.get("top") == "solid" or side_styles.get("bottom") == "solid"
    ) and not compact_mixed_symbol_frame:
        return None
    if (
        line_style == "dashed"
        and aspect < 0.85
        and side_styles.get("left") == "solid"
        and side_styles.get("right") == "dashed"
        and not narrow_symbol_frame
    ):
        return None
    if line_style == "dashed" and side_styles.get("left") == "solid" and side_styles.get("right") == "solid":
        return None
    if (
        line_style == "dashed"
        and aspect < 0.6
        and all(side_styles.get(name) == "dashed" for name in ("top", "bottom", "left", "right"))
        and interior_density > 0.1
        and min(
            float(edge_metrics["top"].get("span_ratio", 0.0)),
            float(edge_metrics["right"].get("span_ratio", 0.0)),
        )
        < 0.95
    ):
        return None
    if (
        line_style == "dashed"
        and side_styles.get("left") == "dashed"
        and side_styles.get("right") == "solid"
        and aspect < 2.0
        and interior_density > 0.105
    ):
        return None

    if line_style == "dashed":
        corners_ok, corner_metrics = validate_dashed_corners(binary, bbox, cfg)
    else:
        corners_ok, corner_metrics = validate_l_corners(binary, bbox, cfg)
    if not corners_ok:
        return None

    def expected_side_style(name: str) -> str | None:
        value = str(side_styles.get(name, line_style))
        if value == "short_dash":
            return "dashed"
        if value in {"solid", "dashed"}:
            return value
        return str(line_style) if line_style in {"solid", "dashed"} else None

    top_ref = refine_side_edge(binary, "horizontal", x1, x2, y1, cfg, expected_side_style("top"))
    bottom_ref = refine_side_edge(binary, "horizontal", x1, x2, y2, cfg, expected_side_style("bottom"))
    left_ref = refine_side_edge(binary, "vertical", y1, y2, x1, cfg, expected_side_style("left"))
    right_ref = refine_side_edge(binary, "vertical", y1, y2, x2, cfg, expected_side_style("right"))
    if top_ref is None or bottom_ref is None or left_ref is None or right_ref is None:
        if not narrow_symbol_frame:
            return None

        def narrow_symbol_ref(
            name: str,
            orientation: str,
            axis_start: int,
            axis_end: int,
            normal_center: int,
        ) -> dict[str, Any]:
            normal_start = int(normal_center) - 1
            normal_end = int(normal_center) + 2
            metrics = edge_band_projection_metrics(
                binary,
                orientation,
                int(axis_start),
                int(axis_end),
                int(normal_start),
                int(normal_end),
                cfg,
            )
            return {
                **metrics,
                "orientation": str(orientation),
                "axis_start": int(axis_start),
                "axis_end": int(axis_end),
                "normal_start": int(normal_start),
                "normal_end": int(normal_end),
                "normal_center": float(normal_center),
                "width_px": int(normal_end - normal_start),
                "normal_distance_px": 0.0,
                "local_narrow_symbol_side": str(name),
            }

        if top_ref is None:
            top_ref = narrow_symbol_ref("top", "horizontal", x1, x2, y1)
        if bottom_ref is None:
            bottom_ref = narrow_symbol_ref("bottom", "horizontal", x1, x2, y2)
        if left_ref is None:
            left_ref = narrow_symbol_ref("left", "vertical", y1, y2, x1)
        if right_ref is None:
            right_ref = narrow_symbol_ref("right", "vertical", y1, y2, x2)
    refined_metrics = {
        "top": top_ref,
        "bottom": bottom_ref,
        "left": left_ref,
        "right": right_ref,
    }

    refined_x1 = int(round(float(left_ref["normal_center"])))
    refined_x2 = int(round(float(right_ref["normal_center"])))
    refined_y1 = int(round(float(top_ref["normal_center"])))
    refined_y2 = int(round(float(bottom_ref["normal_center"])))
    if abs(refined_x1 - x1) > int(cfg.corner_tolerance_px):
        return None
    if abs(refined_x2 - x2) > int(cfg.corner_tolerance_px):
        return None
    if abs(refined_y1 - y1) > int(cfg.corner_tolerance_px):
        return None
    if abs(refined_y2 - y2) > int(cfg.corner_tolerance_px):
        return None

    top_ref = {
        **top_ref,
        "orientation": "horizontal",
        "axis_start": int(x1),
        "axis_end": int(x2),
    }
    bottom_ref = {
        **bottom_ref,
        "orientation": "horizontal",
        "axis_start": int(x1),
        "axis_end": int(x2),
    }
    left_ref = {
        **left_ref,
        "orientation": "vertical",
        "axis_start": int(y1),
        "axis_end": int(y2),
    }
    right_ref = {
        **right_ref,
        "orientation": "vertical",
        "axis_start": int(y1),
        "axis_end": int(y2),
    }

    top = refined_edge(synthetic_edge("horizontal", x1, x2, y1), top_ref)
    bottom = refined_edge(synthetic_edge("horizontal", x1, x2, y2), bottom_ref)
    left_edge = refined_edge(synthetic_edge("vertical", y1, y2, x1), left_ref)
    right_edge = refined_edge(synthetic_edge("vertical", y1, y2, x2), right_ref)
    edge_metrics = refined_metrics
    coverages = [float(item.get("coverage", 0.0)) for item in edge_metrics.values()]
    gaps = [int(item.get("max_gap_px", 0)) for item in edge_metrics.values()]
    line_thickness = int(
        max(
            1,
            min(
                int(cfg.max_side_thickness_px),
                int(
                    round(
                        float(
                            np.median(
                                [
                                    int(top_ref.get("width_px", 1)),
                                    int(bottom_ref.get("width_px", 1)),
                                    int(left_ref.get("width_px", 1)),
                                    int(right_ref.get("width_px", 1)),
                                ]
                            )
                        )
                    )
                ),
            ),
        )
    )
    confidence_value = round(float(min(0.99, np.mean(coverages))), 4)
    score_value = round(float(np.mean(coverages) - 0.01 * max(gaps) + 0.03), 4)
    if narrow_symbol_frame:
        confidence_value = round(float(min(0.85, 0.52 + 0.002 * float(height))), 4)
        score_value = round(float(0.4 + 0.003 * float(height) - 0.001 * float(width)), 4)
    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "type": "closed_rectangular_border_frame",
        "confidence": confidence_value,
        "score": score_value,
        "line_thickness_px": int(line_thickness),
        "geometry": {
            "width_px": int(width),
            "height_px": int(height),
            "aspect_ratio": round(float(width / max(1, height)), 4),
            "line_style": str(line_style),
            "dashed_frame_like": str(line_style) == "dashed",
            "seed_axis": "local_adjusted",
            "small_dashed_candidate": True,
            "min_side_length_used_px": int(cfg.small_dashed_min_side_length_px),
            "local_adjust_source": str(source),
            "local_narrow_symbol_frame": bool(narrow_symbol_frame),
            "corner_mask_extension_px": int(cfg.corner_mask_extension_px),
            "candidate_span_ratios": {
                name: round(float(item.get("span_ratio", 0.0)), 4)
                for name, item in edge_metrics.items()
            },
            "edge_refinement": {
                name: {
                    "normal_center": float(
                        {"top": y1, "bottom": y2, "left": x1, "right": x2}[name]
                    ),
                    "width_px": int(line_thickness),
                    "span_ratio": round(float(item.get("span_ratio", 0.0)), 4),
                    "coverage": round(float(item.get("coverage", 0.0)), 4),
                    "endpoint_gap_px": int(item.get("endpoint_gap_px", 0)),
                    "normal_valid_ratio": round(float(item.get("normal_valid_ratio", 0.0)), 4),
                    "min_normal_span_ratio": round(float(item.get("min_normal_span_ratio", 0.0)), 4),
                    "invalid_normal_count": int(item.get("invalid_normal_count", 0)),
                    "normal_distance_px": 0.0,
                    "style_hint": str(item.get("style_hint", "unknown")),
                    "local_side_style": str(side_styles.get(name, "unknown")),
                }
                for name, item in edge_metrics.items()
            },
        },
        "edges": {
            "top": top,
            "bottom": bottom,
            "left": left_edge,
            "right": right_edge,
        },
        "edge_metrics": edge_metrics,
        "frame_style": frame_style,
        "corner_metrics": corner_metrics,
    }


def recovery_projection_metrics(
    binary: np.ndarray,
    orientation: str,
    axis_start: int,
    axis_end: int,
    normal_center: int,
    cfg: RectConfig,
) -> dict[str, Any]:
    h, w = binary.shape
    band = max(int(cfg.edge_band_px), 4)
    if axis_end <= axis_start:
        return {"coverage": 0.0, "span_ratio": 0.0, "max_gap_px": 0, "style_hint": "unknown"}
    if orientation == "horizontal":
        box = clip_box([axis_start, normal_center - band, axis_end, normal_center + band + 1], w, h)
    else:
        box = clip_box([normal_center - band, axis_start, normal_center + band + 1, axis_end], w, h)
    if box is None:
        return {"coverage": 0.0, "span_ratio": 0.0, "max_gap_px": 0, "style_hint": "unknown"}
    x1, y1, x2, y2 = box
    local = binary[y1:y2, x1:x2]
    projection = local.sum(axis=0) > 0 if orientation == "horizontal" else local.sum(axis=1) > 0
    runs = contiguous_true_runs(projection)
    gaps = contiguous_true_runs(~projection)
    inner_gaps = [(int(a), int(b)) for a, b in gaps if a > 0 and b < projection.size]
    inner_gap_lengths = [int(b - a) for a, b in inner_gaps]
    active = int(np.count_nonzero(projection))
    coverage = float(active / max(1, projection.size))
    if runs:
        active_start, active_end = int(runs[0][0]), int(runs[-1][1])
        span_ratio = float((active_end - active_start) / max(1, projection.size))
    else:
        active_start, active_end = 0, 0
        span_ratio = 0.0
    max_gap = max(inner_gap_lengths, default=0)
    endpoint_gap_px = int(active_start + max(0, projection.size - active_end))
    dash_gaps = [value for value in inner_gap_lengths if int(value) > int(cfg.solid_edge_max_gap_px)]
    solid_like = (
        coverage >= float(cfg.solid_edge_min_coverage)
        and max_gap <= int(cfg.solid_edge_max_gap_px)
        and endpoint_gap_px <= int(cfg.medium_recovery_endpoint_slack_px)
    )
    dashed_like = (
        coverage >= float(cfg.medium_recovery_min_coverage)
        and span_ratio >= float(cfg.medium_recovery_min_span_ratio)
        and max_gap <= int(cfg.medium_recovery_max_gap_px)
        and bool(dash_gaps)
    )
    short_dash_like = (
        coverage >= float(cfg.medium_recovery_min_coverage)
        and span_ratio >= float(cfg.medium_recovery_min_span_ratio)
        and max_gap <= int(cfg.medium_recovery_max_gap_px)
        and endpoint_gap_px <= int(cfg.medium_recovery_endpoint_slack_px)
    )
    style_hint = "solid" if solid_like else "dashed" if dashed_like or short_dash_like else "unknown"
    return {
        "accepted": bool(solid_like or dashed_like or short_dash_like),
        "style_hint": str(style_hint),
        "coverage": round(float(coverage), 4),
        "span_ratio": round(float(span_ratio), 4),
        "max_gap_px": int(max_gap),
        "inner_gap_lengths_px": inner_gap_lengths,
        "ink_run_lengths_px": [int(b - a) for a, b in runs],
        "active_pixels": int(active),
        "active_axis_start": int(axis_start + active_start),
        "active_axis_end": int(axis_start + active_end),
        "endpoint_gap_px": int(endpoint_gap_px),
        "normal_valid_ratio": 1.0,
        "min_normal_span_ratio": round(float(span_ratio), 4),
        "invalid_normal_count": 0,
        "bbox": [int(v) for v in box],
    }


def medium_large_frame_from_bbox(
    binary: np.ndarray,
    bbox: Sequence[int],
    cfg: RectConfig,
    source: str,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = int(x2 - x1)
    height = int(y2 - y1)
    area = int(width * height)
    if width < int(cfg.medium_recovery_min_side_px) or height < int(cfg.medium_recovery_min_side_px):
        return None
    if width > int(cfg.medium_recovery_max_side_px) or height > int(cfg.medium_recovery_max_side_px):
        return None
    if area < int(cfg.medium_recovery_min_area_px) or area > int(cfg.medium_recovery_max_area_px):
        return None
    aspect = float(width / max(1, height))
    if aspect < 0.32 or aspect > 8.0:
        return None

    edge_metrics = {
        "top": recovery_projection_metrics(binary, "horizontal", x1, x2, y1, cfg),
        "bottom": recovery_projection_metrics(binary, "horizontal", x1, x2, y2, cfg),
        "left": recovery_projection_metrics(binary, "vertical", y1, y2, x1, cfg),
        "right": recovery_projection_metrics(binary, "vertical", y1, y2, x2, cfg),
    }
    if any(not bool(item.get("accepted", False)) for item in edge_metrics.values()):
        return None
    if any(float(item.get("coverage", 0.0)) < float(cfg.medium_recovery_min_coverage) for item in edge_metrics.values()):
        return None
    if any(float(item.get("span_ratio", 0.0)) < float(cfg.medium_recovery_min_span_ratio) for item in edge_metrics.values()):
        return None
    if any(int(item.get("max_gap_px", 0)) > int(cfg.medium_recovery_max_gap_px) for item in edge_metrics.values()):
        return None
    if any(int(item.get("endpoint_gap_px", 0)) > int(cfg.medium_recovery_endpoint_slack_px) for item in edge_metrics.values()):
        return None

    side_styles = {name: str(item.get("style_hint", "unknown")) for name, item in edge_metrics.items()}
    solid_count = sum(1 for value in side_styles.values() if value == "solid")
    dashed_count = sum(1 for value in side_styles.values() if value == "dashed")
    if solid_count >= 3 and dashed_count <= 1:
        line_style = "solid"
    elif dashed_count >= 2:
        line_style = "dashed"
    else:
        return None

    interior_density = local_adjust_interior_density(binary, bbox)
    if line_style == "solid" and width <= 135 and height <= 190 and interior_density > 0.055:
        return None
    if line_style == "solid" and interior_density > 0.13:
        return None
    if line_style == "solid" and area < 22000 and interior_density > 0.09:
        return None
    if line_style == "dashed" and interior_density > 0.16:
        return None
    if line_style == "dashed" and area < 14000 and interior_density > 0.09:
        return None

    corner_metrics: list[dict[str, Any]]
    if line_style == "dashed":
        corners_ok, corner_metrics = validate_dashed_corners(binary, bbox, cfg)
        if not corners_ok:
            return None
    else:
        corners_ok, corner_metrics = validate_l_corners(binary, bbox, cfg)
        if not corners_ok:
            return None

    line_thickness = 2
    refs = {
        "top": {
            **edge_metrics["top"],
            "orientation": "horizontal",
            "axis_start": int(x1),
            "axis_end": int(x2),
            "normal_start": int(y1 - 1),
            "normal_end": int(y1 + 2),
            "normal_center": float(y1),
            "width_px": 3,
            "normal_distance_px": 0.0,
        },
        "bottom": {
            **edge_metrics["bottom"],
            "orientation": "horizontal",
            "axis_start": int(x1),
            "axis_end": int(x2),
            "normal_start": int(y2 - 1),
            "normal_end": int(y2 + 2),
            "normal_center": float(y2),
            "width_px": 3,
            "normal_distance_px": 0.0,
        },
        "left": {
            **edge_metrics["left"],
            "orientation": "vertical",
            "axis_start": int(y1),
            "axis_end": int(y2),
            "normal_start": int(x1 - 1),
            "normal_end": int(x1 + 2),
            "normal_center": float(x1),
            "width_px": 3,
            "normal_distance_px": 0.0,
        },
        "right": {
            **edge_metrics["right"],
            "orientation": "vertical",
            "axis_start": int(y1),
            "axis_end": int(y2),
            "normal_start": int(x2 - 1),
            "normal_end": int(x2 + 2),
            "normal_center": float(x2),
            "width_px": 3,
            "normal_distance_px": 0.0,
        },
    }
    top = refined_edge(synthetic_edge("horizontal", x1, x2, y1, line_thickness), refs["top"])
    bottom = refined_edge(synthetic_edge("horizontal", x1, x2, y2, line_thickness), refs["bottom"])
    left_edge = refined_edge(synthetic_edge("vertical", y1, y2, x1, line_thickness), refs["left"])
    right_edge = refined_edge(synthetic_edge("vertical", y1, y2, x2, line_thickness), refs["right"])
    coverages = [float(item.get("coverage", 0.0)) for item in edge_metrics.values()]
    gaps = [int(item.get("max_gap_px", 0)) for item in edge_metrics.values()]
    score_value = round(float(np.mean(coverages) - 0.004 * max(gaps) + 0.075), 4)
    frame_style = {
        "accepted": True,
        "style": str(line_style),
        "edges": {
            name: {
                "style": str(side_styles[name]),
                "gap_lengths_px": [int(value) for value in edge_metrics[name].get("inner_gap_lengths_px", [])],
                "median_gap_px": round(
                    float(np.median(edge_metrics[name].get("inner_gap_lengths_px", [0]))),
                    4,
                )
                if edge_metrics[name].get("inner_gap_lengths_px")
                else 0.0,
                "reason": None,
            }
            for name in ("top", "bottom", "left", "right")
        },
        "reason": "medium_large_recovery_accepted",
        "local_side_styles": side_styles,
    }
    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "type": "closed_rectangular_border_frame",
        "confidence": round(float(min(0.96, np.mean(coverages))), 4),
        "score": score_value,
        "line_thickness_px": int(line_thickness),
        "geometry": {
            "width_px": int(width),
            "height_px": int(height),
            "aspect_ratio": round(float(aspect), 4),
            "line_style": str(line_style),
            "dashed_frame_like": str(line_style) == "dashed",
            "seed_axis": "medium_large_recovery",
            "medium_large_recovery_source": str(source),
            "interior_density": round(float(interior_density), 4),
            "corner_mask_extension_px": int(cfg.corner_mask_extension_px),
            "candidate_span_ratios": {
                name: round(float(item.get("span_ratio", 0.0)), 4)
                for name, item in edge_metrics.items()
            },
            "edge_refinement": {
                name: {
                    "normal_center": round(float(item.get("normal_center", 0.0)), 4),
                    "width_px": int(item.get("width_px", 0)),
                    "span_ratio": round(float(item.get("span_ratio", 0.0)), 4),
                    "coverage": round(float(item.get("coverage", 0.0)), 4),
                    "endpoint_gap_px": int(item.get("endpoint_gap_px", 0)),
                    "normal_valid_ratio": round(float(item.get("normal_valid_ratio", 0.0)), 4),
                    "min_normal_span_ratio": round(float(item.get("min_normal_span_ratio", 0.0)), 4),
                    "invalid_normal_count": int(item.get("invalid_normal_count", 0)),
                    "normal_distance_px": 0.0,
                    "style_hint": str(item.get("style_hint", "unknown")),
                }
                for name, item in refs.items()
            },
        },
        "edges": {
            "top": top,
            "bottom": bottom,
            "left": left_edge,
            "right": right_edge,
        },
        "edge_metrics": edge_metrics,
        "frame_style": frame_style,
        "corner_metrics": corner_metrics,
    }


def frame_iou(a: Sequence[int], b: Sequence[int]) -> float:
    inter = overlap_area(a, b)
    union = float(box_area(a) + box_area(b) - inter)
    return 0.0 if union <= 0 else float(inter / union)


def dedupe_frames(frames: list[dict[str, Any]], cfg: RectConfig) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for frame in sorted(
        frames,
        key=lambda item: (
            str(item.get("geometry", {}).get("seed_axis", "")) == "medium_large_recovery",
            -float(item["score"]),
            box_area(item["bbox"]),
        ),
    ):
        duplicate = False
        frame_recovery = str(frame.get("geometry", {}).get("seed_axis", "")) == "medium_large_recovery"
        for existing in kept:
            area_ratio = float(min(box_area(frame["bbox"]), box_area(existing["bbox"]))) / float(
                max(1, max(box_area(frame["bbox"]), box_area(existing["bbox"])))
            )
            fw = int(frame["bbox"][2] - frame["bbox"][0])
            fh = int(frame["bbox"][3] - frame["bbox"][1])
            ew = int(existing["bbox"][2] - existing["bbox"][0])
            eh = int(existing["bbox"][3] - existing["bbox"][1])
            shares_existing_dimension = (
                float(min(fw, ew) / max(1, max(fw, ew))) > 0.72
                or float(min(fh, eh) / max(1, max(fh, eh))) > 0.72
            )
            if frame_iou(frame["bbox"], existing["bbox"]) > float(cfg.dedupe_iou):
                duplicate = True
                break
            if (
                inside_ratio(frame["bbox"], existing["bbox"]) > float(cfg.dedupe_inside_ratio)
                and not (frame_recovery and area_ratio < 0.55 and not shares_existing_dimension)
            ):
                duplicate = True
                break
            if (
                area_ratio >= 0.55
                and inside_ratio(existing["bbox"], frame["bbox"]) > float(cfg.dedupe_inside_ratio)
            ):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(frame)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["bbox"][3] - item["bbox"][1]))
    return kept


def is_dense_internal_solid_symbol_frame(binary: np.ndarray, frame: dict[str, Any]) -> bool:
    if str(frame.get("geometry", {}).get("line_style", "solid")) != "solid":
        return False
    x1, y1, x2, y2 = [int(v) for v in frame["bbox"]]
    width = int(x2 - x1)
    height = int(y2 - y1)
    area = int(width * height)
    density = local_adjust_interior_density(binary, frame["bbox"])
    compact_dense = width <= 135 and height <= 190 and density > 0.3
    tiny_dense = area < 5200 and density > 0.3
    return bool(compact_dense or tiny_dense)


def detect_rects(binary: np.ndarray, cfg: RectConfig) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    h_edges = extract_edge_candidates(binary, "horizontal", cfg)
    v_edges = extract_edge_candidates(binary, "vertical", cfg)
    if int(cfg.composite_bridge_gap_px) > int(cfg.bridge_gap_px):
        composite_cfg = replace(cfg, bridge_gap_px=int(cfg.composite_bridge_gap_px))
        h_seen = {tuple(int(v) for v in edge["bbox"]) for edge in h_edges}
        v_seen = {tuple(int(v) for v in edge["bbox"]) for edge in v_edges}
        for edge in extract_edge_candidates(binary, "horizontal", composite_cfg):
            key = tuple(int(v) for v in edge["bbox"])
            if key not in h_seen:
                h_edges.append({**edge, "composite_bridge_gap_px": int(cfg.composite_bridge_gap_px)})
                h_seen.add(key)
        for edge in extract_edge_candidates(binary, "vertical", composite_cfg):
            key = tuple(int(v) for v in edge["bbox"])
            if key not in v_seen:
                v_edges.append({**edge, "composite_bridge_gap_px": int(cfg.composite_bridge_gap_px)})
                v_seen.add(key)
    small_min_side_length = min(int(cfg.min_side_length_px), int(cfg.small_dashed_min_side_length_px))
    h_edge_keys = {tuple(int(v) for v in edge["bbox"]) for edge in h_edges}
    h_edges_small_only = extract_edge_candidates(binary, "horizontal", cfg, small_min_side_length)
    v_edges_small = extract_edge_candidates(binary, "vertical", cfg, small_min_side_length)
    h_edges_small = h_edges + [
        edge
        for edge in h_edges_small_only
        if tuple(int(v) for v in edge["bbox"]) not in h_edge_keys
    ]
    band = int(cfg.edge_band_px)
    frames: list[dict[str, Any]] = []
    rejected_horizontal_pairs = 0
    rejected_vertical_pairs = 0
    rejection_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1

    def try_add_frame(
        top: dict[str, Any],
        bottom: dict[str, Any],
        left_edge: dict[str, Any],
        right_edge: dict[str, Any],
        pair_metrics: dict[str, Any],
        seed_axis: str,
        edge_search_tol: int,
        min_side_length_px: int | None = None,
        force_dashed: bool = False,
        small_dashed_candidate: bool = False,
    ) -> None:
        local_min_side_length = int(
            cfg.min_side_length_px if min_side_length_px is None else min_side_length_px
        )
        if top is bottom:
            reject("cap_edges_collapsed")
            return
        if left_edge is right_edge:
            reject("side_edges_collapsed")
            return

        top_y = int(np.floor(edge_normal_center(top)))
        bottom_y = int(np.ceil(edge_normal_center(bottom)))
        if bottom_y < top_y:
            top_y, bottom_y = bottom_y, top_y
            top, bottom = bottom, top
        height = int(bottom_y - top_y)
        if height < int(local_min_side_length):
            reject("height_below_min")
            return

        x1 = int(np.floor(edge_normal_center(left_edge)))
        x2 = int(np.ceil(edge_normal_center(right_edge)))
        if x2 < x1:
            x1, x2 = x2, x1
            left_edge, right_edge = right_edge, left_edge
        width = int(x2 - x1)
        if width < int(local_min_side_length):
            reject("width_below_min")
            return
        if int(width * height) < int(cfg.min_frame_area_px):
            reject("area_below_min")
            return

        coarse_candidate_span_ratios = {
            "top": edge_candidate_span_ratio(top, x1, x2),
            "bottom": edge_candidate_span_ratio(bottom, x1, x2),
            "left": edge_candidate_span_ratio(left_edge, top_y, bottom_y),
            "right": edge_candidate_span_ratio(right_edge, top_y, bottom_y),
        }
        if any(float(value) < float(cfg.min_candidate_span_ratio) for value in coarse_candidate_span_ratios.values()):
            reject("edge_candidate_span_below_min")
            return

        coarse_edge_metrics = {
            "top": axis_projection_coverage(binary, "horizontal", x1, x2, top_y, band),
            "bottom": axis_projection_coverage(binary, "horizontal", x1, x2, bottom_y, band),
            "left": axis_projection_coverage(binary, "vertical", top_y, bottom_y, x1, band),
            "right": axis_projection_coverage(binary, "vertical", top_y, bottom_y, x2, band),
        }
        if any(float(item["coverage"]) < float(cfg.min_edge_coverage) for item in coarse_edge_metrics.values()):
            reject("edge_coverage_below_min")
            return
        if any(int(item["max_gap_px"]) > int(cfg.max_edge_gap_px) for item in coarse_edge_metrics.values()):
            reject("edge_gap_above_max")
            return

        coarse_frame_style = classify_frame_style(coarse_edge_metrics, cfg)
        if force_dashed:
            expected_styles = {name: "dashed" for name in ("top", "bottom", "left", "right")}
        elif not bool(coarse_frame_style["accepted"]):
            reason = str(coarse_frame_style.get("reason", "frame_style_rejected"))
            coarse_styles = {
                name: str(item.get("style", "unknown"))
                for name, item in coarse_frame_style.get("edges", {}).items()
            }
            dashed_count = sum(1 for value in coarse_styles.values() if value == "dashed")
            solid_count = sum(1 for value in coarse_styles.values() if value == "solid")
            if reason == "mixed_edge_styles" and dashed_count >= 2:
                expected_styles = {name: "dashed" for name in ("top", "bottom", "left", "right")}
            elif solid_count >= 1 and dashed_count >= 1:
                # Genuine conflict: some edges confidently solid, others confidently
                # dashed -- not just coarse-probe noise. Reject as before.
                reject(reason)
                return
            else:
                # The coarse probe (a single fixed-position row/column scan, before any
                # per-edge search) classified some edges "unknown" -- often just noise
                # right at a threshold boundary, not a real style conflict (no edge was
                # confidently solid AND confidently dashed at once). Coverage/gap floors
                # were already checked above, so there is real edge structure here; let
                # the finer per-edge refinement below determine the true style instead
                # of rejecting on this coarse pass.
                expected_styles = None
        else:
            expected_styles = {
                name: str(coarse_frame_style["style"])
                for name in ("top", "bottom", "left", "right")
            }

        refined = refine_frame_edges(
            binary,
            top,
            bottom,
            left_edge,
            right_edge,
            x1,
            top_y,
            x2,
            bottom_y,
            cfg,
            expected_styles,
        )
        if refined is None:
            reject("edge_refine_failed")
            return
        x1 = int(refined["x1"])
        x2 = int(refined["x2"])
        top_y = int(refined["top_y"])
        bottom_y = int(refined["bottom_y"])
        top = refined["top"]
        bottom = refined["bottom"]
        left_edge = refined["left"]
        right_edge = refined["right"]
        width = int(x2 - x1)
        height = int(bottom_y - top_y)
        if width < int(local_min_side_length):
            reject("width_below_min")
            return
        if height < int(local_min_side_length):
            reject("height_below_min")
            return
        if int(width * height) < int(cfg.min_frame_area_px):
            reject("area_below_min")
            return
        if bool(small_dashed_candidate) and int(width * height) > int(cfg.small_dashed_max_area_px):
            reject("small_dashed_area_above_max")
            return

        candidate_span_ratios = {
            name: float(item.get("span_ratio", 0.0))
            for name, item in refined["metrics"].items()
        }
        if any(float(value) < float(cfg.min_candidate_span_ratio) for value in candidate_span_ratios.values()):
            reject("edge_candidate_span_below_min")
            return

        edge_metrics = {
            name: {
                "coverage": float(item.get("coverage", 0.0)),
                "dashed_effective_coverage": float(item.get("dashed_effective_coverage", item.get("coverage", 0.0))),
                "max_gap_px": int(item.get("max_gap_px", 0)),
                "inner_gap_lengths_px": [int(value) for value in item.get("inner_gap_lengths_px", [])],
                "ink_run_lengths_px": [int(value) for value in item.get("ink_run_lengths_px", [])],
                "active_pixels": int(item.get("active_pixels", 0)),
                "span_ratio": float(item.get("span_ratio", 0.0)),
                "endpoint_gap_px": int(item.get("endpoint_gap_px", 0)),
                "normal_valid_ratio": float(item.get("normal_valid_ratio", 0.0)),
                "min_normal_span_ratio": float(item.get("min_normal_span_ratio", 0.0)),
                "invalid_normal_count": int(item.get("invalid_normal_count", 0)),
                "style_hint": str(item.get("style_hint", "unknown")),
            }
            for name, item in refined["metrics"].items()
        }
        if any(float(item["coverage"]) < float(cfg.min_edge_coverage) for item in edge_metrics.values()):
            reject("edge_coverage_below_min")
            return
        if any(int(item["max_gap_px"]) > int(cfg.max_edge_gap_px) for item in edge_metrics.values()):
            reject("edge_gap_above_max")
            return

        frame_style = classify_frame_style(edge_metrics, cfg)
        if not bool(frame_style["accepted"]):
            reason = str(frame_style.get("reason", "frame_style_rejected"))
            edge_styles = {
                name: str(item.get("style", "unknown"))
                for name, item in frame_style.get("edges", {}).items()
            }
            small_mixed_dashed = (
                bool(small_dashed_candidate)
                and reason == "mixed_edge_styles"
                and edge_styles.get("top") == "dashed"
                and edge_styles.get("bottom") == "dashed"
                and edge_styles.get("left") in {"solid", "dashed"}
                and edge_styles.get("right") in {"solid", "dashed"}
            )
            if not small_mixed_dashed:
                reject(reason)
                return
            dashed_medians = [
                float(item.get("median_gap_px", 0.0))
                for name, item in frame_style.get("edges", {}).items()
                if name in {"top", "bottom"} and str(item.get("style")) == "dashed"
            ]
            frame_style = {
                **frame_style,
                "accepted": True,
                "style": "dashed",
                "reason": "small_dashed_mixed_edges_accepted",
                "gap_median_range_px": round(float(max(dashed_medians) - min(dashed_medians)), 4)
                if len(dashed_medians) >= 2
                else 0.0,
            }
        if force_dashed and str(frame_style["style"]) != "dashed":
            reject("small_dashed_style_not_dashed")
            return
        if str(frame_style["style"]) == "dashed":
            thick_candidate_edges = [
                name
                for name, edge in {
                    "top": top,
                    "bottom": bottom,
                    "left": left_edge,
                    "right": right_edge,
                }.items()
                if int(edge.get("thickness_px", 0)) > int(cfg.max_side_thickness_px) // 2
            ]
            if len(thick_candidate_edges) > int(cfg.dashed_edge_max_thick_candidates):
                reject("dashed_candidate_edges_too_thick")
                return

        bbox = [int(x1), int(top_y), int(x2), int(bottom_y)]
        if str(frame_style["style"]) == "dashed":
            corners_ok, corner_metrics = validate_dashed_corners(binary, bbox, cfg)
        else:
            corners_ok, corner_metrics = validate_l_corners(binary, bbox, cfg)
        if not corners_ok:
            for item in corner_metrics:
                if not bool(item["accepted"]):
                    reject(str(item.get("reason", "corner_not_l_shape")))
            return

        curved_edges = []
        if edge_is_curved(binary, "horizontal", x1, x2, top_y, band):
            curved_edges.append("top")
        if edge_is_curved(binary, "horizontal", x1, x2, bottom_y, band):
            curved_edges.append("bottom")
        if edge_is_curved(binary, "vertical", top_y, bottom_y, x1, band):
            curved_edges.append("left")
        if edge_is_curved(binary, "vertical", top_y, bottom_y, x2, band):
            curved_edges.append("right")
        if curved_edges:
            reject("edge_curved_not_straight")
            return

        coverages = [float(item["coverage"]) for item in edge_metrics.values()]
        gaps = [int(item["max_gap_px"]) for item in edge_metrics.values()]
        line_thickness = int(
            max(
                1,
                round(
                    float(
                        np.median(
                            [
                                int(top["thickness_px"]),
                                int(bottom["thickness_px"]),
                                int(left_edge["thickness_px"]),
                                int(right_edge["thickness_px"]),
                            ]
                        )
                    )
                ),
            )
        )
        line_thickness = min(int(cfg.max_side_thickness_px), int(line_thickness))
        pair_key = "horizontal_pair" if "horizontal" in str(seed_axis) else "vertical_pair"
        frames.append(
            {
                "bbox": bbox,
                "type": "closed_rectangular_border_frame",
                "confidence": round(float(min(0.99, np.mean(coverages))), 4),
                "score": round(float(np.mean(coverages) - 0.01 * max(gaps)), 4),
                "line_thickness_px": int(line_thickness),
                "geometry": {
                    "width_px": int(width),
                    "height_px": int(height),
                    "aspect_ratio": round(float(width / max(1, height)), 4),
                    "line_style": str(frame_style["style"]),
                    "dashed_frame_like": str(frame_style["style"]) == "dashed",
                    "seed_axis": str(seed_axis),
                    "small_dashed_candidate": bool(small_dashed_candidate),
                    "min_side_length_used_px": int(local_min_side_length),
                    pair_key: pair_metrics,
                    "edge_search_tolerance_px": int(edge_search_tol),
                    "side_search_tolerance_px": int(edge_search_tol),
                    "corner_mask_extension_px": int(cfg.corner_mask_extension_px),
                    "candidate_span_ratios": {
                        name: round(float(value), 4)
                        for name, value in candidate_span_ratios.items()
                    },
                    "edge_refinement": {
                        name: {
                            "normal_center": round(float(item.get("normal_center", 0.0)), 4),
                            "width_px": int(item.get("width_px", 0)),
                            "span_ratio": round(float(item.get("span_ratio", 0.0)), 4),
                            "coverage": round(float(item.get("coverage", 0.0)), 4),
                            "endpoint_gap_px": int(item.get("endpoint_gap_px", 0)),
                            "normal_valid_ratio": round(float(item.get("normal_valid_ratio", 0.0)), 4),
                            "min_normal_span_ratio": round(float(item.get("min_normal_span_ratio", 0.0)), 4),
                            "invalid_normal_count": int(item.get("invalid_normal_count", 0)),
                            "normal_distance_px": round(float(item.get("normal_distance_px", 0.0)), 4),
                            "style_hint": str(item.get("style_hint", "unknown")),
                        }
                        for name, item in refined["metrics"].items()
                    },
                },
                "edges": {
                    "top": top,
                    "bottom": bottom,
                    "left": left_edge,
                    "right": right_edge,
                },
                "edge_metrics": edge_metrics,
                "frame_style": frame_style,
                "corner_metrics": corner_metrics,
            }
        )

    for top in h_edges:
        tx1, ty1, tx2, ty2 = [int(v) for v in top["bbox"]]
        top_y = int(np.floor(0.5 * float(ty1 + ty2)))
        for bottom in h_edges:
            if bottom is top:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in bottom["bbox"]]
            bottom_y = int(np.ceil(0.5 * float(by1 + by2)))
            if bottom_y <= top_y:
                continue
            height = int(bottom_y - top_y)
            if height < int(cfg.min_side_length_px):
                reject("height_below_min")
                continue
            pair_metrics = horizontal_pair_alignment(top, bottom, cfg)
            if not bool(pair_metrics["accepted"]):
                rejected_horizontal_pairs += 1
                reject(str(pair_metrics.get("reason", "horizontal_pair_rejected")))
                continue

            side_search_tol = max(
                int(cfg.edge_endpoint_tolerance_px) * 3,
                int(cfg.corner_tolerance_px) * 4,
                int(cfg.side_search_min_tolerance_px),
            )
            left_edge = best_side_edge(
                v_edges,
                top_y,
                bottom_y,
                min(tx1, bx1) - side_search_tol,
                max(tx1, bx1) + side_search_tol,
                cfg,
            )
            right_edge = best_side_edge(
                v_edges,
                top_y,
                bottom_y,
                min(tx2, bx2) - side_search_tol,
                max(tx2, bx2) + side_search_tol,
                cfg,
            )
            if left_edge is None:
                reject("missing_left_edge")
                continue
            if right_edge is None:
                reject("missing_right_edge")
                continue

            try_add_frame(top, bottom, left_edge, right_edge, pair_metrics, "horizontal", side_search_tol)

    for left_edge in v_edges:
        lx1, ly1, lx2, ly2 = [int(v) for v in left_edge["bbox"]]
        left_x = int(np.floor(0.5 * float(lx1 + lx2)))
        for right_edge in v_edges:
            if right_edge is left_edge:
                continue
            rx1, ry1, rx2, ry2 = [int(v) for v in right_edge["bbox"]]
            right_x = int(np.ceil(0.5 * float(rx1 + rx2)))
            if right_x <= left_x:
                continue
            width = int(right_x - left_x)
            if width < int(cfg.min_side_length_px):
                reject("width_below_min")
                continue
            pair_metrics = vertical_pair_alignment(left_edge, right_edge, cfg)
            if not bool(pair_metrics["accepted"]):
                rejected_vertical_pairs += 1
                reject(str(pair_metrics.get("reason", "vertical_pair_rejected")))
                continue

            cap_search_tol = max(
                int(cfg.edge_endpoint_tolerance_px) * 3,
                int(cfg.corner_tolerance_px) * 4,
                int(cfg.side_search_min_tolerance_px),
            )
            top_edge = best_cap_edge(
                h_edges,
                left_x,
                right_x,
                min(ly1, ry1) - cap_search_tol,
                max(ly1, ry1) + cap_search_tol,
                cfg,
            )
            bottom_edge = best_cap_edge(
                h_edges,
                left_x,
                right_x,
                min(ly2, ry2) - cap_search_tol,
                max(ly2, ry2) + cap_search_tol,
                cfg,
            )
            if top_edge is None:
                reject("missing_top_edge")
                continue
            if bottom_edge is None:
                reject("missing_bottom_edge")
                continue

            try_add_frame(top_edge, bottom_edge, left_edge, right_edge, pair_metrics, "vertical", cap_search_tol)

    if int(cfg.small_dashed_min_side_length_px) < int(cfg.min_side_length_px):
        small_side_search_tol = max(
            int(cfg.edge_endpoint_tolerance_px) * 2,
            int(cfg.corner_tolerance_px) * 3,
            int(cfg.small_dashed_min_side_length_px),
        )
        for top in h_edges_small:
            tx1, ty1, tx2, ty2 = [int(v) for v in top["bbox"]]
            top_y = int(np.floor(0.5 * float(ty1 + ty2)))
            for bottom in h_edges_small:
                if bottom is top:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in bottom["bbox"]]
                bottom_y = int(np.ceil(0.5 * float(by1 + by2)))
                if bottom_y <= top_y:
                    continue
                height = int(bottom_y - top_y)
                if height < int(cfg.small_dashed_min_side_length_px):
                    reject("small_dashed_height_below_min")
                    continue
                if height >= int(cfg.min_side_length_px):
                    continue
                overlap = overlap_len(tx1, tx2, bx1, bx2)
                if overlap < float(cfg.min_side_length_px):
                    reject("small_dashed_horizontal_overlap_below_min")
                    continue
                pair_metrics = horizontal_pair_alignment(top, bottom, cfg)
                if not bool(pair_metrics["accepted"]):
                    rejected_horizontal_pairs += 1
                    reject(str(pair_metrics.get("reason", "small_dashed_horizontal_pair_rejected")))
                    continue

                left_edge = best_side_edge(
                    v_edges_small,
                    top_y,
                    bottom_y,
                    min(tx1, bx1) - small_side_search_tol,
                    max(tx1, bx1) + small_side_search_tol,
                    cfg,
                )
                right_edge = best_side_edge(
                    v_edges_small,
                    top_y,
                    bottom_y,
                    min(tx2, bx2) - small_side_search_tol,
                    max(tx2, bx2) + small_side_search_tol,
                    cfg,
                )
                if left_edge is None:
                    left_edge = synthetic_edge(
                        "vertical",
                        top_y,
                        bottom_y,
                        int(min(tx1, bx1)),
                    )
                if right_edge is None:
                    right_edge = synthetic_edge(
                        "vertical",
                        top_y,
                        bottom_y,
                        int(max(tx2, bx2)),
                    )

                try_add_frame(
                    top,
                    bottom,
                    left_edge,
                    right_edge,
                    pair_metrics,
                    "small_dashed_horizontal",
                    small_side_search_tol,
                    min_side_length_px=int(cfg.small_dashed_min_side_length_px),
                    force_dashed=True,
                    small_dashed_candidate=True,
                )

    frames = dedupe_frames(frames, cfg)
    base_filtered_frames: list[dict[str, Any]] = []
    for frame in frames:
        cur = canonical_rect_mask(binary, frame, cfg)
        colored_gap_validation = validate_colored_dashed_gaps(cur, frame, cfg)
        frame["colored_gap_validation"] = colored_gap_validation
        if not bool(colored_gap_validation.get("accepted", True)):
            reject(str(colored_gap_validation.get("reason", "colored_gap_rejected")))
            continue
        frame["mask_bbox"] = mask_bbox(cur)
        frame["mask_pixel_count"] = int(cur.sum())
        base_filtered_frames.append(frame)
    frames = base_filtered_frames

    local_adjust_candidates = 0
    local_adjust_accepted = 0
    medium_recovery_candidates = 0
    medium_recovery_accepted = 0
    local_seen: set[tuple[int, int, int, int]] = set()
    medium_seen: set[tuple[int, int, int, int]] = set()

    def add_local_adjusted_bbox(bbox: Sequence[int], source: str) -> None:
        nonlocal local_adjust_candidates, local_adjust_accepted
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return
        key = (int(x1), int(y1), int(x2), int(y2))
        if key in local_seen:
            return
        local_seen.add(key)
        local_adjust_candidates += 1
        for existing in frames:
            if str(existing.get("geometry", {}).get("seed_axis", "")) == "local_adjusted":
                continue
            existing_box = existing["bbox"]
            if frame_iou(key, existing_box) > float(cfg.dedupe_iou):
                return
            if inside_ratio(key, existing_box) > float(cfg.dedupe_inside_ratio):
                return
            if inside_ratio(existing_box, key) > float(cfg.dedupe_inside_ratio):
                return
        frame = local_adjusted_frame_from_bbox(binary, key, cfg, source)
        if frame is None:
            return
        local_mask = canonical_rect_mask(binary, frame, cfg)
        colored_gap_validation = validate_colored_dashed_gaps(local_mask, frame, cfg)
        if not bool(colored_gap_validation.get("accepted", True)):
            return
        frame["colored_gap_validation"] = colored_gap_validation
        frame["mask_bbox"] = mask_bbox(local_mask)
        frame["mask_pixel_count"] = int(local_mask.sum())
        frames.append(frame)
        local_adjust_accepted += 1

    local_min = int(cfg.small_dashed_min_side_length_px)
    local_max = int(cfg.local_adjust_max_side_px)
    local_tol = max(int(cfg.edge_endpoint_tolerance_px) * 2, int(cfg.corner_tolerance_px) * 3, local_min)
    local_edge_max_len = int(local_max + local_tol * 2)

    for top in h_edges_small if bool(cfg.enable_local_adjust) else []:
        if int(top.get("length_px", 0)) > local_edge_max_len:
            continue
        tx1, ty1, tx2, ty2 = [int(v) for v in top["bbox"]]
        top_y = int(round(edge_normal_center(top)))
        for bottom in h_edges_small:
            if bottom is top:
                continue
            if int(bottom.get("length_px", 0)) > local_edge_max_len:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in bottom["bbox"]]
            bottom_y = int(round(edge_normal_center(bottom)))
            if bottom_y <= top_y:
                continue
            height = int(bottom_y - top_y)
            if height < local_min or height > local_max:
                continue
            overlap = overlap_len(tx1, tx2, bx1, bx2)
            if overlap < float(local_min) * 0.7:
                continue
            x_positions: set[int] = {
                int(round(max(tx1, bx1))),
                int(round(min(tx2, bx2))),
            }
            search_x1 = min(tx1, bx1) - local_tol
            search_x2 = max(tx2, bx2) + local_tol
            for edge in v_edges_small:
                if edge["orientation"] != "vertical":
                    continue
                x = int(round(edge_normal_center(edge)))
                if x < search_x1 or x > search_x2:
                    continue
                y1, y2 = edge_axis_range(edge)
                if overlap_len(y1, y2, top_y, bottom_y) < float(height) * 0.35:
                    continue
                x_positions.add(int(x))
                ex1, _, ex2, _ = [int(v) for v in edge["bbox"]]
                for edge_x in (ex1, ex2):
                    if search_x1 <= int(edge_x) <= search_x2:
                        x_positions.add(int(edge_x))
            anchors = [
                float(max(tx1, bx1)),
                float(min(tx2, bx2)),
                float(min(tx1, bx1)),
                float(max(tx2, bx2)),
            ]
            xs = sorted(
                sorted(x_positions, key=lambda value: min(abs(float(value) - anchor) for anchor in anchors))[:12]
            )
            for i, x1 in enumerate(xs):
                for x2 in xs[i + 1 :]:
                    width = int(x2 - x1)
                    if width < local_min or width > local_max:
                        continue
                    if int(width * height) > int(cfg.local_adjust_max_area_px):
                        continue
                    add_local_adjusted_bbox([x1, top_y, x2, bottom_y], "horizontal_pair")

    if bool(cfg.enable_local_adjust) and bool(cfg.local_adjust_vertical_seed):
        for left_edge in v_edges_small:
            if int(left_edge.get("length_px", 0)) > local_edge_max_len:
                continue
            lx1, ly1, lx2, ly2 = [int(v) for v in left_edge["bbox"]]
            left_x = int(round(edge_normal_center(left_edge)))
            for right_edge in v_edges_small:
                if right_edge is left_edge:
                    continue
                if int(right_edge.get("length_px", 0)) > local_edge_max_len:
                    continue
                rx1, ry1, rx2, ry2 = [int(v) for v in right_edge["bbox"]]
                right_x = int(round(edge_normal_center(right_edge)))
                if right_x <= left_x:
                    continue
                width = int(right_x - left_x)
                if width < local_min or width > local_max:
                    continue
                overlap = overlap_len(ly1, ly2, ry1, ry2)
                if overlap < float(local_min) * 0.7:
                    continue
                y_positions: set[int] = {
                    int(round(max(ly1, ry1))),
                    int(round(min(ly2, ry2))),
                }
                search_y1 = min(ly1, ry1) - local_tol
                search_y2 = max(ly2, ry2) + local_tol
                for edge in h_edges_small:
                    if edge["orientation"] != "horizontal":
                        continue
                    y = int(round(edge_normal_center(edge)))
                    if y < search_y1 or y > search_y2:
                        continue
                    x1, x2 = edge_axis_range(edge)
                    if overlap_len(x1, x2, left_x, right_x) < float(width) * 0.35:
                        continue
                    y_positions.add(int(y))
                anchors = [
                    float(max(ly1, ry1)),
                    float(min(ly2, ry2)),
                    float(min(ly1, ry1)),
                    float(max(ly2, ry2)),
                ]
                ys = sorted(
                    sorted(y_positions, key=lambda value: min(abs(float(value) - anchor) for anchor in anchors))[:12]
                )
                for i, y1 in enumerate(ys):
                    for y2 in ys[i + 1 :]:
                        height = int(y2 - y1)
                        if height < local_min or height > local_max:
                            continue
                        if int(width * height) > int(cfg.local_adjust_max_area_px):
                            continue
                        add_local_adjusted_bbox([left_x, y1, right_x, y2], "vertical_pair")

    def add_medium_recovery_bbox(bbox: Sequence[int], source: str) -> None:
        nonlocal medium_recovery_candidates, medium_recovery_accepted
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return
        key = (int(x1), int(y1), int(x2), int(y2))
        if key in medium_seen:
            return
        medium_seen.add(key)
        area = int((x2 - x1) * (y2 - y1))
        if area < int(cfg.medium_recovery_min_area_px) or area > int(cfg.medium_recovery_max_area_px):
            return
        medium_recovery_candidates += 1
        for existing in frames:
            if frame_iou(key, existing["bbox"]) > float(cfg.dedupe_iou):
                return
        frame = medium_large_frame_from_bbox(binary, key, cfg, source)
        if frame is None:
            return
        recovery_mask = canonical_rect_mask(binary, frame, cfg)
        colored_gap_validation = validate_colored_dashed_gaps(recovery_mask, frame, cfg)
        if not bool(colored_gap_validation.get("accepted", True)):
            reject(str(colored_gap_validation.get("reason", "medium_recovery_colored_gap_rejected")))
            return
        frame["colored_gap_validation"] = colored_gap_validation
        frame["mask_bbox"] = mask_bbox(recovery_mask)
        frame["mask_pixel_count"] = int(recovery_mask.sum())
        frames.append(frame)
        medium_recovery_accepted += 1

    medium_min = int(cfg.medium_recovery_min_side_px)
    medium_max = int(cfg.medium_recovery_max_side_px)
    medium_tol = max(int(cfg.side_search_min_tolerance_px), int(cfg.edge_endpoint_tolerance_px) * 4)
    for top in h_edges if bool(cfg.enable_medium_recovery) else []:
        tx1, ty1, tx2, ty2 = [int(v) for v in top["bbox"]]
        top_y = int(round(edge_normal_center(top)))
        top_len = int(tx2 - tx1)
        if top_len < medium_min or top_len > medium_max + medium_tol * 2:
            continue
        for bottom in h_edges:
            if bottom is top:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in bottom["bbox"]]
            bottom_y = int(round(edge_normal_center(bottom)))
            if bottom_y <= top_y:
                continue
            bottom_len = int(bx2 - bx1)
            height = int(bottom_y - top_y)
            if bottom_len < medium_min or bottom_len > medium_max + medium_tol * 2:
                continue
            if height < medium_min or height > medium_max:
                continue
            if int(max(top_len, bottom_len) * height) > int(cfg.medium_recovery_max_area_px) * 2:
                continue
            overlap = overlap_len(tx1, tx2, bx1, bx2)
            if overlap < max(24, min(top_len, bottom_len) * 0.25):
                continue
            search_x1 = min(tx1, bx1) - medium_tol
            search_x2 = max(tx2, bx2) + medium_tol
            x_positions: set[int] = set()
            for edge in v_edges:
                x = int(round(edge_normal_center(edge)))
                if x < search_x1 or x > search_x2:
                    continue
                y1e, y2e = edge_axis_range(edge)
                if overlap_len(y1e, y2e, top_y, bottom_y) < max(18, height * 0.22):
                    continue
                x_positions.add(int(x))
                ex1, _, ex2, _ = [int(v) for v in edge["bbox"]]
                if search_x1 <= int(ex1) <= search_x2:
                    x_positions.add(int(ex1))
                if search_x1 <= int(ex2) <= search_x2:
                    x_positions.add(int(ex2))
            anchors = [float(min(tx1, bx1)), float(max(tx1, bx1)), float(min(tx2, bx2)), float(max(tx2, bx2))]
            xs = sorted(
                sorted(x_positions, key=lambda value: min(abs(float(value) - anchor) for anchor in anchors))[:16]
            )
            left_anchors = [float(tx1), float(bx1)]
            right_anchors = [float(tx2), float(bx2)]
            endpoint_tol = max(18, min(42, medium_tol))
            for i, x1 in enumerate(xs):
                if min(abs(float(x1) - anchor) for anchor in left_anchors) > endpoint_tol:
                    continue
                for x2 in xs[i + 1 :]:
                    if min(abs(float(x2) - anchor) for anchor in right_anchors) > endpoint_tol:
                        continue
                    width = int(x2 - x1)
                    if width < medium_min or width > medium_max:
                        continue
                    add_medium_recovery_bbox([x1, top_y, x2, bottom_y], "horizontal_pair")

    for left_edge in v_edges if bool(cfg.enable_medium_recovery) else []:
        lx1, ly1, lx2, ly2 = [int(v) for v in left_edge["bbox"]]
        left_x = int(round(edge_normal_center(left_edge)))
        left_len = int(ly2 - ly1)
        if left_len < medium_min or left_len > medium_max + medium_tol * 2:
            continue
        for right_edge in v_edges:
            if right_edge is left_edge:
                continue
            rx1, ry1, rx2, ry2 = [int(v) for v in right_edge["bbox"]]
            right_x = int(round(edge_normal_center(right_edge)))
            if right_x <= left_x:
                continue
            right_len = int(ry2 - ry1)
            width = int(right_x - left_x)
            if right_len < medium_min or right_len > medium_max + medium_tol * 2:
                continue
            if width < medium_min or width > medium_max:
                continue
            overlap = overlap_len(ly1, ly2, ry1, ry2)
            if overlap < max(24, min(left_len, right_len) * 0.25):
                continue
            search_y1 = min(ly1, ry1) - medium_tol
            search_y2 = max(ly2, ry2) + medium_tol
            y_positions: set[int] = set()
            for edge in h_edges:
                y = int(round(edge_normal_center(edge)))
                if y < search_y1 or y > search_y2:
                    continue
                x1e, x2e = edge_axis_range(edge)
                if overlap_len(x1e, x2e, left_x, right_x) < max(18, width * 0.22):
                    continue
                y_positions.add(int(y))
                _, ey1, _, ey2 = [int(v) for v in edge["bbox"]]
                if search_y1 <= int(ey1) <= search_y2:
                    y_positions.add(int(ey1))
                if search_y1 <= int(ey2) <= search_y2:
                    y_positions.add(int(ey2))
            anchors = [float(min(ly1, ry1)), float(max(ly1, ry1)), float(min(ly2, ry2)), float(max(ly2, ry2))]
            ys = sorted(
                sorted(y_positions, key=lambda value: min(abs(float(value) - anchor) for anchor in anchors))[:16]
            )
            top_anchors = [float(ly1), float(ry1)]
            bottom_anchors = [float(ly2), float(ry2)]
            endpoint_tol = max(18, min(42, medium_tol))
            for i, y1 in enumerate(ys):
                if min(abs(float(y1) - anchor) for anchor in top_anchors) > endpoint_tol:
                    continue
                for y2 in ys[i + 1 :]:
                    if min(abs(float(y2) - anchor) for anchor in bottom_anchors) > endpoint_tol:
                        continue
                    height = int(y2 - y1)
                    if height < medium_min or height > medium_max:
                        continue
                    add_medium_recovery_bbox([left_x, y1, right_x, y2], "vertical_pair")

    frames = dedupe_frames(frames, cfg)
    filtered_frames: list[tuple[dict[str, Any], np.ndarray]] = []
    for frame in frames:
        if is_dense_internal_solid_symbol_frame(binary, frame):
            reject("dense_internal_solid_symbol_frame")
            continue
        cur = canonical_rect_mask(binary, frame, cfg)
        colored_gap_validation = validate_colored_dashed_gaps(cur, frame, cfg)
        frame["colored_gap_validation"] = colored_gap_validation
        if not bool(colored_gap_validation.get("accepted", True)):
            reject(str(colored_gap_validation.get("reason", "colored_gap_rejected")))
            continue
        side_widths = [int(value) for value in frame.get("mask_side_widths_px", {}).values() if int(value) > 0]
        if side_widths:
            frame["mask_line_width_px"] = int(round(float(np.median(side_widths))))
        frame["mask_bbox"] = mask_bbox(cur)
        frame["mask_pixel_count"] = int(cur.sum())
        filtered_frames.append((frame, cur))

    frames = []
    mask = np.zeros_like(binary, dtype=bool)
    for index, (frame, cur) in enumerate(filtered_frames, start=1):
        frame["id"] = f"rect_{index:04d}"
        frames.append(frame)
        mask |= cur

    return frames, mask, {
        "horizontal_edge_candidates": int(len(h_edges)),
        "vertical_edge_candidates": int(len(v_edges)),
        "rejected_horizontal_pairs": int(rejected_horizontal_pairs),
        "rejected_vertical_pairs": int(rejected_vertical_pairs),
        "local_adjust_candidates": int(local_adjust_candidates),
        "local_adjust_accepted": int(local_adjust_accepted),
        "medium_recovery_candidates": int(medium_recovery_candidates),
        "medium_recovery_accepted": int(medium_recovery_accepted),
        "rejection_counts": rejection_counts,
        "rect_count": int(len(frames)),
        "rect_mask_pixels": int(mask.sum()),
    }


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def make_review_image(rgb: np.ndarray, rect_mask_image: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[rect_mask_image] = np.array(RED, dtype=np.uint8)
    return out


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def frame_border_erase_mask(binary: np.ndarray, frame: dict[str, Any], cfg: RectConfig) -> np.ndarray:
    # canonical_rect_mask measures the border's true ink width per side, but that
    # measurement can undershoot by a pixel or two (antialiasing, thin dash ends),
    # leaving a faint sliver of the original line behind. Dilate a couple of
    # pixels so the erased region always fully covers the real stroke.
    border = canonical_rect_mask(binary, frame, cfg)
    return dilate_mask(border, 2)


def frame_full_erase_mask(binary: np.ndarray, frame: dict[str, Any], cfg: RectConfig) -> np.ndarray:
    h, w = binary.shape
    out = frame_border_erase_mask(binary, frame, cfg)
    x1, y1, x2, y2 = [int(v) for v in frame["bbox"]]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2 + 1), min(h, y2 + 1)
    if x2c > x1c and y2c > y1c:
        out[y1c:y2c, x1c:x2c] = True
    return out


def make_clean_image(rgb: np.ndarray, fill_mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[fill_mask] = 255
    return out


def make_mask_image(rect_mask_image: np.ndarray) -> np.ndarray:
    gray = np.where(rect_mask_image, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def make_module_image(rgb: np.ndarray, binary: np.ndarray, frame: dict[str, Any], cfg: RectConfig) -> np.ndarray:
    h, w = binary.shape
    x1, y1, x2, y2 = [int(v) for v in frame["bbox"]]
    pad = max(6, int(frame.get("line_thickness_px", 1)) + 4)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad + 1), min(h, y2 + pad + 1)
    border_mask = frame_border_erase_mask(binary, frame, cfg)
    crop = rgb[cy1:cy2, cx1:cx2].copy()
    crop[border_mask[cy1:cy2, cx1:cx2]] = 255
    return crop


def rect_json(frame: dict[str, Any]) -> dict[str, Any]:
    edges = frame.get("edges", {})
    return {
        "id": str(frame["id"]),
        "type": str(frame["type"]),
        "bbox": [int(v) for v in frame["bbox"]],
        "confidence": round(float(frame["confidence"]), 4),
        "geometry": {
            **frame.get("geometry", {}),
            "line_thickness_px": int(frame.get("line_thickness_px", 1)),
        },
        "edges": {
            name: {
                "bbox": [int(v) for v in edge["bbox"]],
                "orientation": str(edge["orientation"]),
                "length_px": int(edge["length_px"]),
                "thickness_px": int(edge["thickness_px"]),
            }
            for name, edge in edges.items()
        },
        "mask": {
            "bbox": None if frame.get("mask_bbox") is None else [int(v) for v in frame["mask_bbox"]],
            "pixel_count": int(frame.get("mask_pixel_count", 0)),
            "line_width_px": int(frame.get("mask_line_width_px", frame.get("line_thickness_px", 1))),
            "side_bands_px": {
                str(name): int(value)
                for name, value in frame.get("mask_side_bands_px", {}).items()
            },
            "side_widths_px": {
                str(name): int(value)
                for name, value in frame.get("mask_side_widths_px", {}).items()
            },
            "side_offsets_px": {
                str(name): int(value)
                for name, value in frame.get("mask_side_offsets_px", {}).items()
            },
            "corner_supplements": frame.get("mask_corner_supplements", {}),
        },
        "quality": {
            "frame_style": frame.get("frame_style", {}),
            "colored_gap_validation": frame.get("colored_gap_validation", {}),
            "edge_metrics": frame.get("edge_metrics", {}),
            "corner_metrics": frame.get("corner_metrics", []),
        },
    }


def analyze_page(rgb: np.ndarray, page_number: int, cfg: RectConfig) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    binary = binarize(rgb, cfg.ink_threshold)
    frames, mask, detection_debug = detect_rects(binary, cfg)
    debug = {
        "page": int(page_number),
        "stage": STAGE_NAME,
        "config": asdict(cfg),
        "summary": {
            "foreground_pixels": int(binary.sum()),
            "rects": int(len(frames)),
            "rect_mask_pixels": int(mask.sum()),
        },
        "detection_debug": detection_debug,
        "rects": [rect_json(frame) for frame in frames],
    }
    return frames, mask, debug


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> None:
    pdf_stem = pdf_path.stem
    input_dir = resolve_input_dir(pdf_stem)
    pages = select_pages(args.pages, available_pages(input_dir))
    output = make_output_dirs(pdf_stem, args.preserve)
    cfg = RectConfig(
        ink_threshold=args.ink_threshold,
        min_side_length_px=args.min_side_length_px,
        max_side_thickness_px=args.max_side_thickness_px,
        corner_tolerance_px=args.corner_tolerance_px,
        edge_band_px=args.edge_band_px,
        min_edge_coverage=args.min_edge_coverage,
        max_edge_gap_px=args.max_edge_gap_px,
        bridge_gap_px=args.bridge_gap_px,
        composite_bridge_gap_px=args.composite_bridge_gap_px,
        dedupe_iou=args.dedupe_iou,
        dedupe_inside_ratio=args.dedupe_inside_ratio,
        min_frame_area_px=args.min_frame_area_px,
        corner_probe_px=args.corner_probe_px,
        corner_required_coverage=args.corner_required_coverage,
        corner_forbidden_coverage=args.corner_forbidden_coverage,
        corner_forbidden_min_active_px=args.corner_forbidden_min_active_px,
        edge_endpoint_tolerance_px=args.edge_endpoint_tolerance_px,
        solid_edge_min_coverage=args.solid_edge_min_coverage,
        solid_edge_max_gap_px=args.solid_edge_max_gap_px,
        dashed_edge_min_gaps=args.dashed_edge_min_gaps,
        dashed_gap_tolerance_px=args.dashed_gap_tolerance_px,
        dashed_mask_min_side_coverage=args.dashed_mask_min_side_coverage,
        dashed_mask_coverage_max_area_px=args.dashed_mask_coverage_max_area_px,
        dashed_terminal_stub_max_px=args.dashed_terminal_stub_max_px,
        dashed_terminal_stub_max_count=args.dashed_terminal_stub_max_count,
        dashed_terminal_stub_max_area_px=args.dashed_terminal_stub_max_area_px,
        dashed_edge_max_thick_candidates=args.dashed_edge_max_thick_candidates,
        edge_refine_search_px=args.edge_refine_search_px,
        refined_edge_min_span_ratio=args.refined_edge_min_span_ratio,
        corner_mask_extension_px=args.corner_mask_extension_px,
        corner_supplement_normal_tolerance_px=args.corner_supplement_normal_tolerance_px,
        corner_supplement_min_density=args.corner_supplement_min_density,
        small_dashed_min_side_length_px=args.small_dashed_min_side_length_px,
        small_dashed_max_area_px=args.small_dashed_max_area_px,
        local_adjust_min_area_px=args.local_adjust_min_area_px,
        local_adjust_max_area_px=args.local_adjust_max_area_px,
        local_adjust_max_side_px=args.local_adjust_max_side_px,
        local_adjust_min_coverage=args.local_adjust_min_coverage,
        local_adjust_min_span_ratio=args.local_adjust_min_span_ratio,
        min_candidate_span_ratio=args.min_candidate_span_ratio,
        side_search_min_tolerance_px=args.side_search_min_tolerance_px,
        medium_recovery_min_area_px=args.medium_recovery_min_area_px,
        medium_recovery_max_area_px=args.medium_recovery_max_area_px,
        medium_recovery_min_side_px=args.medium_recovery_min_side_px,
        medium_recovery_max_side_px=args.medium_recovery_max_side_px,
        medium_recovery_min_coverage=args.medium_recovery_min_coverage,
        medium_recovery_min_span_ratio=args.medium_recovery_min_span_ratio,
        medium_recovery_max_gap_px=args.medium_recovery_max_gap_px,
        medium_recovery_endpoint_slack_px=args.medium_recovery_endpoint_slack_px,
    )

    print(f"\nProcessing: {pdf_stem}")
    print(f"Input circuit_images: {input_dir}")
    print(f"Output: {output['root']}")
    print(f"Pages: {pages}")
    print(f"  preserve existing output: {args.preserve}")
    print(f"  images: {output['images']}")
    print(f"  circuit_images: {output['circuit_images']}")
    print(f"  module_images: {output['module_images']}")
    print(f"  masks: {output['masks']}")
    print(f"  json: {output['json']}")
    print(f"  debug: {output['debug']}")

    review_pages: list[np.ndarray] = []
    result_pages: list[np.ndarray] = []
    summary_pages: list[dict[str, Any]] = []

    for page_number in pages:
        print(f"  Page {page_number}")
        rgb = load_rgb(input_dir / f"page_{page_number:03d}.png")
        frames, rect_mask_image, debug = analyze_page(rgb, page_number, cfg)
        binary = binarize(rgb, cfg.ink_threshold)
        review = make_review_image(rgb, rect_mask_image)
        fill_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
        for frame in frames:
            fill_mask |= frame_full_erase_mask(binary, frame, cfg)
        clean = make_clean_image(rgb, fill_mask)
        mask_img = make_mask_image(rect_mask_image)

        image_path = output["images"] / f"page_{page_number:03d}.png"
        clean_path = output["circuit_images"] / f"page_{page_number:03d}.png"
        mask_path = output["masks"] / f"page_{page_number:03d}.png"
        json_path = output["json"] / f"page_{page_number:03d}.json"
        debug_path = output["debug"] / f"page_{page_number:03d}.json"

        module_dir = output["module_images"] / f"page_{page_number:03d}"
        module_dir.mkdir(parents=True, exist_ok=True)
        module_paths: list[str] = []
        for index, frame in enumerate(frames, start=1):
            module = make_module_image(rgb, binary, frame, cfg)
            module_path = module_dir / f"module_{index:04d}.png"
            save_rgb(module_path, module)
            module_paths.append(str(module_path))

        save_rgb(image_path, review)
        save_rgb(clean_path, clean)
        save_rgb(mask_path, mask_img)
        save_json(
            json_path,
            {
                "pdf": pdf_path.name,
                "page": int(page_number),
                "stage": STAGE_NAME,
                "source": {
                    "stage": PHASE1_STAGE,
                    "circuit_image_path": str(input_dir / f"page_{page_number:03d}.png"),
                },
                "image_width": int(rgb.shape[1]),
                "image_height": int(rgb.shape[0]),
                "review_image_path": str(image_path),
                "circuit_image_path": str(clean_path),
                "mask_image_path": str(mask_path),
                "module_image_paths": module_paths,
                "summary": {
                    "num_rects": int(len(frames)),
                    "rect_mask_pixels": int(rect_mask_image.sum()),
                },
                "rects": [rect_json(frame) for frame in frames],
            },
        )
        save_json(debug_path, debug)

        review_pages.append(review)
        result_pages.append(clean)
        summary_pages.append(
            {
                "page": int(page_number),
                "rect_count": int(len(frames)),
                "rect_mask_pixels": int(rect_mask_image.sum()),
                "review_image_path": f"images/page_{page_number:03d}.png",
                "circuit_image_path": f"circuit_images/page_{page_number:03d}.png",
                "mask_image_path": f"masks/page_{page_number:03d}.png",
                "module_image_dir": f"module_images/page_{page_number:03d}",
                "json_path": f"json/page_{page_number:03d}.json",
                "debug_path": f"debug/page_{page_number:03d}.json",
            }
        )
        print(f"    rects={len(frames)} mask_pixels={int(rect_mask_image.sum())}")

    if review_pages:
        image_to_pdf(review_pages, output["root"] / "review.pdf")
        image_to_pdf(result_pages, output["root"] / "result.pdf")
    save_json(
        output["json"] / "summary.json",
        {
            "pdf": pdf_path.name,
            "stage": STAGE_NAME,
            "input_stage": PHASE1_STAGE,
            "input_circuit_images": str(input_dir),
            "output_root": str(output["root"]),
            "pages": summary_pages,
            "config": asdict(cfg),
        },
    )
    print(f"Review PDF: {output['root'] / 'review.pdf'}")
    print(f"Result PDF: {output['root'] / 'result.pdf'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=str, default="bmw-328i-1997.pdf")
    p.add_argument("--pages", type=str, default="1-2")
    p.add_argument("-p", "--preserve", action="store_true", help="Preserve the output folder instead of clearing it first.")
    p.add_argument("--ink-threshold", type=int, default=210)
    p.add_argument("--min-side-length-px", type=int, default=32)
    p.add_argument("--max-side-thickness-px", type=int, default=12)
    p.add_argument("--corner-tolerance-px", type=int, default=8)
    p.add_argument("--edge-band-px", type=int, default=5)
    p.add_argument("--min-edge-coverage", type=float, default=0.58)
    p.add_argument("--max-edge-gap-px", type=int, default=20)
    p.add_argument("--bridge-gap-px", type=int, default=17)
    p.add_argument("--composite-bridge-gap-px", type=int, default=21)
    p.add_argument("--dedupe-iou", type=float, default=0.82)
    p.add_argument("--dedupe-inside-ratio", type=float, default=0.88)
    p.add_argument("--min-frame-area-px", type=int, default=1200)
    p.add_argument("--corner-probe-px", type=int, default=16)
    p.add_argument("--corner-required-coverage", type=float, default=0.45)
    p.add_argument("--corner-forbidden-coverage", type=float, default=0.65)
    p.add_argument("--corner-forbidden-min-active-px", type=int, default=7)
    p.add_argument("--edge-endpoint-tolerance-px", type=int, default=10)
    p.add_argument("--solid-edge-min-coverage", type=float, default=0.9)
    p.add_argument("--solid-edge-max-gap-px", type=int, default=3)
    p.add_argument("--dashed-edge-min-gaps", type=int, default=1)
    p.add_argument("--dashed-gap-tolerance-px", type=int, default=6)
    p.add_argument("--dashed-mask-min-side-coverage", type=float, default=0.62)
    p.add_argument("--dashed-mask-coverage-max-area-px", type=int, default=20000)
    p.add_argument("--dashed-terminal-stub-max-px", type=int, default=2)
    p.add_argument("--dashed-terminal-stub-max-count", type=int, default=2)
    p.add_argument("--dashed-terminal-stub-max-area-px", type=int, default=5000)
    p.add_argument("--dashed-edge-max-thick-candidates", type=int, default=1)
    p.add_argument("--edge-refine-search-px", type=int, default=10)
    p.add_argument("--refined-edge-min-span-ratio", type=float, default=0.95)
    p.add_argument("--corner-mask-extension-px", type=int, default=1)
    p.add_argument("--corner-supplement-normal-tolerance-px", type=int, default=3)
    p.add_argument("--corner-supplement-min-density", type=float, default=0.45)
    p.add_argument("--small-dashed-min-side-length-px", type=int, default=33)
    p.add_argument("--small-dashed-max-area-px", type=int, default=5000)
    p.add_argument("--local-adjust-min-area-px", type=int, default=1200)
    p.add_argument("--local-adjust-max-area-px", type=int, default=16000)
    p.add_argument("--local-adjust-max-side-px", type=int, default=180)
    p.add_argument("--local-adjust-min-coverage", type=float, default=0.55)
    p.add_argument("--local-adjust-min-span-ratio", type=float, default=0.78)
    p.add_argument("--min-candidate-span-ratio", type=float, default=0.55)
    p.add_argument("--side-search-min-tolerance-px", type=int, default=60)
    p.add_argument("--medium-recovery-min-area-px", type=int, default=8000)
    p.add_argument("--medium-recovery-max-area-px", type=int, default=180000)
    p.add_argument("--medium-recovery-min-side-px", type=int, default=50)
    p.add_argument("--medium-recovery-max-side-px", type=int, default=900)
    p.add_argument("--medium-recovery-min-coverage", type=float, default=0.48)
    p.add_argument("--medium-recovery-min-span-ratio", type=float, default=0.72)
    p.add_argument("--medium-recovery-max-gap-px", type=int, default=34)
    p.add_argument("--medium-recovery-endpoint-slack-px", type=int, default=28)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    print("Page range:", "ALL" if args.pages is None else args.pages)
    if args.pdf is not None:
        process_pdf(resolve_pdf(args.pdf), args)
        return

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF found under:\n{INPUT_DIR}")
    for pdf_path in pdf_files:
        process_pdf(pdf_path, args)


if __name__ == "__main__":
    main()
