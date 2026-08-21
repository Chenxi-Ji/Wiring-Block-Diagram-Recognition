import argparse
import shutil
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from pipeline_io import image_to_pdf, parse_page_range, render_pdf_page, save_json


DEFAULT_DPI = 300
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / Path(__file__).stem


# ============================================================
# IO / PDF utils
# ============================================================


# ============================================================
# Basic helpers
# ============================================================

def binarize_page(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray < threshold


def make_binary_output_image(img, threshold):
    binary = binarize_page(img, threshold)
    gray = np.where(binary, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def clip_box(box, w, h):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def expand_box(box, pad, w, h):
    x1, y1, x2, y2 = box
    return clip_box([x1 - pad, y1 - pad, x2 + pad, y2 + pad], w, h)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def union_box(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def merge_adjacent_line_boxes(boxes, max_gap, min_y_overlap_ratio=0.45):
    boxes = sorted([b for b in boxes if b is not None], key=lambda b: (b[1], b[0]))
    merged = []
    for box in boxes:
        placed = False
        for i, cur in enumerate(merged):
            y_overlap = _overlap_len(box[1], box[3], cur[1], cur[3])
            min_h = max(1, min(box[3] - box[1], cur[3] - cur[1]))
            gap = max(0, max(box[0], cur[0]) - min(box[2], cur[2]))
            if y_overlap >= min_y_overlap_ratio * min_h and gap <= max_gap:
                merged[i] = union_box([cur, box])
                placed = True
                break
        if not placed:
            merged.append(box)
    return sorted(merged, key=lambda b: (b[1], b[0]))


def mask_to_runs(mask):
    runs = []
    ys = np.where(mask.any(axis=1))[0]
    for y in ys:
        xs = np.where(mask[y])[0]
        if len(xs) == 0:
            continue
        start = int(xs[0])
        prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                runs.append([int(y), start, prev + 1])
                start = x
                prev = x
        runs.append([int(y), start, prev + 1])
    return runs


def get_components(binary):
    num_labels, label_img, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    comps = []
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        if area <= 0:
            continue
        bbox_area = max(1, int(bw) * int(bh))
        comps.append({
            "label_id": int(label_id),
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "width": int(bw),
            "height": int(bh),
            "area": int(area),
            "density": float(area) / float(bbox_area),
            "cx": float(cents[label_id][0]),
            "cy": float(cents[label_id][1]),
        })
    return comps, label_img


def draw_mask_rgb(img, mask, color, alpha):
    out = img.copy()
    if not mask.any():
        return out
    color_arr = np.array(color, dtype=np.float32)
    pix = out[mask].astype(np.float32)
    out[mask] = np.clip((1.0 - alpha) * pix + alpha * color_arr, 0, 255).astype(np.uint8)
    return out


# ============================================================
# Line extraction for outer frame
# ============================================================

def _line_debug(line):
    return {
        "bbox": [int(v) for v in line["bbox"]],
        "orientation": line["orientation"],
        "length": int(line["length"]),
        "thickness": int(line["thickness"]),
        "center": [float(line["cx"]), float(line["cy"])],
    }


def extract_axis_lines(binary, args, debug=None):
    h, w = binary.shape
    src = binary.astype(np.uint8)

    # Close first, then open.  This bridges small breaks in faint/unclear frame edges,
    # but the later long-line filters still reject text and small components.
    bridge = max(1, int(getattr(args, "frame_bridge_gap", 1)))
    if bridge > 1:
        h_src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)))
        v_src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)))
    else:
        h_src, v_src = src, src

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.frame_open_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, args.frame_open_kernel_len))
    h_mask = cv2.morphologyEx(h_src, cv2.MORPH_OPEN, h_kernel).astype(bool)
    v_mask = cv2.morphologyEx(v_src, cv2.MORPH_OPEN, v_kernel).astype(bool)

    lines = []
    rejected = [] if debug is not None else None
    for cur_mask, orientation in [(h_mask, "horizontal"), (v_mask, "vertical")]:
        comps, label_img = get_components(cur_mask)
        for c in comps:
            x1, y1, x2, y2 = c["bbox"]
            bw, bh = x2 - x1, y2 - y1
            length, thickness = (bw, bh) if orientation == "horizontal" else (bh, bw)
            reasons = []
            if thickness < args.frame_min_thickness or thickness > args.frame_max_thickness:
                if thickness < args.frame_min_thickness:
                    reasons.append("thickness_below_min")
                else:
                    reasons.append("thickness_above_max")
            if length < args.frame_open_kernel_len:
                reasons.append("length_below_open_kernel")
            if reasons:
                if rejected is not None:
                    rejected.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "orientation": orientation,
                        "length": int(length),
                        "thickness": int(thickness),
                        "component_area": int(c["area"]),
                        "reasons": reasons,
                    })
                continue
            lines.append({
                "bbox": [x1, y1, x2, y2],
                "orientation": orientation,
                "length": int(length),
                "thickness": int(thickness),
                "cx": 0.5 * (x1 + x2),
                "cy": 0.5 * (y1 + y2),
                "mask": label_img == c["label_id"],
            })
    if debug is not None:
        debug["accepted_axis_lines"] = [_line_debug(line) for line in lines]
        debug["rejected_axis_lines"] = rejected
    return lines


def line_to_json(line):
    if line is None:
        return None
    data = {
        "bbox": [int(v) for v in line["bbox"]],
        "orientation": line["orientation"],
        "length": int(line["length"]),
        "thickness": int(line["thickness"]),
        "center": [float(line["cx"]), float(line["cy"])],
    }
    if "inferred" in line:
        data["inferred"] = bool(line["inferred"])
    if "adjusted" in line:
        data["adjusted"] = bool(line["adjusted"])
    if "source_bbox" in line:
        data["source_bbox"] = [int(v) for v in line["source_bbox"]]
    if "erase_bbox" in line:
        data["erase_bbox"] = [int(v) for v in line["erase_bbox"]]
    if "inference" in line:
        data["inference"] = line["inference"]
    return data


def alignment_error(top, bottom, left, right):
    tx1, ty1, tx2, ty2 = top["bbox"]
    bx1, by1, bx2, by2 = bottom["bbox"]
    lx1, ly1, lx2, ly2 = left["bbox"]
    rx1, ry1, rx2, ry2 = right["bbox"]

    left_x = 0.5 * (lx1 + lx2)
    right_x = 0.5 * (rx1 + rx2)
    top_y = 0.5 * (ty1 + ty2)
    bottom_y = 0.5 * (by1 + by2)

    err = 0.0
    err += abs(tx1 - left_x) + abs(bx1 - left_x)
    err += abs(tx2 - right_x) + abs(bx2 - right_x)
    err += abs(ly1 - top_y) + abs(ry1 - top_y)
    err += abs(ly2 - bottom_y) + abs(ry2 - bottom_y)
    return float(err)


def rectangle_from_sides(top, bottom, left, right, w, h):
    x1 = int(round(np.median([top["bbox"][0], bottom["bbox"][0], left["cx"]])))
    x2 = int(round(np.median([top["bbox"][2], bottom["bbox"][2], right["cx"]])))
    y1 = int(round(np.median([top["cy"], left["bbox"][1], right["bbox"][1]])))
    y2 = int(round(np.median([bottom["cy"], left["bbox"][3], right["bbox"][3]])))
    return clip_box([x1, y1, x2, y2], w, h)


def _overlap_len(a1, a2, b1, b2):
    return max(0.0, min(float(a2), float(b2)) - max(float(a1), float(b1)))


def _overlap_area(a, b):
    return _overlap_len(a[0], a[2], b[0], b[2]) * _overlap_len(a[1], a[3], b[1], b[3])


def _box_iou(a, b):
    inter = _overlap_area(a, b)
    den = box_area(a) + box_area(b) - inter
    return 0.0 if den <= 0 else inter / den


def _inside_ratio(a, b):
    return 0.0 if box_area(a) <= 0 else _overlap_area(a, b) / float(box_area(a))


def _aligned_edge_count(a, b, tol):
    return sum([
        abs(a[0] - b[0]) <= tol,
        abs(a[1] - b[1]) <= tol,
        abs(a[2] - b[2]) <= tol,
        abs(a[3] - b[3]) <= tol,
    ])


def _frame_candidate_debug(cand):
    if cand is None:
        return None
    sides = {}
    for side in ["top", "bottom", "left", "right"]:
        line = cand.get(side)
        sides[side] = None if line is None else line_to_json(line)
    return {
        "bbox": [int(v) for v in cand["bbox"]],
        "kind": str(cand.get("kind", "closed")),
        "area_ratio": float(cand.get("area_ratio", 0.0)),
        "score": float(cand.get("score", 0.0)),
        "alignment_error": float(cand.get("alignment_error", 0.0)),
        "partial": bool(cand.get("partial", False)),
        "fallback": bool(cand.get("fallback", False)),
        "sides": sides,
        "calibration": cand.get("calibration"),
    }


def _rejected_frame_debug(cand, reasons):
    return {
        **_frame_candidate_debug(cand),
        "reasons": list(reasons),
    }


def _intersects_any_frame(box, frame):
    if frame is None or box is None:
        return False
    return any(_overlap_area(box, fr["bbox"]) > 0 for fr in _frame_parts(frame))


def _clip_info_box_to_frame_side(box, fr, side, w, h, args):
    if box is None:
        return None
    x1, y1, x2, y2 = fr["bbox"]
    b = list(box)
    if side == "top":
        b[3] = min(b[3], y1 - args.info_outside_tol)
    elif side == "bottom":
        b[1] = max(b[1], y2 + args.info_bl_outside_gap)
    elif side == "left":
        b[2] = min(b[2], x1 - args.info_outside_tol)
    return clip_box(b, w, h)


def _partial_alignment_error(rect, top=None, bottom=None, left=None, right=None):
    x1, y1, x2, y2 = rect
    err, n = 0.0, 0
    if top is not None:
        err += abs(top["bbox"][0] - x1) + abs(top["bbox"][2] - x2) + abs(top["cy"] - y1); n += 3
    if bottom is not None:
        err += abs(bottom["bbox"][0] - x1) + abs(bottom["bbox"][2] - x2) + abs(bottom["cy"] - y2); n += 3
    if left is not None:
        err += abs(left["cx"] - x1) + abs(left["bbox"][1] - y1) + abs(left["bbox"][3] - y2); n += 3
    if right is not None:
        err += abs(right["cx"] - x2) + abs(right["bbox"][1] - y1) + abs(right["bbox"][3] - y2); n += 3
    return err / max(1, n)


def _rect_from_sides(top, bottom, left, right, w, h):
    xs1, xs2, ys1, ys2 = [], [], [], []
    if top is not None:
        xs1.append(top["bbox"][0]); xs2.append(top["bbox"][2]); ys1.append(top["cy"])
    if bottom is not None:
        xs1.append(bottom["bbox"][0]); xs2.append(bottom["bbox"][2]); ys2.append(bottom["cy"])
    if left is not None:
        xs1.append(left["cx"]); ys1.append(left["bbox"][1]); ys2.append(left["bbox"][3])
    if right is not None:
        xs2.append(right["cx"]); ys1.append(right["bbox"][1]); ys2.append(right["bbox"][3])
    if not xs1 or not xs2 or not ys1 or not ys2:
        return None

    x1 = left["cx"] if left is not None else np.median(xs1)
    x2 = right["cx"] if right is not None else np.median(xs2)
    y1 = top["cy"] if top is not None else np.median(ys1)
    y2 = bottom["cy"] if bottom is not None else np.median(ys2)
    return clip_box([x1, y1, x2, y2], w, h)


def _make_frame_candidate(binary, args, top=None, bottom=None, left=None, right=None, kind="partial", min_side_count=None):
    h, w = binary.shape
    side_count = sum(x is not None for x in [top, bottom, left, right])
    min_side_count = getattr(args, "frame_min_side_count", 3) if min_side_count is None else min_side_count
    if side_count < min_side_count:
        return None
    rect = _rect_from_sides(top, bottom, left, right, w, h)
    if rect is None:
        return None
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    area_ratio = (rw * rh) / float(w * h)
    min_wr = getattr(args, "frame_min_rect_width_ratio", args.frame_min_width_ratio)
    min_hr = getattr(args, "frame_min_rect_height_ratio", args.frame_min_height_ratio)
    if rw < min_wr * w or rh < min_hr * h:
        return None
    if area_ratio < args.frame_min_area_ratio:
        return None
    err = _partial_alignment_error(rect, top, bottom, left, right)
    tol = getattr(args, "frame_partial_alignment_tol", args.frame_alignment_tol * 4.0)
    if err > tol:
        return None
    return {
        "bbox": rect,
        "score": float(1000.0 * area_ratio + 80.0 * side_count - err),
        "area_ratio": float(area_ratio),
        "alignment_error": float(err),
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "partial": side_count < 4,
        "kind": kind,
    }


def _median_side_thickness(frame, args):
    vals = [
        int(frame[side]["thickness"])
        for side in ["top", "bottom", "left", "right"]
        if frame.get(side) is not None
    ]
    if vals:
        return max(1, int(round(float(np.median(vals)))))
    return max(1, int(getattr(args, "frame_inferred_thickness", 2)))


def _make_calibrated_line(side, x1, y1, x2, y2, thickness, source=None, inference=None):
    if source is not None:
        thickness = max(1, int(source["thickness"]))
    else:
        thickness = max(1, int(round(thickness)))
    half = thickness // 2
    extra = thickness - half
    inferred = source is None
    if side in ["top", "bottom"]:
        bbox = [int(round(x1)), int(round(y1)) - half, int(round(x2)), int(round(y1)) + extra]
        orientation = "horizontal"
        length = max(0, bbox[2] - bbox[0])
    else:
        bbox = [int(round(x1)) - half, int(round(y1)), int(round(x1)) + extra, int(round(y2))]
        orientation = "vertical"
        length = max(0, bbox[3] - bbox[1])
    line = {
        "bbox": bbox,
        "orientation": orientation,
        "length": int(length),
        "thickness": int(thickness),
        "cx": 0.5 * (bbox[0] + bbox[2]),
        "cy": 0.5 * (bbox[1] + bbox[3]),
        "inferred": inferred,
        "adjusted": True,
    }
    if source is not None:
        line["source_bbox"] = [int(v) for v in source["bbox"]]
        line["source_center"] = [float(source["cx"]), float(source["cy"])]
        line["erase_bbox"] = [int(v) for v in source["bbox"]]
    if inference is not None:
        line["inference"] = inference
    return line


def _trim_existing_edge_erase_bboxes_to_corners(frame, w, h):
    sides = {side: frame.get(side) for side in ["top", "bottom", "left", "right"]}
    if any(line is None for line in sides.values()):
        return

    top = sides["top"]
    bottom = sides["bottom"]
    left = sides["left"]
    right = sides["right"]

    horizontal_min_x = min(left["bbox"][0], right["bbox"][0])
    horizontal_max_x = max(left["bbox"][2], right["bbox"][2])
    vertical_min_y = min(top["bbox"][1], bottom["bbox"][1])
    vertical_max_y = max(top["bbox"][3], bottom["bbox"][3])

    for side, line in sides.items():
        if line.get("inferred", False) or "erase_bbox" not in line:
            continue
        x1, y1, x2, y2 = line["erase_bbox"]
        if side in ["top", "bottom"]:
            clipped = [max(x1, horizontal_min_x), y1, min(x2, horizontal_max_x), y2]
        else:
            clipped = [x1, max(y1, vertical_min_y), x2, min(y2, vertical_max_y)]
        box = clip_box(clipped, w, h)
        if box is not None:
            line["erase_bbox"] = box


def _edge_band_box(side, coord, lo, hi, thickness, w, h, corner_ignore):
    thickness = max(1, int(round(thickness)))
    half = thickness // 2
    extra = thickness - half
    lo = int(round(lo))
    hi = int(round(hi))
    corner_ignore = max(0, int(corner_ignore))
    if side in ["top", "bottom"]:
        x1 = min(w, max(0, lo + corner_ignore))
        x2 = min(w, max(0, hi - corner_ignore))
        y1 = min(h, max(0, int(round(coord)) - half))
        y2 = min(h, max(0, int(round(coord)) + extra))
    else:
        x1 = min(w, max(0, int(round(coord)) - half))
        x2 = min(w, max(0, int(round(coord)) + extra))
        y1 = min(h, max(0, lo + corner_ignore))
        y2 = min(h, max(0, hi - corner_ignore))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _edge_has_ink(binary, side, coord, lo, hi, thickness, args):
    h, w = binary.shape
    corner_ignore = getattr(args, "frame_infer_corner_ignore_px", 18)
    band = _edge_band_box(side, coord, lo, hi, thickness, w, h, corner_ignore)
    if band is None:
        return False, None, 0
    x1, y1, x2, y2 = band
    ink_count = int(binary[y1:y2, x1:x2].sum())
    return ink_count > 0, band, ink_count


def _edge_cleanest_thickness(binary, side, coord, lo, hi, preferred_thickness, args):
    preferred_thickness = max(1, int(round(preferred_thickness)))
    attempts = []
    for thickness in range(preferred_thickness, 0, -1):
        has_ink, band, ink_count = _edge_has_ink(binary, side, coord, lo, hi, thickness, args)
        attempts.append({
            "thickness": int(thickness),
            "ink_count": int(ink_count),
            "band": band,
        })
        if not has_ink:
            return int(thickness), band, 0, attempts
    return None, attempts[-1]["band"] if attempts else None, attempts[-1]["ink_count"] if attempts else 0, attempts


def _find_clean_inferred_edge(binary, side, initial_coord, lo, hi, thickness, span_len, args):
    h, w = binary.shape
    max_inward = max(0, int(round(span_len / 3.0)))
    if side == "bottom":
        inward_step, outward_step, min_coord, max_coord = -1, 1, 0, h - 1
    elif side == "top":
        inward_step, outward_step, min_coord, max_coord = 1, -1, 0, h - 1
    elif side == "right":
        inward_step, outward_step, min_coord, max_coord = -1, 1, 0, w - 1
    else:
        inward_step, outward_step, min_coord, max_coord = 1, -1, 0, w - 1

    initial_coord = int(round(initial_coord))
    attempts = []
    preferred_thickness = max(1, int(round(thickness)))

    for dist in range(0, max_inward + 1):
        coord = initial_coord + inward_step * dist
        if coord < min_coord or coord > max_coord:
            break
        clean_thickness, band, ink_count, thickness_attempts = _edge_cleanest_thickness(
            binary, side, coord, lo, hi, preferred_thickness, args
        )
        if len(attempts) < 12:
            attempts.append({
                "coord": int(coord),
                "direction": "inward",
                "ink_count": int(ink_count),
                "band": band,
                "thickness_attempts": thickness_attempts,
            })
        if clean_thickness is not None:
            return int(coord), int(clean_thickness), {
                "initial_coord": int(initial_coord),
                "final_coord": int(coord),
                "preferred_thickness": int(preferred_thickness),
                "final_thickness": int(clean_thickness),
                "search_result": "clean_inward",
                "shift_pixels": int(coord - initial_coord),
                "max_inward_pixels": int(max_inward),
                "sampled_attempts": attempts,
            }

    dist = 1
    while True:
        coord = initial_coord + outward_step * dist
        if coord < min_coord or coord > max_coord:
            break
        clean_thickness, band, ink_count, thickness_attempts = _edge_cleanest_thickness(
            binary, side, coord, lo, hi, preferred_thickness, args
        )
        if len(attempts) < 24:
            attempts.append({
                "coord": int(coord),
                "direction": "outward",
                "ink_count": int(ink_count),
                "band": band,
                "thickness_attempts": thickness_attempts,
            })
        if clean_thickness is not None:
            return int(coord), int(clean_thickness), {
                "initial_coord": int(initial_coord),
                "final_coord": int(coord),
                "preferred_thickness": int(preferred_thickness),
                "final_thickness": int(clean_thickness),
                "search_result": "clean_outward",
                "shift_pixels": int(coord - initial_coord),
                "max_inward_pixels": int(max_inward),
                "sampled_attempts": attempts,
            }
        dist += 1

    return int(initial_coord), 1, {
        "initial_coord": int(initial_coord),
        "final_coord": int(initial_coord),
        "preferred_thickness": int(preferred_thickness),
        "final_thickness": 1,
        "search_result": "no_clean_position_found",
        "shift_pixels": 0,
        "max_inward_pixels": int(max_inward),
        "sampled_attempts": attempts,
    }


def _estimate_frame_coords(frame, w, h):
    top, bottom, left, right = frame.get("top"), frame.get("bottom"), frame.get("left"), frame.get("right")
    rect = frame.get("bbox")
    x1 = float(rect[0]) if rect is not None else 0.0
    y1 = float(rect[1]) if rect is not None else 0.0
    x2 = float(rect[2]) if rect is not None else float(w)
    y2 = float(rect[3]) if rect is not None else float(h)

    if left is not None:
        x1 = float(left["cx"])
    elif top is not None and bottom is not None:
        x1 = float(np.median([top["bbox"][0], bottom["bbox"][0]]))

    if right is not None:
        x2 = float(right["cx"])
    elif top is not None and bottom is not None:
        x2 = float(np.median([top["bbox"][2], bottom["bbox"][2]]))

    if top is not None:
        y1 = float(top["cy"])
    elif left is not None and right is not None:
        y1 = float(np.median([left["bbox"][1], right["bbox"][1]]))

    if bottom is not None:
        y2 = float(bottom["cy"])
    elif left is not None and right is not None:
        y2 = float(np.median([left["bbox"][3], right["bbox"][3]]))

    return clip_box([x1, y1, x2, y2], w, h)


def calibrate_frame_edges(binary, frame, args):
    if frame is None:
        return None
    h, w = binary.shape
    calibrated = frame.copy()
    rect = _estimate_frame_coords(frame, w, h)
    if rect is None:
        return frame

    x1, y1, x2, y2 = rect
    inferred_thickness = _median_side_thickness(frame, args)
    side_thickness = {
        side: int(frame[side]["thickness"]) if frame.get(side) is not None else int(inferred_thickness)
        for side in ["top", "bottom", "left", "right"]
    }
    calibration = {
        "initial_bbox": [int(v) for v in rect],
        "preferred_inferred_thickness": int(inferred_thickness),
        "inferred_edges": {},
        "corner_aligned": True,
    }

    if frame.get("top") is None:
        y1, found_thickness, info = _find_clean_inferred_edge(
            binary, "top", y1, x1, x2, side_thickness["top"], y2 - y1, args
        )
        side_thickness["top"] = found_thickness
        calibration["inferred_edges"]["top"] = info
    if frame.get("bottom") is None:
        y2, found_thickness, info = _find_clean_inferred_edge(
            binary, "bottom", y2, x1, x2, side_thickness["bottom"], y2 - y1, args
        )
        side_thickness["bottom"] = found_thickness
        calibration["inferred_edges"]["bottom"] = info
    if frame.get("left") is None:
        x1, found_thickness, info = _find_clean_inferred_edge(
            binary, "left", x1, y1, y2, side_thickness["left"], x2 - x1, args
        )
        side_thickness["left"] = found_thickness
        calibration["inferred_edges"]["left"] = info
    if frame.get("right") is None:
        x2, found_thickness, info = _find_clean_inferred_edge(
            binary, "right", x2, y1, y2, side_thickness["right"], x2 - x1, args
        )
        side_thickness["right"] = found_thickness
        calibration["inferred_edges"]["right"] = info

    rect = clip_box([x1, y1, x2, y2], w, h)
    if rect is None:
        calibration["failed"] = "invalid_calibrated_bbox"
        calibrated["calibration"] = calibration
        return calibrated
    x1, y1, x2, y2 = rect
    calibration["final_bbox"] = [int(v) for v in rect]

    source_top = frame.get("top")
    source_bottom = frame.get("bottom")
    source_left = frame.get("left")
    source_right = frame.get("right")
    calibrated["top"] = _make_calibrated_line("top", x1, y1, x2, y1, side_thickness["top"], source=source_top, inference=calibration["inferred_edges"].get("top"))
    calibrated["bottom"] = _make_calibrated_line("bottom", x1, y2, x2, y2, side_thickness["bottom"], source=source_bottom, inference=calibration["inferred_edges"].get("bottom"))
    calibrated["left"] = _make_calibrated_line("left", x1, y1, x1, y2, side_thickness["left"], source=source_left, inference=calibration["inferred_edges"].get("left"))
    calibrated["right"] = _make_calibrated_line("right", x2, y1, x2, y2, side_thickness["right"], source=source_right, inference=calibration["inferred_edges"].get("right"))
    _trim_existing_edge_erase_bboxes_to_corners(calibrated, w, h)
    calibrated["bbox"] = [int(v) for v in rect]
    calibrated["area_ratio"] = float(box_area(rect) / float(w * h))
    calibrated["partial"] = any(frame.get(side) is None for side in ["top", "bottom", "left", "right"])
    calibration["line_thickness_by_side"] = {side: int(v) for side, v in side_thickness.items()}
    calibrated["calibration"] = calibration
    return calibrated


def calibrate_frame_tree(binary, frame, args):
    if frame is None:
        return None
    if "frames" not in frame:
        return calibrate_frame_edges(binary, frame, args)
    frames = [calibrate_frame_edges(binary, fr, args) for fr in frame["frames"]]
    frames = [fr for fr in frames if fr is not None]
    if not frames:
        return frame
    primary = max(frames, key=lambda z: box_area(z["bbox"])).copy()
    primary["frames"] = frames
    primary["multi_frame"] = True
    primary["bbox"] = union_box([z["bbox"] for z in frames])
    h, w = binary.shape
    primary["area_ratio"] = float(box_area(primary["bbox"]) / float(w * h))
    primary["score"] = float(sum(z.get("score", 0.0) for z in frames))
    primary["kind"] = "multi_frame"
    return primary


def _select_frame_candidates(candidates, args, debug=None):
    selected = []
    # Important: when a small false frame is fully inside a real big frame,
    # always keep the big one.  Therefore containment replacement is handled
    # explicitly instead of blindly keeping the highest-score candidate first.
    for c in sorted(candidates, key=lambda x: (box_area(x["bbox"]), x["score"]), reverse=True):
        b = c["bbox"]
        keep = True
        k = 0
        while k < len(selected):
            s = selected[k]
            sb = s["bbox"]
            b_in_s = _inside_ratio(b, sb)
            s_in_b = _inside_ratio(sb, b)
            b_sides = sum(c.get(side) is not None for side in ["top", "bottom", "left", "right"])
            s_sides = sum(s.get(side) is not None for side in ["top", "bottom", "left", "right"])
            if b_in_s > args.frame_select_inside_thresh and box_area(b) <= box_area(sb):
                area_ratio = box_area(b) / float(max(1, box_area(sb)))
                shared_edges = _aligned_edge_count(b, sb, args.frame_select_refine_edge_tol)
                if (
                    area_ratio >= args.frame_select_refine_area_ratio
                    and b_sides > s_sides
                    and shared_edges >= args.frame_select_refine_min_edges
                ):
                    if debug is not None:
                        debug["rejected_frame_candidates"].append(_rejected_frame_debug(
                            s,
                            ["replaced_by_more_complete_contained_candidate"],
                        ))
                    selected.pop(k)
                    continue
                if debug is not None:
                    debug["rejected_frame_candidates"].append(_rejected_frame_debug(
                        c,
                        ["inside_larger_selected_frame"],
                    ))
                keep = False
                break
            if s_in_b > args.frame_select_inside_thresh and box_area(b) > box_area(sb):
                if debug is not None:
                    debug["rejected_frame_candidates"].append(_rejected_frame_debug(
                        s,
                        ["contained_by_larger_candidate"],
                    ))
                selected.pop(k)
                continue
            if _box_iou(b, sb) > args.frame_select_iou_thresh:
                if b_sides > s_sides or (b_sides == s_sides and c["score"] > s["score"]):
                    if debug is not None:
                        debug["rejected_frame_candidates"].append(_rejected_frame_debug(
                            s,
                            ["overlapped_by_higher_score_candidate"],
                        ))
                    selected.pop(k)
                    continue
                if debug is not None:
                    debug["rejected_frame_candidates"].append(_rejected_frame_debug(
                        c,
                        ["overlaps_selected_frame_with_lower_score"],
                    ))
                keep = False
                break
            k += 1
        if keep:
            selected.append(c)
    return selected[:args.frame_max_boxes]


def find_outer_frame(binary, args):
    h, w = binary.shape
    debug = {
        "stage": "frame_detection",
        "accepted_axis_lines": [],
        "rejected_axis_lines": [],
        "rejected_frame_lines": [],
        "accepted_frame_candidates": [],
        "rejected_frame_candidates": [],
        "selected_frames": [],
        "notes": [
            "Closed, missing-edge, L-shaped, and fallback frame candidates are accepted.",
        ],
    }
    lines = extract_axis_lines(binary, args, debug=debug)

    min_lw = getattr(args, "frame_min_line_width_ratio", args.frame_min_width_ratio)
    min_lh = getattr(args, "frame_min_line_height_ratio", args.frame_min_height_ratio)
    h_lines = [ln for ln in lines if ln["orientation"] == "horizontal" and ln["length"] >= min_lw * w]
    v_lines = [ln for ln in lines if ln["orientation"] == "vertical" and ln["length"] >= min_lh * h]

    for ln in lines:
        reasons = []
        if ln["orientation"] == "horizontal" and ln["length"] < min_lw * w:
            reasons.append("horizontal_length_below_frame_ratio")
        if ln["orientation"] == "vertical" and ln["length"] < min_lh * h:
            reasons.append("vertical_length_below_frame_ratio")
        if reasons:
            debug["rejected_frame_lines"].append({
                **_line_debug(ln),
                "reasons": reasons,
            })

    candidates = []

    def add_candidate(top=None, bottom=None, left=None, right=None, kind="partial", min_side_count=None, score_bonus=0.0, reason_context=None):
        cand = _make_frame_candidate(binary, args, top, bottom, left, right, kind, min_side_count=min_side_count)
        if cand is None:
            debug["rejected_frame_candidates"].append({
                "kind": kind,
                "sides": {
                    "top": line_to_json(top),
                    "bottom": line_to_json(bottom),
                    "left": line_to_json(left),
                    "right": line_to_json(right),
                },
                "reasons": ["candidate_validation_failed"],
                "context": reason_context,
            })
            return None
        if score_bonus:
            cand["score"] += float(score_bonus)
        candidates.append(cand)
        return cand

    # Case A: normal closed rectangle.
    for i, top0 in enumerate(h_lines):
        for bottom0 in h_lines[i + 1:]:
            top, bottom = (top0, bottom0) if top0["cy"] <= bottom0["cy"] else (bottom0, top0)
            if bottom["cy"] - top["cy"] < args.frame_min_height_ratio * h:
                debug["rejected_frame_candidates"].append({
                    "kind": "closed",
                    "sides": {
                        "top": line_to_json(top),
                        "bottom": line_to_json(bottom),
                        "left": None,
                        "right": None,
                    },
                    "reasons": ["horizontal_pair_height_below_min_ratio"],
                })
                continue
            for j, left0 in enumerate(v_lines):
                for right0 in v_lines[j + 1:]:
                    left, right = (left0, right0) if left0["cx"] <= right0["cx"] else (right0, left0)
                    if right["cx"] - left["cx"] < args.frame_min_width_ratio * w:
                        debug["rejected_frame_candidates"].append({
                            "kind": "closed",
                            "sides": {
                                "top": line_to_json(top),
                                "bottom": line_to_json(bottom),
                                "left": line_to_json(left),
                                "right": line_to_json(right),
                            },
                            "reasons": ["vertical_pair_width_below_min_ratio"],
                        })
                        continue
                    rect = rectangle_from_sides(top, bottom, left, right, w, h)
                    if rect is None:
                        debug["rejected_frame_candidates"].append({
                            "kind": "closed",
                            "sides": {
                                "top": line_to_json(top),
                                "bottom": line_to_json(bottom),
                                "left": line_to_json(left),
                                "right": line_to_json(right),
                            },
                            "reasons": ["invalid_rectangle_from_sides"],
                        })
                        continue
                    area_ratio = box_area(rect) / float(w * h)
                    if area_ratio < args.frame_min_area_ratio:
                        debug["rejected_frame_candidates"].append({
                            "bbox": [int(v) for v in rect],
                            "kind": "closed",
                            "area_ratio": float(area_ratio),
                            "sides": {
                                "top": line_to_json(top),
                                "bottom": line_to_json(bottom),
                                "left": line_to_json(left),
                                "right": line_to_json(right),
                            },
                            "reasons": ["area_ratio_below_min"],
                        })
                        continue
                    err = alignment_error(top, bottom, left, right)
                    if err > args.frame_alignment_tol * 8:
                        debug["rejected_frame_candidates"].append({
                            "bbox": [int(v) for v in rect],
                            "kind": "closed",
                            "area_ratio": float(area_ratio),
                            "alignment_error": float(err),
                            "sides": {
                                "top": line_to_json(top),
                                "bottom": line_to_json(bottom),
                                "left": line_to_json(left),
                                "right": line_to_json(right),
                            },
                            "reasons": ["alignment_error_above_closed_tolerance"],
                        })
                        continue
                    cand = _make_frame_candidate(binary, args, top, bottom, left, right, "closed")
                    if cand is not None:
                        cand["alignment_error"] = float(err)
                        cand["score"] = float(1000.0 * cand["area_ratio"] + 320.0 - err)
                        candidates.append(cand)
                    else:
                        debug["rejected_frame_candidates"].append({
                            "bbox": [int(v) for v in rect],
                            "kind": "closed",
                            "area_ratio": float(area_ratio),
                            "alignment_error": float(err),
                            "sides": {
                                "top": line_to_json(top),
                                "bottom": line_to_json(bottom),
                                "left": line_to_json(left),
                                "right": line_to_json(right),
                            },
                            "reasons": ["candidate_validation_failed"],
                        })

    # Case B: one horizontal edge is missing/unclear; infer it from a pair of vertical sides.
    min_x_overlap_ratio = getattr(args, "frame_partial_overlap_ratio", 0.55)
    for j, left0 in enumerate(v_lines):
        for right0 in v_lines[j + 1:]:
            left, right = (left0, right0) if left0["cx"] <= right0["cx"] else (right0, left0)
            x1, x2 = left["cx"], right["cx"]
            if x2 - x1 < args.frame_min_width_ratio * w:
                debug["rejected_frame_candidates"].append({
                    "kind": "missing_horizontal_edge",
                    "sides": {
                        "top": None,
                        "bottom": None,
                        "left": line_to_json(left),
                        "right": line_to_json(right),
                    },
                    "reasons": ["vertical_pair_width_below_min_ratio"],
                })
                continue
            span_w = x2 - x1
            hs = [ln for ln in h_lines if _overlap_len(ln["bbox"][0], ln["bbox"][2], x1, x2) >= min_x_overlap_ratio * span_w]
            if not hs:
                debug["rejected_frame_candidates"].append({
                    "kind": "missing_horizontal_edge",
                    "sides": {
                        "top": None,
                        "bottom": None,
                        "left": line_to_json(left),
                        "right": line_to_json(right),
                    },
                    "reasons": ["no_horizontal_line_overlaps_vertical_span"],
                })
            for horizontal in hs:
                matched = False
                if abs(horizontal["cy"] - min(left["bbox"][1], right["bbox"][1])) <= args.frame_partial_alignment_tol:
                    add_candidate(
                        top=horizontal,
                        left=left,
                        right=right,
                        kind="missing_bottom",
                        reason_context="top line aligns with vertical starts",
                    )
                    matched = True
                if abs(horizontal["cy"] - max(left["bbox"][3], right["bbox"][3])) <= args.frame_partial_alignment_tol:
                    add_candidate(
                        bottom=horizontal,
                        left=left,
                        right=right,
                        kind="missing_top",
                        reason_context="bottom line aligns with vertical ends",
                    )
                    matched = True
                if not matched:
                    debug["rejected_frame_candidates"].append({
                        "kind": "missing_horizontal_edge",
                        "sides": {
                            "top": line_to_json(horizontal),
                            "bottom": None,
                            "left": line_to_json(left),
                            "right": line_to_json(right),
                        },
                        "reasons": ["horizontal_line_not_aligned_to_vertical_pair"],
                    })

    # Case C: one vertical edge is missing/unclear; infer it from top/bottom horizontal sides.
    min_y_overlap_ratio = getattr(args, "frame_partial_overlap_ratio", 0.55)
    for i, top0 in enumerate(h_lines):
        for bottom0 in h_lines[i + 1:]:
            top, bottom = (top0, bottom0) if top0["cy"] <= bottom0["cy"] else (bottom0, top0)
            if bottom["cy"] - top["cy"] < args.frame_min_height_ratio * h:
                debug["rejected_frame_candidates"].append({
                    "kind": "missing_vertical_edge",
                    "sides": {
                        "top": line_to_json(top),
                        "bottom": line_to_json(bottom),
                        "left": None,
                        "right": None,
                    },
                    "reasons": ["horizontal_pair_height_below_min_ratio"],
                })
                continue
            x1 = np.median([top["bbox"][0], bottom["bbox"][0]])
            x2 = np.median([top["bbox"][2], bottom["bbox"][2]])
            if x2 - x1 < args.frame_min_width_ratio * w:
                debug["rejected_frame_candidates"].append({
                    "kind": "missing_vertical_edge",
                    "bbox": [int(round(x1)), int(round(top["cy"])), int(round(x2)), int(round(bottom["cy"]))],
                    "sides": {
                        "top": line_to_json(top),
                        "bottom": line_to_json(bottom),
                        "left": None,
                        "right": None,
                    },
                    "reasons": ["horizontal_pair_width_below_min_ratio"],
                })
                continue
            span_h = bottom["cy"] - top["cy"]
            vs = [ln for ln in v_lines if _overlap_len(ln["bbox"][1], ln["bbox"][3], top["cy"], bottom["cy"]) >= min_y_overlap_ratio * span_h]
            lefts = [ln for ln in vs if abs(ln["cx"] - x1) <= args.frame_partial_alignment_tol]
            rights = [ln for ln in vs if abs(ln["cx"] - x2) <= args.frame_partial_alignment_tol]
            if lefts:
                add_candidate(
                    top=top,
                    bottom=bottom,
                    left=min(lefts, key=lambda z: abs(z["cx"] - x1)),
                    kind="missing_right",
                    reason_context="left line aligns with horizontal starts",
                )
            if rights:
                add_candidate(
                    top=top,
                    bottom=bottom,
                    right=min(rights, key=lambda z: abs(z["cx"] - x2)),
                    kind="missing_left",
                    reason_context="right line aligns with horizontal ends",
                )
            if not lefts and not rights:
                debug["rejected_frame_candidates"].append({
                    "kind": "missing_vertical_edge",
                    "bbox": [int(round(x1)), int(round(top["cy"])), int(round(x2)), int(round(bottom["cy"]))],
                    "sides": {
                        "top": line_to_json(top),
                        "bottom": line_to_json(bottom),
                        "left": None,
                        "right": None,
                    },
                    "reasons": ["no_vertical_line_aligned_to_horizontal_pair"],
                })

    # Case D: Mitchell-style open/L frame: only one long horizontal and one long vertical border are visible.
    l_min_area = getattr(args, "frame_l_min_area_ratio", max(args.frame_min_area_ratio, 0.30))
    for hh in h_lines:
        for vv in v_lines:
            pairs = []
            if abs(vv["cx"] - hh["bbox"][0]) <= args.frame_partial_alignment_tol:
                if abs(vv["bbox"][1] - hh["cy"]) <= args.frame_partial_alignment_tol:
                    pairs.append((hh, None, vv, None, "l_top_left"))
                if abs(vv["bbox"][3] - hh["cy"]) <= args.frame_partial_alignment_tol:
                    pairs.append((None, hh, vv, None, "l_bottom_left"))
            if abs(vv["cx"] - hh["bbox"][2]) <= args.frame_partial_alignment_tol:
                if abs(vv["bbox"][1] - hh["cy"]) <= args.frame_partial_alignment_tol:
                    pairs.append((hh, None, None, vv, "l_top_right"))
                if abs(vv["bbox"][3] - hh["cy"]) <= args.frame_partial_alignment_tol:
                    pairs.append((None, hh, None, vv, "l_bottom_right"))
            if not pairs:
                continue
            for top, bottom, left, right, kind in pairs:
                cand = _make_frame_candidate(binary, args, top, bottom, left, right, kind, min_side_count=2)
                if cand is None:
                    debug["rejected_frame_candidates"].append({
                        "kind": kind,
                        "sides": {
                            "top": line_to_json(top),
                            "bottom": line_to_json(bottom),
                            "left": line_to_json(left),
                            "right": line_to_json(right),
                        },
                        "reasons": ["candidate_validation_failed"],
                    })
                    continue
                if cand["area_ratio"] < l_min_area:
                    debug["rejected_frame_candidates"].append(_rejected_frame_debug(
                        cand,
                        ["l_frame_area_ratio_below_min"],
                    ))
                    continue
                cand["score"] += 60.0
                candidates.append(cand)

    candidates = [c for c in candidates if c is not None]
    debug["accepted_frame_candidates"] = [_frame_candidate_debug(c) for c in candidates]
    selected = _select_frame_candidates(candidates, args, debug=debug)
    if selected:
        ordered = sorted(selected, key=lambda z: (z["bbox"][1], z["bbox"][0]))
        if len(ordered) == 1:
            calibrated = calibrate_frame_tree(binary, ordered[0], args)
            debug["selected_frames"] = [_frame_candidate_debug(calibrated)]
            return calibrated, lines, debug
        primary = max(ordered, key=lambda z: box_area(z["bbox"])).copy()
        primary["frames"] = ordered
        primary["multi_frame"] = True
        primary["bbox"] = union_box([z["bbox"] for z in ordered])
        primary["area_ratio"] = float(box_area(primary["bbox"]) / float(w * h))
        primary["score"] = float(sum(z["score"] for z in ordered))
        calibrated = calibrate_frame_tree(binary, primary, args)
        debug["selected_frames"] = [_frame_candidate_debug(c) for c in _frame_parts(calibrated)]
        return calibrated, lines, debug

    # Last-resort fallback: use extreme long lines if side matching fails.
    if len(h_lines) >= 2 and len(v_lines) >= 2:
        top = min(h_lines, key=lambda x: x["cy"])
        bottom = max(h_lines, key=lambda x: x["cy"])
        left = min(v_lines, key=lambda x: x["cx"])
        right = max(v_lines, key=lambda x: x["cx"])
        rect = rectangle_from_sides(top, bottom, left, right, w, h)
        if rect is not None:
            area_ratio = box_area(rect) / float(w * h)
            if area_ratio >= args.frame_min_area_ratio:
                err = alignment_error(top, bottom, left, right)
                fallback = {
                    "bbox": rect,
                    "score": float(-err),
                    "area_ratio": float(area_ratio),
                    "alignment_error": float(err),
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "fallback": True,
                    "kind": "fallback_extreme_lines",
                }
                fallback = calibrate_frame_tree(binary, fallback, args)
                debug["selected_frames"] = [_frame_candidate_debug(fallback)]
                debug["notes"].append("Selected fallback frame from extreme long axis lines.")
                return fallback, lines, debug
            debug["rejected_frame_candidates"].append({
                "bbox": [int(v) for v in rect],
                "kind": "fallback_extreme_lines",
                "area_ratio": float(area_ratio),
                "sides": {
                    "top": line_to_json(top),
                    "bottom": line_to_json(bottom),
                    "left": line_to_json(left),
                    "right": line_to_json(right),
                },
                "reasons": ["fallback_area_ratio_below_min"],
            })

    debug["notes"].append("No frame candidate passed all filters.")
    return None, lines, debug


def _frame_parts(frame):
    if frame is None:
        return []
    return frame.get("frames", [frame])


def make_frame_mask(frame, shape, args):
    mask = np.zeros(shape, dtype=bool)
    if frame is None:
        return mask
    h, w = shape
    for fr in _frame_parts(frame):
        for side in ["top", "bottom", "left", "right"]:
            line = fr.get(side)
            if line is None:
                continue
            for raw_box in [line["bbox"], line.get("erase_bbox")]:
                if raw_box is None:
                    continue
                box = clip_box(raw_box, w, h)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                mask[y1:y2, x1:x2] = True
    return mask


# ============================================================
# Title detection
# ============================================================

def detect_title_boxes(binary, frame, args):
    h, w = binary.shape
    y_cap = int(args.title_search_max_ratio * h)
    if frame is not None:
        _, frame_y1, _, _ = frame["bbox"]
        y_max = min(y_cap, max(0, frame_y1 - args.title_frame_gap))
    else:
        y_max = y_cap

    y_min = int(args.title_search_min_ratio * h)
    if y_max <= y_min + 5:
        return {
            "search_region": [0, y_min, w, y_max],
            "line_boxes": [],
            "block_bbox": None,
        }

    region = binary[y_min:y_max, :].copy()

    comps, label_img = get_components(region)
    text_mask = np.zeros(region.shape, dtype=bool)
    x_center_min = args.title_center_x_min_ratio * w
    x_center_max = args.title_center_x_max_ratio * w

    for c in comps:
        x1, y1, x2, y2 = c["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cx = c["cx"]
        if c["area"] < args.title_min_component_area:
            continue
        if bh < args.title_min_component_h or bh > args.title_max_component_h:
            continue
        if not (x_center_min <= cx <= x_center_max):
            continue
        text_mask[label_img == c["label_id"]] = True

    if not text_mask.any():
        return {
            "search_region": [0, y_min, w, y_max],
            "line_boxes": [],
            "block_bbox": None,
        }

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (args.title_merge_kernel_w, args.title_merge_kernel_h),
    )
    merged = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    line_comps, line_labels = get_components(merged)

    line_boxes = []
    for c in line_comps:
        x1, y1, x2, y2 = c["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cx = 0.5 * (x1 + x2)
        if bw < args.title_min_line_width:
            continue
        if bh < args.title_min_line_height or bh > args.title_max_line_height:
            continue
        if not (x_center_min <= cx <= x_center_max):
            continue
        box = expand_box([x1, y1 + y_min, x2, y2 + y_min], args.title_box_pad, w, h)
        if box is not None:
            line_boxes.append(box)

    line_boxes = _select_top_center_title_block(line_boxes, w, h, args)
    block_bbox = union_box(line_boxes)
    if block_bbox is not None:
        block_bbox = expand_box(block_bbox, args.title_block_pad, w, h)

    return {
        "search_region": [0, y_min, w, y_max],
        "line_boxes": line_boxes,
        "block_bbox": block_bbox,
    }


def _select_top_center_title_block(line_boxes, w, h, args):
    if not line_boxes:
        return []

    center_x = 0.5 * w
    strict_min = args.title_strict_center_x_min_ratio * w
    strict_max = args.title_strict_center_x_max_ratio * w

    ordered = sorted(line_boxes, key=lambda b: (b[1], b[0]))
    seeds = []
    for i, b in enumerate(ordered):
        cx = 0.5 * (b[0] + b[2])
        crosses_center = b[0] <= center_x <= b[2]
        strict_centered = strict_min <= cx <= strict_max
        if crosses_center or strict_centered:
            seeds.append(i)

    if not seeds:
        return []

    seed_i = seeds[0]
    block = [ordered[seed_i]]
    block_top = ordered[seed_i][1]
    prev = ordered[seed_i]
    max_block_h = int(args.title_block_max_height_ratio * h)

    for b in ordered[seed_i + 1:]:
        gap = b[1] - prev[3]
        if gap > args.title_block_line_gap:
            break
        if b[3] - block_top > max_block_h:
            break
        block.append(b)
        prev = b

    return sorted(block, key=lambda b: (b[1], b[0]))



# ============================================================
# Frame information text detection
# ============================================================

def detect_text_boxes_in_region(binary, region_box, remove_mask, args, min_width=None, keep=None, prefer_y=None):
    h, w = binary.shape
    rb = clip_box(region_box, w, h)
    if rb is None:
        return []
    x1, y1, x2, y2 = rb
    region = binary[y1:y2, x1:x2].copy()
    if remove_mask is not None:
        region &= ~remove_mask[y1:y2, x1:x2]

    comps, label_img = get_components(region)
    text_mask = np.zeros(region.shape, dtype=bool)
    for c in comps:
        bx1, by1, bx2, by2 = c["bbox"]
        bw, bh = bx2 - bx1, by2 - by1
        if c["area"] < args.info_min_component_area:
            continue
        if bh < args.info_min_component_h or bh > args.info_max_component_h:
            continue
        # Reject long wire/frame fragments; keep compact text glyphs.
        if (bw >= args.info_long_line_w and bh <= args.info_long_line_h) or (bh >= args.info_long_line_w and bw <= args.info_long_line_h):
            continue
        text_mask[label_img == c["label_id"]] = True

    if not text_mask.any():
        return []

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.info_merge_kernel_w, args.info_merge_kernel_h))
    merged = cv2.dilate(text_mask.astype(np.uint8), k, iterations=1).astype(bool)
    line_comps, line_labels = get_components(merged)

    boxes = []
    min_width = args.info_min_line_width if min_width is None else min_width
    for c in line_comps:
        bx1, by1, bx2, by2 = c["bbox"]
        ink = text_mask & (line_labels == c["label_id"])
        if not ink.any():
            continue
        ys, xs = np.where(ink)
        bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        bw, bh = bx2 - bx1, by2 - by1
        if bw < min_width or bh < args.info_min_line_height or bh > args.info_max_line_height:
            continue
        b = expand_box([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1], args.info_box_pad, w, h)
        if b is not None:
            boxes.append(b)

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    if prefer_y is not None:
        boxes = sorted(boxes, key=lambda b: (abs(0.5 * (b[1] + b[3]) - prefer_y), b[1], b[0]))
    if keep is not None:
        boxes = boxes[:keep]
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _make_frame_fill_mask(frame, shape):
    mask = np.zeros(shape, dtype=bool)
    if frame is None:
        return mask
    h, w = shape
    for fr in _frame_parts(frame):
        b = clip_box(fr["bbox"], w, h)
        if b is None:
            continue
        x1, y1, x2, y2 = b
        mask[y1:y2, x1:x2] = True
    return mask


def detect_frame_info_boxes(binary, frame, frame_mask, args):
    h, w = binary.shape
    items = []
    if frame is None:
        return items

    # Red frame-info boxes must describe text OUTSIDE the circuit frame.
    # Remove the whole interior, not only the frame border pixels.
    outside_remove_mask = _make_frame_fill_mask(frame, binary.shape) | frame_mask

    for idx, fr in enumerate(_frame_parts(frame)):
        x1, y1, x2, y2 = fr["bbox"]
        fw, fh = x2 - x1, y2 - y1

        # Case 2: text around the top-left corner, but outside the frame.
        tl_w = max(args.info_tl_min_w, int(args.info_tl_w_ratio * fw))
        tl_h = max(args.info_tl_min_h, int(args.info_tl_h_ratio * fh))
        tl_region = [
            max(0, x1 - args.info_tl_left_pad),
            max(0, y1 - args.info_tl_above),
            min(w, x1 + tl_w),
            min(h, y1 + tl_h),
        ]
        top_left = detect_text_boxes_in_region(
            binary, tl_region, outside_remove_mask, args,
            min_width=args.info_min_corner_line_width,
            keep=None,
            prefer_y=y1 - 0.5 * args.info_tl_above,
        )
        top_left = [b for b in top_left if b[2] <= x1 + args.info_tl_left_tol or b[3] <= y1 + args.info_outside_tol]
        top_left = [
            b for b in (
                _clip_info_box_to_frame_side(b, fr, "top" if 0.5 * (b[1] + b[3]) < y1 else "left", w, h, args)
                for b in top_left
            )
            if b is not None and not _intersects_any_frame(b, frame)
        ]
        top_left = sorted(
            top_left,
            key=lambda b: (abs(0.5 * (b[1] + b[3]) - (y1 - 0.5 * args.info_tl_above)), b[1], b[0]),
        )[:args.info_top_left_max_boxes]
        top_left = sorted(top_left, key=lambda b: (b[1], b[0]))

        # Case 3: module/circuit title below the bottom-left frame corner.
        # Start from y2 so labels inside the frame, such as connector names,
        # cannot be selected as red frame info.
        bl_x1 = max(0, x1 - args.info_bl_left_pad)
        bl_x2 = min(w, x1 + max(args.info_bl_min_w, int(args.info_bl_w_ratio * fw)))
        bl_region = [bl_x1, min(h, y2 + args.info_bl_outside_gap), bl_x2, min(h, y2 + args.info_bl_below)]
        bottom_left = detect_text_boxes_in_region(
            binary, bl_region, outside_remove_mask, args,
            min_width=args.info_min_bottom_line_width,
            keep=None,
            prefer_y=y2 + args.info_bottom_prefer_offset,
        )
        bottom_left = [b for b in bottom_left if b[1] >= y2 - args.info_outside_tol]
        bottom_left = [
            b for b in (
                _clip_info_box_to_frame_side(b, fr, "bottom", w, h, args)
                for b in bottom_left
            )
            if b is not None and not _intersects_any_frame(b, frame)
        ]
        bottom_left = merge_adjacent_line_boxes(bottom_left, args.info_bottom_join_gap)
        bottom_left = sorted(
            bottom_left,
            key=lambda b: (abs(0.5 * (b[1] + b[3]) - (y2 + args.info_bottom_prefer_offset)), b[1], b[0]),
        )[:args.info_bottom_left_max_boxes]
        bottom_left = sorted(bottom_left, key=lambda b: (b[1], b[0]))

        items.append({
            "frame_index": int(idx),
            "frame_bbox": [int(v) for v in fr["bbox"]],
            "top_left_region": [int(v) for v in clip_box(tl_region, w, h)],
            "bottom_left_region": [int(v) for v in clip_box(bl_region, w, h)],
            "top_left_boxes": [[int(v) for v in b] for b in top_left],
            "bottom_left_boxes": [[int(v) for v in b] for b in bottom_left],
        })
    return items

# ============================================================
# Rendering and clean image
# ============================================================

def make_review_image(img, frame, frame_mask, args):
    vis = img.copy()

    if frame_mask.any():
        vis[frame_mask] = (0, 180, 0)

    return vis


def erase_frame_border(clean, frame, args):
    h, w = clean.shape[:2]
    for fr in _frame_parts(frame):
        for side in ["top", "bottom", "left", "right"]:
            line = fr.get(side)
            if line is None:
                continue
            box = clip_box(line.get("erase_bbox", line["bbox"]), w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            clean[y1:y2, x1:x2] = 255


def make_circuit_image(img, frame, args):
    circuit = np.full_like(img, 255)
    if frame is None:
        return img.copy()

    h, w = img.shape[:2]
    for fr in _frame_parts(frame):
        x1, y1, x2, y2 = fr["bbox"]
        pad = args.frame_clean_pad
        inner = clip_box([x1 + pad, y1 + pad, x2 - pad, y2 - pad], w, h)
        if inner is None:
            continue
        ix1, iy1, ix2, iy2 = inner
        circuit[iy1:iy2, ix1:ix2] = img[iy1:iy2, ix1:ix2]
    erase_frame_border(circuit, frame, args)
    return circuit


def make_sheet_image(img, frame, frame_mask, args):
    sheet = np.full_like(img, 255)
    if frame is None:
        return img.copy()

    h, w = img.shape[:2]
    outside_mask = np.ones((h, w), dtype=bool)
    for fr in _frame_parts(frame):
        box = clip_box(fr["bbox"], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        outside_mask[y1:y2, x1:x2] = False
    outside_mask &= ~frame_mask
    sheet[outside_mask] = img[outside_mask]
    return sheet


def strip_masks_for_json(obj):
    if isinstance(obj, list):
        return [strip_masks_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: strip_masks_for_json(v) for k, v in obj.items() if k != "mask"}
    return obj


def _frame_json(fr):
    thickness_by_side = {
        side: None if fr.get(side) is None else int(fr[side]["thickness"])
        for side in ["top", "bottom", "left", "right"]
    }
    present_thickness = [v for v in thickness_by_side.values() if v is not None]
    return {
        "bbox": [int(v) for v in fr["bbox"]],
        "area_ratio": float(fr.get("area_ratio", 0.0)),
        "score": float(fr.get("score", 0.0)),
        "alignment_error": float(fr.get("alignment_error", 0.0)),
        "partial": bool(fr.get("partial", False)),
        "fallback": bool(fr.get("fallback", False)),
        "kind": str(fr.get("kind", "closed")),
        "calibration": fr.get("calibration"),
        "line_thickness": {
            "by_side": thickness_by_side,
            "min": None if not present_thickness else int(min(present_thickness)),
            "max": None if not present_thickness else int(max(present_thickness)),
            "median": None if not present_thickness else float(np.median(present_thickness)),
        },
        "sides": {
            "top": line_to_json(fr.get("top")),
            "bottom": line_to_json(fr.get("bottom")),
            "left": line_to_json(fr.get("left")),
            "right": line_to_json(fr.get("right")),
        },
    }


def locate_page(img, args):
    binary = binarize_page(img, args.ink_threshold)
    frame, all_lines, frame_debug = find_outer_frame(binary, args)
    frame_mask = make_frame_mask(frame, binary.shape, args)

    frame_entries = []
    if frame is None:
        outer_frame_json = None
    else:
        parts = _frame_parts(frame)
        inner_boxes = []
        h, w = binary.shape
        for fr in parts:
            b = fr["bbox"]
            inner = clip_box(
                [b[0] + args.frame_clean_pad, b[1] + args.frame_clean_pad,
                 b[2] - args.frame_clean_pad, b[3] - args.frame_clean_pad],
                w,
                h,
            )
            if inner is not None:
                inner_boxes.append(inner)
            entry = _frame_json(fr)
            entry["inner_bbox"] = None if inner is None else [int(v) for v in inner]
            frame_entries.append(entry)
        outer_frame_json = {
            "bbox": [int(v) for v in frame["bbox"]],
            "inner_bbox": None if not inner_boxes else [int(v) for v in union_box(inner_boxes)],
            "area_ratio": float(frame.get("area_ratio", 0.0)),
            "score": float(frame.get("score", 0.0)),
            "alignment_error": float(frame.get("alignment_error", 0.0)),
            "multi_frame": bool(frame.get("multi_frame", False)),
            "fallback": bool(frame.get("fallback", False)),
            "kind": str(frame.get("kind", "multi_frame" if frame.get("multi_frame", False) else "closed")),
            "sub_frames": frame_entries,
            "inner_bboxes": [[int(v) for v in b] for b in inner_boxes],
        }

    result_json = {
        "outer_frame": outer_frame_json,
        "frames": frame_entries,
        "summary": {
            "num_axis_lines": len(all_lines),
            "num_frame_boxes": 0 if frame is None else len(_frame_parts(frame)),
            "detection_mode": "closed_partial_l_fallback",
        },
    }

    frame_debug["summary"] = {
        "num_axis_lines": len(all_lines),
        "num_rejected_axis_lines": len(frame_debug.get("rejected_axis_lines", [])),
        "num_rejected_frame_lines": len(frame_debug.get("rejected_frame_lines", [])),
        "num_accepted_frame_candidates": len(frame_debug.get("accepted_frame_candidates", [])),
        "num_rejected_frame_candidates": len(frame_debug.get("rejected_frame_candidates", [])),
        "num_selected_frames": 0 if frame is None else len(_frame_parts(frame)),
    }

    return result_json, frame_debug, frame, frame_mask


# ============================================================
# PDF processing
# ============================================================

def process_pdf(pdf_path, args, page_range=None):
    pdf_name = pdf_path.stem
    output_root = OUTPUT_DIR / pdf_name
    json_dir = output_root / "json"
    debug_dir = output_root / "debug"
    image_dir = output_root / "image"
    circuit_image_dir = output_root / "circuit_images"
    sheet_image_dir = output_root / "sheet_images"

    if output_root.exists() and not args.preserve:
        resolved_output = output_root.resolve()
        resolved_stage = OUTPUT_DIR.resolve()
        if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
            raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    circuit_image_dir.mkdir(parents=True, exist_ok=True)
    sheet_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {pdf_name}")
    print(f"Output: {output_root}")
    print(f"  preserve existing output: {args.preserve}")
    print(f"  json: {json_dir}")
    print(f"  debug: {debug_dir}")
    print(f"  image: {image_dir}")
    print(f"  circuit_images: {circuit_image_dir}")
    print(f"  sheet_images: {sheet_image_dir}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_page, end_page = (1, total_pages) if page_range is None else (page_range[0], min(page_range[1], total_pages))

    review_pages = []
    circuit_pages = []
    sheet_pages = []

    for page_idx in range(start_page - 1, end_page):
        page_no = page_idx + 1
        print(f"  Page {page_no}/{total_pages}")

        img = render_pdf_page(doc[page_idx], args.dpi)
        output_img = make_binary_output_image(img, args.ink_threshold)
        page_json, debug_json, frame, frame_mask = locate_page(img, args)

        review_img = make_review_image(output_img, frame, frame_mask, args)
        circuit_img = make_circuit_image(output_img, frame, args)
        sheet_img = make_sheet_image(output_img, frame, frame_mask, args)

        page_json = {
            "page": page_no,
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),
            **page_json,
        }
        debug_json = {
            "page": page_no,
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),
            **debug_json,
        }

        json_path = json_dir / f"page_{page_no:03d}.json"
        debug_path = debug_dir / f"page_{page_no:03d}.json"
        png_path = image_dir / f"page_{page_no:03d}.png"
        circuit_png_path = circuit_image_dir / f"page_{page_no:03d}.png"
        sheet_png_path = sheet_image_dir / f"page_{page_no:03d}.png"

        save_json(json_path, page_json)
        save_json(debug_path, debug_json)
        Image.fromarray(review_img).save(png_path)
        Image.fromarray(circuit_img).save(circuit_png_path)
        Image.fromarray(sheet_img).save(sheet_png_path)

        print(
            f"    saved: json/{json_path.name}, debug/{debug_path.name}, "
            f"image/{png_path.name}, circuit_images/{circuit_png_path.name}, "
            f"sheet_images/{sheet_png_path.name}"
        )

        review_pages.append(review_img)
        circuit_pages.append(circuit_img)
        sheet_pages.append(sheet_img)

    review_pdf = output_root / "review.pdf"
    result_pdf = output_root / "result.pdf"
    sheet_pdf = output_root / "sheet.pdf"
    image_to_pdf(review_pages, review_pdf)
    image_to_pdf(circuit_pages, result_pdf)
    image_to_pdf(sheet_pages, sheet_pdf)

    print(f"Finished: {pdf_name}")
    print(f"Review PDF: {review_pdf}")
    print(f"Result PDF: {result_pdf}")
    print(f"Sheet PDF: {sheet_pdf}")

# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--pdf", type=str, default="bmw-328i-1997.pdf")
    p.add_argument("--pages", type=str, default=None)
    p.add_argument("-p", "--preserve", action="store_true", help="Preserve the output folder instead of clearing it first.")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)

    p.add_argument("--ink-threshold", type=int, default=210)

    p.add_argument("--frame-open-kernel-len", type=int, default=80)
    p.add_argument("--frame-bridge-gap", type=int, default=24)
    p.add_argument("--frame-min-width-ratio", type=float, default=0.55)
    p.add_argument("--frame-min-height-ratio", type=float, default=0.35)
    p.add_argument("--frame-min-line-width-ratio", type=float, default=0.55)
    p.add_argument("--frame-min-line-height-ratio", type=float, default=0.28)
    p.add_argument("--frame-min-rect-width-ratio", type=float, default=0.55)
    p.add_argument("--frame-min-rect-height-ratio", type=float, default=0.28)
    p.add_argument("--frame-min-area-ratio", type=float, default=0.20)
    p.add_argument("--frame-l-min-area-ratio", type=float, default=0.30)
    p.add_argument("--frame-min-thickness", type=int, default=1)
    p.add_argument("--frame-max-thickness", type=int, default=12)
    p.add_argument("--frame-alignment-tol", type=float, default=28.0)
    p.add_argument("--frame-partial-alignment-tol", type=float, default=70.0)
    p.add_argument("--frame-partial-overlap-ratio", type=float, default=0.55)
    p.add_argument("--frame-min-side-count", type=int, default=3)
    p.add_argument("--frame-max-boxes", type=int, default=4)
    p.add_argument("--frame-select-iou-thresh", type=float, default=0.65)
    p.add_argument("--frame-select-inside-thresh", type=float, default=0.85)
    p.add_argument("--frame-select-refine-area-ratio", type=float, default=0.65)
    p.add_argument("--frame-select-refine-edge-tol", type=int, default=12)
    p.add_argument("--frame-select-refine-min-edges", type=int, default=2)
    p.add_argument("--frame-mask-pad", type=int, default=5)
    p.add_argument("--frame-clean-pad", type=int, default=0)
    p.add_argument("--frame-clean-border-pad", type=int, default=0)
    p.add_argument("--frame-inferred-thickness", type=int, default=2)
    p.add_argument("--frame-infer-corner-ignore-px", type=int, default=18)

    p.add_argument("--title-frame-gap", type=int, default=30)
    p.add_argument("--title-search-min-ratio", type=float, default=0.02)
    p.add_argument("--title-search-max-ratio", type=float, default=0.25)
    p.add_argument("--title-center-x-min-ratio", type=float, default=0.25)
    p.add_argument("--title-center-x-max-ratio", type=float, default=0.75)
    p.add_argument("--title-strict-center-x-min-ratio", type=float, default=0.35)
    p.add_argument("--title-strict-center-x-max-ratio", type=float, default=0.65)
    p.add_argument("--title-block-line-gap", type=int, default=120)
    p.add_argument("--title-block-max-height-ratio", type=float, default=0.12)
    p.add_argument("--title-min-component-area", type=int, default=8)
    p.add_argument("--title-min-component-h", type=int, default=5)
    p.add_argument("--title-max-component-h", type=int, default=80)
    p.add_argument("--title-merge-kernel-w", type=int, default=65)
    p.add_argument("--title-merge-kernel-h", type=int, default=7)
    p.add_argument("--title-min-line-width", type=int, default=80)
    p.add_argument("--title-min-line-height", type=int, default=8)
    p.add_argument("--title-max-line-height", type=int, default=70)
    p.add_argument("--title-box-pad", type=int, default=4)
    p.add_argument("--title-block-pad", type=int, default=2)


    p.add_argument("--info-tl-w-ratio", type=float, default=0.30)
    p.add_argument("--info-tl-h-ratio", type=float, default=0.03)
    p.add_argument("--info-tl-min-w", type=int, default=520)
    p.add_argument("--info-tl-min-h", type=int, default=45)
    p.add_argument("--info-tl-left-pad", type=int, default=260)
    p.add_argument("--info-tl-above", type=int, default=85)
    p.add_argument("--info-tl-left-tol", type=int, default=75)
    p.add_argument("--info-outside-tol", type=int, default=3)
    p.add_argument("--info-bl-w-ratio", type=float, default=0.48)
    p.add_argument("--info-bl-min-w", type=int, default=360)
    p.add_argument("--info-bl-left-pad", type=int, default=420)
    p.add_argument("--info-bl-above", type=int, default=5)  # kept for CLI compatibility
    p.add_argument("--info-bl-outside-gap", type=int, default=2)
    p.add_argument("--info-bl-below", type=int, default=110)
    p.add_argument("--info-bottom-prefer-offset", type=int, default=28)
    p.add_argument("--info-bottom-join-gap", type=int, default=90)
    p.add_argument("--info-min-component-area", type=int, default=4)
    p.add_argument("--info-min-component-h", type=int, default=3)
    p.add_argument("--info-max-component-h", type=int, default=45)
    p.add_argument("--info-long-line-w", type=int, default=80)
    p.add_argument("--info-long-line-h", type=int, default=3)
    p.add_argument("--info-merge-kernel-w", type=int, default=42)
    p.add_argument("--info-merge-kernel-h", type=int, default=7)
    p.add_argument("--info-min-line-width", type=int, default=18)
    p.add_argument("--info-min-corner-line-width", type=int, default=14)
    p.add_argument("--info-min-bottom-line-width", type=int, default=60)
    p.add_argument("--info-min-line-height", type=int, default=5)
    p.add_argument("--info-max-line-height", type=int, default=55)
    p.add_argument("--info-box-pad", type=int, default=4)
    p.add_argument("--info-top-left-max-boxes", type=int, default=1)
    p.add_argument("--info-bottom-left-max-boxes", type=int, default=1)
    p.add_argument("--info-draw-thickness", type=int, default=2)

    p.add_argument("--overlay-alpha", type=float, default=0.85)
    p.add_argument("--frame-draw-thickness", type=int, default=2)
    p.add_argument("--title-draw-thickness", type=int, default=2)
    p.add_argument("--title-block-draw-thickness", type=int, default=2)
    p.add_argument("--draw-title-block", action="store_true", default=False)

    return p


def main():
    args = build_argparser().parse_args()
    page_range = parse_page_range(args.pages) if args.pages is not None else None

    print("Page range:", "ALL" if page_range is None else f"{page_range[0]}-{page_range[1]}")
    print(f"DPI: {args.dpi}")

    if args.pdf is not None:
        pdf_path = INPUT_DIR / args.pdf
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found:\n{pdf_path}")
        process_pdf(pdf_path, args, page_range=page_range)
        return

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF found under:\n{INPUT_DIR}")

    for pdf_path in pdf_files:
        process_pdf(pdf_path, args, page_range=page_range)


if __name__ == "__main__":
    main()
