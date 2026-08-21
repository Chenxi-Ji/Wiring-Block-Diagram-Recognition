#!/usr/bin/env python3
"""Detect and remove high-confidence text from 05_endpoint circuit images.

Inputs:
  * outputs/05_endpoint/<pdf_stem>/circuit_images/page_NNN.png

Outputs:
  * outputs/06_text/<pdf_stem>/images/page_NNN.png (05_endpoint circuit image + detected-text overlay)
  * outputs/06_text/<pdf_stem>/circuit_images/page_NNN.png (optional, with --save-circuit-images)
  * outputs/06_text/<pdf_stem>/json/page_NNN.json (text position + content)
  * outputs/06_text/<pdf_stem>/debug/page_NNN.json (raw OCR output, rejections, erasure stats)
  * outputs/06_text/<pdf_stem>/review.pdf
  * outputs/06_text/<pdf_stem>/result.pdf (optional, with --save-circuit-images)

This stage runs PaddleOCR directly on the whole page (no tile-splitting stage
exists in the current pipeline). Detected text is classified (wire color code /
number / other text) and quality-filtered the same way as backup/07_text.py's
recognition logic. Only text whose OCR confidence is >= --erase-min-confidence
(strict, default 0.80) has its ink pixels erased from the result image; every
kept detection (regardless of erasure) is recorded in json/ and drawn in
images/. Erasure only removes the text's own ink pixels (via a per-polygon
gray threshold + small dilation), not the whole bounding box, so nearby wire
lines are not damaged.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from pipeline_io import image_to_pdf, parse_page_range, save_json


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_DIR = ROOT_DIR / "outputs" / Path(__file__).stem
INPUT_STAGE = "05_endpoint"
STAGE_NAME = Path(__file__).stem

WIRE_COLOR_TOKENS = {
    "BLK", "BLU", "BRN", "GRN", "GRY", "ORG", "PNK", "RED", "TAN", "TR", "VIO", "WHT", "YEL",
}
WIRE_COLOR_ALIASES = {
    "BK": "BLK", "SW": "BLK", "BL": "BLU", "BLL": "BLU", "BU": "BLU", "BR": "BRN",
    "BIRN": "BRN", "8RN": "BRN", "GN": "GRN", "GR": "GRY", "GY": "GRY", "OR": "ORG",
    "PK": "PNK", "RS": "PNK", "R": "RED", "RT": "RED", "TN": "TAN", "V": "VIO",
    "VI": "VIO", "W": "WHT", "WS": "WHT", "Y": "YEL", "GE": "YEL",
}
SPECIAL_WIRE_LABELS = {"NCA"}
COLOR_PATTERN = "|".join(sorted(WIRE_COLOR_TOKENS))
READABLE_TEXT_RE = re.compile(r"^[A-Z0-9][A-Z0-9 _/\-().,+&]+$")

REVIEW_COLORS = {
    "wire_color": (0, 90, 255),
    "number": (255, 150, 0),
    "other_text": (0, 170, 0),
}
ERASED_OVERLAY_COLOR = (200, 30, 30)


@dataclass(frozen=True)
class TextConfig:
    dpi: int = 300
    lang: str = "en"
    min_confidence: float = 0.35
    erase_min_confidence: float = 0.80
    text_gray_threshold: int = 245
    erase_ink_pad_px: int = 1
    line_probe_pad_px: int = 10
    line_shape_min_aspect_ratio: float = 4.0
    line_shape_max_thickness_px: int = 5
    line_shape_min_length_px: int = 22
    edge_divider_coverage_ratio: float = 0.92
    edge_divider_band_px: int = 3
    thick_line_min_thickness_px: int = 6
    thick_line_min_fill_ratio: float = 0.85
    tight_crossing_line_min_alnum: int = 3
    text_det_limit_side_len: int = 4000
    text_det_thresh: float = 0.25
    text_det_box_thresh: float = 0.42
    text_det_unclip_ratio: float = 1.8
    use_clahe: bool = True
    refined_ink_padding_px: int = 2
    dedupe_iou_threshold: float = 0.82
    max_text_height_ratio: float = 1.3
    strong_text_height_ratio: float = 1.8
    short_text_max_alnum: int = 3
    single_letter_min_confidence: float = 0.95
    single_digit_min_confidence: float = 0.85
    short_alnum_min_confidence: float = 0.80
    short_wire_color_min_confidence: float = 0.55


@dataclass
class TextRecord:
    id: str
    page: int
    text: str
    normalized_text: str
    category: str
    category_confidence: float
    ocr_confidence: float
    bbox: list[int]
    raw_bbox: list[int]
    polygon: list[list[int]]
    erased: bool
    ink_pixel_count: int
    debug: dict[str, Any]


# ============================================================
# IO / paths
# ============================================================

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
    direct = ROOT_DIR / "outputs" / INPUT_STAGE / pdf_stem / "circuit_images"
    if direct.is_dir():
        return direct
    stage_root = ROOT_DIR / "outputs" / INPUT_STAGE
    if stage_root.is_dir():
        for candidate in stage_root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == pdf_stem.lower():
                image_dir = candidate / "circuit_images"
                if image_dir.is_dir():
                    return image_dir
    raise FileNotFoundError(
        f"Missing {INPUT_STAGE} circuit images: {direct}. "
        f"Run: python scripts/05_endpoint.py --pdf {pdf_stem}.pdf"
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
        raise ValueError(f"Requested page(s) {missing} are missing from {INPUT_STAGE}/circuit_images")
    return requested


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    resolved_output = root.resolve()
    resolved_stage = OUTPUT_DIR.resolve()
    if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
    import shutil
    shutil.rmtree(root)


def make_output_dirs(pdf_stem: str, preserve: bool, save_circuit_images: bool) -> dict[str, Path]:
    root = OUTPUT_DIR / pdf_stem
    if root.exists() and not preserve:
        clear_output_root(root)
    paths = {
        "root": root,
        "images": root / "images",
        "json": root / "json",
        "debug": root / "debug",
    }
    if save_circuit_images:
        paths["circuit_images"] = root / "circuit_images"
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


# ============================================================
# Geometry helpers
# ============================================================

def clip_bbox(box: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    return [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]


def bbox_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    denom = bbox_area(a) + bbox_area(b) - inter
    return float(inter / max(1, denom))


def bbox_to_poly(box: Sequence[int]) -> list[list[int]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def poly_bbox(poly: Sequence[Sequence[float]]) -> list[int]:
    pts = np.asarray(poly, dtype=float)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2 + 1)), int(math.ceil(y2 + 1))]


# ============================================================
# Text normalization / classification (ported from backup/07_text.py)
# ============================================================

def normalize_text_token(text: str) -> str:
    norm = text.strip().upper()
    norm = norm.replace("\\", "/").replace("|", "/")
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip(" \t\r\n:;,.[]{}")


def compact_token(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text_token(text))


def canonical_wire_part(part: str) -> str | None:
    clean = re.sub(r"[^A-Z]", "", part.upper())
    if clean in WIRE_COLOR_TOKENS:
        return clean
    return WIRE_COLOR_ALIASES.get(clean)


def normalize_wire_sequence(text: str) -> tuple[str, list[str]] | None:
    compact = normalize_text_token(text)
    compact = compact.replace("\\", "/").replace("-", "/")
    compact = re.sub(r"\s*/\s*", "/", compact)
    compact = re.sub(r"[()]", "", compact)
    compact = compact.replace(" ", "")
    compact = re.sub(r"\b0RG\b", "ORG", compact)
    compact = re.sub(r"\b8RN\b", "BRN", compact)
    compact = re.sub(r"\b6RN\b", "GRN", compact)
    direct_alias = canonical_wire_part(compact)
    if direct_alias is not None:
        return direct_alias, [direct_alias]
    if "/" not in compact:
        compact = re.sub(r"(?<=[A-Z])I(?=[A-Z])", "/", compact)
    for token in sorted(WIRE_COLOR_TOKENS):
        compact = re.sub(rf"{token}[I1](?=$|(?:{COLOR_PATTERN}))", f"{token}/", compact)
    if compact in SPECIAL_WIRE_LABELS:
        return compact, [compact]
    if not re.fullmatch(r"[A-Z0-9]+(?:/[A-Z0-9]+){0,3}", compact):
        return None
    parts = compact.split("/")
    if len(parts) > 4:
        return None
    canonical = [canonical_wire_part(part) for part in parts]
    if any(part is None for part in canonical):
        return None
    tokens = [str(part) for part in canonical]
    return "/".join(tokens), tokens


def normalize_wire_color_token(text: str) -> tuple[str, float, list[str]]:
    norm = normalize_text_token(text)
    if not norm:
        return "", 0.0, []
    direct = normalize_wire_sequence(norm)
    if direct is not None:
        normalized, tokens = direct
        return normalized, 0.99 if normalized in SPECIAL_WIRE_LABELS else 0.98, tokens
    alt = re.fullmatch(r"(.+?)\s*\(?\s*OR\s+(.+?)\s*\)?", norm)
    if alt:
        left = normalize_wire_sequence(alt.group(1))
        right = normalize_wire_sequence(alt.group(2))
        if left is not None and right is not None:
            return f"{left[0]} OR {right[0]}", 0.94, [*left[1], *right[1]]
    return norm, 0.0, []


def is_readable_text(text: str) -> bool:
    norm = normalize_text_token(text)
    compact = compact_token(text)
    if not norm or not re.search(r"[A-Z0-9]", norm):
        return False
    if len(compact) == 1:
        return compact.isalnum()
    if len(re.sub(r"[^A-Z0-9]", "", compact)) == 0:
        return False
    return bool(READABLE_TEXT_RE.fullmatch(norm) or re.search(r"[A-Z]{2,}", norm) or re.search(r"\d", norm))


def classify_text(text: str) -> tuple[str, str, float, dict[str, Any]]:
    normalized_color, color_confidence, color_tokens = normalize_wire_color_token(text)
    if color_confidence > 0:
        return "wire_color", normalized_color, color_confidence, {
            "rule": "wire_color_token",
            "canonical_wire_tokens": color_tokens,
        }
    compact = compact_token(text)
    if re.fullmatch(r"\d{1,3}", compact):
        return "number", compact, 0.90, {"rule": "pure_integer"}
    normalized = normalize_text_token(text)
    confidence = 0.72 if is_readable_text(text) else 0.35
    return "other_text", normalized, confidence, {"rule": "fallback_other_text"}


def alnum_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_text_token(text))


# ============================================================
# PaddleOCR prediction parsing (ported from backup/07_text.py)
# ============================================================

def as_prediction_payload(item: Any) -> Any:
    data = getattr(item, "json", None)
    if callable(data):
        try:
            return data()
        except TypeError:
            pass
    if isinstance(data, dict):
        return data
    return item


def polygon_from_any(value: Any) -> list[list[int]] | None:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 2:
        return None
    pts = arr[:4, :2]
    return [[int(round(float(x))), int(round(float(y)))] for x, y in pts]


def text_score_from_any(value: Any) -> tuple[str, float] | None:
    if isinstance(value, dict):
        text = None
        score = None
        for key in ("text", "rec_text", "label", "word", "transcription", "content"):
            if value.get(key) is not None:
                text = str(value[key])
                break
        for key in ("confidence", "conf", "score", "rec_score", "prob", "probability"):
            if value.get(key) is not None:
                try:
                    score = float(value[key])
                    if score > 1.0:
                        score /= 100.0
                except (TypeError, ValueError):
                    score = None
                break
        if text is not None:
            return text, float(score if score is not None else 1.0)
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[0], str):
            try:
                return value[0], float(value[1])
            except (TypeError, ValueError):
                return value[0], 1.0
        if len(value) >= 2 and isinstance(value[1], str):
            try:
                return value[1], float(value[2]) if len(value) >= 3 else 1.0
            except (TypeError, ValueError):
                return value[1], 1.0
    return None


def deduplicate_predictions(
    predictions: Iterable[tuple[list[list[int]], str, float]],
    iou_threshold: float = 0.82,
) -> list[tuple[list[list[int]], str, float]]:
    ordered = sorted(predictions, key=lambda item: (-float(item[2]), poly_bbox(item[0])[1], poly_bbox(item[0])[0], item[1]))
    kept: list[tuple[list[list[int]], str, float]] = []
    kept_boxes: list[list[int]] = []
    for item in ordered:
        box = poly_bbox(item[0])
        if any(bbox_iou(box, existing) >= iou_threshold for existing in kept_boxes):
            continue
        kept.append(item)
        kept_boxes.append(box)
    return sorted(kept, key=lambda item: (poly_bbox(item[0])[1], poly_bbox(item[0])[0], item[1]))


def normalize_prediction(prediction: Any) -> list[tuple[list[list[int]], str, float]]:
    out: list[tuple[list[list[int]], str, float]] = []

    def append_text_poly(poly: Any, text_value: Any, score_value: Any = 1.0) -> bool:
        pts = polygon_from_any(poly)
        if pts is None:
            return False
        text = str(text_value).strip()
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 1.0
        if score > 1.0:
            score /= 100.0
        out.append((pts, text, score))
        return True

    def append_detection_only(poly: Any) -> bool:
        pts = polygon_from_any(poly)
        if pts is None:
            return False
        out.append((pts, "", 0.0))
        return True

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        payload = as_prediction_payload(item)
        if isinstance(payload, dict):
            rec_polys = payload.get("rec_polys") or payload.get("text_polys") or payload.get("boxes") or []
            dt_polys = payload.get("dt_polys") or []
            texts = payload.get("rec_texts") or payload.get("texts") or []
            scores = payload.get("rec_scores") or payload.get("scores") or []
            if rec_polys and texts:
                recognized_boxes: list[list[int]] = []
                for index, poly in enumerate(rec_polys):
                    if index >= len(texts):
                        append_detection_only(poly)
                        continue
                    if append_text_poly(poly, texts[index], scores[index] if index < len(scores) else 1.0):
                        recognized_boxes.append(poly_bbox(polygon_from_any(poly) or []))
                for poly in dt_polys:
                    pts = polygon_from_any(poly)
                    if pts is None:
                        continue
                    box = poly_bbox(pts)
                    if any(bbox_iou(box, known) >= 0.75 for known in recognized_boxes):
                        continue
                    append_detection_only(poly)
                return
            if dt_polys and texts and len(dt_polys) == len(texts):
                for index, poly in enumerate(dt_polys):
                    append_text_poly(poly, texts[index], scores[index] if index < len(scores) else 1.0)
                return
            if dt_polys:
                for poly in dt_polys:
                    append_detection_only(poly)
                return
            for value in payload.values():
                visit(value, depth + 1)
            return
        if isinstance(payload, (list, tuple)):
            if len(payload) >= 2:
                pts = polygon_from_any(payload[0])
                parsed = text_score_from_any(payload[1])
                if pts is not None and parsed is not None:
                    out.append((pts, parsed[0], parsed[1]))
                    return
            for value in payload:
                visit(value, depth + 1)

    visit(prediction)
    return deduplicate_predictions(out)


def create_ocr_engine(cfg: TextConfig, paddle_kwargs_json: str | None) -> tuple[Any, dict[str, Any]]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit("PaddleOCR is required. Install it with: pip install paddlepaddle paddleocr") from exc

    attempts: list[dict[str, Any]] = []
    if paddle_kwargs_json:
        import json as json_module
        parsed = json_module.loads(paddle_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--paddle-kwargs-json must decode to an object")
        attempts.append(parsed)
    attempts.extend(
        [
            {
                "lang": cfg.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "text_rec_score_thresh": cfg.min_confidence,
                "text_det_limit_side_len": cfg.text_det_limit_side_len,
                "text_det_thresh": cfg.text_det_thresh,
                "text_det_box_thresh": cfg.text_det_box_thresh,
                "text_det_unclip_ratio": cfg.text_det_unclip_ratio,
            },
            {
                "lang": cfg.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "text_rec_score_thresh": cfg.min_confidence,
                "text_det_limit_side_len": cfg.text_det_limit_side_len,
                "text_det_thresh": cfg.text_det_thresh,
                "text_det_box_thresh": cfg.text_det_box_thresh,
                "text_det_unclip_ratio": cfg.text_det_unclip_ratio,
            },
            {"lang": cfg.lang, "use_angle_cls": False, "show_log": False},
            {"lang": cfg.lang, "use_angle_cls": False},
        ]
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs), kwargs
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to initialize PaddleOCR. Last error: {last_error}")


def run_ocr_once(engine: Any, rgb: np.ndarray) -> list[tuple[list[list[int]], str, float]]:
    errors: list[str] = []
    if hasattr(engine, "predict"):
        for image in (rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            try:
                detections = normalize_prediction(engine.predict(image))
                if detections:
                    return detections
            except Exception as exc:
                errors.append(str(exc))
    if hasattr(engine, "ocr"):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for kwargs in ({"cls": False}, {}):
            try:
                detections = normalize_prediction(engine.ocr(bgr, **kwargs))
                if detections:
                    return detections
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        print(f"  [ocr] no detections/errors: {errors[-3:]}")
    return []


def enhance_ocr_image(rgb: np.ndarray, use_clahe: bool) -> np.ndarray:
    if not use_clahe:
        return rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# ============================================================
# Ink-based bbox refinement / erasure mask
# ============================================================

def ink_mask_in_polygon(page_gray: np.ndarray, poly: list[list[int]], cfg: TextConfig) -> np.ndarray:
    """Boolean mask (full page shape) of dark ("ink") pixels inside the polygon."""
    height, width = page_gray.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    pts = np.asarray(poly, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return (page_gray < int(cfg.text_gray_threshold)) & (mask > 0)


def component_is_line_like(ys: np.ndarray, xs: np.ndarray, cfg: TextConfig) -> bool:
    """True if a connected ink component's shape looks like a drawn line/tick rather
    than a text glyph, via two independent geometric tests on its minimum-area rotated
    bounding rectangle:

    1. Thin & elongated in ANY direction (not just axis-aligned) -- e.g. a diagonal
       arrow shaft or tick mark. A minimum absolute length is required so short
       punctuation strokes that are themselves thin (e.g. "/", "(", ")") aren't mistaken
       for a line -- a real intruding wire/tick is usually longer than a single
       character's own stroke.
    2. A near-perfect filled rectangle (pixel count close to its own bounding rect's
       area) whose thickness exceeds a normal text stroke's width -- e.g. a wire lead or
       terminal tick drawn as a solid bar. Real glyphs are comparatively thin-stroked and
       irregular (curves, gaps, serifs), so they never fill their bounding rect this
       completely at this thickness.
    """
    if xs.size < 3:
        return False
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    thickness = min(rw, rh)
    length = max(rw, rh)
    if thickness <= 0:
        return False

    if length >= float(cfg.line_shape_min_length_px):
        aspect = length / thickness
        if thickness <= float(cfg.line_shape_max_thickness_px) and aspect >= float(cfg.line_shape_min_aspect_ratio):
            return True

    rect_area = rw * rh
    if rect_area > 0:
        fill_ratio = xs.size / rect_area
        if thickness >= float(cfg.thick_line_min_thickness_px) and fill_ratio >= float(cfg.thick_line_min_fill_ratio):
            return True

    return False


def isolate_text_ink(
    raw_bbox: list[int],
    poly: list[list[int]],
    page_gray: np.ndarray,
    cfg: TextConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boolean mask (full page shape) of ink believed to be actual text glyph strokes.

    A wire, dash, or connector tick that merely passes through or touches the label is
    typically still connected to the rest of the wire network just outside the label's
    own footprint. We probe a small crop padded around the raw detection, find connected
    ink components in that crop, and drop:
      1. any component that touches the crop's edge -- it necessarily continues beyond
         the probed area, so it is presumed to be line/connector ink; and
      2. any remaining component whose shape is thin & elongated (in any direction),
         e.g. a diagonal arrow/tick fully contained in the crop but not part of a glyph.
    Both reasons are checked independently and never restored, even if that leaves no
    text ink at all for this detection (e.g. every stroke of a misread capacitor/connector
    symbol touches the probe edge, or is itself a curved/thin non-glyph shape): a
    detection whose ink cannot be confidently attributed to real text ends up with no
    erasure, rather than risk damaging a line or a misread component symbol. This trades
    away erasing some very tightly-bounded real labels (where OCR's polygon already hugs
    the glyphs so closely that real ink reaches the probe pad) in exchange for never
    erasing something that turns out not to be text.
    """
    height, width = page_gray.shape[:2]
    poly_ink = ink_mask_in_polygon(page_gray, poly, cfg)
    pad = max(0, int(cfg.line_probe_pad_px))
    crop_box = clip_bbox(
        [raw_bbox[0] - pad, raw_bbox[1] - pad, raw_bbox[2] + pad, raw_bbox[3] + pad], width, height
    )
    cx1, cy1, cx2, cy2 = crop_box
    if cx2 <= cx1 or cy2 <= cy1:
        return poly_ink, {"method": "invalid_probe_crop"}

    crop_gray = page_gray[cy1:cy2, cx1:cx2]
    crop_ink = crop_gray < int(cfg.text_gray_threshold)
    if not crop_ink.any():
        return poly_ink, {"method": "no_ink_in_probe_crop"}

    num_labels, labels = cv2.connectedComponents(crop_ink.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return poly_ink, {"method": "no_ink_components_in_probe_crop"}

    h, w = crop_ink.shape
    boundary_labels: set[int] = set()
    boundary_labels.update(labels[0, :].tolist())
    boundary_labels.update(labels[h - 1, :].tolist())
    boundary_labels.update(labels[:, 0].tolist())
    boundary_labels.update(labels[:, w - 1].tolist())
    boundary_labels.discard(0)

    line_shaped_labels: set[int] = set()
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if component_is_line_like(ys, xs, cfg):
            line_shaped_labels.add(label)

    exclude_labels = boundary_labels | line_shaped_labels
    if not exclude_labels:
        return poly_ink, {"method": "no_excluded_components"}

    keep_local = crop_ink & ~np.isin(labels, list(exclude_labels))
    text_ink = np.zeros((height, width), dtype=bool)
    text_ink[cy1:cy2, cx1:cx2] = keep_local

    return text_ink, {
        "method": "boundary_and_shape_component_exclusion",
        "boundary_excluded_components": int(len(boundary_labels)),
        "line_shaped_excluded_components": int(len(line_shaped_labels)),
        "probe_pad_px": int(pad),
        "remaining_ink": bool(text_ink.any()),
    }


def bbox_orientation(bbox: Sequence[int]) -> str:
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    return "vertical" if height > width * 1.35 else "horizontal"


def strip_perpendicular_crossing_lines(
    text_ink: np.ndarray,
    raw_bbox: list[int],
    orientation: str,
    pad: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove rows/columns that demonstrably continue past the label on both sides.

    A wire (e.g. a horizontal bus line) crossing straight through a label -- not just
    attached at one end, but passing behind the whole box -- has ink at a given row (for
    horizontal text) or column (for vertical text) that reaches all the way to BOTH edges
    of a probe crop around the raw bbox (padded by `pad`, which may be 0 to test the raw
    bbox's own edges directly). Requiring both edges (not just a coverage ratio measured
    only inside the raw bbox) is deliberately conservative: dense, tightly-kerned glyph
    rows/columns of real text essentially never extend to both edges at once, whereas a
    genuine crossing wire does by construction -- and often already defines the raw bbox's
    own edges, since a loose OCR polygon absorbed the visible stretch of wire ink.
    """
    height, width = text_ink.shape
    pad = max(0, int(pad))
    x1, y1, x2, y2 = clip_bbox(
        [raw_bbox[0] - pad, raw_bbox[1] - pad, raw_bbox[2] + pad, raw_bbox[3] + pad], width, height
    )
    if x2 <= x1 or y2 <= y1:
        return text_ink, {"method": "invalid_probe_crop"}
    local = text_ink[y1:y2, x1:x2]
    h, w = local.shape
    if h < 3 or w < 3:
        return text_ink, {"method": "too_small_to_check"}
    edge = 2
    if orientation == "horizontal":
        left_touch = local[:, :edge].any(axis=1)
        right_touch = local[:, -edge:].any(axis=1)
        line_rows = left_touch & right_touch
        if not line_rows.any():
            return text_ink, {"method": "no_full_span_rows"}
        cleaned_local = local.copy()
        cleaned_local[line_rows, :] = False
        removed_count = int(np.count_nonzero(line_rows))
    else:
        top_touch = local[:edge, :].any(axis=0)
        bottom_touch = local[-edge:, :].any(axis=0)
        line_cols = top_touch & bottom_touch
        if not line_cols.any():
            return text_ink, {"method": "no_full_span_cols"}
        cleaned_local = local.copy()
        cleaned_local[:, line_cols] = False
        removed_count = int(np.count_nonzero(line_cols))
    if not cleaned_local.any():
        return text_ink, {"method": "fallback_all_rows_or_cols_spanned", "removed_count": removed_count}
    out = text_ink.copy()
    out[y1:y2, x1:x2] = cleaned_local
    return out, {"method": "full_span_crossing_lines_stripped", "removed_count": removed_count, "probe_pad_px": pad}


def strip_edge_band_divider_lines(
    text_ink: np.ndarray,
    raw_bbox: list[int],
    orientation: str,
    cfg: TextConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strip a near-unbroken, near-full-width row/column sitting right at the raw bbox's
    own top/bottom (or left/right) edge -- e.g. a table/grid divider line running directly
    along the edge of a compact pin-number or wire-color cell. Unlike a wire crossing
    through the middle, this doesn't need to reach outside the bbox at all: a table
    divider is drawn flush with the cell edge. The high coverage threshold keeps this
    from tripping on real glyph tops/bottoms, which -- even when several characters'
    caps align on nearly the same row -- still have small inter-letter gaps a true
    divider line does not.
    """
    x1, y1, x2, y2 = raw_bbox
    local = text_ink[y1:y2, x1:x2]
    h, w = local.shape
    if h < 5 or w < 5:
        return text_ink, {"method": "too_small_to_check"}
    threshold = float(cfg.edge_divider_coverage_ratio)
    band = max(1, int(cfg.edge_divider_band_px))
    cleaned_local = local.copy()
    removed_rows_or_cols = 0
    if orientation == "horizontal":
        coverage = local.sum(axis=1) / max(1, w)
        edge_rows = np.zeros(h, dtype=bool)
        edge_rows[:band] = True
        edge_rows[-band:] = True
        divider_rows = edge_rows & (coverage >= threshold)
        if divider_rows.any():
            cleaned_local[divider_rows, :] = False
            removed_rows_or_cols = int(np.count_nonzero(divider_rows))
    else:
        coverage = local.sum(axis=0) / max(1, h)
        edge_cols = np.zeros(w, dtype=bool)
        edge_cols[:band] = True
        edge_cols[-band:] = True
        divider_cols = edge_cols & (coverage >= threshold)
        if divider_cols.any():
            cleaned_local[:, divider_cols] = False
            removed_rows_or_cols = int(np.count_nonzero(divider_cols))
    if removed_rows_or_cols == 0:
        return text_ink, {"method": "no_edge_divider_found"}
    if not cleaned_local.any():
        return text_ink, {"method": "fallback_divider_would_empty_mask", "removed_count": removed_rows_or_cols}
    out = text_ink.copy()
    out[y1:y2, x1:x2] = cleaned_local
    return out, {
        "method": "edge_divider_line_stripped",
        "removed_count": removed_rows_or_cols,
        "coverage_threshold": threshold,
        "band_px": band,
    }


def refine_bbox_from_ink(
    raw_bbox: list[int],
    poly: list[list[int]],
    page_gray: np.ndarray,
    text: str,
    cfg: TextConfig,
) -> tuple[list[int], dict[str, Any], np.ndarray]:
    height, width = page_gray.shape[:2]
    x1, y1, x2, y2 = clip_bbox(raw_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        empty = np.zeros((height, width), dtype=bool)
        return [x1, y1, x2, y2], {"method": "invalid_raw_bbox_kept"}, empty
    text_ink, isolation_debug = isolate_text_ink(raw_bbox, poly, page_gray, cfg)
    orientation = bbox_orientation([x1, y1, x2, y2])
    text_ink, crossing_debug = strip_perpendicular_crossing_lines(
        text_ink, [x1, y1, x2, y2], orientation, cfg.line_probe_pad_px
    )
    # A crossing line's visible stretch often already defines the raw bbox's own edges
    # (a loose OCR polygon absorbed it), so also check with zero padding -- but only for
    # labels with enough characters that a single glyph can't plausibly span the whole
    # raw bbox on its own, to avoid nuking short pin-number style labels.
    compact_len = len(re.sub(r"[^A-Za-z0-9]", "", text))
    tight_crossing_debug: dict[str, Any] = {"method": "skipped_short_text", "compact_len": compact_len}
    if compact_len >= int(cfg.tight_crossing_line_min_alnum):
        text_ink, tight_crossing_debug = strip_perpendicular_crossing_lines(
            text_ink, [x1, y1, x2, y2], orientation, 0
        )
    text_ink, edge_divider_debug = strip_edge_band_divider_lines(text_ink, [x1, y1, x2, y2], orientation, cfg)
    ys, xs = np.where(text_ink)
    line_debug = {
        "text_ink_isolation": isolation_debug,
        "crossing_line_removal": crossing_debug,
        "tight_crossing_line_removal": tight_crossing_debug,
        "edge_divider_removal": edge_divider_debug,
        "orientation": orientation,
    }
    if xs.size < 2 or ys.size < 2:
        return [x1, y1, x2, y2], {
            "method": "raw_bbox_kept_no_clear_ink",
            "ink_pixels": int(xs.size),
            **line_debug,
        }, text_ink
    pad = max(0, int(cfg.refined_ink_padding_px))
    refined = clip_bbox(
        [int(xs.min()) - pad, int(ys.min()) - pad, int(xs.max()) + pad + 1, int(ys.max()) + pad + 1],
        width,
        height,
    )
    if bbox_area(refined) < max(4, bbox_area(raw_bbox) * 0.03):
        return [x1, y1, x2, y2], {
            "method": "raw_bbox_kept_refinement_too_small",
            "candidate_refined_bbox": refined,
            "ink_pixels": int(xs.size),
            **line_debug,
        }, text_ink
    return refined, {
        "method": "foreground_text_ink",
        "ink_pixels": int(xs.size),
        "raw_area_px": int(bbox_area(raw_bbox)),
        "refined_area_px": int(bbox_area(refined)),
        **line_debug,
    }, text_ink


# ============================================================
# Quality filtering (ported from backup/07_text.py, tile-fields removed)
# ============================================================

def record_orientation(record: TextRecord) -> str:
    width = max(1, int(record.bbox[2]) - int(record.bbox[0]))
    height = max(1, int(record.bbox[3]) - int(record.bbox[1]))
    return "vertical" if height > width * 1.35 else "horizontal"


def record_text_height(record: TextRecord, orientation: str | None = None) -> int:
    orientation = orientation or record_orientation(record)
    width = max(1, int(record.bbox[2]) - int(record.bbox[0]))
    height = max(1, int(record.bbox[3]) - int(record.bbox[1]))
    return int(width if orientation == "vertical" else height)


def raw_record_text_height(record: TextRecord, orientation: str | None = None) -> int:
    orientation = orientation or record_orientation(record)
    width = max(1, int(record.raw_bbox[2]) - int(record.raw_bbox[0]))
    height = max(1, int(record.raw_bbox[3]) - int(record.raw_bbox[1]))
    return int(width if orientation == "vertical" else height)


def median_or(values: list[float], fallback: float) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0]
    if not clean:
        return float(fallback)
    return float(np.median(np.asarray(clean, dtype=np.float32)))


def compute_page_text_stats(records: Sequence[TextRecord]) -> dict[str, Any]:
    horizontal: list[float] = []
    vertical: list[float] = []
    skipped: Counter[str] = Counter()
    for record in records:
        compact = alnum_text(record.text)
        if len(compact) < 2:
            skipped["too_short"] += 1
            continue
        if "?" in record.text:
            skipped["question_mark"] += 1
            continue
        if record.ocr_confidence < 0.70:
            skipped["low_confidence"] += 1
            continue
        orientation = record_orientation(record)
        height = float(record_text_height(record, orientation))
        if height < 4 or height > 80:
            skipped["implausible_height"] += 1
            continue
        if orientation == "vertical":
            vertical.append(height)
        else:
            horizontal.append(height)
    h_median = median_or(horizontal, 16.0)
    v_median = median_or(vertical, h_median)
    return {
        "horizontal_median_height_px": round(float(h_median), 3),
        "vertical_median_height_px": round(float(v_median), 3),
        "horizontal_sample_count": int(len(horizontal)),
        "vertical_sample_count": int(len(vertical)),
        "skipped_counts": dict(skipped),
    }


def required_confidence_for_record(record: TextRecord, cfg: TextConfig) -> tuple[float, str]:
    compact = alnum_text(record.text)
    if record.category == "wire_color":
        if len(compact) <= int(cfg.short_text_max_alnum):
            return float(cfg.short_wire_color_min_confidence), "short_wire_color"
        return float(cfg.min_confidence), "wire_color"
    if re.fullmatch(r"\d", compact):
        return float(cfg.single_digit_min_confidence), "single_digit"
    if re.fullmatch(r"[A-Z]", compact):
        return float(cfg.single_letter_min_confidence), "single_letter"
    if re.fullmatch(r"\d+[A-Z]+", compact) or re.fullmatch(r"[A-Z]+\d+", compact):
        return float(cfg.short_alnum_min_confidence), "short_digit_letter"
    if len(compact) <= int(cfg.short_text_max_alnum):
        return float(cfg.short_alnum_min_confidence), "short_alnum"
    return float(cfg.min_confidence), "default"


def text_record_debug_json(record: TextRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "type": "text",
        "category": record.category,
        "text": record.text,
        "normalized_text": record.normalized_text,
        "ocr_confidence": float(record.ocr_confidence),
        "category_confidence": float(record.category_confidence),
        "bbox": record.bbox,
        "raw_bbox": record.raw_bbox,
        "polygon": record.polygon,
        "erased": bool(record.erased),
        "ink_pixel_count": int(record.ink_pixel_count),
        "debug": record.debug,
    }


def quality_filter_text_records(
    records: list[TextRecord],
    cfg: TextConfig,
) -> tuple[list[TextRecord], list[dict[str, Any]], dict[str, Any]]:
    stats = compute_page_text_stats(records)
    h_median = float(stats["horizontal_median_height_px"])
    v_median = float(stats["vertical_median_height_px"])
    kept: list[TextRecord] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        orientation = record_orientation(record)
        text_height = float(record_text_height(record, orientation))
        raw_text_height = float(raw_record_text_height(record, orientation))
        page_height = v_median if orientation == "vertical" else h_median
        height_ratio = float(text_height / max(1.0, page_height))
        raw_height_ratio = float(raw_text_height / max(1.0, page_height))
        effective_height_ratio = min(height_ratio, raw_height_ratio)
        compact = alnum_text(record.text)
        required_confidence, confidence_rule = required_confidence_for_record(record, cfg)
        quality = {
            "orientation": orientation,
            "text_height_px": round(float(text_height), 3),
            "raw_text_height_px": round(float(raw_text_height), 3),
            "page_reference_height_px": round(float(page_height), 3),
            "height_ratio": round(float(height_ratio), 4),
            "raw_height_ratio": round(float(raw_height_ratio), 4),
            "effective_height_ratio": round(float(effective_height_ratio), 4),
            "required_confidence": round(float(required_confidence), 4),
            "confidence_rule": confidence_rule,
            "alnum_text": compact,
        }
        record.debug = {**record.debug, "quality_gate": quality}

        reason: str | None = None
        if not compact and re.fullmatch(r"[\W_?？]+", record.text.strip() or ""):
            reason = "punctuation_or_question_only"
        elif record.text.strip() and set(record.text.strip()) <= {"?", "？"}:
            reason = "question_mark_only"
        elif effective_height_ratio > float(cfg.strong_text_height_ratio):
            reason = "strong_text_height_outlier"
        elif effective_height_ratio > float(cfg.max_text_height_ratio):
            if len(compact) <= 1:
                reason = "text_height_outlier"
            elif record.ocr_confidence < required_confidence:
                reason = "text_height_outlier_low_confidence"
        elif record.ocr_confidence < required_confidence:
            reason = "low_confidence_for_short_text"
        elif re.fullmatch(r"[A-Z]", compact) and record.category != "wire_color":
            reason = "single_letter_non_wire_text"

        if reason is None:
            kept.append(record)
            continue
        reason_counts[reason] += 1
        rejected_item = text_record_debug_json(record)
        rejected_item["reason"] = reason
        rejected_item["quality_gate"] = quality
        rejected.append(rejected_item)

    stats["quality_gate_config"] = {
        "max_text_height_ratio": float(cfg.max_text_height_ratio),
        "strong_text_height_ratio": float(cfg.strong_text_height_ratio),
        "single_letter_min_confidence": float(cfg.single_letter_min_confidence),
        "single_digit_min_confidence": float(cfg.single_digit_min_confidence),
        "short_alnum_min_confidence": float(cfg.short_alnum_min_confidence),
        "short_wire_color_min_confidence": float(cfg.short_wire_color_min_confidence),
    }
    stats["rejected_counts"] = dict(reason_counts)
    stats["num_rejected"] = int(len(rejected))
    stats["num_kept"] = int(len(kept))
    return kept, rejected, stats


def dedupe_text_records(records: list[TextRecord], threshold: float) -> tuple[list[TextRecord], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda item: (-item.ocr_confidence, bbox_area(item.bbox), item.bbox[1], item.bbox[0], item.text))
    kept: list[TextRecord] = []
    rejected: list[dict[str, Any]] = []
    for record in ordered:
        duplicate_of: TextRecord | None = None
        for previous in kept:
            if bbox_iou(record.bbox, previous.bbox) >= threshold:
                duplicate_of = previous
                break
        if duplicate_of is None:
            kept.append(record)
            continue
        rejected.append(
            {
                "id": record.id,
                "text": record.text,
                "bbox": record.bbox,
                "reason": "duplicate_detection",
                "duplicate_of": duplicate_of.id,
            }
        )
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.id))
    for index, record in enumerate(kept, start=1):
        record.id = f"p{record.page:03d}_text_{index:04d}"
    return kept, rejected


# ============================================================
# Drawing
# ============================================================

def draw_label(out: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.42) -> None:
    y = max(14, int(y))
    cv2.putText(out, label, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, label, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def add_legend(out: np.ndarray) -> None:
    items = [
        ("wire_color", REVIEW_COLORS["wire_color"]),
        ("number", REVIEW_COLORS["number"]),
        ("other_text", REVIEW_COLORS["other_text"]),
    ]
    x = 14
    y = 18
    line_h = 24
    max_text = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for label, _ in items)
    box_w = max_text + 42
    box_h = line_h * len(items) + 12
    cv2.rectangle(out, (x - 8, y - 14), (x + box_w, y - 14 + box_h), (255, 255, 255), -1)
    cv2.rectangle(out, (x - 8, y - 14), (x + box_w, y - 14 + box_h), (190, 190, 190), 1)
    for index, (label, color) in enumerate(items):
        yy = y + index * line_h
        cv2.rectangle(out, (x, yy - 9), (x + 18, yy + 5), color, -1)
        cv2.putText(out, label, (x + 26, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)


def draw_text_records(out: np.ndarray, records: Sequence[TextRecord]) -> None:
    h, w = out.shape[:2]
    for record in records:
        color = REVIEW_COLORS.get(record.category, REVIEW_COLORS["other_text"])
        box = clip_bbox(record.bbox, w, h)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2, cv2.LINE_AA)
        label = f"{record.text}"[:32]
        draw_label(out, label, box[0], box[1] - 4, color, scale=0.38)


# ============================================================
# Per-page processing
# ============================================================

def process_page(
    pdf_name: str,
    page_number: int,
    input_dir: Path,
    output_paths: dict[str, Path],
    engine: Any,
    cfg: TextConfig,
) -> dict[str, Any]:
    root = output_paths["root"]
    input_path = input_dir / f"page_{page_number:03d}.png"
    page_rgb = load_rgb(input_path)
    page_gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    height, width = page_rgb.shape[:2]

    ocr_input = enhance_ocr_image(page_rgb, cfg.use_clahe)
    raw = run_ocr_once(engine, ocr_input)

    raw_predictions: list[dict[str, Any]] = []
    ignored_detections: list[dict[str, Any]] = []
    records: list[TextRecord] = []
    text_ink_by_record_id: dict[int, np.ndarray] = {}

    for poly, text, score in raw:
        raw_bbox = clip_bbox(poly_bbox(poly), width, height)
        raw_predictions.append(
            {
                "polygon": [[int(x), int(y)] for x, y in poly],
                "raw_bbox": raw_bbox,
                "raw_text": str(text),
                "raw_confidence": float(score),
            }
        )
        if not str(text).strip() or float(score) < cfg.min_confidence:
            ignored_detections.append(
                {
                    "polygon": [[int(x), int(y)] for x, y in poly],
                    "raw_bbox": raw_bbox,
                    "raw_text": str(text),
                    "raw_confidence": float(score),
                    "reason": "empty_or_low_confidence",
                }
            )
            continue
        refined_bbox, refinement, text_ink = refine_bbox_from_ink(raw_bbox, poly, page_gray, str(text), cfg)
        category, normalized, category_confidence, class_debug = classify_text(text)
        record = TextRecord(
            id=f"p{page_number:03d}_text_raw_{len(records) + 1:04d}",
            page=int(page_number),
            text=str(text).strip(),
            normalized_text=normalized,
            category=category,
            category_confidence=float(category_confidence),
            ocr_confidence=float(score),
            bbox=refined_bbox,
            raw_bbox=raw_bbox,
            polygon=[[int(x), int(y)] for x, y in poly],
            erased=False,
            ink_pixel_count=int(np.count_nonzero(text_ink)),
            debug={
                "bbox_refinement": refinement,
                "classification": class_debug,
            },
        )
        text_ink_by_record_id[id(record)] = text_ink
        records.append(record)

    records, quality_rejections, page_text_stats = quality_filter_text_records(records, cfg)
    records, duplicate_rejections = dedupe_text_records(records, cfg.dedupe_iou_threshold)

    # Determine erasure: only records whose OCR confidence clears the strict
    # erase threshold get their own ink pixels erased (not the whole bbox).
    erase_mask = np.zeros((height, width), dtype=bool)
    pad = max(0, int(cfg.erase_ink_pad_px))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1)) if pad > 0 else None
    for record in records:
        ink = text_ink_by_record_id.get(id(record))
        if ink is None:
            ink = ink_mask_in_polygon(page_gray, record.polygon, cfg)
        record.ink_pixel_count = int(np.count_nonzero(ink))
        if record.ocr_confidence >= cfg.erase_min_confidence:
            record.erased = True
            if dilate_kernel is not None:
                ink = cv2.dilate(ink.astype(np.uint8), dilate_kernel).astype(bool)
            erase_mask |= ink

    circuit_image = page_rgb.copy()
    circuit_image[erase_mask] = 255

    review_image = page_rgb.copy()
    draw_text_records(review_image, records)
    add_legend(review_image)

    image_path = output_paths["images"] / f"page_{page_number:03d}.png"
    circuit_image_path = (
        output_paths["circuit_images"] / f"page_{page_number:03d}.png"
        if "circuit_images" in output_paths
        else None
    )
    json_path = output_paths["json"] / f"page_{page_number:03d}.json"
    debug_path = output_paths["debug"] / f"page_{page_number:03d}.json"

    save_rgb(image_path, review_image)
    if circuit_image_path is not None:
        save_rgb(circuit_image_path, circuit_image)

    category_counts = Counter(record.category for record in records)
    erased_count = int(sum(1 for record in records if record.erased))
    save_json(
        json_path,
        {
            "pdf": pdf_name,
            "page": int(page_number),
            "stage": STAGE_NAME,
            "source": {
                "stage": INPUT_STAGE,
                "circuit_image_path": str(input_path),
            },
            "image_width": int(width),
            "image_height": int(height),
            "review_image_path": str(image_path),
            "circuit_image_path": str(circuit_image_path) if circuit_image_path is not None else None,
            "summary": {
                "num_texts": int(len(records)),
                "category_counts": dict(category_counts),
                "erased_count": erased_count,
                "kept_not_erased_count": int(len(records) - erased_count),
                "erase_mask_pixels": int(np.count_nonzero(erase_mask)),
            },
            "texts": [text_record_debug_json(record) for record in records],
        },
    )
    save_json(
        debug_path,
        {
            "pdf": pdf_name,
            "page": int(page_number),
            "stage": STAGE_NAME,
            "config": asdict(cfg),
            "page_text_stats": page_text_stats,
            "raw_ocr_predictions": raw_predictions,
            "ignored_detections": ignored_detections,
            "quality_rejections": quality_rejections,
            "duplicate_rejections": duplicate_rejections,
            "review_colors": {
                "wire_color": "blue",
                "number": "orange",
                "other_text": "green",
                "erased_overlay": "red",
            },
        },
    )

    return {
        "page": int(page_number),
        "num_texts": int(len(records)),
        "category_counts": dict(category_counts),
        "erased_count": erased_count,
        "review_image": review_image,
        "circuit_image": circuit_image,
        "review_image_path": rel_path(image_path, root),
        "circuit_image_path": rel_path(circuit_image_path, root) if circuit_image_path is not None else None,
        "json_path": f"json/page_{page_number:03d}.json",
        "debug_path": f"debug/page_{page_number:03d}.json",
    }


# ============================================================
# Main
# ============================================================

def build_config(args: argparse.Namespace) -> TextConfig:
    return TextConfig(
        dpi=max(1, int(args.dpi)),
        lang=str(args.lang),
        min_confidence=max(0.0, min(1.0, float(args.min_confidence))),
        erase_min_confidence=max(0.0, min(1.0, float(args.erase_min_confidence))),
        text_gray_threshold=max(0, min(255, int(args.text_gray_threshold))),
        erase_ink_pad_px=max(0, int(args.erase_ink_pad_px)),
        line_probe_pad_px=max(0, int(args.line_probe_pad_px)),
        line_shape_min_aspect_ratio=max(1.0, float(args.line_shape_min_aspect_ratio)),
        line_shape_max_thickness_px=max(1, int(args.line_shape_max_thickness_px)),
        line_shape_min_length_px=max(1, int(args.line_shape_min_length_px)),
        edge_divider_coverage_ratio=max(0.0, min(1.0, float(args.edge_divider_coverage_ratio))),
        edge_divider_band_px=max(1, int(args.edge_divider_band_px)),
        thick_line_min_thickness_px=max(1, int(args.thick_line_min_thickness_px)),
        thick_line_min_fill_ratio=max(0.0, min(1.0, float(args.thick_line_min_fill_ratio))),
        tight_crossing_line_min_alnum=max(1, int(args.tight_crossing_line_min_alnum)),
        text_det_limit_side_len=max(1, int(args.text_det_limit_side_len)),
        text_det_thresh=float(args.text_det_thresh),
        text_det_box_thresh=float(args.text_det_box_thresh),
        text_det_unclip_ratio=float(args.text_det_unclip_ratio),
        use_clahe=bool(args.use_clahe),
        refined_ink_padding_px=max(0, int(args.refined_ink_padding_px)),
        dedupe_iou_threshold=max(0.0, min(1.0, float(args.dedupe_iou_threshold))),
        max_text_height_ratio=max(1.0, float(args.max_text_height_ratio)),
        strong_text_height_ratio=max(1.0, float(args.strong_text_height_ratio)),
        short_text_max_alnum=max(1, int(args.short_text_max_alnum)),
        single_letter_min_confidence=max(0.0, min(1.0, float(args.single_letter_min_confidence))),
        single_digit_min_confidence=max(0.0, min(1.0, float(args.single_digit_min_confidence))),
        short_alnum_min_confidence=max(0.0, min(1.0, float(args.short_alnum_min_confidence))),
        short_wire_color_min_confidence=max(0.0, min(1.0, float(args.short_wire_color_min_confidence))),
    )


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> None:
    pdf_stem = pdf_path.stem
    input_dir = resolve_input_dir(pdf_stem)
    pages = select_pages(args.pages, available_pages(input_dir))
    output = make_output_dirs(pdf_stem, args.preserve, args.save_circuit_images)
    cfg = build_config(args)

    print(f"\nProcessing: {pdf_stem}")
    print(f"Input circuit_images ({INPUT_STAGE}): {input_dir}")
    print(f"Output: {output['root']}")
    print(f"Pages: {pages}")
    print(f"  preserve existing output: {args.preserve}")
    print(f"  images: {output['images']}")
    print(f"  circuit_images: {output['circuit_images'] if args.save_circuit_images else 'disabled'}")
    print(f"  json: {output['json']}")
    print(f"  debug: {output['debug']}")
    print(f"  erase_min_confidence: {cfg.erase_min_confidence}")

    print("Initializing PaddleOCR...")
    engine, ocr_kwargs = create_ocr_engine(cfg, args.paddle_kwargs_json)
    print(f"PaddleOCR kwargs: {ocr_kwargs}")

    review_pages: list[np.ndarray] = []
    result_pages: list[np.ndarray] = []
    summary_pages: list[dict[str, Any]] = []
    total_category_counts: Counter[str] = Counter()
    total_erased = 0

    for page_number in pages:
        print(f"  Page {page_number}")
        result = process_page(pdf_path.name, page_number, input_dir, output, engine, cfg)
        review_pages.append(result.pop("review_image"))
        circuit_image = result.pop("circuit_image")
        if args.save_circuit_images:
            result_pages.append(circuit_image)
        total_category_counts.update(result["category_counts"])
        total_erased += result["erased_count"]
        summary_pages.append(result)
        print(
            f"    texts={result['num_texts']} categories={result['category_counts']} erased={result['erased_count']}"
        )

    if review_pages:
        image_to_pdf(review_pages, output["root"] / "review.pdf")
    if args.save_circuit_images and result_pages:
        image_to_pdf(result_pages, output["root"] / "result.pdf")
    save_json(
        output["json"] / "summary.json",
        {
            "pdf": pdf_path.name,
            "stage": STAGE_NAME,
            "input_stage": INPUT_STAGE,
            "input_circuit_images": str(input_dir),
            "output_root": str(output["root"]),
            "circuit_images_saved": bool(args.save_circuit_images),
            "pages": summary_pages,
            "category_counts": dict(total_category_counts),
            "total_erased": int(total_erased),
            "config": asdict(cfg),
            "ocr_engine": "PaddleOCR",
            "ocr_kwargs": ocr_kwargs,
        },
    )
    print(f"Review PDF: {output['root'] / 'review.pdf'}")
    if args.save_circuit_images:
        print(f"Result PDF: {output['root'] / 'result.pdf'}")
    else:
        print("Result PDF: disabled")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=str, default="bmw-328i-1997.pdf")
    p.add_argument("--pages", type=str, default="1-5")
    p.add_argument("-p", "--preserve", action="store_true", help="Preserve the output folder instead of clearing it first.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--lang", default="en")
    p.add_argument("--min-confidence", type=float, default=0.35, help="Minimum OCR confidence for a detection to be recorded at all.")
    p.add_argument("--erase-min-confidence", type=float, default=0.80, help="Strict OCR confidence threshold; only text at/above this is erased from the result image.")
    p.add_argument("--save-circuit-images", action="store_true", help="Also write per-page erased PNGs under circuit_images/. Disabled by default.")
    p.add_argument("--text-gray-threshold", type=int, default=245, help="Grayscale value below which a pixel counts as text ink.")
    p.add_argument("--erase-ink-pad-px", type=int, default=1, help="Dilation (px) applied to erased ink pixels to fully remove anti-aliased edges.")
    p.add_argument("--line-probe-pad-px", type=int, default=10, help="Padding (px) around each raw detection used to detect wire/connector ink touching the probe crop edge, which is excluded from text ink so erasure never eats into nearby lines.")
    p.add_argument("--line-shape-min-aspect-ratio", type=float, default=4.0, help="Connected ink components thinner/more elongated than this (any direction) are treated as a line/tick mark, not a text glyph.")
    p.add_argument("--line-shape-max-thickness-px", type=int, default=5, help="Max thickness (px) for a component to be considered line-like under --line-shape-min-aspect-ratio.")
    p.add_argument("--line-shape-min-length-px", type=int, default=22, help="Minimum length (px) for a thin component to be treated as a line rather than a short punctuation stroke like '/', '(', ')'.")
    p.add_argument("--edge-divider-coverage-ratio", type=float, default=0.92, help="Ink coverage at/above which a row/column right at the raw bbox's own edge is treated as a table/grid divider line, not text.")
    p.add_argument("--edge-divider-band-px", type=int, default=3, help="How many rows/columns from each edge of the raw bbox to check for a divider line.")
    p.add_argument("--thick-line-min-thickness-px", type=int, default=6, help="Min thickness (px) for a solid, near-perfectly-rectangular component to be treated as a drawn line/tick rather than a text glyph (which is comparatively thin-stroked).")
    p.add_argument("--thick-line-min-fill-ratio", type=float, default=0.85, help="Min fraction of its own bounding rectangle a component must fill to be treated as a solid line/tick under --thick-line-min-thickness-px.")
    p.add_argument("--tight-crossing-line-min-alnum", type=int, default=3, help="Minimum alnum character count before also checking for a crossing line touching the raw (unpadded) bbox edges; shorter labels skip this check to avoid false positives.")
    p.add_argument("--text-det-limit-side-len", type=int, default=4000)
    p.add_argument("--text-det-thresh", type=float, default=0.25)
    p.add_argument("--text-det-box-thresh", type=float, default=0.42)
    p.add_argument("--text-det-unclip-ratio", type=float, default=1.8)
    p.add_argument("--use-clahe", action=argparse.BooleanOptionalAction, default=True, help="Apply CLAHE contrast enhancement before OCR.")
    p.add_argument("--refined-ink-padding-px", type=int, default=2)
    p.add_argument("--dedupe-iou-threshold", type=float, default=0.82)
    p.add_argument("--max-text-height-ratio", type=float, default=1.3)
    p.add_argument("--strong-text-height-ratio", type=float, default=1.8)
    p.add_argument("--short-text-max-alnum", type=int, default=3)
    p.add_argument("--single-letter-min-confidence", type=float, default=0.95)
    p.add_argument("--single-digit-min-confidence", type=float, default=0.85)
    p.add_argument("--short-alnum-min-confidence", type=float, default=0.80)
    p.add_argument("--short-wire-color-min-confidence", type=float, default=0.55)
    p.add_argument("--paddle-kwargs-json", default=None)
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
