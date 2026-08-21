from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from pipeline_io import image_to_pdf, parse_page_range, save_json


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
SOURCE_STAGE_DIR = ROOT_DIR / "outputs" / "01_circuit"
OUTPUT_DIR = ROOT_DIR / "outputs" / Path(__file__).stem


# ============================================================
# Geometry / OCR normalization
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


def bbox_intersects(a: Sequence[int], b: Sequence[int]) -> bool:
    return min(int(a[2]), int(b[2])) > max(int(a[0]), int(b[0])) and min(int(a[3]), int(b[3])) > max(int(a[1]), int(b[1]))


def bbox_union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return [
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    ]


def expand_bbox(box: Sequence[int], pad: int, width: int, height: int) -> list[int] | None:
    return clip_bbox([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], width, height)


def poly_bbox(poly: Sequence[Sequence[float]]) -> list[int]:
    pts = np.asarray(poly, dtype=float)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2 + 1)), int(math.ceil(y2 + 1))]


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
    ordered = sorted(
        predictions,
        key=lambda item: (-float(item[2]), poly_bbox(item[0])[1], poly_bbox(item[0])[0], item[1]),
    )
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
        if not text:
            return False
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 1.0
        if score > 1.0:
            score /= 100.0
        out.append((pts, text, score))
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
                for index, poly in enumerate(rec_polys):
                    if index < len(texts):
                        append_text_poly(poly, texts[index], scores[index] if index < len(scores) else 1.0)
                return
            if dt_polys and texts and len(dt_polys) == len(texts):
                for index, poly in enumerate(dt_polys):
                    append_text_poly(poly, texts[index], scores[index] if index < len(scores) else 1.0)
                return
            parsed = text_score_from_any(payload)
            poly = payload.get("poly") or payload.get("polygon") or payload.get("bbox")
            if parsed is not None and poly is not None:
                append_text_poly(poly, parsed[0], parsed[1])
                return
            for value in payload.values():
                visit(value, depth + 1)
            return
        if isinstance(payload, (list, tuple)):
            if len(payload) >= 2:
                pts = polygon_from_any(payload[0])
                parsed = text_score_from_any(payload[1])
                if pts is not None and parsed is not None:
                    append_text_poly(pts, parsed[0], parsed[1])
                    return
            for value in payload:
                visit(value, depth + 1)

    visit(prediction)
    return deduplicate_predictions(out)


# ============================================================
# OCR region proposal
# ============================================================

def binarize_sheet(img: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray < int(threshold)


def merge_nearby_boxes(boxes: Sequence[Sequence[int]], gap: int, width: int, height: int) -> list[list[int]]:
    expanded = [expand_bbox(box, gap, width, height) for box in boxes]
    items = [(list(box), ex) for box, ex in zip(boxes, expanded) if ex is not None]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[list[int], list[int]]] = []
        for box, ex in sorted(items, key=lambda item: (item[0][1], item[0][0])):
            placed = False
            for index, (cur, cur_ex) in enumerate(merged):
                if bbox_intersects(ex, cur_ex):
                    new_box = bbox_union(cur, box)
                    new_ex = expand_bbox(new_box, gap, width, height) or new_box
                    merged[index] = (new_box, new_ex)
                    changed = True
                    placed = True
                    break
            if not placed:
                merged.append((box, ex))
        items = merged
    return [box for box, _ in sorted(items, key=lambda item: (item[0][1], item[0][0]))]


def detect_ocr_regions(img: np.ndarray, args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height, width = img.shape[:2]
    ink = binarize_sheet(img, args.ink_threshold)
    total_ink = int(ink.sum())
    debug = {
        "ink_threshold": int(args.ink_threshold),
        "total_ink_pixels": total_ink,
        "region_merge_kernel": [int(args.region_merge_kernel_w), int(args.region_merge_kernel_h)],
        "region_merge_iterations": int(args.region_merge_iterations),
        "region_padding": int(args.region_padding),
        "region_merge_gap": int(args.region_merge_gap),
        "min_region_ink_pixels": int(args.min_region_ink_pixels),
    }
    if total_ink < int(args.min_page_ink_pixels):
        debug["skipped"] = "page_ink_below_min_page_ink_pixels"
        return [], debug

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, int(args.region_merge_kernel_w)), max(1, int(args.region_merge_kernel_h))),
    )
    region_mask = cv2.dilate(ink.astype(np.uint8), kernel, iterations=max(1, int(args.region_merge_iterations))).astype(bool)
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(region_mask.astype(np.uint8), 8)

    boxes: list[list[int]] = []
    rejected = 0
    for label_id in range(1, num_labels):
        x, y, bw, bh, _area = stats[label_id]
        raw_box = [int(x), int(y), int(x + bw), int(y + bh)]
        padded = expand_bbox(raw_box, int(args.region_padding), width, height)
        if padded is None:
            rejected += 1
            continue
        x1, y1, x2, y2 = padded
        ink_count = int(ink[y1:y2, x1:x2].sum())
        if ink_count < int(args.min_region_ink_pixels):
            rejected += 1
            continue
        boxes.append(padded)

    boxes = merge_nearby_boxes(boxes, int(args.region_merge_gap), width, height)

    regions: list[dict[str, Any]] = []
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        local_ink = ink[y1:y2, x1:x2]
        ys, xs = np.where(local_ink)
        if len(xs) == 0:
            rejected += 1
            continue
        tight = [int(x1 + xs.min()), int(y1 + ys.min()), int(x1 + xs.max() + 1), int(y1 + ys.max() + 1)]
        regions.append({
            "id": f"region_{index:03d}",
            "bbox": [int(v) for v in box],
            "tight_ink_bbox": tight,
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "area": int((x2 - x1) * (y2 - y1)),
            "ink_pixels": int(local_ink.sum()),
        })

    if len(regions) > int(args.max_regions):
        tight_boxes = [region["tight_ink_bbox"] for region in regions]
        merged = expand_bbox(
            [
                min(box[0] for box in tight_boxes),
                min(box[1] for box in tight_boxes),
                max(box[2] for box in tight_boxes),
                max(box[3] for box in tight_boxes),
            ],
            int(args.region_padding),
            width,
            height,
        )
        if merged is not None:
            x1, y1, x2, y2 = merged
            regions = [{
                "id": "region_001",
                "bbox": merged,
                "tight_ink_bbox": [
                    min(box[0] for box in tight_boxes),
                    min(box[1] for box in tight_boxes),
                    max(box[2] for box in tight_boxes),
                    max(box[3] for box in tight_boxes),
                ],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "area": int((x2 - x1) * (y2 - y1)),
                "ink_pixels": int(ink[y1:y2, x1:x2].sum()),
                "merged_from_too_many_regions": True,
            }]
            debug["merged_all_regions"] = f"region_count_exceeded_max_regions_{int(args.max_regions)}"

    debug["initial_components"] = int(num_labels - 1)
    debug["rejected_components"] = int(rejected)
    debug["num_regions"] = int(len(regions))
    return regions, debug


def shift_polygon(poly: Sequence[Sequence[int]], dx: int, dy: int) -> list[list[int]]:
    return [[int(x) + int(dx), int(y) + int(dy)] for x, y in poly]


# ============================================================
# OCR / review rendering
# ============================================================

def create_ocr_engine(args) -> tuple[Any, dict[str, Any]]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit("PaddleOCR is required. Install it with: pip install paddlepaddle paddleocr") from exc

    attempts: list[dict[str, Any]] = []
    if args.paddle_kwargs_json:
        parsed = json.loads(args.paddle_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("--paddle-kwargs-json must decode to an object")
        attempts.append(parsed)

    attempts.extend([
        {
            "lang": args.lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_rec_score_thresh": args.min_confidence,
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_limit_type": args.text_det_limit_type,
            "text_det_thresh": args.text_det_thresh,
            "text_det_box_thresh": args.text_det_box_thresh,
            "text_det_unclip_ratio": args.text_det_unclip_ratio,
        },
        {
            "lang": args.lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "text_rec_score_thresh": args.min_confidence,
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_limit_type": args.text_det_limit_type,
            "text_det_thresh": args.text_det_thresh,
            "text_det_box_thresh": args.text_det_box_thresh,
            "text_det_unclip_ratio": args.text_det_unclip_ratio,
        },
        {"lang": args.lang, "use_angle_cls": False, "show_log": False},
        {"lang": args.lang, "use_angle_cls": False},
    ])

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
        print(f"    OCR returned no detections; last error: {errors[-1]}")
    return []


def draw_ocr_review(img: np.ndarray, records: Sequence[dict[str, Any]], args, regions: Sequence[dict[str, Any]] | None = None) -> np.ndarray:
    out = img.copy()
    if args.draw_regions and regions is not None:
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 190, 0), 1, cv2.LINE_AA)
    color = (255, 0, 0)
    for record in records:
        poly = np.asarray(record["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=args.draw_thickness, lineType=cv2.LINE_AA)
        if args.draw_labels:
            x1, y1, _, _ = record["bbox"]
            label = str(record["text"])[:32]
            cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def records_from_detections(
    detections: Sequence[tuple[list[list[int]], str, float]],
    width: int,
    height: int,
    page_no: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (poly, text, score) in enumerate(detections, start=1):
        text = str(text).strip()
        score = float(score)
        if not text or score < min_confidence:
            continue
        bbox = clip_bbox(poly_bbox(poly), width, height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        records.append({
            "id": f"page_{page_no:03d}_text_{index:04d}",
            "text": text,
            "confidence": score,
            "bbox": bbox,
            "polygon": [[int(x), int(y)] for x, y in poly],
        })
    return records


# ============================================================
# Stage IO
# ============================================================

def resolve_sheet_image_dir(pdf_path: Path) -> Path:
    return SOURCE_STAGE_DIR / pdf_path.stem / "sheet_images"


def page_image_path(sheet_dir: Path, page_no: int) -> Path:
    return sheet_dir / f"page_{page_no:03d}.png"


def infer_available_page_range(sheet_dir: Path) -> tuple[int, int]:
    pages = []
    for path in sheet_dir.glob("page_*.png"):
        try:
            pages.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if not pages:
        raise RuntimeError(f"No sheet image pages found under:\n{sheet_dir}")
    return min(pages), max(pages)


def process_pdf(pdf_path: Path, args, page_range=None) -> None:
    pdf_name = pdf_path.stem
    sheet_dir = resolve_sheet_image_dir(pdf_path)
    if not sheet_dir.is_dir():
        raise FileNotFoundError(
            f"01_circuit sheet_images not found:\n{sheet_dir}\n"
            "Run scripts/01_circuit.py for this PDF first."
        )

    output_root = OUTPUT_DIR / pdf_name
    json_dir = output_root / "json"
    image_dir = output_root / "images"

    if output_root.exists() and not args.preserve:
        resolved_output = output_root.resolve()
        resolved_stage = OUTPUT_DIR.resolve()
        if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
            raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    start_page, end_page = infer_available_page_range(sheet_dir) if page_range is None else page_range
    print(f"\nProcessing sheet text: {pdf_name}")
    print(f"Input sheet_images: {sheet_dir}")
    print(f"Output: {output_root}")
    print(f"  preserve existing output: {args.preserve}")
    print(f"  json: {json_dir}")
    print(f"  images: {image_dir}")

    engine, ocr_kwargs = create_ocr_engine(args)
    review_pages: list[np.ndarray] = []

    for page_no in range(start_page, end_page + 1):
        src_path = page_image_path(sheet_dir, page_no)
        if not src_path.is_file():
            print(f"  Page {page_no}: skipped, missing {src_path.name}")
            continue

        print(f"  Page {page_no}")
        img = np.array(Image.open(src_path).convert("RGB"))
        height, width = img.shape[:2]
        regions, region_debug = detect_ocr_regions(img, args)
        detections: list[tuple[list[list[int]], str, float]] = []
        total_raw_detections = 0
        page_ocr_start = time.perf_counter()
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            crop = img[y1:y2, x1:x2]
            region_start = time.perf_counter()
            local = run_ocr_once(engine, crop)
            elapsed_ms = int(round((time.perf_counter() - region_start) * 1000))
            total_raw_detections += len(local)
            mapped = [(shift_polygon(poly, x1, y1), text, score) for poly, text, score in local]
            detections.extend(mapped)
            region["raw_ocr_detections"] = int(len(local))
            region["ocr_elapsed_ms"] = elapsed_ms

        detections = deduplicate_predictions(detections)
        records = records_from_detections(detections, width, height, page_no, args.min_confidence)
        for region in regions:
            region["num_texts"] = int(sum(
                bbox_intersects(region["bbox"], record["bbox"])
                for record in records
            ))

        review = draw_ocr_review(img, records, args, regions=regions)
        json_path = json_dir / f"page_{page_no:03d}.json"
        image_path = image_dir / f"page_{page_no:03d}.png"

        save_json(json_path, {
            "page": int(page_no),
            "image_width": int(width),
            "image_height": int(height),
            "source_image": str(src_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "ocr_engine": "PaddleOCR",
            "ocr_kwargs": ocr_kwargs,
            "summary": {
                "ocr_mode": "region_crops",
                "num_ocr_regions": int(len(regions)),
                "raw_ocr_detections": int(total_raw_detections),
                "deduped_ocr_detections": int(len(detections)),
                "num_texts": int(len(records)),
                "min_confidence": float(args.min_confidence),
                "ocr_elapsed_seconds": round(float(time.perf_counter() - page_ocr_start), 3),
            },
            "region_detection": region_debug,
            "ocr_regions": regions,
            "texts": records,
        })
        Image.fromarray(review).save(image_path)
        review_pages.append(review)
        print(f"    saved: json/{json_path.name}, images/{image_path.name}, texts={len(records)}")

    review_pdf = output_root / "review.pdf"
    image_to_pdf(review_pages, review_pdf)
    print(f"Finished: {pdf_name}")
    print(f"Review PDF: {review_pdf}")


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--pdf", type=str, default="bmw-328i-1997.pdf")
    p.add_argument("--pages", type=str, default=None)
    p.add_argument("-p", "--preserve", action="store_true", help="Preserve the output folder instead of clearing it first.")

    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--ink-threshold", type=int, default=210)
    p.add_argument("--min-page-ink-pixels", type=int, default=8)
    p.add_argument("--min-region-ink-pixels", type=int, default=8)
    p.add_argument("--region-merge-kernel-w", type=int, default=96)
    p.add_argument("--region-merge-kernel-h", type=int, default=28)
    p.add_argument("--region-merge-iterations", type=int, default=1)
    p.add_argument("--region-padding", type=int, default=20)
    p.add_argument("--region-merge-gap", type=int, default=8)
    p.add_argument("--max-regions", type=int, default=80)
    p.add_argument("--text-det-limit-side-len", type=int, default=4000)
    p.add_argument("--text-det-limit-type", type=str, default="max")
    p.add_argument("--text-det-thresh", type=float, default=0.2)
    p.add_argument("--text-det-box-thresh", type=float, default=0.3)
    p.add_argument("--text-det-unclip-ratio", type=float, default=1.8)
    p.add_argument("--paddle-kwargs-json", default=None)

    p.add_argument("--draw-thickness", type=int, default=2)
    p.add_argument("--draw-labels", action="store_true", default=False)
    p.add_argument("--draw-regions", action="store_true", default=False)

    return p


def main():
    args = build_argparser().parse_args()
    page_range = parse_page_range(args.pages) if args.pages is not None else None

    print("Page range:", "ALL" if page_range is None else f"{page_range[0]}-{page_range[1]}")

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
