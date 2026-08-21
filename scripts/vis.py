#!/usr/bin/env python3
"""Aggregate and visualize recognized circuit content from stages 03-06.

Inputs:
  * outputs/01_circuit/<pdf_stem>/circuit_images/page_NNN.png
  * outputs/03_rect/<pdf_stem>/json/page_NNN.json
  * outputs/04_wire/<pdf_stem>/json/page_NNN.json
  * outputs/05_endpoint/<pdf_stem>/json/page_NNN.json
  * outputs/06_text/<pdf_stem>/json/page_NNN.json
  * outputs/06_text/<pdf_stem>/circuit_images/page_NNN.png

Outputs:
  * outputs/vis/<pdf_stem>/images/page_NNN.png
      01_circuit circuit image with 03-06 recognition overlays
  * outputs/vis/<pdf_stem>/circuit_images/page_NNN.png
      black/white final circuit image after 03-06 removal
  * outputs/vis/<pdf_stem>/module_images/
      copied directly from 03_rect/module_images
  * outputs/vis/<pdf_stem>/review.pdf
      combined images/page_NNN.png
  * outputs/vis/<pdf_stem>/json/page_NNN.json
      recognized content collected from stages 03-06
  * outputs/vis/<pdf_stem>/debug/page_NNN.json
      source paths, counts, and overlay/removal diagnostics for correction
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from pipeline_io import image_to_pdf, save_json


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "inputs"
OUTPUT_ROOT = ROOT_DIR / "outputs"
STAGE_NAME = Path(__file__).stem

BASE_STAGE = "01_circuit"
RECT_STAGE = "03_rect"
WIRE_STAGE = "04_wire"
ENDPOINT_STAGE = "05_endpoint"
TEXT_STAGE = "06_text"

RECT_COLOR = (220, 0, 0)
WIRE_SOLID_COLOR = (0, 190, 0)
WIRE_DASHED_COLOR = (255, 145, 0)
ENDPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "connection_dot": (0, 120, 255),
    "round_terminal_object": (0, 80, 255),
    "ground_terminal": (255, 0, 0),
    "small_connected_structure": (0, 210, 210),
    "arrow_structure": (0, 120, 255),
    "brace_structure": (190, 0, 125),
    "rectangle_frame": (170, 120, 45),
}
TEXT_COLORS: dict[str, tuple[int, int, int]] = {
    "wire_color": (0, 90, 255),
    "number": (255, 150, 0),
    "other_text": (0, 170, 0),
}

WIRE_TYPES = {
    "solid_wire_seed",
    "solid_wire_extension",
    "solid_wire_diagonal_extension",
    "dashed_wire_group",
    "solid_wire",
    "dash_group",
}
ENDPOINT_CLASSIFICATION_TYPES = {"connected_endpoint", "disconnected_endpoint"}
PHYSICAL_ENDPOINT_TYPES = set(ENDPOINT_COLORS)


def resolve_pdf(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path, ROOT_DIR / path, INPUT_DIR / path]
    if path.suffix.lower() != ".pdf":
        candidates.append(INPUT_DIR / f"{value}.pdf")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"PDF not found: {value}. Looked directly and in {INPUT_DIR}")


def resolve_stage_pdf_dir(stage: str, pdf_stem: str) -> Path:
    direct = OUTPUT_ROOT / stage / pdf_stem
    if direct.is_dir():
        return direct
    stage_root = OUTPUT_ROOT / stage
    if stage_root.is_dir():
        for candidate in stage_root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == pdf_stem.lower():
                return candidate
    raise FileNotFoundError(f"Missing stage output: {direct}")


def resolve_base_circuit_dir(pdf_stem: str) -> Path:
    root = resolve_stage_pdf_dir(BASE_STAGE, pdf_stem)
    circuit_dir = root / "circuit_images"
    if not circuit_dir.is_dir():
        raise FileNotFoundError(f"Missing base circuit_images: {circuit_dir}")
    return circuit_dir


def resolve_stage_json_path(stage: str, pdf_stem: str, page: int) -> Path:
    root = resolve_stage_pdf_dir(stage, pdf_stem)
    return root / "json" / f"page_{page:03d}.json"


def available_pages(image_dir: Path) -> list[int]:
    pages: set[int] = set()
    for path in image_dir.glob("page_*.png"):
        parts = path.stem.split("_")
        for part in reversed(parts):
            if part.isdigit():
                pages.add(int(part))
                break
    if not pages:
        raise FileNotFoundError(f"No page_NNN.png files found under: {image_dir}")
    return sorted(pages)


def parse_pages(spec: str | None, available: Sequence[int]) -> list[int]:
    available_set = set(int(page) for page in available)
    if spec is None:
        return sorted(available_set)
    requested: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(value.strip()) for value in part.split("-", 1)]
            if start <= 0 or end < start:
                raise ValueError
            requested.update(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError
            requested.add(page)
    missing = sorted(requested - available_set)
    if missing:
        raise FileNotFoundError(f"Requested page(s) missing from {BASE_STAGE}/circuit_images: {missing}")
    if not requested:
        raise ValueError(f"No pages selected from spec: {spec}")
    return sorted(requested)


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    resolved_output = root.resolve()
    resolved_stage = (OUTPUT_ROOT / STAGE_NAME).resolve()
    if resolved_output == resolved_stage or resolved_stage not in resolved_output.parents:
        raise RuntimeError(f"Refusing to clear unsafe output path: {resolved_output}")
    shutil.rmtree(root)


def make_output_dirs(pdf_stem: str, preserve: bool) -> dict[str, Path]:
    root = OUTPUT_ROOT / STAGE_NAME / pdf_stem
    if root.exists() and not preserve:
        clear_output_root(root)
    paths = {
        "root": root,
        "images": root / "images",
        "circuit_images": root / "circuit_images",
        "module_images": root / "module_images",
        "json": root / "json",
        "debug": root / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def clip_bbox(box: Sequence[Any], shape: tuple[int, int]) -> list[int] | None:
    if not isinstance(box, Sequence) or len(box) != 4:
        return None
    h, w = shape
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def runs_to_mask(runs: Sequence[Sequence[Any]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for run in runs or []:
        if not isinstance(run, Sequence) or len(run) != 3:
            continue
        y, x1, x2 = [int(value) for value in run]
        if 0 <= y < shape[0]:
            mask[y, max(0, x1) : min(shape[1], x2)] = True
    return mask


def paste_local_mask(full: np.ndarray, bbox: Sequence[Any], runs: Sequence[Sequence[Any]]) -> bool:
    box = clip_bbox(bbox, full.shape)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    local = runs_to_mask(runs, (y2 - y1, x2 - x1))
    if not bool(np.any(local)):
        return False
    full[y1:y2, x1:x2] |= local
    return True


def black_pixels_in_bbox(source_rgb: np.ndarray, bbox: Sequence[Any], threshold: int = 128) -> np.ndarray:
    mask = np.zeros(source_rgb.shape[:2], dtype=bool)
    box = clip_bbox(bbox, source_rgb.shape[:2])
    if box is None:
        return mask
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(source_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    mask[y1:y2, x1:x2] = gray < int(threshold)
    return mask


def mask_from_record(
    record: dict[str, Any],
    shape: tuple[int, int],
    *,
    allow_bbox_fallback: bool = True,
    source_rgb: np.ndarray | None = None,
    black_pixel_fallback: bool = False,
) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    top_mask = record.get("mask", {}) or {}
    bbox = top_mask.get("bbox")
    runs = top_mask.get("runs") or top_mask.get("pixel_runs")
    if isinstance(bbox, list) and runs and paste_local_mask(full, bbox, runs):
        return full

    attrs = record.get("attributes", {}) or {}
    candidates = [
        ("wire_pixel_bbox", "wire_pixel_runs"),
        ("pixel_bbox", "pixel_runs"),
        ("edge_expansion_bbox", "edge_expansion_pixel_runs"),
        ("mask_bbox", "mask_runs"),
    ]
    for bbox_key, runs_key in candidates:
        bbox = attrs.get(bbox_key) or record.get(bbox_key)
        runs = attrs.get(runs_key) or record.get(runs_key)
        if isinstance(bbox, list) and runs and paste_local_mask(full, bbox, runs):
            return full

    if black_pixel_fallback and source_rgb is not None:
        return black_pixels_in_bbox(source_rgb, record.get("bbox", []))
    if not allow_bbox_fallback:
        return full

    box = clip_bbox(record.get("bbox", []), shape)
    if box is not None:
        x1, y1, x2, y2 = box
        full[y1:y2, x1:x2] = True
    return full


def paint_mask(out: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> int:
    pixels = int(np.count_nonzero(mask))
    if pixels:
        out[mask] = np.array(color, dtype=np.uint8)
    return pixels


def load_rect_mask(rect_root: Path, page: int, shape: tuple[int, int]) -> np.ndarray:
    path = rect_root / "masks" / f"page_{page:03d}.png"
    if path.is_file():
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if gray.shape == shape:
            return gray < 128
    return np.zeros(shape, dtype=bool)


def draw_bbox(out: np.ndarray, box: Sequence[Any], color: tuple[int, int, int], thickness: int = 2) -> None:
    clipped = clip_bbox(box, out.shape[:2])
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, int(thickness), cv2.LINE_AA)


def draw_label(out: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.38) -> None:
    y = max(14, int(y))
    cv2.putText(out, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def color_for_wire(wire: dict[str, Any]) -> tuple[int, int, int]:
    if str(wire.get("type", "")) == "dashed_wire_group":
        return WIRE_DASHED_COLOR
    return WIRE_SOLID_COLOR


def text_color(text: dict[str, Any]) -> tuple[int, int, int]:
    return TEXT_COLORS.get(str(text.get("category", "other_text")), TEXT_COLORS["other_text"])


def endpoint_visual_objects(endpoint_json: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for obj in endpoint_json.get("objects", []) or []:
        obj_type = str(obj.get("type", ""))
        if obj_type in WIRE_TYPES or obj_type.startswith("solid_wire"):
            continue
        if obj_type in ENDPOINT_CLASSIFICATION_TYPES:
            continue
        if obj_type not in PHYSICAL_ENDPOINT_TYPES:
            continue
        objects.append(obj)
    return objects


def endpoint_classification_records(endpoint_json: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        obj
        for obj in endpoint_json.get("objects", []) or []
        if str(obj.get("type", "")) in ENDPOINT_CLASSIFICATION_TYPES
    ]


def make_black_white(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bw = np.where(gray < int(threshold), 0, 255).astype(np.uint8)
    return np.repeat(bw[:, :, None], 3, axis=2)


def copy_module_images(rect_root: Path, output: dict[str, Path]) -> int:
    source = rect_root / "module_images"
    dest = output["module_images"]
    if not source.is_dir():
        return 0
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return sum(1 for path in dest.rglob("*.png"))


def source_debug_paths(pdf_stem: str, page: int) -> dict[str, str | None]:
    paths: dict[str, str | None] = {}
    for stage in (RECT_STAGE, WIRE_STAGE, ENDPOINT_STAGE, TEXT_STAGE):
        try:
            root = resolve_stage_pdf_dir(stage, pdf_stem)
        except FileNotFoundError:
            paths[stage] = None
            continue
        path = root / "debug" / f"page_{page:03d}.json"
        paths[stage] = str(path) if path.is_file() else None
    return paths


def summarize_stage_counts(
    rect_json: dict[str, Any],
    wire_json: dict[str, Any],
    endpoint_json: dict[str, Any],
    text_json: dict[str, Any],
    endpoint_objects: Sequence[dict[str, Any]],
    endpoint_classifications: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rect_count": int(len(rect_json.get("rects", []) or [])),
        "wire_count": int(len(wire_json.get("wires", []) or [])),
        "endpoint_object_count": int(len(endpoint_objects)),
        "endpoint_classification_count": int(len(endpoint_classifications)),
        "net_count": int(len(endpoint_json.get("nets", []) or [])),
        "text_count": int(len(text_json.get("texts", []) or [])),
        "text_category_counts": dict(Counter(str(item.get("category", "other_text")) for item in text_json.get("texts", []) or [])),
        "endpoint_object_type_counts": dict(Counter(str(item.get("type", "")) for item in endpoint_objects)),
        "endpoint_classification_type_counts": dict(Counter(str(item.get("type", "")) for item in endpoint_classifications)),
        "wire_type_counts": dict(Counter(str(item.get("type", "")) for item in wire_json.get("wires", []) or [])),
    }


def process_page(
    pdf_path: Path,
    page: int,
    base_image_dir: Path,
    rect_root: Path,
    text_root: Path,
    output: dict[str, Path],
    black_threshold: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    base_path = base_image_dir / f"page_{page:03d}.png"
    final_circuit_path = text_root / "circuit_images" / f"page_{page:03d}.png"
    rect_json_path = resolve_stage_json_path(RECT_STAGE, pdf_path.stem, page)
    wire_json_path = resolve_stage_json_path(WIRE_STAGE, pdf_path.stem, page)
    endpoint_json_path = resolve_stage_json_path(ENDPOINT_STAGE, pdf_path.stem, page)
    text_json_path = resolve_stage_json_path(TEXT_STAGE, pdf_path.stem, page)

    base = load_rgb(base_path)
    rect_json = read_json(rect_json_path)
    wire_json = read_json(wire_json_path)
    endpoint_json = read_json(endpoint_json_path)
    text_json = read_json(text_json_path)
    endpoint_objects = endpoint_visual_objects(endpoint_json)
    endpoint_classifications = endpoint_classification_records(endpoint_json)

    review = base.copy()
    overlay_pixels: dict[str, int] = {}

    rect_mask = load_rect_mask(rect_root, page, base.shape[:2])
    overlay_pixels["03_rect"] = paint_mask(review, rect_mask, RECT_COLOR)
    for rect in rect_json.get("rects", []) or []:
        draw_bbox(review, rect.get("bbox", []), RECT_COLOR, 2)

    wire_pixels = 0
    for wire in wire_json.get("wires", []) or []:
        wire_pixels += paint_mask(review, mask_from_record(wire, base.shape[:2]), color_for_wire(wire))
    overlay_pixels["04_wire"] = int(wire_pixels)

    endpoint_pixels = 0
    for obj in endpoint_objects:
        color = ENDPOINT_COLORS.get(str(obj.get("type", "")), (0, 210, 210))
        endpoint_mask = mask_from_record(
            obj,
            base.shape[:2],
            allow_bbox_fallback=False,
            source_rgb=base,
            black_pixel_fallback=True,
        )
        endpoint_pixels += paint_mask(review, endpoint_mask, color)
    overlay_pixels["05_endpoint"] = int(endpoint_pixels)

    text_boxes = 0
    for text in text_json.get("texts", []) or []:
        color = text_color(text)
        box = clip_bbox(text.get("bbox", []), base.shape[:2])
        if box is None:
            continue
        text_boxes += 1
        draw_bbox(review, box, color, 2)
        label = str(text.get("normalized_text") or text.get("text") or "")[:32]
        if label:
            draw_label(review, label, box[0], box[1] - 4, color)
    overlay_pixels["06_text_boxes"] = int(text_boxes)

    final_circuit = make_black_white(load_rgb(final_circuit_path), black_threshold)
    review_path = output["images"] / f"page_{page:03d}.png"
    circuit_path = output["circuit_images"] / f"page_{page:03d}.png"
    save_rgb(review_path, review)
    save_rgb(circuit_path, final_circuit)

    counts = summarize_stage_counts(rect_json, wire_json, endpoint_json, text_json, endpoint_objects, endpoint_classifications)
    module_paths = [
        rel_path(path, output["root"])
        for path in sorted((output["module_images"] / f"page_{page:03d}").glob("*.png"))
    ]
    page_json = {
        "pdf": pdf_path.name,
        "page": int(page),
        "stage": STAGE_NAME,
        "source_stages": [RECT_STAGE, WIRE_STAGE, ENDPOINT_STAGE, TEXT_STAGE],
        "base_image": {
            "stage": BASE_STAGE,
            "path": str(base_path),
            "meaning": "black/white 01_circuit circuit image used as review background",
        },
        "image_width": int(base.shape[1]),
        "image_height": int(base.shape[0]),
        "review_image_path": rel_path(review_path, output["root"]),
        "circuit_image_path": rel_path(circuit_path, output["root"]),
        "module_image_paths": module_paths,
        "summary": counts,
        "recognized": {
            "rects": copy.deepcopy(rect_json.get("rects", []) or []),
            "wires": copy.deepcopy(wire_json.get("wires", []) or []),
            "endpoint_objects": copy.deepcopy(endpoint_objects),
            "endpoint_classification_records": copy.deepcopy(endpoint_classifications),
            "nets": copy.deepcopy(endpoint_json.get("nets", []) or []),
            "texts": copy.deepcopy(text_json.get("texts", []) or []),
        },
    }
    debug_json = {
        "pdf": pdf_path.name,
        "page": int(page),
        "stage": STAGE_NAME,
        "source_json_paths": {
            RECT_STAGE: str(rect_json_path),
            WIRE_STAGE: str(wire_json_path),
            ENDPOINT_STAGE: str(endpoint_json_path),
            TEXT_STAGE: str(text_json_path),
        },
        "source_debug_paths": source_debug_paths(pdf_path.stem, page),
        "input_images": {
            "base_01_circuit": str(base_path),
            "final_06_text_circuit": str(final_circuit_path),
            "rect_mask": str(rect_root / "masks" / f"page_{page:03d}.png"),
        },
        "output_images": {
            "review": rel_path(review_path, output["root"]),
            "circuit": rel_path(circuit_path, output["root"]),
        },
        "overlay_pixels_or_counts": overlay_pixels,
        "counts": counts,
        "black_white_threshold": int(black_threshold),
        "notes": [
            "images/page_NNN.png is an overlay only; no detection or connectivity is recomputed.",
            "circuit_images/page_NNN.png is copied from 06_text/circuit_images and thresholded to black/white.",
            "module_images are copied from 03_rect/module_images.",
        ],
    }
    return review, page_json, debug_json


def write_legend(path: Path) -> None:
    lines = [f"{STAGE_NAME} visual legend", ""]
    lines.append(f"03_rect: #{RECT_COLOR[0]:02X}{RECT_COLOR[1]:02X}{RECT_COLOR[2]:02X} rectangular/module frames")
    lines.append(f"04_wire.solid: #{WIRE_SOLID_COLOR[0]:02X}{WIRE_SOLID_COLOR[1]:02X}{WIRE_SOLID_COLOR[2]:02X} solid wires")
    lines.append(f"04_wire.dashed: #{WIRE_DASHED_COLOR[0]:02X}{WIRE_DASHED_COLOR[1]:02X}{WIRE_DASHED_COLOR[2]:02X} dashed wires")
    for key, color in sorted(ENDPOINT_COLORS.items()):
        lines.append(f"05_endpoint.{key}: #{color[0]:02X}{color[1]:02X}{color[2]:02X}")
    for key, color in sorted(TEXT_COLORS.items()):
        lines.append(f"06_text.{key}: #{color[0]:02X}{color[1]:02X}{color[2]:02X}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual aggregate stage for 03_rect through 06_text.")
    parser.add_argument("--pdf", type=str, default="bmw-328i-1997.pdf")
    parser.add_argument("--pages", type=str, default="1-5")
    parser.add_argument("-p", "--preserve", action="store_true", help="Preserve the output folder instead of clearing it first.")
    parser.add_argument("--black-threshold", type=int, default=245, help="Threshold used to force output circuit_images to black/white.")
    return parser


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> None:
    base_image_dir = resolve_base_circuit_dir(pdf_path.stem)
    rect_root = resolve_stage_pdf_dir(RECT_STAGE, pdf_path.stem)
    text_root = resolve_stage_pdf_dir(TEXT_STAGE, pdf_path.stem)
    pages = parse_pages(args.pages, available_pages(base_image_dir))
    output = make_output_dirs(pdf_path.stem, args.preserve)
    modules_copied = copy_module_images(rect_root, output)

    print(f"\nProcessing: {pdf_path.stem}")
    print(f"Base circuit_images ({BASE_STAGE}): {base_image_dir}")
    print(f"Output: {output['root']}")
    print(f"Pages: {pages}")
    print(f"Copied module images: {modules_copied}")

    review_pages: list[np.ndarray] = []
    summary_pages: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    total_text_categories: Counter[str] = Counter()

    for page in pages:
        print(f"  Page {page}")
        review, page_json, debug_json = process_page(
            pdf_path=pdf_path,
            page=page,
            base_image_dir=base_image_dir,
            rect_root=rect_root,
            text_root=text_root,
            output=output,
            black_threshold=args.black_threshold,
        )
        save_json(output["json"] / f"page_{page:03d}.json", page_json)
        save_json(output["debug"] / f"page_{page:03d}.json", debug_json)
        review_pages.append(review)

        summary = page_json["summary"]
        total_counts.update(
            {
                "rects": int(summary["rect_count"]),
                "wires": int(summary["wire_count"]),
                "endpoint_objects": int(summary["endpoint_object_count"]),
                "nets": int(summary["net_count"]),
                "texts": int(summary["text_count"]),
            }
        )
        total_text_categories.update(summary.get("text_category_counts", {}) or {})
        summary_pages.append(
            {
                "page": int(page),
                **summary,
                "review_image_path": f"images/page_{page:03d}.png",
                "circuit_image_path": f"circuit_images/page_{page:03d}.png",
                "json_path": f"json/page_{page:03d}.json",
                "debug_path": f"debug/page_{page:03d}.json",
            }
        )
        print(
            "    rects={rect_count} wires={wire_count} endpoints={endpoint_object_count} "
            "texts={text_count} nets={net_count}".format(**summary)
        )

    if review_pages:
        image_to_pdf(review_pages, output["root"] / "review.pdf")
    write_legend(output["root"] / "legend")
    save_json(
        output["json"] / "summary.json",
        {
            "pdf": pdf_path.name,
            "stage": STAGE_NAME,
            "base_stage": BASE_STAGE,
            "source_stages": [RECT_STAGE, WIRE_STAGE, ENDPOINT_STAGE, TEXT_STAGE],
            "output_root": str(output["root"]),
            "pages": summary_pages,
            "totals": dict(total_counts),
            "text_category_totals": dict(total_text_categories),
            "module_images_copied": int(modules_copied),
        },
    )
    save_json(
        output["debug"] / "summary.json",
        {
            "pdf": pdf_path.name,
            "pages": pages,
            "base_circuit_images": str(base_image_dir),
            "rect_module_images": str(rect_root / "module_images"),
            "final_circuit_images": str(text_root / "circuit_images"),
            "output_root": str(output["root"]),
            "module_images_copied": int(modules_copied),
        },
    )
    print(f"Review PDF: {output['root'] / 'review.pdf'}")


def main() -> None:
    args = build_argparser().parse_args()
    print("Page range:", "ALL" if args.pages is None else args.pages)
    process_pdf(resolve_pdf(args.pdf), args)


if __name__ == "__main__":
    main()
