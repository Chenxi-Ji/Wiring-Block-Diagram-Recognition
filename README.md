# ParseDiagram

ParseDiagram is an experimental pipeline for parsing dense automotive wiring diagram PDFs. The active implementation lives in `scripts/`. The pipeline decomposes each page into progressively more structured artifacts: the circuit region, sheet/title text, rectangular module frames, wire segments, endpoint objects, local nets, OCR text, and final review visualizations.

The project is designed around inspectable intermediate outputs. Each stage writes per-page images, JSON records, and debugging artifacts under `outputs/<stage_name>/<pdf_stem>/`.

## Environment

Use Python 3.11.9. The direct dependencies are pinned in `requirements.txt`:

```txt
numpy==2.3.5
opencv-python-headless==4.13.0.92
pillow==12.2.0
pymupdf==1.27.2.3
reportlab==5.0.0
paddlepaddle==3.2.0
paddleocr==3.7.0
```

Install the environment:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `opencv-python-headless` is used intentionally because the pipeline does not require OpenCV GUI features.
- Avoid mixing `opencv-python-headless` with `opencv-python` or `opencv-contrib-python` in the same environment.
- PaddleOCR depends on PaddlePaddle. If your CPU/GPU setup requires a different PaddlePaddle wheel, install the matching build from the official PaddlePaddle instructions.
- Exact reproduction can still be affected by transitive dependency resolution and PaddleOCR model/cache changes.

## Directory Layout

Place source PDFs under:

```txt
inputs/
```

Each stage writes to:

```txt
outputs/<stage_name>/<pdf_stem>/
```

Examples:

```txt
outputs/01_circuit/bmw-328i-1997/
outputs/03_rect/bmw-328i-1997/
outputs/vis/bmw-328i-1997/
```

Most scripts support:

- `--pdf`: input PDF filename.
- `--pages`: page range, usually in `start-end` format, for example `1-5`.
- `-p` / `--preserve`: keep the existing output directory instead of clearing it before the run.

For reproducible runs, pass `--pdf` and `--pages` explicitly instead of relying on script defaults.

## Pipeline Stages

### 1. `01_circuit.py`

Purpose: locate the main circuit frame in each PDF page, establish page coordinates, and split the page into circuit content and sheet/title content.

Input:

```txt
inputs/<pdf_name>.pdf
```

Outputs:

```txt
outputs/01_circuit/<pdf_stem>/image/page_NNN.png
outputs/01_circuit/<pdf_stem>/circuit_images/page_NNN.png
outputs/01_circuit/<pdf_stem>/sheet_images/page_NNN.png
outputs/01_circuit/<pdf_stem>/json/page_NNN.json
outputs/01_circuit/<pdf_stem>/debug/page_NNN.json
outputs/01_circuit/<pdf_stem>/review.pdf
outputs/01_circuit/<pdf_stem>/result.pdf
outputs/01_circuit/<pdf_stem>/sheet.pdf
```

Output meaning:

- `image/`: frame-detection review images.
- `circuit_images/`: main circuit-region images used by later circuit stages.
- `sheet_images/`: page-level sheet/title content outside the circuit frame, used by `02_sheet_text.py`.
- `json/`: frame geometry, sub-frame records, coordinates, line widths, and calibration data.
- `debug/`: frame candidates, rejection reasons, and diagnostic statistics.

### 2. `02_sheet_text.py`

Purpose: run OCR on the `01_circuit` sheet/title images to extract page title, vehicle, system name, and other sheet-level text outside the circuit frame.

Input:

```txt
outputs/01_circuit/<pdf_stem>/sheet_images/page_NNN.png
```

Outputs:

```txt
outputs/02_sheet_text/<pdf_stem>/images/page_NNN.png
outputs/02_sheet_text/<pdf_stem>/json/page_NNN.json
outputs/02_sheet_text/<pdf_stem>/review.pdf
```

Output meaning:

- `images/`: sheet OCR review images.
- `json/`: OCR regions, PaddleOCR settings, text content, confidence, bounding boxes, and polygons.
- `review.pdf`: combined multi-page OCR review.

### 3. `03_rect.py`

Purpose: detect rectangular component, switch, and module frames from `01_circuit/circuit_images`. This stage also exports one crop per detected module frame.

Input:

```txt
outputs/01_circuit/<pdf_stem>/circuit_images/page_NNN.png
```

Outputs:

```txt
outputs/03_rect/<pdf_stem>/images/page_NNN.png
outputs/03_rect/<pdf_stem>/circuit_images/page_NNN.png
outputs/03_rect/<pdf_stem>/module_images/page_NNN/module_NNNN.png
outputs/03_rect/<pdf_stem>/masks/page_NNN.png
outputs/03_rect/<pdf_stem>/json/page_NNN.json
outputs/03_rect/<pdf_stem>/debug/page_NNN.json
outputs/03_rect/<pdf_stem>/review.pdf
outputs/03_rect/<pdf_stem>/result.pdf
```

Output meaning:

- `images/`: rectangle-detection review images.
- `circuit_images/`: circuit images after detected frames and their interiors are erased; used by `04_wire.py`.
- `module_images/`: cropped module regions with their borders erased. These crops are the key input for module-level circuit analysis and component extraction.
- `masks/`: rectangle-frame masks.
- `json/`: rectangle bounding boxes, side records, confidence-related data, and module crop paths.
- `debug/`: rectangle candidates, edge coverage, rejection reasons, and diagnostic details.

### 4. `04_wire.py`

Purpose: detect wire structures from `03_rect/circuit_images`. The output normalizes solid wire seeds, solid wire extensions, diagonal extensions, and dashed wire groups into a common wire JSON format.

Input:

```txt
outputs/03_rect/<pdf_stem>/circuit_images/page_NNN.png
```

Outputs:

```txt
outputs/04_wire/<pdf_stem>/images/page_NNN.png
outputs/04_wire/<pdf_stem>/json/page_NNN.json
outputs/04_wire/<pdf_stem>/debug/page_NNN.json
outputs/04_wire/<pdf_stem>/review.pdf
outputs/04_wire/<pdf_stem>/result.pdf
```

Output meaning:

- `images/`: wire-detection review images with solid and dashed wires marked in different colors.
- `json/`: a `wires` list containing wire type, bounding box, centerline, mask runs, geometry, and connection records.
- `debug/`: detector debug data, adapter diagnostics, and connection diagnostics.
- `review.pdf`: combined labeled wire review.
- `result.pdf`: combined unlabeled wire overlay.

### 5. `05_endpoint.py`

Purpose: classify wire endpoints and stitch local nets using `03_rect` and `04_wire` artifacts. This stage identifies connected and disconnected endpoints, suppresses rectangle-owned wire or dash groups, records endpoint-attached physical objects, and assigns local net IDs.

Inputs:

```txt
outputs/03_rect/<pdf_stem>/circuit_images/page_NNN.png
outputs/04_wire/<pdf_stem>/json/page_NNN.json
outputs/03_rect/<pdf_stem>/json/page_NNN.json
```

Outputs:

```txt
outputs/05_endpoint/<pdf_stem>/image/page_NNN.png
outputs/05_endpoint/<pdf_stem>/circuit_images/page_NNN.png
outputs/05_endpoint/<pdf_stem>/json/page_NNN.json
outputs/05_endpoint/<pdf_stem>/legend
outputs/05_endpoint/<pdf_stem>/review.pdf
outputs/05_endpoint/<pdf_stem>/result.pdf
```

Output meaning:

- `image/`: endpoint and physical-object review images.
- `circuit_images/`: endpoint-processed circuit images used by `06_text.py`.
- `json/`: wire objects, endpoint classification records, physical endpoint objects, and local nets.
- `legend`: color legend for review images.
- `review.pdf` / `result.pdf`: combined endpoint review and result PDFs.

Endpoint-attached object types include connection dots, round terminals, ground terminals, arrows, braces, and small connected structures.

### 6. `06_text.py`

Purpose: run PaddleOCR on the full `05_endpoint/circuit_images` page, classify detected text, and erase only high-confidence text ink. Text is categorized as wire color code, number, or other text.

Input:

```txt
outputs/05_endpoint/<pdf_stem>/circuit_images/page_NNN.png
```

Outputs:

```txt
outputs/06_text/<pdf_stem>/images/page_NNN.png
outputs/06_text/<pdf_stem>/circuit_images/page_NNN.png
outputs/06_text/<pdf_stem>/json/page_NNN.json
outputs/06_text/<pdf_stem>/debug/page_NNN.json
outputs/06_text/<pdf_stem>/json/summary.json
outputs/06_text/<pdf_stem>/review.pdf
outputs/06_text/<pdf_stem>/result.pdf
```

Output meaning:

- `images/`: OCR detection review images.
- `circuit_images/`: circuit images after high-confidence text ink has been erased.
- `json/page_NNN.json`: text content, normalized text, category, OCR confidence, bounding box, polygon, erase status, and ink-pixel statistics.
- `debug/`: raw OCR output, filtering/rejection reasons, bounding-box refinement, text-ink isolation, and erase statistics.
- `json/summary.json`: aggregate text category counts and total erased count.

Important parameters:

- `--min-confidence`: minimum OCR confidence for keeping detections, default `0.35`.
- `--erase-min-confidence`: strict OCR confidence threshold for erasing text ink, default `0.80`.
- `--text-gray-threshold`: grayscale threshold used to identify text ink, default `245`.

### 7. `vis.py`

Purpose: aggregate and visualize recognized content. This stage does not recompute detection or connectivity. It reads JSON from `03_rect`, `04_wire`, `05_endpoint`, and `06_text`, then writes final review overlays, aggregate JSON, debug JSON, and copied module crops.

Inputs:

```txt
outputs/01_circuit/<pdf_stem>/circuit_images/page_NNN.png
outputs/03_rect/<pdf_stem>/json/page_NNN.json
outputs/04_wire/<pdf_stem>/json/page_NNN.json
outputs/05_endpoint/<pdf_stem>/json/page_NNN.json
outputs/06_text/<pdf_stem>/json/page_NNN.json
outputs/06_text/<pdf_stem>/circuit_images/page_NNN.png
outputs/03_rect/<pdf_stem>/module_images/
```

Outputs:

```txt
outputs/vis/<pdf_stem>/images/page_NNN.png
outputs/vis/<pdf_stem>/circuit_images/page_NNN.png
outputs/vis/<pdf_stem>/module_images/page_NNN/module_NNNN.png
outputs/vis/<pdf_stem>/json/page_NNN.json
outputs/vis/<pdf_stem>/debug/page_NNN.json
outputs/vis/<pdf_stem>/json/summary.json
outputs/vis/<pdf_stem>/debug/summary.json
outputs/vis/<pdf_stem>/legend
outputs/vis/<pdf_stem>/review.pdf
```

Output meaning:

- `images/`: primary human-review overlays. These images draw rectangles, wires, endpoint objects, and OCR text on the `01_circuit` base circuit image.
- `circuit_images/`: thresholded final circuit images generated from `06_text/circuit_images`.
- `module_images/`: module crops copied from `03_rect/module_images`. These are the preferred starting point for module-internal circuit parsing, component extraction, and local graph recovery.
- `json/page_NNN.json`: aggregate recognized content, including `rects`, `wires`, `endpoint_objects`, `endpoint_classification_records`, `nets`, and `texts`.
- `debug/`: source JSON paths, source debug paths, overlay statistics, and summary counts.
- `json/summary.json`: cross-page totals for rectangles, wires, endpoint objects, nets, texts, text categories, and copied module images.

## Running the Pipeline

Place the PDF in `inputs/`, then run:

```powershell
python scripts/01_circuit.py --pdf bmw-328i-1997.pdf --pages 1-5
python scripts/02_sheet_text.py --pdf bmw-328i-1997.pdf --pages 1-5
python scripts/03_rect.py --pdf bmw-328i-1997.pdf --pages 1-5
python scripts/04_wire.py --pdf bmw-328i-1997.pdf --pages 1-5
python scripts/05_endpoint.py --pdf bmw-328i-1997.pdf --pages 1-5
python scripts/06_text.py --pdf bmw-328i-1997.pdf --pages 1-5
python scripts/vis.py --pdf bmw-328i-1997.pdf --pages 1-5
```

To rerun one stage while keeping existing output files in that stage directory:

```powershell
python scripts/03_rect.py --pdf bmw-328i-1997.pdf --pages 1-5 -p
```

Without `-p`, most stages clear `outputs/<stage_name>/<pdf_stem>/` before writing new results.

## Output Directory Reference

Common output directories:

- `image/` or `images/`: review images for human inspection.
- `circuit_images/`: circuit images passed to later stages or final thresholded circuit images.
- `sheet_images/`: sheet/title regions produced by `01_circuit.py`.
- `module_images/`: module crops produced by `03_rect.py` and copied by `vis.py`.
- `masks/`: structure masks.
- `json/`: per-page structured records and summary files.
- `debug/`: candidates, filtering decisions, rejection reasons, runtime settings, and diagnostics.
- `legend`: color legend for visual review outputs.
- `review.pdf`: combined review images.
- `result.pdf`: combined stage result images. The exact visual meaning depends on the stage.

## `pipeline_io.py`

`pipeline_io.py` is a shared helper module, not a standalone pipeline stage. It provides:

- `render_pdf_page`: render a PyMuPDF page to a NumPy RGB image.
- `save_json`: write JSON with UTF-8 encoding.
- `image_to_pdf`: combine image arrays into a PDF.
- `parse_page_range`: parse `start-end` page ranges.

## Interpreting Results

Counts in `outputs/` are artifact counts, not accuracy metrics. For example, `outputs/vis/<pdf_stem>/json/summary.json` reports counts for detected rectangles, wires, endpoint objects, local nets, and OCR text. These counts are useful for debugging and version comparison, but they are not a substitute for ground-truth evaluation.

## Current Focus

The pipeline currently emphasizes structured, inspectable decomposition of dense wiring diagrams. Important follow-up work includes:

- validating endpoint classifications and local nets against a ground-truth wiring graph;
- improving OCR cleanup while preserving nearby wire geometry;
- expanding robustness across different diagram styles and scan qualities;
- parsing module crops from `module_images/`, including internal wire analysis, component extraction, local connectivity recovery, and integration of module-level graphs with page-level nets.
