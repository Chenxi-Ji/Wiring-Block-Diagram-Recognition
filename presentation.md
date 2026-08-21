<style>
body {
  max-width: 1120px;
  margin: 0 auto;
  padding: 28px 22px 80px;
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.48;
  color: #172026;
  background: #fbfbf8;
}
h1 {
  font-size: 44px;
  margin: 44px 0 14px;
  letter-spacing: 0;
}
h2 {
  font-size: 30px;
  margin: 40px 0 12px;
  border-top: 1px solid #d9e1e5;
  padding-top: 28px;
}
h3 {
  font-size: 23px;
  margin: 32px 0 12px;
}
p, li {
  font-size: 18px;
}
.lead {
  font-size: 23px;
  font-weight: 600;
  color: #263238;
}
.figure {
  margin: 18px 0 26px;
}
.figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid #d8dee3;
  background: white;
}
.caption {
  margin-top: 8px;
  color: #546e7a;
  font-size: 15px;
}
.color-legend {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px 16px;
  margin: 12px 0 24px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  color: #33444c;
}
.swatch {
  width: 16px;
  height: 16px;
  border: 1px solid #8a969d;
  flex: 0 0 16px;
}
.two {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
  align-items: start;
}
.callout {
  border-left: 6px solid #00897b;
  background: #eef8f5;
  padding: 16px 20px;
  margin: 22px 0;
}
.stats {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin: 18px 0;
}
.stat {
  background: white;
  border: 1px solid #d8dee3;
  padding: 12px 14px;
}
.stat strong {
  display: block;
  font-size: 24px;
}
code {
  background: #eef1f3;
  padding: 1px 4px;
}
@media (max-width: 900px) {
  .stats {
    grid-template-columns: repeat(2, 1fr);
  }
}
@media (max-width: 760px) {
  .two, .stats, .color-legend {
    grid-template-columns: 1fr;
  }
  h1 {
    font-size: 34px;
  }
}
</style>

# Circuit Diagram Processing

<p class="lead">The pipeline parses dense wiring diagrams by progressively removing and recording visual structures: page frame, sheet text, module rectangles, wires, endpoint objects, and OCR text. Its goal is to recover connection relationships and module-level information from complex diagrams, while producing smaller module crops that can be passed to downstream circuit-analysis tools.</p>

The figures below use `bmw-328i-1997` page 1 to illustrate the processing flow. The `outputs/` directory includes results for pages 1-5 from three circuit sets: `bmw-328i-1997`, `honda-accord-1994`, and `mitsubishi-montero-sport-1997-1999`. The pipeline consists of `01_circuit.py` through `06_text.py`, followed by `vis.py`.

## Key challenge

<div class="two">
  <div class="figure">
    <img src="other circuit.png" alt="Representative clean schematic example">
    <div class="caption">A cleaner schematic-style example: visual primitives are relatively separated.</div>
  </div>
  <div class="figure">
    <img src="outputs/01_circuit/bmw-328i-1997/circuit_images/page_001.png" alt="BMW benchmark page 001 circuit image">
    <div class="caption">`bmw-328i-1997/page_001`: text, wires, boxes, connection dots, terminals, and symbols occupy the same large visual field.</div>
  </div>
</div>

Recognition is easier when components, wires, labels, and page metadata are already separated. In these pages, those primitives overlap heavily. A direct one-shot recognizer must solve layout, topology, OCR, and symbol separation at the same time.

## Main method

<div class="callout">
The method is based on computational geometry and OCR rather than large models or trained recognition networks for specific circuit components. It avoids recognizing every object from the original page in one pass. It first establishes page/circuit coordinates, then records or removes high-confidence structures in a fixed order. Each stage receives a cleaner image and structured JSON from previous stages, which makes connection extraction and module localization easier to verify.
</div>

The active processing order is:

1. `01_circuit.py`: locate page/circuit frame and split circuit content from sheet-level content.
2. `02_sheet_text.py`: OCR the outside-sheet/title region from `01_circuit/sheet_images`.
3. `03_rect.py`: detect module/component rectangles and export module crops.
4. `04_wire.py`: detect solid wires, diagonal extensions, dashed wire groups, and basic wire contacts.
5. `05_endpoint.py`: classify wire endpoints, infer endpoint-attached physical objects, and assign local net IDs.
6. `06_text.py`: run whole-page OCR on the cleaned endpoint image and erase only high-confidence text ink.
7. `vis.py`: combine `03_rect`, `04_wire`, `05_endpoint`, and `06_text` into final review JSON/images and copied module crops.

The workflow below uses the active 01-06+vis sequence. Tile splitting, large-model inference, and trained component-specific recognition networks are not part of these outputs.

# Pipeline

## Stage 01 - Circuit and sheet separation

<div class="figure">
  <img src="outputs/01_circuit/bmw-328i-1997/image/page_001.png" alt="Stage 01 circuit frame review">
  <div class="caption">`01_circuit`: detects the main circuit frame and creates both `circuit_images` and `sheet_images`.</div>
</div>

Role: separate the dense circuit region from the surrounding sheet/title region. Goal: provide a stable page coordinate system and clean input images for downstream OCR, rectangle detection, and wire analysis.

## Stage 02 - Sheet text OCR

<div class="figure">
  <img src="outputs/02_sheet_text/bmw-328i-1997/images/page_001.png" alt="Stage 02 sheet text OCR review">
  <div class="caption">`02_sheet_text`: OCRs the sheet/title region rather than the dense circuit body.</div>
</div>

Role: extract sheet-level text from the outside-sheet/title region rather than from the dense circuit body. Goal: record document metadata while keeping later circuit parsing focused on the circuit region.

## Stage 03 - Rectangle and module frames

<div class="figure">
  <img src="outputs/03_rect/bmw-328i-1997/images/page_001.png" alt="Stage 03 rectangle detection review">
  <div class="caption">`03_rect`: detects rectangular/module frames, writes masks, and exports `module_images` crops with borders erased.</div>
</div>

Role: detect component, switch, and module frames before wire extraction. Goal: remove rectangular borders from the working image and export module crops that can be analyzed as smaller local circuit regions.

## Stage 04 - Solid and dashed wires

<div class="figure">
  <img src="outputs/04_wire/bmw-328i-1997/images/page_001.png" alt="Stage 04 wire detection review">
  <div class="caption">`04_wire`: detects solid wire seeds/extensions, diagonal extensions, dashed wire groups, and connection contacts from `03_rect/circuit_images`.</div>
</div>

Role: detect wire geometry from the rectangle-cleaned circuit image, including solid wires, diagonal extensions, dashed wire groups, and basic contacts. Goal: convert visual line structures into normalized wire records for endpoint and net analysis.

## Stage 05 - Endpoints and local nets

<div class="figure">
  <img src="outputs/05_endpoint/bmw-328i-1997/image/page_001.png" alt="Stage 05 endpoint classification review">
  <div class="caption">`05_endpoint`: classifies endpoint states, infers endpoint-attached structures, and assigns local net IDs.</div>
</div>

Role: classify wire endpoints and infer endpoint-attached physical structures from the wire and rectangle outputs. Goal: build local connectivity records while keeping distinct ground terminals and disconnected endpoints explicit.

## Stage 06 - Whole-page OCR and text erasure

<div class="figure">
  <img src="outputs/06_text/bmw-328i-1997/images/page_001.png" alt="Stage 06 OCR text review">
  <div class="caption">`06_text`: detects page-level circuit text and erases high-confidence text ink from the endpoint-cleaned circuit image.</div>
</div>

Role: run OCR on `05_endpoint/circuit_images` and categorize detections as wire color codes, numbers, or other text. Goal: preserve recognized text as structured data while removing high-confidence text ink from the image for downstream visual parsing.

## Final visualization - Combined review

<div class="figure">
  <img src="outputs/vis/bmw-328i-1997/images/page_001.png" alt="Final combined visual review">
  <div class="caption">`vis.py`: overlays recognized rectangles, wires, endpoint objects, and text on the `01_circuit` base image.</div>
</div>

Role: combine the outputs from `03_rect`, `04_wire`, `05_endpoint`, and `06_text` into one review layer. Goal: provide a single visual and JSON representation for checking recognized rectangles, wires, endpoint objects, OCR text, and copied module crops without recomputing detection.

Color meanings in the `vis.py` combined review:

<div class="color-legend">
  <div class="legend-item"><span class="swatch" style="background:#DC0000"></span>Rectangular/module frames</div>
  <div class="legend-item"><span class="swatch" style="background:#00BE00"></span>Solid wires</div>
  <div class="legend-item"><span class="swatch" style="background:#FF9100"></span>Dashed wire groups</div>
  <div class="legend-item"><span class="swatch" style="background:#0078FF"></span>Connection dots and arrow structures</div>
  <div class="legend-item"><span class="swatch" style="background:#0050FF"></span>Round terminal objects</div>
  <div class="legend-item"><span class="swatch" style="background:#FF0000"></span>Ground terminals</div>
  <div class="legend-item"><span class="swatch" style="background:#00D2D2"></span>Small connected endpoint structures</div>
  <div class="legend-item"><span class="swatch" style="background:#BE007D"></span>Brace structures</div>
  <div class="legend-item"><span class="swatch" style="background:#AA782D"></span>Rectangle-frame suppression boundaries</div>
  <div class="legend-item"><span class="swatch" style="background:#005AFF"></span>OCR wire-color labels</div>
  <div class="legend-item"><span class="swatch" style="background:#FF9600"></span>OCR numbers</div>
  <div class="legend-item"><span class="swatch" style="background:#00AA00"></span>Other OCR text</div>
</div>

The module crops are intended for downstream analysis. After the global page is decomposed into frames, wires, endpoints, nets, and text, each crop contains a smaller local circuit problem. Users can process these crops with tools designed for small-scale circuit recognition or component-level analysis.

# Results

The manually reviewed coverage check uses the first five pages from each of three circuit sets: `bmw-328i-1997`, `honda-accord-1994`, and `mitsubishi-montero-sport-1997-1999` (15 pages total). From visual inspection, recognized detections were almost all correct, so this table focuses on retrieval coverage: for each metric, the denominator is `recognized + manually observed missed`.

| Metric | BMW | Honda | Mitsubishi | Total |
|---|---:|---:|---:|---:|
| Box | 64 / 64 = **100.0%** | 76 / 79 = **96.2%** | 61 / 63 = **96.8%** | 201 / 206 = **97.6%** |
| Solid wire | 598 / 607 = **98.5%** | 461 / 471 = **97.9%** | 291 / 313 = **93.0%** | 1350 / 1391 = **97.1%** |
| Dash wire | 45 / 47 = **95.7%** | 53 / 64 = **82.8%** | 17 / 24 = **70.8%** | 115 / 135 = **85.2%** |

The box metric corresponds to `03_rect.py`. The solid-wire and dash-wire metrics correspond to `04_wire.py`; they are reported separately because solid wires and dashed wire groups are different visual primitives with different failure modes.

# Limitations

### 1. Rule and threshold dependence

The method relies on geometry rules: line length, side coverage, corner tolerance, wire width estimates, endpoint search radii, and OCR confidence thresholds. These settings work for the reviewed diagrams, but they need broader validation across other diagram families.

### 2. OCR and erasure sensitivity

`06_text.py` erases only high-confidence text ink to avoid damaging nearby wires. This conservative behavior reduces accidental wire damage, but it can leave low-confidence text residuals or miss labels when OCR quality drops.

# Final Summary

- The workflow is `01_circuit` -> `02_sheet_text` -> `03_rect` -> `04_wire` -> `05_endpoint` -> `06_text` -> `vis`.
- The pipeline extracts page regions, module frames, wires, endpoints, nets, and OCR text through staged image cleanup and JSON handoff.
- `vis.py` produces review images and combined JSON without recomputing detection.
- `vis/module_images` contains smaller module crops that can be passed to other small-scale circuit analysis tools.
- Remaining work includes broader validation, OCR cleanup, and endpoint/net ground truth evaluation.
