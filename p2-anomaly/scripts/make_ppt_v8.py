"""Create deliverables/AIAA3201_P2_Intermediate_v8.pptx from v6: only slide 4
is upgraded to the real Part 1 OK/NG decision demo. Every number shown comes
from results/part1_template/demo_chewinggum_1shot_seed0.json (one real run,
self-checked against the committed baseline summary).
"""
import copy
import json
import os

from pptx import Presentation
from pptx.util import Emu, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V8 = os.path.join(ROOT, "deliverables", "AIAA3201_P2_Intermediate_v8.pptx")
V6 = os.path.join(ROOT, "deliverables", "AIAA3201_P2_Intermediate_v6.pptx")
FIG = os.path.join(ROOT, "p2-anomaly", "results", "figs",
                   "part1_okng_demo_chewinggum_1shot_seed0.png")
META = json.load(open(os.path.join(ROOT, "p2-anomaly", "results", "part1_template",
                                   "demo_chewinggum_1shot_seed0.json")))

THR = META["threshold"]
n, d = META["samples"][0], META["samples"][1]
# predictions come straight from the JSON; a misjudgment is kept as-is

prs = Presentation(V6)
slide = prs.slides[3]

# ---- title ----
for sh in slide.shapes:
    if sh.name == "title":
        r = sh.text_frame.paragraphs[0].runs[0]
        r.text = "Part 1 — Real OK/NG Decision Demo"

# ---- remove old demo visuals / placeholder wording ----
DROP = {"dm_lab_in", "dm_lab_hm", "fig_demo_input", "fig_demo_heat",
        "dm_imgcap", "dm_normcap", "dm_pending", "dm_rule"}
for sh in list(slide.shapes):
    if sh.name in DROP:
        sh._element.getparent().remove(sh._element)

# ---- four-grid demo figure, full content width ----
top_h = Emu(int(11094415 / 3.3806366047745358))  # keep PNG aspect ratio
slide.shapes.add_picture(FIG, Emu(548640), Emu(1400000), width=Emu(11094415),
                         height=top_h)

# ---- traceability table (reuse the existing 4x2 table) ----
tbl_shape = next(sh for sh in slide.shapes if sh.has_table)
tbl = tbl_shape.table
tbl_shape.left, tbl_shape.top = Emu(548640), Emu(1400000 + top_h + 120000)
tbl.columns[0].width = Emu(3600000)
tbl.columns[1].width = Emu(11094415 - 3600000)
rows = [
    ("Sample", f"Score vs threshold ({THR:.3f}) → decision"),
    (n["filename"].split("/")[-1] + "   (" + "/".join(n["filename"].split("/")[-2:]) + ")",
     f"{n['score']:.3f}  {'<' if n['prediction']=='OK' else '>'}  {THR:.3f}  →  {n['prediction']}   (ground truth: {n['ground_truth']})"),
    (d["filename"].split("/")[-1] + "   (" + "/".join(d["filename"].split("/")[-2:]) + ")",
     f"{d['score']:.3f}  {'<' if d['prediction']=='OK' else '>'}  {THR:.3f}  →  {d['prediction']}   (ground truth: {d['ground_truth']})"),
    ("Setup", "Part 1 template-difference baseline · chewinggum · 1-shot · seed 0"),
]
for row, (c0, c1) in zip(tbl.rows, rows):
    for cell, text in zip(row.cells, (c0, c1)):
        cell.text = text
        cell.margin_top = cell.margin_bottom = Emu(18000)
        cell.margin_left = cell.margin_right = Emu(70000)
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(12)
                run.font.name = "Calibri"
for row in tbl.rows:
    row.height = Emu(228000)

# ---- rule + config strip (replaces the old pending/limitation wording) ----
proto = next(sh for sh in slide.shapes if sh.name == "dm_protocol")
tf = proto.text_frame
tf.clear()
p0 = tf.paragraphs[0]
run = p0.add_run()
run.text = ("Decision rule: score > threshold → NG     •     chewinggum, 1-shot, seed 0"
            "     •     threshold = 95th percentile of 20 held-out calibration normals"
            "     •     image score = mean of top 1% of the anomaly map")
f = run.font
f.size = Pt(13)
f.name = "Calibri"
f.bold = False
f.color.rgb = __import__("pptx.dml.color", fromlist=["RGBColor"]).RGBColor(0x23, 0x2E, 0x39)
proto.top, proto.height = Emu(5860000), Emu(420000)

prs.save(V8)
print("saved", V8)
