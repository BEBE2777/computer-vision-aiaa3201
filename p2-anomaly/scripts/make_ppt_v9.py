"""Build deliverables/AIAA3201_P2_Intermediate_v9_part1_video.pptx from v8.

Scope: slides 1-4 only (the Part 1 main line). Slides 5-11 are carried through
unchanged, and no animation is added to them. Adds 14 click-triggered entrance
steps (fade / wipe-from-left) through p:timing XML.

Every number on slide 4 comes from
results/part1_template/demo_chewinggum_1shot_seed0.json; the defective sample is
kept as the real false negative it is. Figure crops are derived at run time from
committed assets, so nothing extra needs to be checked in.
"""
import io
import json
import os

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.util import Emu, Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "p2-anomaly", "results")
V8 = os.path.join(ROOT, "deliverables", "AIAA3201_P2_Intermediate_v8.pptx")
OUT = os.path.join(ROOT, "deliverables", "AIAA3201_P2_Intermediate_v9_part1_video.pptx")
FIGS = os.path.join(RES, "figs", "part1_okng_demo_chewinggum_1shot_seed0")
MONTAGE = os.path.join(RES, "figs", "chewinggum_montage.png")
META = json.load(open(os.path.join(RES, "part1_template",
                                   "demo_chewinggum_1shot_seed0.json")))


def cropped(path, box):
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.crop(box).save(buf, format="PNG")
    buf.seek(0)
    return buf


MROWS = [cropped(MONTAGE, (0, r * 448, 1344, (r + 1) * 448)) for r in (1, 3, 5)]
NHEAT = cropped(os.path.join(FIGS, "normal_heatmap.png"), (46, 82, 762, 688))
DHEAT = cropped(os.path.join(FIGS, "defective_heatmap.png"), (46, 82, 762, 688))

NAVY = RGBColor(0x0F, 0x3B, 0x63)
INK = RGBColor(0x23, 0x2E, 0x39)
MUTED = RGBColor(0x5B, 0x68, 0x75)
ACCENT = RGBColor(0xB0, 0x3A, 0x1E)
LINE = RGBColor(0xC6, 0xD2, 0xDE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PANEL = RGBColor(0xEE, 0xF3, 0xF8)
FONT = "Calibri"
E = 914400

THR = META["threshold"]
nrm, dft = META["samples"][0], META["samples"][1]
assert nrm["prediction"] == "OK" and nrm["ground_truth"] == "normal"
assert dft["prediction"] == "OK" and dft["ground_truth"] == "defective"   # the real miss
AUROC = META["self_check"]["committed_image_auroc"] * 100.0

prs = Presentation(V8)
S = list(prs.slides)


def by_name(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def drop(slide, names):
    for sh in list(slide.shapes):
        if sh.name in names:
            sh._element.getparent().remove(sh._element)


def set_text(sh, text, size=None, color=None, bold=None, align=None):
    tf = sh.text_frame
    tf.clear()                                  # drop every paragraph, not just runs
    while len(tf.paragraphs) > 1:
        tf.paragraphs[-1]._p.getparent().remove(tf.paragraphs[-1]._p)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    if size:
        r.font.size = Pt(size)
    if color is not None:
        r.font.color.rgb = color
    if bold is not None:
        r.font.bold = bold
    if align is not None:
        p.alignment = align
    return r


def tb(slide, name, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for item in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", align)
        if item.get("space_after") is not None:
            p.space_after = Pt(item["space_after"])
        if item.get("line"):
            p.line_spacing = item["line"]
        for (t, sz, col, bold) in item["runs"]:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(sz)
            r.font.color.rgb = col
            r.font.bold = bold
            r.font.name = FONT
    box.name = name
    return box


# ============================================================ SLIDE 1  (Opening)
s1 = S[0]
# geometry already minimal; only tighten the flow so it reads as three steps
by_name(s1, "cover_title").top = Inches(1.35)
by_name(s1, "cover_sub").top = Inches(2.38)
by_name(s1, "students_label").top = Inches(3.30)
by_name(s1, "speakers").top = Inches(3.62)
by_name(s1, "task_def").top = Inches(4.66)
for nm in ("cover_flow_1", "cover_flow_2", "cover_flow_3"):
    by_name(s1, nm).top = Inches(5.62)
s1_steps = [
    [("cover_title", "fade"), ("cover_sub", "fade")],
    [("task_def", "fade")],
    [("cover_flow_1", "wipe"), ("cover_flow_arrow_1", "fade"),
     ("cover_flow_2", "wipe"), ("cover_flow_arrow_2", "fade"),
     ("cover_flow_3", "wipe")],
]

# ============================================================ SLIDE 2  (Method)
s2 = S[1]
set_text(by_name(s2, "p1_protocol"),
         "Calibration \u2014 20 held-out normal images per category set the 95th-percentile threshold.",
         size=15, color=MUTED)
by_name(s2, "p1_protocol").top = Inches(5.62)
by_name(s2, "p1_protocol").height = Inches(0.60)
s2_steps = [
    [("p1_flow_1", "wipe"), ("p1_flow_arrow_1", "fade"), ("p1_flow_2", "wipe"),
     ("p1_flow_arrow_2", "fade"), ("p1_flow_3", "wipe"),
     ("p1_flow_arrow_3", "fade"), ("p1_flow_4", "wipe"),
     ("p1_flow_arrow_4", "fade"), ("p1_flow_5", "wipe")],
    [("p1_notes", "fade")],
    [("p1_rule", "fade"), ("p1_protocol", "fade")],
]

# ============================================================ SLIDE 3  (Evidence)
s3 = S[2]
drop(s3, {"fig_montage"})
CELL_H = 4.26 / 3.0
for k in range(3):
    pic = s3.shapes.add_picture(MROWS[k],
                                Inches(6.40), Inches(2.15 + k * CELL_H),
                                Inches(4.26), Inches(CELL_H))
    pic.name = f"mrow_{k}"
s3_steps = [
    [("p1_table", "fade"), ("p1_table_cap", "fade"), ("p1_sens", "fade")],
    [("mrow_0", "fade"), ("rowlab_1", "fade"), ("mrow_1", "fade"), ("rowlab_2", "fade"),
     ("mrow_2", "fade"), ("rowlab_3", "fade")],
    [("annot_arrow", "fade"), ("annot_text", "fade")],
    [("p1_conclusion", "fade")],
]

# ============================================================ SLIDE 4  (Real demo)
s4 = S[3]
drop(s4, {"Picture 15", "dm_table", "dm_protocol"})
by_name(s4, "title").top = Inches(0.42)
by_name(s4, "title").height = Inches(0.55)

rule = tb(s4, "dm_rule", 0.60, 1.00, 12.13, 0.62,
          [dict(runs=[("Decision rule: score > threshold \u2192 NG     \u2022     "
                       "chewinggum \u00b7 1-shot \u00b7 seed 0     \u2022     threshold = 95th "
                       "percentile of 20 held-out calibration normals     \u2022     image score "
                       "= mean of top 1% of the anomaly map", 13, MUTED, False)])],
          wrap=True)

PH = 2.45
panels = [
    ("cap_n_in", "p_n_in", "Normal input", os.path.join(FIGS, "normal_input.jpg"), 1.0),
    ("cap_n_hm", "p_n_hm", "Difference map", NHEAT, 1.1815),
    ("cap_d_in", "p_d_in", "Defective input", os.path.join(FIGS, "defective_input.jpg"), 1.0),
    ("cap_d_hm", "p_d_hm", "Difference map", DHEAT, 1.1815),
]
xs = [0.60, 3.46, 6.79, 9.65]
for (cap_nm, pic_nm, cap, path, ratio), x in zip(panels, xs):
    tb(s4, cap_nm, x, 1.70, PH * ratio, 0.28,
       [dict(runs=[(cap, 14, NAVY, True)])], align=PP_ALIGN.CENTER, wrap=False)
    pic = s4.shapes.add_picture(path, Inches(x), Inches(2.02),
                                Inches(PH * ratio), Inches(PH))
    pic.name = pic_nm

tbl_shape = s4.shapes.add_table(3, 4, Inches(0.60), Inches(4.62), Inches(8.10), Inches(1.26))
tbl = tbl_shape.table
tbl_shape.name = "dm_table"
hdr = ["Sample", "Ground truth", "Image score", "Decision"]
body = [[nrm["filename"].split("chewinggum/")[-1], nrm["ground_truth"],
         f"{nrm['score']:.6f}", f"{nrm['prediction']}"],
        [dft["filename"].split("chewinggum/")[-1], dft["ground_truth"],
         f"{dft['score']:.6f}", f"{dft['prediction']} \u2014 missed"]]
for c in range(4):
    tbl.columns[c].width = Inches([2.30, 1.85, 1.85, 2.10][c])
for r in range(3):
    tbl.rows[r].height = Inches(0.42)
for r in range(3):
    for c in range(4):
        cell = tbl.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else PANEL)
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = hdr[c] if r == 0 else body[r - 1][c]
        run.font.name = FONT
        run.font.size = Pt(18)
        run.font.bold = (r == 0) or (r == 2 and c == 3)
        run.font.color.rgb = WHITE if r == 0 else (ACCENT if (r == 2 and c == 3) else INK)
tblPr = tbl._tbl.find(qn('a:tblPr'))
tblPr.set('firstRow', '1')
tblPr.set('bandRow', '0')
sid = tblPr.find(qn('a:tableStyleId'))
if sid is None:
    sid = tblPr.makeelement(qn('a:tableStyleId'), {})
    tblPr.append(sid)
sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"

tb(s4, "thr_note", 8.90, 4.66, 3.83, 0.75,
   [dict(runs=[("Threshold = ", 16, INK, False), (f"{THR:.6f}", 16, INK, True),
               ("\nBoth scores sit below it.", 16, INK, False)], line=1.15)],
   align=PP_ALIGN.RIGHT)

tb(s4, "dm_concl", 0.60, 5.98, 12.13, 0.36,
   [dict(runs=[("The defective sample is missed because its score remains below the "
                "calibration threshold.", 18, ACCENT, True)])], wrap=False)
tb(s4, "dm_limit", 0.60, 6.38, 12.13, 0.55,
   [dict(runs=[("Samples are the first normal and first defective test images in deterministic "
                "filename order, chosen independently of score. This deterministic example "
                f"reflects the low {AUROC:.2f}% Image AUROC for 1-shot chewinggum.",
                14, MUTED, False)], line=1.12)])

s4_steps = [
    [("title", "fade"), ("dm_rule", "fade")],
    [("cap_n_in", "fade"), ("p_n_in", "fade"), ("cap_n_hm", "fade"), ("p_n_hm", "fade")],
    [("cap_d_in", "fade"), ("p_d_in", "fade"), ("cap_d_hm", "fade"), ("p_d_hm", "fade")],
    [("dm_table", "fade"), ("thr_note", "fade"), ("dm_concl", "fade"), ("dm_limit", "fade")],
]

# ============================================================ timing injection
NSMAP = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
         "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
         "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
P = "{%s}" % NSMAP["p"]
FILTER = {"fade": "fade", "wipe": "wipe(right)"}
PRESET = {"fade": "10", "wipe": "22"}
SUBTYPE = {"fade": "0", "wipe": "2"}
_id = [10]


def nid():
    _id[0] += 1
    return str(_id[0])


def el(tag, parent=None, **attrs):
    node = etree.Element(P + tag, nsmap=NSMAP) if parent is None else \
        etree.SubElement(parent, P + tag)
    for k, v in attrs.items():
        node.set(k, v)
    return node


def effect(spid, kind, node_type):
    par = el("par")
    ctn = el("cTn", par, id=nid(), presetClass="entr", presetID=PRESET[kind],
             presetSubtype=SUBTYPE[kind], fill="hold", grpId="0", nodeType=node_type)
    el("cond", el("stCondLst", ctn), delay="0")
    body = el("childTnLst", ctn)

    setel = el("set", body)
    cb = el("cBhvr", setel)
    c2 = el("cTn", cb, id=nid(), dur="1", fill="hold")
    el("cond", el("stCondLst", c2), delay="0")
    el("spTgt", el("tgtEl", cb), spid=str(spid))
    el("attrName", el("attrNameLst", cb)).text = "style.visibility"
    el("strVal", el("to", setel), val="visible")

    ae = el("animEffect", body, transition="in", filter=FILTER[kind])
    cb2 = el("cBhvr", ae)
    el("cTn", cb2, id=nid(), dur="400")
    el("spTgt", el("tgtEl", cb2), spid=str(spid))
    return par


def click_group(step, slide):
    """step = [(shape_name, kind), ...]; first effect is clickEffect, rest chain after."""
    outer = el("par")
    oc = el("cTn", outer, id=nid(), fill="hold")
    el("cond", el("stCondLst", oc), delay="indefinite")
    inner = el("par", el("childTnLst", oc))
    ic = el("cTn", inner, id=nid(), fill="hold")
    el("cond", el("stCondLst", ic), delay="0")
    lst = el("childTnLst", ic)

    names = {sh.name: sh.shape_id for sh in slide.shapes}
    for i, (nm, kind) in enumerate(step):
        if nm not in names:
            raise KeyError(f"shape {nm!r} not found on slide")
        lst.append(effect(names[nm], kind, "clickEffect" if i == 0 else "afterEffect"))
    return outer


def add_timing(slide, steps):
    sld = slide._element
    for old in sld.findall(P + "timing"):
        sld.remove(old)
    timing = el("timing")
    root_par = el("par", el("tnLst", timing))
    root_ctn = el("cTn", root_par, id="1", dur="indefinite", restart="never",
                  nodeType="tmRoot")
    seq = el("seq", el("childTnLst", root_ctn), concurrent="1", nextAc="seek")
    seq_ctn = el("cTn", seq, id="2", dur="indefinite", nodeType="mainSeq")
    seq_lst = el("childTnLst", seq_ctn)
    for step in steps:
        seq_lst.append(click_group(step, slide))
    for tag, evt in (("prevCondLst", "onPrev"), ("nextCondLst", "onNext")):
        cond = el("cond", el(tag, seq), evt=evt, delay="0")
        el("sldTgt", el("tgtEl", cond))
    # schema order inside p:sld: cSld, clrMapOvr, transition, timing
    for tag in ("transition", "clrMapOvr", "cSld"):
        anchor = sld.find(P + tag)
        if anchor is not None:
            anchor.addnext(timing)
            break
    else:
        raise RuntimeError("no anchor element found for p:timing")


for slide, steps in ((S[0], s1_steps), (S[1], s2_steps), (S[2], s3_steps), (S[3], s4_steps)):
    add_timing(slide, steps)

# notes for slide 4 must describe the real misclassification
S[3].notes_slide.notes_text_frame.text = (
    "[Slide 4 \u00b7 1:40\u20132:00 \u00b7 Speaker 1] This is a real Part 1 decision on two chewinggum "
    "test images, one shot, seed zero. The calibration threshold is the ninety-fifth percentile "
    f"of twenty held-out normals, {THR:.6f}. The normal image scores {nrm['score']:.6f} and is "
    f"correctly accepted as OK. The defective image scores {dft['score']:.6f}, still below the "
    "threshold, so it is also predicted OK: a genuine false negative, not a staged example. The "
    f"two images are the first normal and first defective files in deterministic filename order, "
    f"selected independently of score. This behaviour is consistent with the {AUROC:.2f} percent "
    "image AUROC that Part 1 reaches on one-shot chewinggum.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
prs.save(OUT)
steps_total = len(s1_steps) + len(s2_steps) + len(s3_steps) + len(s4_steps)
fx = sum(len(st) for st in s1_steps + s2_steps + s3_steps + s4_steps)
print("saved", OUT)
print(f"slides={len(prs.slides._sldIdLst)} click-steps={steps_total} effects={fx}")
