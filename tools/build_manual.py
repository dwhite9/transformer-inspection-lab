#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py", "pygments"]
# ///
"""Build the single-file HTML lab manual from the detailed markdown.

The markdown file is the source of truth. This script converts it to one
self-contained HTML page (glossary tooltips, diagrams, syntax highlighting).
Edit the .md, then re-run this script. Do not edit the .html by hand.

Usage (from anywhere; uv installs the dependencies listed above automatically):

    uv run build_manual.py
    uv run build_manual.py [input.md [template.html [output.html]]]

With no arguments it reads ../manuals/model-inspection-lab-manual-detailed.md and
manual-template.html (in this folder), and writes
model-inspection-lab-manual-detailed.html next to the markdown, in ../manuals/.

Two conventions handled here, on top of plain markdown:

  * A relative link to a sibling manual, [text](other-manual.md), is rewritten to
    other-manual.html in the built page (when other-manual.md exists next to the source),
    so the same link works in the markdown and in the HTML.
  * A blockquote that starts with "**Notice!!**" gets the "notice" style (a prominent box).
"""
import html
import json
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
if len(args) > 3 or any(a in ("-h", "--help") for a in args):
    sys.exit(__doc__)
src_path = Path(args[0]) if len(args) > 0 else HERE.parent / "manuals" / "model-inspection-lab-manual-detailed.md"
tpl_path = Path(args[1]) if len(args) > 1 else HERE / "manual-template.html"
out_path = Path(args[2]) if len(args) > 2 else src_path.with_suffix(".html")
src = src_path.read_text(encoding="utf-8")

# ---------------------------------------------------------------- markdown setup
fmt = HtmlFormatter(nowrap=True)


def hl(code, lang, attrs):
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except Exception:
        lexer = TextLexer()
    body = highlight(code, lexer, fmt)
    return f'<pre class="hl" data-lang="{html.escape(lang or "text")}"><code>{body}</code></pre>\n'


md = MarkdownIt("commonmark", {"html": True, "highlight": hl}).enable("table")
md.renderer.rules["table_open"] = lambda tokens, idx, options, env: '<div class="table-wrap"><table>\n'
md.renderer.rules["table_close"] = lambda tokens, idx, options, env: "</table></div>\n"

# ------------------------------------------------------- split out the glossary
lines = src.split("\n")
g = next(i for i, l in enumerate(lines) if l.startswith("# Part 0: Glossary"))
p = next(i for i, l in enumerate(lines) if l.startswith("# Part 0.5"))
pre_md = "\n".join(lines[: g + 1])
gl_lines = lines[g + 1 : p]
while gl_lines and gl_lines[-1].strip() in ("", "---"):
    gl_lines.pop()
post_md = "---\n\n" + "\n".join(lines[p:])

# Group the glossary lines into blocks: ("heading", text) or ("entry", text).
blocks, cur, in_fence = [], None, False
for ln in gl_lines:
    if ln.startswith("```"):
        in_fence = not in_fence
        if cur:
            cur[1].append(ln)
        continue
    if not in_fence and re.match(r"^\*\*.+?\.\*\*", ln):
        cur = ("entry", [ln])
        blocks.append(cur)
    elif not in_fence and ln.startswith(("## ", "### ")):
        cur = ("heading", [ln])
        blocks.append(cur)
    elif cur:
        cur[1].append(ln)


# ------------------------------------------------------------------ figure (SVG)
def figure_tensor_shapes():
    """Four panels: scalar, vector, matrix, 3-axis tensor. CSS classes live in the template."""
    C = 34  # cell size

    def grid(x, y, rows, cols, hi=(), cls="dg-cell"):
        out = []
        for r in range(rows):
            for c in range(cols):
                k = "dg-hi" if r in hi else cls
                out.append(f'<rect class="{k}" x="{x + c * C}" y="{y + r * C}" width="{C}" height="{C}" rx="3"/>')
        return "".join(out)

    def text(x, y, s, cls, anchor="middle", extra=""):
        return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}" {extra}>{s}</text>'

    def arrow(x1, y1, x2, y2):
        return f'<line class="dg-axis" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#dg-arr)"/>'

    def head(ox, oy, title, axes, shape, gloss):
        return (text(ox + 160, oy + 24, title, "dg-title") + text(ox + 160, oy + 44, axes, "dg-sub")
                + text(ox + 160, oy + 236, shape, "dg-shape") + text(ox + 160, oy + 258, gloss, "dg-sub"))

    p = []
    # panel 1: scalar
    ox, oy = 0, 0
    p.append(head(ox, oy, "Scalar", "0 axes", "shape ()", "one number"))
    p.append(grid(ox + 143, oy + 115, 1, 1) + text(ox + 160, oy + 138, "7", "dg-num"))
    # panel 2: vector
    ox, oy = 320, 0
    p.append(head(ox, oy, "Vector", "1 axis", "shape (4,)", "a row of 4 numbers"))
    p.append(arrow(ox + 92, oy + 92, ox + 228, oy + 92) + text(ox + 160, oy + 84, "axis 0 (size 4)", "dg-axl"))
    p.append(grid(ox + 92, oy + 105, 1, 4, hi=(0,)))
    p.append(f'<path class="dg-brace" d="M{ox + 92} {oy + 154} v6 h136 v-6"/>')
    p.append(text(ox + 160, oy + 180, "length 4 = a &ldquo;4-dimensional&rdquo; vector", "dg-axl"))
    # panel 3: matrix
    ox, oy = 0, 290
    p.append(head(ox, oy, "Matrix", "2 axes", "shape (3, 4)", "3 vectors of 4 numbers"))
    p.append(arrow(ox + 92, oy + 72, ox + 228, oy + 72) + text(ox + 160, oy + 64, "axis 1 (size 4)", "dg-axl"))
    p.append(arrow(ox + 80, oy + 82, ox + 80, oy + 184))
    p.append(text(ox + 68, oy + 133, "axis 0 (size 3)", "dg-axl", extra=f'transform="rotate(-90 {ox + 68} {oy + 133})"'))
    p.append(grid(ox + 92, oy + 82, 3, 4, hi=(0,)))
    p.append(text(ox + 236, oy + 104, "&larr; one vector", "dg-axl", anchor="start"))
    # panel 4: 3-axis tensor
    ox, oy = 320, 290
    p.append(head(ox, oy, "3-axis tensor", "3 axes", "shape (2, 3, 4)", "2 matrices = 6 vectors of 4"))
    p.append(grid(ox + 106, oy + 74, 3, 4, cls="dg-cell dg-back"))
    p.append(grid(ox + 80, oy + 100, 3, 4, hi=(0,)))
    p.append(arrow(ox + 106, oy + 66, ox + 242, oy + 66) + text(ox + 174, oy + 58, "axis 2 (size 4)", "dg-axl"))
    p.append(arrow(ox + 68, oy + 100, ox + 68, oy + 202))
    p.append(text(ox + 56, oy + 151, "axis 1 (size 3)", "dg-axl", extra=f'transform="rotate(-90 {ox + 56} {oy + 151})"'))
    p.append(arrow(ox + 222, oy + 206, ox + 248, oy + 180))
    p.append(text(ox + 256, oy + 190, "axis 0", "dg-axl", anchor="start"))
    p.append(text(ox + 256, oy + 204, "(size 2)", "dg-axl", anchor="start"))
    # legend
    p.append(f'<rect class="dg-hi" x="226" y="568" width="22" height="22" rx="3"/>')
    p.append(text(256, 584, "= one vector", "dg-sub", anchor="start"))

    defs = ('<defs><marker id="dg-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="dg-arrowhead"/></marker></defs>')
    svg = (f'<svg viewBox="0 0 640 600" role="img" aria-label="Four example tensors: a scalar with shape (), '
           f'a vector with shape (4,), a matrix with shape (3, 4) and a 3-axis tensor with shape (2, 3, 4)">'
           + defs + "".join(p) + "</svg>")
    return '<figure class="diagram">' + svg + "</figure>"



# ---------------------------------------------------- figures: attention heatmap
_VIRIDIS = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]


def viridis(t):
    """Approximate matplotlib's viridis colormap for t in [0, 1]. Returns (css colour, text colour)."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_VIRIDIS, _VIRIDIS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            rgb = tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
            break
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", ("#1d1f23" if lum > 140 else "#ffffff")


def _svg_text(x, y, s, cls, anchor="middle", extra=""):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}" {extra}>{s}</text>'


def figure_attention_heatmap():
    """A made-up 5-token attention matrix with numbers in the cells, annotated step by step."""
    toks = ["[CLS]", "The", "service", "crashed", "[SEP]"]
    W = [  # each row sums to 1.00; invented for teaching, not model output
        [0.10, 0.05, 0.35, 0.40, 0.10],
        [0.10, 0.30, 0.45, 0.10, 0.05],
        [0.05, 0.20, 0.25, 0.40, 0.10],
        [0.05, 0.10, 0.60, 0.15, 0.10],
        [0.30, 0.05, 0.15, 0.30, 0.20],
    ]
    ox, oy, cw, ch = 140, 96, 66, 42
    p = [_svg_text(ox + 5 * cw / 2, 26, "column = the token being looked AT", "dg-axl"),
         _svg_text(40, oy + 5 * ch / 2, "row = the token doing the LOOKING", "dg-axl",
                   extra=f'transform="rotate(-90 40 {oy + 5 * ch / 2})"'),
         _svg_text(ox + 5 * cw + 46, oy - 14, "row total", "dg-sub")]
    for j, t in enumerate(toks):
        p.append(_svg_text(ox + j * cw + cw / 2, oy - 14, t, "dg-sub"))
    for i, t in enumerate(toks):
        p.append(_svg_text(ox - 10, oy + i * ch + ch / 2 + 5, t, "dg-sub", anchor="end"))
        for j in range(5):
            colour, ink = viridis(W[i][j] / 0.6)
            p.append(f'<rect x="{ox + j * cw}" y="{oy + i * ch}" width="{cw}" height="{ch}" fill="{colour}" '
                     f'stroke="var(--surface)" stroke-width="1.5"/>')
            p.append(f'<text x="{ox + j * cw + cw / 2}" y="{oy + i * ch + ch / 2 + 5}" text-anchor="middle" '
                     f'fill="{ink}" class="dg-heatnum">{W[i][j]:.2f}</text>')
        p.append(_svg_text(ox + 5 * cw + 46, oy + i * ch + ch / 2 + 5, f"{sum(W[i]):.2f}", "dg-sub"))
    # highlight the row for "crashed"
    p.append(f'<rect class="dg-outline" x="{ox - 2}" y="{oy + 3 * ch - 2}" width="{5 * cw + 4}" height="{ch + 4}" rx="3"/>')
    y0 = oy + 5 * ch + 34
    p.append(_svg_text(ox + 5 * cw / 2, y0, "Outlined row: \u201ccrashed\u201d puts 0.60 of its attention on \u201cservice\u201d.", "dg-note", anchor="middle"))
    p.append(_svg_text(ox + 5 * cw / 2, y0 + 22, "(Numbers are invented for teaching, not model output.)", "dg-sub", anchor="middle"))
    h = y0 + 34
    svg = (f'<svg viewBox="0 0 700 {h}" role="img" aria-label="A worked example of an attention heatmap with five tokens">'
           + "".join(p) + "</svg>")
    return '<figure class="diagram">' + svg + "</figure>"


def figure_attention_patterns():
    """Four small 6x6 patterns people commonly see: self, previous token, sink, broad."""
    n, c = 6, 24
    pats = [
        ("Self", "bright diagonal", lambda i, j: 1.0 if i == j else 0.04),
        ("Previous token", "bright just below the diagonal", lambda i, j: 1.0 if j == max(i - 1, 0) else 0.04),
        ("Sink", "one bright column", lambda i, j: 1.0 if j == n - 1 else 0.04),
        ("Broad", "evenly dim, no standout", lambda i, j: 0.35),
    ]
    p = []
    for k, (title, desc, f) in enumerate(pats):
        ox = 26 + k * 170
        p.append(_svg_text(ox + n * c / 2, 24, title, "dg-title"))
        p.append(_svg_text(ox + n * c / 2, 40 + n * c + 24, desc, "dg-sub"))
        for i in range(n):
            for j in range(n):
                colour, _ = viridis(f(i, j))
                p.append(f'<rect x="{ox + j * c}" y="{40 + i * c}" width="{c}" height="{c}" fill="{colour}" '
                         f'stroke="var(--surface)" stroke-width="1"/>')
    svg = ('<svg viewBox="0 0 700 225" role="img" aria-label="Four common attention patterns: self, previous token, sink and broad">'
           + "".join(p) + "</svg>")
    return '<figure class="diagram">' + svg + "</figure>"



# ----------------------------------------------- figures: projection and mixing
def figure_projection_by_hand():
    """y = W.x + b with tiny numbers, row 0 highlighted, calculation written out."""
    C = 44
    W = [[1, 0, 2], [0, 1, -1]]
    x = [1, 2, 3]
    b = [0, 0.5]
    y = [7, -0.5]
    p = []

    def cell(cx, cy, v):
        return (f'<rect class="dg-cell" x="{cx}" y="{cy}" width="{C}" height="{C}" rx="3"/>'
                f'<text class="dg-num" x="{cx + C / 2}" y="{cy + C / 2 + 6}">{v:g}</text>')

    wx, wy, xx, xy, bx, by, yx, yy = 40, 70, 222, 48, 310, 70, 400, 70
    for r in range(2):
        for c in range(3):
            p.append(cell(wx + c * C, wy + r * C, W[r][c]))
    for r in range(3):
        p.append(cell(xx, xy + r * C, x[r]))
    for r in range(2):
        p.append(cell(bx, by + r * C, b[r]))
        p.append(cell(yx, yy + r * C, y[r]))
    p.append(f'<rect class="dg-outline" x="{wx - 2}" y="{wy - 2}" width="{3 * C + 4}" height="{C + 4}" rx="4"/>')
    p.append(f'<rect class="dg-outline" x="{yx - 2}" y="{yy - 2}" width="{C + 4}" height="{C + 4}" rx="4"/>')
    for ox_, txt in [(wx + 3 * C / 2, "W  (2 × 3)"), (xx + C / 2, "x  (3)"), (bx + C / 2, "b  (2)"), (yx + C / 2, "y  (2)")]:
        p.append(_svg_text(ox_, 34, txt, "dg-sub"))
    for ox_, txt in [(197, "·"), (288, "+"), (377, "=")]:
        p.append(_svg_text(ox_, 123, txt, "dg-title"))
    p.append(_svg_text(20, 222, "output 0  =  1×1 + 0×2 + 2×3 + 0  =  7        ← row 0 of W, matched against x", "dg-note", anchor="start"))
    p.append(_svg_text(20, 248, "output 1  =  0×1 + 1×2 + (−1)×3 + 0.5  =  −0.5", "dg-note", anchor="start"))
    svg = ('<svg viewBox="0 0 640 268" role="img" aria-label="A tiny projection: a 2 by 3 matrix W times a 3-number vector x plus a bias b gives a 2-number vector y">'
           + "".join(p) + "</svg>")
    return '<figure class="diagram">' + svg + "</figure>"


def figure_heads_mixed():
    """Three toy heads joined end to end, then mixed by a matrix so every output blends all heads."""
    C = 24
    cols = ["dg-h1", "dg-h2", "dg-h3"]
    defs = ('<defs><marker id="dg-arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="dg-arrowhead"/></marker>'
            '<pattern id="mixpat" width="9" height="24" patternUnits="userSpaceOnUse">'
            + "".join(f'<rect class="{c}" x="{3 * i}" y="0" width="3" height="24"/>' for i, c in enumerate(cols))
            + "</pattern></defs>")
    p = [defs]
    for g, x0 in enumerate([70, 270, 470]):
        p.append(_svg_text(x0 + 2 * C, 28, f"head {g}", "dg-sub"))
        for c in range(4):
            p.append(f'<rect class="{cols[g]}" x="{x0 + c * C}" y="38" width="{C}" height="{C}" rx="3" stroke="var(--surface)" stroke-width="1.5"/>')
    p.append(f'<line class="dg-axis" x1="350" y1="70" x2="350" y2="112" marker-end="url(#dg-arr2)"/>')
    p.append(_svg_text(362, 96, "join end to end", "dg-axl", anchor="start"))
    for i in range(12):
        p.append(f'<rect class="{cols[i // 4]}" x="{206 + i * C}" y="120" width="{C}" height="{C}" rx="3" stroke="var(--surface)" stroke-width="1.5"/>')
    p.append(_svg_text(206 + 12 * C + 14, 137, "joined vector", "dg-sub", anchor="start"))
    p.append(f'<line class="dg-axis" x1="350" y1="152" x2="350" y2="196" marker-end="url(#dg-arr2)"/>')
    p.append(_svg_text(362, 180, "× W_o   (output.dense: a learned 12 × 12 matrix)", "dg-axl", anchor="start"))
    for i in range(12):
        p.append(f'<rect fill="url(#mixpat)" x="{206 + i * C}" y="204" width="{C}" height="{C}" rx="3" stroke="var(--surface)" stroke-width="1.5"/>')
    p.append(_svg_text(206 + 12 * C + 14, 221, "output vector", "dg-sub", anchor="start"))
    p.append(_svg_text(350, 266, "Each output number is a weighted sum of ALL 12 joined numbers,", "dg-note"))
    p.append(_svg_text(350, 288, "so every one contains something from every head (the stripes).", "dg-note"))
    p.append(_svg_text(350, 314, "Toy sizes: 3 heads of 4 numbers. The real DeBERTa layer has 12 heads of 64.", "dg-sub"))
    svg = ('<svg viewBox="0 0 700 326" role="img" aria-label="Three attention heads joined into one vector, then mixed by a matrix so each output number blends all heads">'
           + "".join(p) + "</svg>")
    return '<figure class="diagram">' + svg + "</figure>"


def figure_rag_flow():
    """Two rows: index documents once; answer each question. The embedding model is the shared part."""
    bw, bh, gap, x0 = 140, 56, 36, 24
    p = ['<defs><marker id="dg-arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="dg-arrowhead"/></marker></defs>']

    def box(i, y, lines, cls="dg-box", outline=False):
        x = x0 + i * (bw + gap)
        out = f'<rect class="{cls}" x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8"/>'
        if outline:
            out += f'<rect class="dg-outline" x="{x - 2}" y="{y - 2}" width="{bw + 4}" height="{bh + 4}" rx="10"/>'
        for k, ln in enumerate(lines):
            out += _svg_text(x + bw / 2, y + 24 + k * 17 - (8 if len(lines) == 2 else 0) + (0 if len(lines) == 2 else 0), ln,
                             "dg-boxt" if k == 0 else "dg-sub")
        return out, x

    def arrow(i, y):
        xa = x0 + i * (bw + gap) + bw
        return f'<line class="dg-axis" x1="{xa + 2}" y1="{y + bh / 2}" x2="{xa + gap - 2}" y2="{y + bh / 2}" marker-end="url(#dg-arr3)"/>'

    y1, y2 = 56, 190
    p.append(_svg_text(x0, 28, "A.  Index your documents (once)", "dg-axl", anchor="start"))
    p.append(_svg_text(x0, 162, "B.  Answer a question (every time)", "dg-axl", anchor="start"))
    row1 = [(["Documents", "your own text"], "dg-box", False), (["Chunks", "paragraph-sized"], "dg-box", False),
            (["Embedding model", "vector per chunk"], "dg-box", True), (["Vector database", "vectors + text"], "dg-box", False)]
    row2 = [(["Question", "typed by a user"], "dg-box", False), (["Embedding model", "the SAME one"], "dg-box", True),
            (["Nearest vectors", "-> best chunks"], "dg-box", False), (["LLM (a decoder)", "writes the answer"], "dg-llm", False)]
    for i, (lines, cls, hi) in enumerate(row1):
        b, _ = box(i, y1, lines, cls, hi); p.append(b)
        if i < 3: p.append(arrow(i, y1))
    for i, (lines, cls, hi) in enumerate(row2):
        b, _ = box(i, y2, lines, cls, hi); p.append(b)
        if i < 3: p.append(arrow(i, y2))
    dbx = x0 + 3 * (bw + gap) + bw / 2
    nnx = x0 + 2 * (bw + gap) + bw / 2
    p.append(f'<path class="dg-dash" d="M{dbx} {y1 + bh + 2} C{dbx} {y1 + bh + 40}, {nnx} {y2 - 40}, {nnx} {y2 - 2}" marker-end="url(#dg-arr3)"/>')
    p.append(_svg_text((dbx + nnx) / 2 + 70, 138, "search here", "dg-sub", anchor="start"))
    svg = ('<svg viewBox="0 0 740 270" role="img" aria-label="A retrieval pipeline: documents are chunked, embedded and stored; a question is embedded with the same model, the nearest vectors are found, and an LLM writes the answer">'
           + "".join(p) + "</svg>")
    return '<figure class="diagram">' + svg + "</figure>"

FIGURES = {
    "tensor-shapes": figure_tensor_shapes,
    "attention-heatmap": figure_attention_heatmap,
    "attention-patterns": figure_attention_patterns,
    "projection-by-hand": figure_projection_by_hand,
    "heads-mixed": figure_heads_mixed,
    "rag-flow": figure_rag_flow,
}
FIGURE_RE = re.compile(r"<!-- figure:([a-z-]+):start -->.*?<!-- figure:\1:end -->", re.S)


def expand_figures(text):
    """Replace each marked figure block (plain-text sketch in the markdown) with its SVG."""
    return FIGURE_RE.sub(lambda m: "\n\n" + FIGURES[m.group(1)]() + "\n\n", text)


# Words that must never become tooltip triggers on their own (too ambiguous).
SKIP_PRIMARY = set()
# Extra surface forms for a glossary term (plural -s/-es is handled in the browser).
EXTRAS = {
    "parameter": ["learned parameter"],
    "vocabulary": ["vocab"],
    "subword tokenization": ["subword"],
    "hidden state": ["hidden-state"],
    "layer": ["transformer layer"],
    "linear layer": ["linear map"],
    "q / k / v projections": ["q/k/v projection", "q/k/v projections"],
    "layer normalization": ["layer norm", "normalization", "rmsnorm"],
    "activation function": ["activation"],
    "residual connection": ["residual"],
    "mlp": ["feed-forward"],
    "encoder model": ["encoder"],
    "decoder model": ["decoder"],
    "task head": ["classification head", "lm head", "language-model head", "output head"],
    "pooling": ["pooler", "mean pooling", "cls pooling"],
    "weight tying": ["tied embeddings"],
    "pretraining": ["pretrained"],
    "fine-tuning": ["fine-tune", "fine-tuned"],
    "frozen parameter": ["frozen"],
    "forward hook": ["hook"],
    "pca": ["principal component analysis"],
    "linear probe": ["probe"],
    "gqa": ["grouped-query attention"],
    "rag": ["retrieval"],
    "vector database": ["vector store", "vector db"],
}

glossary, parts = [], []
for kind, body_lines in blocks:
    text = "\n".join(body_lines).rstrip() + "\n"
    rendered = md.render(text)
    if kind == "heading":
        text = expand_figures(text)
        parts.append(md.render(text))
        continue

    label_raw = re.match(r"^\*\*(.+?)\.\*\*", body_lines[0]).group(1)
    label = label_raw.replace("`", "")
    m = re.match(r"^(.*?)\s*\((.*)\)$", label)
    primary, paren = (m.group(1), m.group(2)) if m else (label, "")

    keys = []
    if not primary.startswith("▁"):
        for piece in primary.split(" vs. "):  # "Training vs. inference" -> two terms
            if piece.lower() not in SKIP_PRIMARY:
                keys.append(piece.lower())
        pl = paren.lower()
        if paren and not pl.startswith(("also called", "of ")) and len(paren) > 1:
            keys.append(pl)
        for k in list(keys) + [primary.lower()]:
            keys += EXTRAS.get(k, [])
    keys = sorted(set(keys))

    slug = re.sub(r"[^a-z0-9]+", "-", primary.lower()).strip("-") or "subword-marker"
    # Tooltip body = the rendered entry without its leading bold label.
    body_html = re.sub(r"^<p><strong>.*?</strong>\s*", "<p>", rendered, count=1, flags=re.S)
    glossary.append({"id": slug, "label": label, "keys": keys, "body": body_html.strip()})

    shown = rendered.replace("<strong>", '<strong class="gl-term">', 1)
    parts.append(f'<div class="gl-entry" id="gl-{slug}" data-entry="{slug}">\n{shown}</div>\n')

content = md.render(expand_figures(pre_md)) + "".join(parts) + md.render(expand_figures(post_md))


def link_sibling_pages(page, folder):
    """Point relative links to sibling .md manuals at their built .html pages."""
    def fix(m):
        name, frag = m.group(1), m.group(2) or ""
        return f'href="{name}.html{frag}"' if (folder / f"{name}.md").exists() else m.group(0)
    return re.sub(r'href="([A-Za-z0-9_.-]+)\.md(#[^"]*)?"', fix, page)


content = link_sibling_pages(content, src_path.parent)
content = content.replace("<blockquote>\n<p><strong>Notice!!", '<blockquote class="notice">\n<p><strong>Notice!!')

gloss_json = json.dumps(glossary, ensure_ascii=False).replace("</", "<\\/")
tpl = open(tpl_path, encoding="utf-8").read()
out = tpl.replace("__CONTENT__", content).replace("__GLOSSARY__", gloss_json)
open(out_path, "w", encoding="utf-8").write(out)
print(f"wrote {out_path}: {len(glossary)} glossary entries, "
      f"{sum(len(e['keys']) for e in glossary)} trigger phrases, {len(out)//1024} KiB")
