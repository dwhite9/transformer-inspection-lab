#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py", "pygments"]
# ///
"""Build the single-file HTML linear algebra study guide from its markdown.

The markdown file is the source of truth. Edit the .md, then re-run this script.
Do not edit the .html by hand.

    uv run build_study_guide.py
    uv run build_study_guide.py [input.md [template.html [output.html]]]

With no arguments it reads ../manuals/linear-algebra-study-guide.md and
study-guide-template.html (in this folder) and writes
linear-algebra-study-guide.html next to the markdown, in ../manuals/.

Markdown extensions (on top of the lab-manual conventions):

  {{W}}               a variable with a hover tooltip. The symbol must have an
                      entry under "## Variables and symbols" in the glossary.
  {{q|q₁}}            same tooltip as {{q}}, but displays "q₁".
  [[1, 2; 3, 4]]      a bracketed matrix: ";" separates rows, "," entries.
                      [[1; -1]] is a column vector, [[1, -1]] a row vector.
  <!-- widget:NAME --> an interactive widget (matmul, transform, dot, softmax);
                      the template's script builds it in the browser.
  <!-- figure:NAME:start --> ... <!-- figure:NAME:end -->
                      a plain-text sketch replaced by an SVG figure below.

None of these are expanded inside code spans or fenced code blocks.
"""
import html
import json
import math
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
src_path = Path(args[0]) if len(args) > 0 else HERE.parent / "manuals" / "linear-algebra-study-guide.md"
tpl_path = Path(args[1]) if len(args) > 1 else HERE / "study-guide-template.html"
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


# ------------------------------------------------------ variables and matrices
def var_id(sym):
    """Stable, case-sensitive id for a symbol: W -> var-W, σ -> var-u3c3."""
    return "var-" + "".join(c if (c.isascii() and c.isalnum()) else f"u{ord(c):x}" for c in sym)


# How some symbols are drawn when used inline (default: the symbol itself).
VAR_DISPLAY = {"ᵀ": "<sup>T</sup>"}
VARS_USED = set()


def var_html(key, display=None):
    VARS_USED.add(key)
    shown = display if display is not None else VAR_DISPLAY.get(key, html.escape(key))
    italic = " var-it" if re.fullmatch(r"[A-Za-z]", key) else ""
    return (f'<var class="term var{italic}" tabindex="0" data-id="{var_id(key)}">{shown}</var>')


def matrix_html(body):
    rows = [[c.strip() for c in r.split(",")] for r in body.split(";")]
    ncol = max(len(r) for r in rows)
    cells = "".join(f"<span>{c.replace('-', '−') if re.fullmatch(r'-[0-9.]+', c) else c}</span>"
                    for r in rows for c in r + [""] * (ncol - len(r)))
    return f'<span class="mx" style="--cols:{ncol}">{cells}</span>'


VAR_RE = re.compile(r"\{\{([^|{}]+?)(?:\|([^{}]+?))?\}\}")
MX_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
WIDGET_RE = re.compile(r"<!-- widget:([a-z-]+) -->")
CODE_RE = re.compile(r"(^```.*?^```[ \t]*$|`[^`\n]+`)", re.S | re.M)


def expand_inline(text):
    """Expand {{var}}, [[matrix]] and widget markers everywhere except in code."""
    out, last = [], 0
    for m in CODE_RE.finditer(text):
        out.append(_expand(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_expand(text[last:]))
    return "".join(out)


def _expand(t):
    t = MX_RE.sub(lambda m: matrix_html(m.group(1)), t)
    t = VAR_RE.sub(lambda m: var_html(m.group(1).strip(), m.group(2)), t)
    return WIDGET_RE.sub(lambda m: f'<div class="widget" data-widget="{m.group(1)}"></div>', t)


src = expand_inline(src)

# ------------------------------------------------------- split out the glossary
lines = src.split("\n")
g = next(i for i, l in enumerate(lines) if l.startswith("# Part 0: Glossary"))
p = next(i for i, l in enumerate(lines) if l.startswith("# Part 0.5"))
pre_md = "\n".join(lines[: g + 1])
gl_lines = lines[g + 1 : p]
while gl_lines and gl_lines[-1].strip() in ("", "---"):
    gl_lines.pop()
post_md = "---\n\n" + "\n".join(lines[p:])

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


# ------------------------------------------------------------- SVG helpers
def T(x, y, s, cls, anchor="middle", extra=""):
    return f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}" {extra}>{s}</text>'


def ARROWDEFS(mid, cls="dg-arrowhead"):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="{cls}"/></marker></defs>')


def LINE(x1, y1, x2, y2, cls="dg-axis", mk=None):
    m = f' marker-end="url(#{mk})"' if mk else ""
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"{m}/>'


def CELL(x, y, v, w=40, h=40, cls="dg-cell"):
    v = re.sub(r"^-(?=[0-9])", "−", str(v))
    return (f'<rect class="{cls}" x="{x:g}" y="{y:g}" width="{w}" height="{h}" rx="3"/>'
            + T(x + w / 2, y + h / 2 + 6, v, "dg-num"))


def GRIDM(x, y, M, w=40, h=40, cls=lambda r, c: "dg-cell"):
    return "".join(CELL(x + c * w, y + r * h, f"{M[r][c]:g}" if isinstance(M[r][c], (int, float)) else M[r][c],
                        w, h, cls(r, c)) for r in range(len(M)) for c in range(len(M[0])))


def FIG(svg_inner, vb, label):
    return (f'<figure class="diagram"><svg viewBox="{vb}" role="img" aria-label="{html.escape(label)}">'
            + svg_inner + "</svg></figure>")


# ------------------------------------------------------------------- figures
def fig_vector_arrow():
    """A 2-D vector u = (3, 2) as an arrow with its components and length."""
    s, ox, oy = 50, 70, 250
    p = [ARROWDEFS("fa1")]
    for i in range(0, 5):
        p.append(LINE(ox + i * s, oy, ox + i * s, oy - 4 * s + 20, "dg-grid"))
    for j in range(0, 4):
        p.append(LINE(ox, oy - j * s, ox + 4 * s + 20, oy - j * s, "dg-grid"))
    p.append(LINE(ox, oy, ox + 4.4 * s, oy, "dg-axis", "fa1") + LINE(ox, oy, ox, oy - 3.6 * s, "dg-axis", "fa1"))
    p.append(T(ox + 4.4 * s + 6, oy + 5, "axis 1", "dg-axl", "start") + T(ox, oy - 3.6 * s - 10, "axis 2", "dg-axl"))
    p.append(LINE(ox + 3 * s, oy, ox + 3 * s, oy - 2 * s, "dg-dash") + LINE(ox, oy - 2 * s, ox + 3 * s, oy - 2 * s, "dg-dash"))
    p.append(LINE(ox, oy, ox + 3 * s, oy - 2 * s, "dg-vec", "fa1"))
    p.append(f'<circle class="dg-dot" cx="{ox + 3 * s}" cy="{oy - 2 * s}" r="4"/>')
    p.append(T(ox + 3 * s + 10, oy - 2 * s - 8, "u = [3, 2]", "dg-title", "start"))
    p.append(T(ox + 1.5 * s, oy + 22, "first entry: 3 steps along axis 1", "dg-sub"))
    p.append(T(ox + 3 * s + 8, oy - s, "second entry: 2 up", "dg-sub", "start"))
    lx, ly = ox + 1.25 * s, oy - 1.1 * s - 16
    p.append(T(lx, ly, "‖u‖ = √(3² + 2²) ≈ 3.61", "dg-axl", "middle", f'transform="rotate(-33.7 {lx:g} {ly:g})"'))
    return FIG("".join(p), "0 0 560 290", "The vector u = [3, 2] drawn as an arrow from the origin, with dashed lines showing its two components and its length labelled.")


def fig_dot_product():
    """Multiply matching entries, then add: u·w = 1·4 + 2·(−1) + 3·2 = 8."""
    C = 52
    u, w = [1, 2, 3], [4, -1, 2]
    pr = [a * b for a, b in zip(u, w)]
    x0 = 150
    p = [T(70, 58, "u", "dg-title"), T(70, 128, "w", "dg-title"), T(70, 214, "u⊙w", "dg-title")]
    for i in range(3):
        p.append(CELL(x0 + i * (C + 30), 32, f"{u[i]:g}", C, C, "dg-cell dg-c1"))
        p.append(CELL(x0 + i * (C + 30), 102, f"{w[i]:g}".replace("-", "−"), C, C, "dg-cell dg-c2"))
        p.append(T(x0 + i * (C + 30) + C / 2, 172, "×", "dg-axl"))
        p.append(CELL(x0 + i * (C + 30), 188, f"{pr[i]:g}".replace("-", "−"), C, C, "dg-hi"))
        if i < 2:
            p.append(T(x0 + i * (C + 30) + C + 15, 220, "+", "dg-title"))
    p.append(T(x0 + 3 * (C + 30) + 6, 220, "=  8", "dg-title", "start"))
    p.append(T(330, 278, "Step 1: multiply matching positions.  Step 2: add the products.", "dg-note"))
    p.append(T(330, 302, "The two vectors must have the same length.", "dg-sub"))
    return FIG("".join(p), "0 0 660 318", "Dot product of u = [1, 2, 3] and w = [4, −1, 2]: products 4, −2 and 6 add up to 8.")


def fig_transpose():
    """Row i of M becomes column i of Mᵀ; the diagonal stays put."""
    C = 46
    M = [[1, 2, 3], [4, 5, 6]]
    p = [ARROWDEFS("fa2")]
    p.append(GRIDM(40, 70, M, C, C, lambda r, c: "dg-cell dg-c1" if r == 0 else "dg-cell dg-c2"))
    p.append(T(40 + 1.5 * C, 52, "M   (2 × 3)", "dg-sub"))
    p.append(LINE(40, 70, 40 + 2 * C, 70 + 2 * C, "dg-dash"))
    p.append(LINE(230, 116, 330, 116, "dg-axis", "fa2"))
    p.append(T(280, 104, "flip across", "dg-axl") + T(280, 140, "the diagonal", "dg-axl"))
    Mt = [[M[r][c] for r in range(2)] for c in range(3)]
    p.append(GRIDM(370, 46, Mt, C, C, lambda r, c: "dg-cell dg-c1" if c == 0 else "dg-cell dg-c2"))
    p.append(T(370 + C, 30, "Mᵀ   (3 × 2)", "dg-sub"))
    p.append(LINE(370, 46, 370 + 2 * C, 46 + 2 * C, "dg-dash"))
    p.append(T(300, 222, "Row 1 of M  →  column 1 of Mᵀ.   Row 2  →  column 2.", "dg-note"))
    p.append(T(300, 246, "Entry rule: (Mᵀ)ᵢⱼ = Mⱼᵢ.  The dashed diagonal (1, 5) does not move.", "dg-sub"))
    return FIG("".join(p), "0 0 600 262", "A 2 by 3 matrix M and its 3 by 2 transpose. Its first row (1, 2, 3) becomes the first column; its second row becomes the second column.")


def fig_matvec_views():
    """Row view (dot products) and column view (weighted sum of columns) of W x."""
    C = 40
    W = [[1, 0, 2], [0, 1, -1]]
    x = [1, 2, 3]
    p = [T(20, 20, "Row view: each output is a dot product", "dg-title", "start")]
    p.append(GRIDM(40, 46, W, C, C, lambda r, c: "dg-hi" if r == 0 else "dg-cell"))
    p.append(T(176, 92, "·", "dg-title"))
    p.append(GRIDM(194, 26, [[v] for v in x], C, C, lambda r, c: "dg-hi"))
    p.append(T(250, 92, "=", "dg-title"))
    p.append(GRIDM(268, 46, [[7], [-1]], C, C, lambda r, c: "dg-hi" if r == 0 else "dg-cell"))
    p.append(T(330, 70, "row 1 · x = 1·1 + 0·2 + 2·3 = 7", "dg-note", "start"))
    p.append(T(330, 110, "row 2 · x = 0·1 + 1·2 + (−1)·3 = −1", "dg-note", "start"))
    y0 = 180
    p.append(T(20, y0, "Column view: a weighted sum of W's columns", "dg-title", "start"))
    cols = [[1, 0], [0, 1], [2, -1]]
    xx = 40
    for i in range(3):
        p.append(T(xx + 8, y0 + 66, f"{x[i]} ×", "dg-axl", "start"))
        p.append(GRIDM(xx + 40, y0 + 22, [[v] for v in cols[i]], C, C, lambda r, c, i=i: f"dg-cell dg-c{1 + i % 2}"))
        p.append(T(xx + 98, y0 + 66, "+" if i < 2 else "=", "dg-title"))
        xx += 120
    p.append(GRIDM(xx, y0 + 22, [[7], [-1]], C, C, lambda r, c: "dg-hi"))
    p.append(T(xx + 54, y0 + 50, "1·[1,0] + 2·[0,1] + 3·[2,−1]", "dg-sub", "start"))
    p.append(T(xx + 54, y0 + 72, "= [7, −1]  (same answer)", "dg-sub", "start"))
    return FIG("".join(p), "0 0 700 296", "W times x computed two ways: as two dot products of W's rows with x, and as x's entries weighting W's three columns. Both give [7, −1].")


def fig_matmul_shapes():
    """(m × n)(n × p) = (m × p); the inner sizes must match."""
    p = [ARROWDEFS("fa3")]
    p.append(f'<rect class="dg-box" x="40" y="60" width="150" height="90" rx="4"/>')
    p.append(f'<rect class="dg-hi" x="40" y="88" width="150" height="26" rx="3"/>')
    p.append(T(115, 48, "M:  m × n", "dg-title") + T(115, 134, "row i", "dg-axl"))
    p.append(T(212, 112, "·", "dg-title"))
    p.append(f'<rect class="dg-box" x="232" y="34" width="90" height="150" rx="4"/>')
    p.append(f'<rect class="dg-hi" x="262" y="34" width="26" height="150" rx="3"/>')
    p.append(T(277, 24, "N:  n × p", "dg-title") + T(305, 200, "column j", "dg-axl"))
    p.append(T(346, 112, "=", "dg-title"))
    p.append(f'<rect class="dg-box" x="370" y="60" width="90" height="90" rx="4"/>')
    p.append(f'<rect class="dg-outline" x="400" y="88" width="26" height="26" rx="3"/>')
    p.append(T(415, 48, "MN:  m × p", "dg-title"))
    p.append(T(500, 92, "entry (i, j) =", "dg-note", "start") + T(500, 114, "row i of M · column j of N", "dg-note", "start"))
    p.append(T(330, 238, "(m × n) · (n × p)  =  (m × p)", "dg-shape"))
    p.append(T(330, 262, "the two inner sizes (n and n) must match; the outer sizes give the result's shape", "dg-sub"))
    return FIG("".join(p), "0 0 700 278", "Matrix multiplication: an m by n matrix times an n by p matrix gives an m by p matrix. Entry i, j is row i of the first dotted with column j of the second.")


def fig_outer_product():
    """[1; 2] times [3, 4, 5] gives a 2 × 3 grid of products: rank 1."""
    C = 46
    a, b = [1, 2], [3, 4, 5]
    p = [GRIDM(60, 106, [[v] for v in a], C, C, lambda r, c: "dg-cell dg-c1")]
    p.append(T(83, 92, "u (2 × 1)", "dg-sub"))
    p.append(GRIDM(170, 40, [b], C, C, lambda r, c: "dg-cell dg-c2"))
    p.append(T(170 + 1.5 * C, 28, "wᵀ (1 × 3)", "dg-sub"))
    p.append(T(140, 136, "·", "dg-title"))
    M = [[x * y for y in b] for x in a]
    p.append(GRIDM(170, 106, M, C, C, lambda r, c: "dg-hi"))
    p.append(T(420, 120, "entry (i, j) = uᵢ × wⱼ", "dg-note", "start"))
    p.append(T(420, 146, "row 2 = 2 × row 1", "dg-note", "start"))
    p.append(T(420, 172, "so the result has rank 1", "dg-sub", "start"))
    return FIG("".join(p), "0 0 640 220", "Outer product of the column [1, 2] with the row [3, 4, 5]: a 2 by 3 grid where each entry is a product, and the second row is twice the first.")


def fig_transform_square():
    """A 2 × 2 matrix sends the unit square to a parallelogram; columns are where the axes land."""
    s = 60
    p = [ARROWDEFS("fa4"), ARROWDEFS("fa5", "dg-arrowhead2").replace("<defs>", "<defs>")]

    def panel(ox, oy, M, title):
        q = []
        for i in range(-1, 4):
            q.append(LINE(ox + i * s, oy + s, ox + i * s, oy - 3 * s, "dg-grid"))
        for j in range(-1, 4):
            q.append(LINE(ox - s, oy - j * s, ox + 3 * s, oy - j * s, "dg-grid"))
        (a, b), (c, d) = M
        pts = [(0, 0), (a, c), (a + b, c + d), (b, d)]
        q.append('<polygon class="dg-area" points="' + " ".join(f"{ox + X * s:g},{oy - Y * s:g}" for X, Y in pts) + '"/>')
        q.append(LINE(ox, oy, ox + a * s, oy - c * s, "dg-vec", "fa4"))
        q.append(LINE(ox, oy, ox + b * s, oy - d * s, "dg-vec2", "fa5"))
        q.append(T(ox + s, oy + s + 34, title, "dg-title"))
        return "".join(q)

    p.append(panel(90, 220, [[1, 0], [0, 1]], "before: identity"))
    p.append(panel(420, 220, [[2, 1], [0, 1]], "after M = [[2, 1], [0, 1]]"))
    p.append(T(320, 350, "Blue = column 1 = where [1, 0] lands.   Orange = column 2 = where [0, 1] lands.", "dg-note"))
    p.append(T(320, 374, "The unit square (area 1) becomes a parallelogram of area det M = 2·1 − 1·0 = 2.", "dg-sub"))
    return FIG("".join(p), "0 0 640 390", "Left: the unit square and the two axis arrows. Right: after multiplying by M = [[2, 1], [0, 1]] the arrows become M's columns and the square becomes a parallelogram of area 2.")


def fig_layer():
    """x → W x + b → σ → h, with shapes."""
    p = [ARROWDEFS("fa6")]
    boxes = [("x", "input, n numbers"), ("z = Wx + b", "linear part, m numbers"), ("h = σ(z)", "activation, m numbers")]
    for i, (t1, t2) in enumerate(boxes):
        x = 30 + i * 230
        p.append(f'<rect class="{"dg-llm" if i == 1 else "dg-box"}" x="{x}" y="40" width="180" height="70" rx="8"/>')
        p.append(T(x + 90, 70, t1, "dg-boxt") + T(x + 90, 92, t2, "dg-sub"))
        if i < 2:
            p.append(LINE(x + 182, 75, x + 228, 75, "dg-axis", "fa6"))
    p.append(T(350, 150, "W is m × n (one row per output), b has m entries, σ acts on each entry separately.", "dg-note"))
    p.append(T(350, 174, "Parameters in one layer: m·n weights + m biases.", "dg-sub"))
    return FIG("".join(p), "0 0 700 190", "One neural-network layer: input x, then the linear part z = Wx + b, then the activation h = σ(z).")


def fig_attention():
    """q against each key → scores → softmax → weights → weighted sum of values."""
    p = [ARROWDEFS("fa7")]
    cols = [("q · kᵢ / √d", ["0.707", "0"], "scores s"), ("softmax", ["0.670", "0.330"], "weights α"),
            ("Σ αᵢ vᵢ", ["1.34", "1.32"], "output o")]
    p.append(f'<rect class="dg-box" x="10" y="40" width="150" height="132" rx="8"/>')
    p.append(T(85, 64, "q = [1, 0]", "dg-note"))
    p.append(T(85, 88, "k₁ = [1, 0]", "dg-sub") + T(85, 106, "k₂ = [0, 1]", "dg-sub"))
    p.append(T(85, 128, "v₁ = [2, 0]", "dg-sub") + T(85, 146, "v₂ = [0, 4]", "dg-sub"))
    p.append(T(85, 164, "d = 2", "dg-sub"))
    for i, (op, vals, lab) in enumerate(cols):
        x = 220 + i * 170
        p.append(LINE(x - 50 if i == 0 else x - 70, 104, x + 10, 104, "dg-axis", "fa7"))
        p.append(T(x + 45, 34, op, "dg-axl"))
        for r, v in enumerate(vals):
            p.append(CELL(x + 15, 58 + r * 46, v, 60, 40, "dg-hi" if i == 1 else "dg-cell"))
        p.append(T(x + 45, 172, lab, "dg-sub"))
    p.append(T(380, 210, "Similar key → bigger score → bigger weight → that value dominates the output.", "dg-note"))
    p.append(T(380, 234, "Weights are positive and add to 1, so the output is a blend of the values.", "dg-sub"))
    return FIG("".join(p), "0 0 740 250", "Attention for one query: scores 0.707 and 0, softmax weights 0.670 and 0.330, output [1.34, 1.32].")


def fig_backprop():
    """Forward values left to right, gradients right to left."""
    p = [ARROWDEFS("fa8"), ARROWDEFS("fa9", "dg-arrowhead2")]
    nodes = [("x", 60), ("y = Wx", 260), ("L = ½‖y − t‖²", 500)]
    for t1, x in nodes:
        w = 170 if "L" in t1 else 120
        p.append(f'<rect class="dg-box" x="{x - w / 2}" y="70" width="{w}" height="56" rx="8"/>' + T(x, 104, t1, "dg-boxt"))
    p.append(LINE(122, 86, 198, 86, "dg-axis", "fa8") + LINE(322, 86, 413, 86, "dg-axis", "fa8"))
    p.append(T(360, 56, "forward: compute values", "dg-axl"))
    p.append(LINE(413, 112, 322, 112, "dg-back", "fa9") + LINE(198, 112, 122, 112, "dg-back", "fa9"))
    p.append(T(368, 150, "δ = ∂L/∂y = y − t", "dg-back-t"))
    p.append(T(160, 150, "∂L/∂x = Wᵀ δ", "dg-back-t"))
    p.append(T(260, 186, "∂L/∂W = δ xᵀ  (an outer product)", "dg-back-t"))
    p.append(T(330, 228, "backward: start from L and multiply by each step's local derivative (the chain rule)", "dg-sub"))
    return FIG("".join(p), "0 0 660 244", "A two-step computation graph. Forward arrows carry x to y to the loss L. Backward arrows carry the gradient: delta = y − t, then W-transpose delta to x, and delta x-transpose to W.")


def fig_jacobian():
    """J is m × n: row i = how output i responds, column j = sensitivity to input j."""
    C = 64
    p = []
    labs = [["∂y₁/∂x₁", "∂y₁/∂x₂", "∂y₁/∂x₃"], ["∂y₂/∂x₁", "∂y₂/∂x₂", "∂y₂/∂x₃"]]
    p.append(GRIDM(150, 60, labs, C + 26, 46, lambda r, c: "dg-cell dg-c2" if c == 1 else ("dg-hi" if r == 0 else "dg-cell")))
    p.append(T(150 + 1.5 * (C + 26), 44, "J  (m × n) = (2 outputs × 3 inputs)", "dg-sub"))
    p.append(T(140, 88, "row 1: all ways y₁ changes", "dg-axl", "end"))
    p.append(T(150 + 1.5 * (C + 26), 180, "highlighted column: how every output reacts to x₂", "dg-axl"))
    p.append(T(330, 214, "For y = Wx the Jacobian is just W, because ∂yᵢ/∂xⱼ = Wᵢⱼ.", "dg-note"))
    return FIG("".join(p), "0 0 660 230", "A 2 by 3 Jacobian matrix of partial derivatives: row i holds the derivatives of output i, column j the derivatives with respect to input j.")


def fig_lora():
    """d × d update versus B (d × r) times A (r × d)."""
    p = []
    S = 160
    p.append(f'<rect class="dg-box" x="40" y="40" width="{S}" height="{S}" rx="4"/>')
    p.append(T(40 + S / 2, 126, "ΔW", "dg-title") + T(40 + S / 2, 222, "d × d = d² numbers", "dg-sub"))
    p.append(T(240, 126, "=", "dg-title"))
    p.append(f'<rect class="dg-c1r" x="270" y="40" width="20" height="{S}" rx="3"/>')
    p.append(T(280, 222, "B: d × r", "dg-sub"))
    p.append(T(312, 126, "·", "dg-title"))
    p.append(f'<rect class="dg-c2r" x="330" y="110" width="{S}" height="20" rx="3"/>')
    p.append(T(330 + S / 2, 150, "A: r × d", "dg-sub"))
    p.append(T(330 + S / 2, 222, "2·d·r numbers", "dg-sub"))
    p.append(T(330, 262, "d = 4096, r = 8:   16,777,216  vs  65,536   (≈ 0.39%)", "dg-note"))
    p.append(T(330, 286, "The product has rank at most r: it can only push inputs along r directions.", "dg-sub"))
    return FIG("".join(p), "0 0 660 300", "A full d by d update compared with LoRA's tall thin B times wide thin A. For d = 4096 and r = 8 the factors have 65,536 numbers instead of 16,777,216.")


def fig_svd():
    """Unit circle → rotate (Vᵀ) → stretch (Σ) → rotate (U)."""
    p = [ARROWDEFS("fa10")]
    R = 44

    def panel(cx, title, sub, rx, ry, rot, mark):
        q = [f'<ellipse class="dg-area" cx="{cx}" cy="110" rx="{rx}" ry="{ry}" transform="rotate({rot} {cx} 110)"/>']
        ang = math.radians(rot + mark)
        q.append(LINE(cx, 110, cx + rx * math.cos(ang), 110 + rx * math.sin(ang) * (ry / rx if rx else 1), "dg-vec", "fa10"))
        q.append(T(cx, 196, title, "dg-title") + T(cx, 216, sub, "dg-sub"))
        return "".join(q)

    p.append(panel(80, "start", "unit circle", R, R, 0, -40))
    p.append(panel(230, "Vᵀ", "rotate", R, R, 0, 0))
    p.append(panel(400, "Σ", "stretch the axes", 1.6 * R, 0.6 * R, 0, 0))
    p.append(panel(580, "U", "rotate again", 1.6 * R, 0.6 * R, -30, 0))
    for x in (148, 302, 490):
        p.append(LINE(x - 12, 110, x + 12, 110, "dg-axis", "fa10"))
    p.append(T(330, 256, "M = U Σ Vᵀ: every matrix is rotate → stretch → rotate.", "dg-note"))
    p.append(T(330, 280, "The stretch factors on Σ's diagonal are the singular values.", "dg-sub"))
    return FIG("".join(p), "0 0 660 296", "Singular value decomposition shown as four panels: a unit circle is rotated, stretched into an ellipse, then rotated again.")


FIGURES = {
    "vector-arrow": fig_vector_arrow, "dot-product": fig_dot_product, "transpose": fig_transpose,
    "matvec-views": fig_matvec_views, "matmul-shapes": fig_matmul_shapes, "outer-product": fig_outer_product,
    "transform-square": fig_transform_square, "layer": fig_layer, "attention": fig_attention,
    "backprop": fig_backprop, "jacobian": fig_jacobian, "lora": fig_lora, "svd": fig_svd,
}
FIGURE_RE = re.compile(r"<!-- figure:([a-z-]+):start -->.*?<!-- figure:\1:end -->", re.S)


def expand_figures(text):
    return FIGURE_RE.sub(lambda m: "\n\n" + FIGURES[m.group(1)]() + "\n\n", text)


# ------------------------------------------------------------- glossary terms
SKIP_PRIMARY = set()
EXTRAS = {
    "matrix": ["matrices"],
    "vector": ["vectors"],
    "transpose": ["transposed", "transposing"],
    "dot product": ["inner product"],
    "matrix-vector product": ["matrix-vector multiplication", "matrix-vector multiply"],
    "matrix multiplication": ["matrix product", "matrix multiply", "matmul"],
    "linear map": ["linear transformation", "linear function"],
    "identity matrix": ["identity"],
    "inverse": ["inverse matrix", "invertible", "inverses"],
    "rank": ["low-rank", "full rank", "rank-1", "rank-r"],
    "eigenvector": ["eigenvectors"],
    "eigenvalue": ["eigenvalues"],
    "singular value decomposition": ["singular values", "singular value"],
    "activation function": ["activation", "activations"],
    "layer": ["linear layer", "dense layer", "fully connected layer"],
    "backpropagation": ["backprop", "backward pass"],
    "forward pass": ["forward"],
    "loss": ["loss function"],
    "gradient descent": ["gradient step"],
    "query, key and value vectors": ["query vector", "key vector", "value vector", "queries", "query"],
    "attention": ["self-attention", "attention weights"],
    "jacobian": ["jacobian matrix"],
    "partial derivative": ["partial derivatives"],
    "basis vectors": ["basis", "standard basis"],
    "linear combination": ["weighted sum"],
    "unit vector": ["unit length", "normalized", "normalize"],
}
# In models, "forward" alone is too common a word; keep only the phrase.
EXTRAS.pop("forward pass")

glossary, parts, in_vars = [], [], False
for kind, body_lines in blocks:
    text = "\n".join(body_lines).rstrip() + "\n"
    if kind == "heading":
        if body_lines[0].startswith("## "):
            in_vars = body_lines[0].strip() == "## Variables and symbols"
        parts.append(md.render(expand_figures(text)))
        continue
    rendered = md.render(text)
    label_raw = re.match(r"^\*\*(.+?)\.\*\*", body_lines[0]).group(1)
    label = re.sub(r"<[^>]+>", "", label_raw.replace("`", ""))
    m = re.match(r"^(.*?)\s*\((.*)\)$", label)
    primary, paren = (m.group(1), m.group(2)) if m else (label, "")

    if in_vars:
        # Symbols are triggered only by {{...}} in the markdown, never by text matching.
        slug, keys = var_id(primary), []
    else:
        keys = [k for k in primary.lower().split(" vs. ") if k not in SKIP_PRIMARY]
        pl = paren.lower()
        if paren and not pl.startswith(("also called", "of ", "e.g")) and len(paren) > 1:
            keys.append(pl)
        for k in list(keys) + [primary.lower()]:
            keys += EXTRAS.get(k, [])
        keys = sorted(set(keys))
        slug = re.sub(r"[^a-z0-9]+", "-", primary.lower()).strip("-")

    body_html = re.sub(r"^<p><strong>.*?</strong>\s*", "<p>", rendered, count=1, flags=re.S)
    glossary.append({"id": slug, "label": label, "keys": keys, "body": body_html.strip()})
    shown = rendered.replace("<strong>", '<strong class="gl-term">', 1)
    parts.append(f'<div class="gl-entry{" gl-var" if in_vars else ""}" id="gl-{slug}" data-entry="{slug}">\n{shown}</div>\n')

# ------------------------------------------------------------------- checks
defined = {e["id"] for e in glossary}
missing = sorted(k for k in VARS_USED if var_id(k) not in defined)
if missing:
    sys.exit("No glossary entry under '## Variables and symbols' for: " + ", ".join(missing))
ids = [e["id"] for e in glossary]
dupes = sorted({i for i in ids if ids.count(i) > 1})
if dupes:
    sys.exit("Duplicate glossary ids: " + ", ".join(dupes))

content = md.render(expand_figures(pre_md)) + "".join(parts) + md.render(expand_figures(post_md))
gloss_json = json.dumps(glossary, ensure_ascii=False).replace("</", "<\\/")
tpl = tpl_path.read_text(encoding="utf-8")
out = tpl.replace("__CONTENT__", content).replace("__GLOSSARY__", gloss_json)
out_path.write_text(out, encoding="utf-8")
print(f"wrote {out_path}: {len(glossary)} glossary entries "
      f"({sum(1 for e in glossary if e['id'].startswith('var-'))} symbols), "
      f"{sum(len(e['keys']) for e in glossary)} trigger phrases, {len(VARS_USED)} symbols used, {len(out)//1024} KiB")
