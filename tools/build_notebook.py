#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["nbformat"]
# ///
"""Build a runnable Jupyter notebook from the lab manual's Procedure sections.

The manual's markdown is the source of truth; this notebook is generated from it,
like the HTML page. To change the notebook in the repository, edit the manual and
re-run this script. Do not edit the .ipynb by hand.

    uv run build_notebook.py
    uv run build_notebook.py [manual.md [output.ipynb]]

With no arguments it reads ../manuals/model-inspection-lab-manual-detailed.md and
writes ../notebooks/transformer-inspection-lab.ipynb.

What goes in, per lab:
  * a heading cell: the lab title, its objective, any "Notice!!" box, and a link
    back to the lab in the manual (for the concepts and explanations);
  * the Procedure section: every ```python block becomes a code cell, other code
    blocks (for example the bash setup command) stay as markdown, and the text in
    between (step titles and instructions) becomes markdown cells;
  * the Record questions, followed by an empty cell for the reader's answers.
The answer key is left out on purpose: it stays in the manual.

Cell ids are derived from the lab and position, so rebuilding an unchanged manual
gives a byte-identical notebook (clean git diffs).
"""
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
if len(args) > 2 or any(a in ("-h", "--help") for a in args):
    sys.exit(__doc__)
src_path = Path(args[0]) if len(args) > 0 else HERE.parent / "manuals" / "model-inspection-lab-manual-detailed.md"
out_path = Path(args[1]) if len(args) > 1 else HERE.parent / "notebooks" / "transformer-inspection-lab.ipynb"
md = src_path.read_text(encoding="utf-8")

# Links in the notebook are relative to the notebook's folder, so point them at the manuals.
MANUAL_REL = "../manuals/" + src_path.name
FENCE_RE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
LINK_RE = re.compile(r"\]\(([A-Za-z0-9_.-]+\.md)(#[^)]*)?\)")


def fix_links(text):
    """[x](other.md) in the manual -> [x](../manuals/other.md) in the notebook."""
    return LINK_RE.sub(lambda m: f"](../manuals/{m.group(1)}{m.group(2) or ''})", text)


def github_anchor(heading):
    """The anchor GitHub generates for a markdown heading (lowercase, punctuation dropped)."""
    h = heading.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def section(text, name):
    """The body of '## name' inside one lab, up to the next '## ' heading."""
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


cells = []


def add_md(cell_id, text):
    text = text.strip()
    if text:
        cells.append(new_markdown_cell(fix_links(text), id=cell_id))


def add_code(cell_id, code):
    cells.append(new_code_cell(code.rstrip("\n"), id=cell_id))


# ---------------------------------------------------------------- intro cell
title = md.splitlines()[0].lstrip("# ").strip()
add_md("intro", f"""# Transformer Inspection Lab: the notebook

This notebook holds the runnable code for every lab in **[{title}]({MANUAL_REL})**. Keep the manual open beside it: the manual has the concepts, the glossary (formal definitions of every term), the explanation of each output, and the answer key. The [HTML version of the manual](../manuals/{src_path.stem}.html) shows glossary definitions when you hover over a term.

**How to use it**

- Run the cells **in order, top to bottom**. Later labs reuse variables from earlier ones (`tok`, `base`, `enc`, ...). If you restart the kernel, re-run from the top (the manual's variable table lists which lab defines what).
- Each lab starts with its objective and a link to the matching section of the manual. Read that section's *Concepts* first, run the cells, then compare your output with its *Reading the output*.
- Write your answers to the *Record* questions in the empty cell under each list.
- New to dot products and matrix multiplication? The companion [linear algebra study guide](../manuals/linear-algebra-study-guide.md) covers what the labs need.

*This notebook is generated from the manual by `tools/build_notebook.py`. In the repository, change the manual and rebuild rather than editing the notebook.*""")

# ---------------------------------------------------------------- one block per lab
for m in re.finditer(r"^# (Lab (\d+):[^\n]*)\n", md, re.M):
    heading, n = m.group(1).strip(), int(m.group(2))
    end = md.index("\n---\n", m.end())
    lab = md[m.end():end]
    pid = f"lab{n:02d}"

    head = [f"# {heading}"]
    pre_concepts = lab.split("\n## ", 1)[0]          # notice + objective live above the first ## heading
    for para in re.split(r"\n\s*\n", pre_concepts.strip()):
        if para.startswith(("**Objective", "> **Notice")):
            head.append(para.strip())
    head.append(f"Concepts and explanations: [{heading.split(':')[0]} in the manual]({MANUAL_REL}#{github_anchor(heading)}).")
    add_md(f"{pid}-head", "\n\n".join(head))

    proc, last, k = section(lab, "Procedure"), 0, 0
    for f in FENCE_RE.finditer(proc):
        add_md(f"{pid}-t{k}", proc[last:f.start()])
        lang, body = f.group(1), f.group(2)
        if lang == "python":
            add_code(f"{pid}-c{k}", body)
        else:
            add_md(f"{pid}-m{k}", f"```{lang}\n{body}```")
        last, k = f.end(), k + 1
    add_md(f"{pid}-t{k}", proc[last:])

    record = section(lab, "Record")
    if record:
        add_md(f"{pid}-record", "### Record\n\n" + record)
        cells.append(new_markdown_cell("*Your answers:*\n\n", id=f"{pid}-answers"))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbformat.validate(nb)
out_path.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(out_path))
n_code = sum(c.cell_type == "code" for c in cells)
print(f"wrote {out_path}: {len(cells)} cells ({n_code} code), {out_path.stat().st_size // 1024} KiB")
