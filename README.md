# Transformer Inspection Lab and Linear Algebra Study Guide

Two self-study guides for understanding how transformer models work, by inspecting real models layer by layer.

| Guide | What it covers | Read it |
|---|---|---|
| **Transformer Inspection Lab** | A hands-on lab manual (Labs 0 to 12) using `microsoft/deberta-v3-base` (an encoder) and `Qwen/Qwen2.5-0.5B` (a decoder): tokens, embeddings, hidden states, attention heads, hooks, classification heads, LoRA, and embedding models for retrieval. Includes a glossary with hover tooltips. | [HTML](manuals/model-inspection-lab-manual-detailed.html) / [Markdown](manuals/model-inspection-lab-manual-detailed.md) |
| **Linear Algebra for AI Models** | An interactive study guide for the linear algebra the lab relies on: vectors, dot products, matrices, matrix multiplication, transformations, a neural-network layer, attention, backpropagation and LoRA. | [HTML](manuals/linear-algebra-study-guide.html) / [Markdown](manuals/linear-algebra-study-guide.md) |

Lab 7 links to the study guide. If the matrix-multiplication steps feel unfamiliar, read the guide first.

**Runnable notebook:** [notebooks/transformer-inspection-lab.ipynb](notebooks/transformer-inspection-lab.ipynb) has the code for every lab, in order, with the Record questions and space for your answers. Use it alongside the manual, which has the explanations and the answer key.

## Reading

Open an `.html` file in any browser. Each is a single self-contained page and needs no internet connection or server. Hover over (or tab to) any dotted-underlined term to see its definition.

## Layout

```
manuals/     the guides: markdown sources (the source of truth) and their built HTML pages
notebooks/   the lab notebook, generated from the lab manual
tools/       the scripts and page templates that build the HTML pages and the notebook
```

## Rebuilding

The HTML pages and the notebook are generated. To change a guide, edit its markdown file, then rebuild:

```bash
uv run tools/build_manual.py          # the lab manual's HTML page
uv run tools/build_study_guide.py     # the linear algebra study guide's HTML page
uv run tools/build_notebook.py        # the lab notebook (from the manual's Procedure sections)
tools/build_all.sh                    # all three
```

Each script lists its own dependencies, so [uv](https://docs.astral.sh/uv/) installs what it needs on first run.

Page styling and behaviour live in `tools/manual-template.html` and `tools/study-guide-template.html`. Diagrams are drawn by functions inside the build scripts. Each one replaces a marked block in the markdown (`<!-- figure:NAME:start -->` ... `<!-- figure:NAME:end -->`), and the plain-text sketch left inside the block is what markdown readers see.

## Running the lab

The labs run in Jupyter with PyTorch, `transformers` and `peft`; Lab 0 of the manual gives the setup commands. Open the notebook in `notebooks/` and run it top to bottom, with the manual beside it.
