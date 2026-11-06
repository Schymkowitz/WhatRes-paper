"""Shared figure style for the WhatRes paper figures.
Importing this configures matplotlib for EDITABLE VECTOR output (text stays as text in
Illustrator): PDF/PS fonttype 42 (TrueType), SVG fonttype 'none'. Use savepanel() to write
each panel as both .pdf and .svg next to the calling figure's folder.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401 (re-exported for convenience)

mpl.rcParams.update({
    "pdf.fonttype": 42,      # editable text in Illustrator (not outlines)
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # SVG keeps text as text
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

def savepanel(fig, outdir, name):
    """Write fig as editable vector PDF + SVG (and a PNG preview) into outdir/."""
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")  # quick preview only
    return base

def panel_letters(fig, axes, letters="abcdefgh", size=10):
    """Bold lowercase panel letters at the top-left of each axes' full extent (labels and title
    included). Call after tight_layout(), immediately before savepanel()."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, l in zip(axes, letters):
        bb = ax.get_tightbbox(r); x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, l, fontsize=size, fontweight="bold", ha="left", va="bottom")
