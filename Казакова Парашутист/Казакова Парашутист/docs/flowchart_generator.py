"""Generates a simple flowchart image (block diagram) for the algorithm.

We draw a standard block-scheme using matplotlib patches.
Output: outputs/flowchart_algorithm.png
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


def box(ax, xy, w, h, text, fontsize=10):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def diamond(ax, center, w, h, text, fontsize=10):
    cx, cy = center
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    poly = Polygon(pts, closed=True, linewidth=1.5, facecolor="white")
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, fontsize=9):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text, ha="center", va="bottom", fontsize=fontsize)


def main(out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(9, 12))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Blocks (top to bottom)
    box(ax, (0.18, 0.92), 0.64, 0.05, "Початок")
    box(ax, (0.10, 0.84), 0.80, 0.07, "Ввести параметри: m, H, S, R, h0, t_open, dt, eps\nЗадати константи: g, rho, Cd1, Cd2, alpha")
    box(ax, (0.10, 0.75), 0.80, 0.07, "Обчислити площі: A1(body), A2=πR²\nОбчислити k1, k2 та v_term1, v_term2")
    box(ax, (0.10, 0.66), 0.80, 0.06, "Ініціалізація: t=0, v=v0, h=h0, phase=1")

    diamond(ax, (0.50, 0.56), 0.55, 0.10, "h > 0 ?")
    box(ax, (0.10, 0.45), 0.80, 0.08, "Визначити фазу:\nякщо t ≥ t_open → phase=2, k=k2\nінакше phase=1, k=k1")
    box(ax, (0.10, 0.35), 0.80, 0.08, "Крок інтегрування RK4:\nv ← v + Δv,  h ← h + Δh\n( dv/dt=g-kv², dh/dt=-v )")
    box(ax, (0.10, 0.26), 0.80, 0.06, "Зберегти значення у масиви; t ← t + dt")
    diamond(ax, (0.50, 0.18), 0.70, 0.10, "Досягнуто сталої швидкості?\n|v - v_term|/v_term ≤ eps")
    box(ax, (0.10, 0.08), 0.80, 0.06, "Після циклу: знайти t_const1 і t_const2\nПобудувати графіки/таблиці")
    box(ax, (0.18, 0.01), 0.64, 0.05, "Кінець")

    # Arrows
    arrow(ax, 0.50, 0.92, 0.50, 0.91)
    arrow(ax, 0.50, 0.84, 0.50, 0.82)
    arrow(ax, 0.50, 0.75, 0.50, 0.73)
    arrow(ax, 0.50, 0.66, 0.50, 0.61)

    # Loop arrows around h>0
    arrow(ax, 0.50, 0.51, 0.50, 0.49, text="так")
    arrow(ax, 0.50, 0.45, 0.50, 0.43)
    arrow(ax, 0.50, 0.35, 0.50, 0.33)
    arrow(ax, 0.50, 0.26, 0.50, 0.23)

    # From constant check back to h>0 diamond (continue)
    arrow(ax, 0.85, 0.18, 0.85, 0.56, text="ні")
    arrow(ax, 0.85, 0.56, 0.78, 0.56)

    # From h>0 decision "ні" to post-processing
    arrow(ax, 0.22, 0.56, 0.22, 0.08, text="ні")
    arrow(ax, 0.22, 0.08, 0.50, 0.08)
    arrow(ax, 0.50, 0.08, 0.50, 0.06)
    arrow(ax, 0.50, 0.06, 0.50, 0.01)

    # From constant check "так" to post-processing
    arrow(ax, 0.50, 0.13, 0.50, 0.08, text="так")

    plt.tight_layout()
    outpath = os.path.join(out_dir, "flowchart_algorithm.png")
    plt.savefig(outpath, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
