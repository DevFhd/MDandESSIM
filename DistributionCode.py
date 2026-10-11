import math
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# RELATIVE POSTER SCALING (Huge Text on Compact Canvas)
# ============================================================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,      # Extra large relative title
    "axes.labelsize": 16,      # Extra large relative labels
    "xtick.labelsize": 12,     # Extra large relative ticks
    "ytick.labelsize": 12,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

DIHEDRAL_TYPES = [167, 186, 51, 61, 67, 80]
SORTED_TYPES = sorted(DIHEDRAL_TYPES)

LETTERS = ["A", "B", "C", "D", "E", "F"]
TYPE_TO_LETTER = dict(zip(SORTED_TYPES, LETTERS))

OUTPUT_DIR = "dihedral_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_COLS = 3
N_PLOTS = len(SORTED_TYPES)
N_ROWS = math.ceil(N_PLOTS / N_COLS)

# Fixed, smaller canvas size forces text to visually dominate
fig, axes = plt.subplots(
    N_ROWS, N_COLS, figsize=(14, 8), dpi=300
)

axes = np.atleast_1d(axes).flatten()

for idx, dtype in enumerate(SORTED_TYPES):
    ax = axes[idx]
    letter_label = TYPE_TO_LETTER[dtype]
    data_filename = f"dihedral_type_{dtype}.dat"

    if not os.path.exists(data_filename):
        print(f"Skipping Type {letter_label} ({dtype}): File '{data_filename}' not found.")
        ax.text(
            0.5, 0.5, f"File not found:\n{data_filename}",
            ha="center", va="center", transform=ax.transAxes, fontsize=16
        )
        continue

    centers, pdf = np.loadtxt(data_filename, unpack=True)

    # Thicker curves without line label for legend
    ax.plot(
        centers, pdf, color="#1f77b4", linewidth=3.0
    )
    ax.fill_between(centers, pdf, color="#1f77b4", alpha=0.30)

    # Thicker ticks and spines
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(width=2.0, length=6)

    # Labeling & Fixed Axis Limits
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 0.012)  # Unified maximum y-axis value across all plots
    ax.set_xlabel(r"Dihedral Angle $\phi$ (deg)")
    ax.set_ylabel(r"Probability Density $P(\phi)$")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1.2)
    ax.set_title(f"Type {letter_label}", pad=10)

for idx in range(N_PLOTS, len(axes)):
    fig.delaxes(axes[idx])

# Keeps elements inside the image bounds without shrinking fonts
plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=2.0)

combined_output_file = os.path.join(
    OUTPUT_DIR, "combined_dihedral_plots_large_text.png"
)
plt.savefig(combined_output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved optimized poster plot: {combined_output_file}")
