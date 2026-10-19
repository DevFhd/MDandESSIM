
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "orbital_level_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data extracted from orbital_energies.csv
data = [
    # molecule, torsion, angle_deg, HOMO_eV, LUMO_eV
    ("DR3", "A", 0, -9.2403, -9.2034),
    ("DR3", "A", 90, -9.1415, -9.1133),
    ("DR3", "B", 0, -9.0214, -8.9898),
    ("DR3", "B", 90, -8.8532, -8.8320),
    ("DR3", "C", 0, -9.2593, -9.2229),
    ("DR3", "C", 90, -9.2299, -9.1928),
    ("DR3", "D", 0, -9.2557, -9.2199),
    ("DR3", "D", 90, -9.2416, -9.2051),
    ("OiDTBR", "E", 0, -10.8173, -10.7655),
    ("OiDTBR", "E", 90, -10.8189, -10.7715),
    ("OiDTBR", "F", 0, -10.7668, -10.7161),
    ("OiDTBR", "F", 90, -10.7761, -10.7206),
]

df = pd.DataFrame(
    data,
    columns=["molecule", "torsion", "angle_deg",
             "HOMO_eV", "LUMO_eV"]
)

# Absolute orbital energies
df["abs_HOMO_eV"] = df["HOMO_eV"].abs()
df["abs_LUMO_eV"] = df["LUMO_eV"].abs()



def plot_combined_orbitals():
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Different horizontal positions for each torsion
    positions = {
        ("DR3", "A"): 0,
        ("DR3", "B"): 1,
        ("DR3", "C"): 2,
        ("DR3", "D"): 3,
        ("OiDTBR", "E"): 4,
        ("OiDTBR", "F"): 5,
    }

    colors = {"HOMO": "tab:blue", "LUMO": "tab:orange"}
    offsets = {"HOMO": -0.10, "LUMO": 0.10}
    markers = {0: "o", 90: "s"}

    for _, row in df.iterrows():
        molecule = row["molecule"]
        torsion = row["torsion"]
        angle = row["angle_deg"]
        x = positions[(molecule, torsion)]

        for orbital, column in [
            ("HOMO", "abs_HOMO_eV"),
            ("LUMO", "abs_LUMO_eV"),
        ]:
            y = row[column]

            ax.scatter(
                x + offsets[orbital],
                y,
                color=colors[orbital],
                marker=markers[angle],
                s=75,
                zorder=3,
                label=f"{orbital}, {angle}°",
            )

            ax.annotate(
                f"{y:.3f}",
                (x + offsets[orbital], y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    # X-axis labels
    labels = ["DR3-A", "DR3-B", "DR3-C", "DR3-D",
              "OiDTBR-E", "OiDTBR-F"]
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)

    # Axis limits
    ax.set_ylim(8.5, 11.0)
    ax.set_ylabel("Absolute orbital energy (eV)")
    ax.set_xlabel("Molecule and dihedral")
    ax.set_title("HOMO and LUMO Energies at 0° and 90° Torsion")

    # Grid and legend
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors["HOMO"], markersize=8,
               label="HOMO"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors["LUMO"], markersize=8,
               label="LUMO"),
        Line2D([0], [0], marker="o", color="gray",
               linestyle="None", markersize=7, label="0°"),
        Line2D([0], [0], marker="s", color="gray",
               linestyle="None", markersize=7, label="90°"),
    ]
    ax.legend(handles=legend_items, ncol=4, loc="upper right")

    fig.tight_layout()

    path = OUTPUT_DIR / "combined_HOMO_LUMO.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {path}")


# Generate the combined plot
plot_combined_orbitals()

# Save the plotted data
csv_path = OUTPUT_DIR / "absolute_orbital_energies.csv"
df.to_csv(csv_path, index=False)
print(f"Plotted data saved to: {csv_path}")
