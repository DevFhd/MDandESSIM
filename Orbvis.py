
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


def plot_orbital(molecule, orbital_column, orbital_label, filename):
    subset = df[df["molecule"] == molecule]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for torsion in sorted(subset["torsion"].unique()):
        group = subset[subset["torsion"] == torsion].sort_values(
            "angle_deg"
        )

        ax.plot(
            group["angle_deg"],
            group[orbital_column],
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"Dihedral {torsion}",
        )

        # Annotate each data point
        for _, row in group.iterrows():
            ax.annotate(
                f"{row[orbital_column]:.4f}",
                (row["angle_deg"], row[orbital_column]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Torsion angle (degrees)")
    ax.set_ylabel(f"|{orbital_label}| (eV)")
    ax.set_title(
        f"{molecule}: Absolute {orbital_label} Energy vs Torsion"
    )
    ax.set_xticks(sorted(subset["angle_deg"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dihedral")
    fig.tight_layout()

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {path}")


# Plot 1: DR3 HOMO
plot_orbital(
    molecule="DR3",
    orbital_column="abs_HOMO_eV",
    orbital_label="HOMO",
    filename="DR3_absolute_HOMO.png",
)

# Plot 2: OiDTBR LUMO
plot_orbital(
    molecule="OiDTBR",
    orbital_column="abs_LUMO_eV",
    orbital_label="LUMO",
    filename="OiDTBR_absolute_LUMO.png",
)

# Save the plotted data
csv_path = OUTPUT_DIR / "absolute_orbital_energies.csv"
df.to_csv(csv_path, index=False)

print(f"\nPlotted data saved to: {csv_path}")