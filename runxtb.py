from pathlib import Path
import subprocess
import re
import csv
import sys


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(r"C:\Users\fahad\OneDrive\Desktop\simulations\xTB_torsion_scan")

# xTB executable
XTB = "xtb"

# Molecules / torsions
MOLECULES = {
    "DR3": ["A", "B", "C", "D"],
    "OiDTBR": ["E", "F"],
}

ANGLES = [0, 90]

# xTB settings
XTB_ARGS = [
    "--gfn", "2",
    "--sp",
    "--molden",
]

# Output CSV
CSV_FILE = ROOT / "xtb_results.csv"


# ============================================================
# PARSING
# ============================================================

def parse_xtb_output(text):
    """
    Extract important quantities directly from xTB stdout.
    """

    result = {
        "total_energy_Eh": None,
        "homo_lumo_gap_eV": None,
        "gradient_norm_Eh_A": None,
        "normal_termination": False,
    }

    # --------------------------------------------------------
    # TOTAL ENERGY
    # Example:
    # TOTAL ENERGY       -215.982301147564 Eh
    # --------------------------------------------------------

    m = re.search(
        r"TOTAL ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*Eh",
        text,
        re.IGNORECASE
    )

    if m:
        result["total_energy_Eh"] = float(m.group(1))

    # --------------------------------------------------------
    # HOMO-LUMO GAP
    # Example:
    # HOMO-LUMO GAP
    # 0.036839219456 eV
    #
    # xTB output can vary slightly, so search more flexibly.
    # --------------------------------------------------------

    m = re.search(
        r"HOMO-LUMO GAP\s*[:=]?\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*eV",
        text,
        re.IGNORECASE
    )

    if not m:
        # Try the common multiline format
        m = re.search(
            r"HOMO-LUMO GAP.*?\n\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*eV",
            text,
            re.IGNORECASE
        )

    if m:
        result["homo_lumo_gap_eV"] = float(m.group(1))

    # --------------------------------------------------------
    # GRADIENT NORM
    # --------------------------------------------------------

    m = re.search(
        r"GRADIENT NORM\s*[:=]?\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*Eh",
        text,
        re.IGNORECASE
    )

    if m:
        result["gradient_norm_Eh_A"] = float(m.group(1))

    # --------------------------------------------------------
    # NORMAL TERMINATION
    # --------------------------------------------------------

    if re.search(r"normal termination of xtb", text, re.IGNORECASE):
        result["normal_termination"] = True

    return result


# ============================================================
# RUN ONE CALCULATION
# ============================================================

def run_xtb(xyz_file):
    """
    Run xTB in the directory containing xyz_file.
    """

    workdir = xyz_file.parent

    command = [
        XTB,
        xyz_file.name,
        *XTB_ARGS,
    ]

    print()
    print("=" * 70)
    print(f"Running: {xyz_file}")
    print("=" * 70)
    print("Command:")
    print(" ".join(command))

    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        print()
        print("ERROR: xTB was not found.")
        print("Make sure 'xtb' works in PowerShell.")
        sys.exit(1)

    stdout = completed.stdout
    stderr = completed.stderr

    # Save raw output for reproducibility
    out_file = workdir / f"{xyz_file.stem}_xtb.out"
    out_file.write_text(stdout + "\n\n--- STDERR ---\n" + stderr,
                         encoding="utf-8")

    result = parse_xtb_output(stdout)

    result["return_code"] = completed.returncode

    if completed.returncode != 0:
        result["status"] = "FAILED"
    elif not result["normal_termination"]:
        result["status"] = "WARNING"
    else:
        result["status"] = "OK"

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    for molecule, torsions in MOLECULES.items():

        molecule_dir = ROOT / molecule

        if not molecule_dir.exists():
            print(f"WARNING: directory not found: {molecule_dir}")
            continue

        for torsion in torsions:

            for angle in ANGLES:

                xyz = molecule_dir / f"{torsion}_{angle:03d}deg.xyz"

                if not xyz.exists():
                    print(f"WARNING: missing file: {xyz}")
                    continue

                result = run_xtb(xyz)

                row = {
                    "molecule": molecule,
                    "torsion": torsion,
                    "angle_deg": angle,
                    "file": xyz.name,
                    "status": result["status"],
                    "total_energy_Eh": result["total_energy_Eh"],
                    "homo_lumo_gap_eV": result["homo_lumo_gap_eV"],
                    "gradient_norm_Eh_A": result["gradient_norm_Eh_A"],
                    "return_code": result["return_code"],
                }

                all_results.append(row)

                print()
                print(f"Status       : {result['status']}")
                print(f"Energy       : {result['total_energy_Eh']} Eh")
                print(f"HOMO-LUMO gap: {result['homo_lumo_gap_eV']} eV")
                print(f"Gradient norm: {result['gradient_norm_Eh_A']} Eh")
                print()

    # ========================================================
    # WRITE CSV
    # ========================================================

    if not all_results:
        print("No calculations were completed.")
        return

    fieldnames = [
        "molecule",
        "torsion",
        "angle_deg",
        "file",
        "status",
        "total_energy_Eh",
        "homo_lumo_gap_eV",
        "gradient_norm_Eh_A",
        "return_code",
    ]

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("#" * 80)
    print("FINAL xTB SUMMARY")
    print("#" * 80)

    print(
        f"{'Molecule':<10}"
        f"{'Torsion':<10}"
        f"{'Angle':<10}"
        f"{'Gap (eV)':<15}"
        f"{'Energy (Eh)':<20}"
        f"{'Status':<10}"
    )

    print("-" * 80)

    for r in all_results:

        gap = (
            f"{r['homo_lumo_gap_eV']:.9f}"
            if r["homo_lumo_gap_eV"] is not None
            else "N/A"
        )

        energy = (
            f"{r['total_energy_Eh']:.9f}"
            if r["total_energy_Eh"] is not None
            else "N/A"
        )

        print(
            f"{r['molecule']:<10}"
            f"{r['torsion']:<10}"
            f"{r['angle_deg']:<10}"
            f"{gap:<15}"
            f"{energy:<20}"
            f"{r['status']:<10}"
        )

    print("-" * 80)
    print(f"\nResults saved to:")
    print(CSV_FILE)


if __name__ == "__main__":
    main()