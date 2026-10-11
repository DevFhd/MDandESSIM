from pathlib import Path
import subprocess
import shutil
import sys

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(r"C:\Users\fahad\OneDrive\Desktop\simulations\xTB_torsion_scan")

# xTB executable
XTB = shutil.which("xtb")

if XTB is None:
    # Common fallback locations
    candidates = [
        Path(r"C:\Users\fahad\AppData\Local\Programs\xtb\xtb.exe"),
        Path(r"C:\Users\fahad\bin\xtb.exe"),
    ]

    for candidate in candidates:
        if candidate.exists():
            XTB = str(candidate)
            break

if XTB is None:
    print("ERROR: Could not find xtb.exe on PATH.")
    print("Run: where.exe xtb")
    sys.exit(1)


# ============================================================
# CASES
# ============================================================

CASES = [
    ("DR3",    "A", "A_000deg.xyz"),
    ("DR3",    "A", "A_090deg.xyz"),

    ("DR3",    "B", "B_000deg.xyz"),
    ("DR3",    "B", "B_090deg.xyz"),

    ("DR3",    "C", "C_000deg.xyz"),
    ("DR3",    "C", "C_090deg.xyz"),

    ("DR3",    "D", "D_000deg.xyz"),
    ("DR3",    "D", "D_090deg.xyz"),

    ("OiDTBR", "E", "E_000deg.xyz"),
    ("OiDTBR", "E", "E_090deg.xyz"),

    ("OiDTBR", "F", "F_000deg.xyz"),
    ("OiDTBR", "F", "F_090deg.xyz"),
]


# ============================================================
# FIND INPUT FILE
# ============================================================

def find_xyz(molecule, filename):
    """
    Search recursively under the molecule directory.
    """
    mol_dir = ROOT / molecule

    matches = list(mol_dir.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {mol_dir}"
        )

    if len(matches) > 1:
        print(f"WARNING: Multiple matches for {filename}")
        for m in matches:
            print("  ", m)

    return matches[0]


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(molecule, label, filename):

    source = find_xyz(molecule, filename)

    angle = "000" if "000" in filename else "090"

    outdir = ROOT / "results" / molecule / f"{label}_{angle}deg"
    outdir.mkdir(parents=True, exist_ok=True)

    # Copy XYZ into its own directory.
    xyz_dest = outdir / filename
    shutil.copy2(source, xyz_dest)

    print()
    print("=" * 72)
    print(f"{molecule} / {label} / {angle}°")
    print("=" * 72)
    print(f"Input : {source}")
    print(f"Output: {outdir}")

    # --------------------------------------------------------
    # Run xTB
    # --------------------------------------------------------

    cmd = [
        XTB,
        filename,
        "--gfn",
        "2",
        "--sp",
        "--norestart",
        "--json",
        "--molden",
    ]

    print()
    print("Running:")
    print(" ".join(cmd))
    print()

    with open(outdir / "xtb_stdout.txt", "w", encoding="utf-8") as fout:
        with open(outdir / "xtb_stderr.txt", "w", encoding="utf-8") as ferr:

            result = subprocess.run(
                cmd,
                cwd=outdir,
                stdout=fout,
                stderr=ferr,
                text=True,
            )

    if result.returncode != 0:
        print(f"ERROR: xTB failed for {molecule} {label} {angle}°")
        print(f"Return code: {result.returncode}")
        return False

    # --------------------------------------------------------
    # Validate expected outputs
    # --------------------------------------------------------

    expected = [
        outdir / "molden.input",
        outdir / "xtbout.json",
    ]

    missing = [str(p) for p in expected if not p.exists()]

    if missing:
        print("ERROR: xTB returned successfully but files are missing:")
        for p in missing:
            print("  ", p)
        return False

    print("SUCCESS")
    print("  molden.input:", (outdir / "molden.input").stat().st_size, "bytes")
    print("  xtbout.json :", (outdir / "xtbout.json").stat().st_size, "bytes")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("xTB 12-CASE SINGLE-POINT + MOLDEN GENERATION")
    print("=" * 72)
    print()
    print("xTB:", XTB)
    print("ROOT:", ROOT)
    print()

    results = []

    for molecule, label, filename in CASES:
        try:
            success = run_case(molecule, label, filename)
        except Exception as exc:
            print()
            print("ERROR:")
            print(exc)
            success = False

        results.append(
            (molecule, label, filename, success)
        )

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    for molecule, label, filename, success in results:
        status = "OK" if success else "FAILED"
        angle = "0°" if "000" in filename else "90°"
        print(f"{status:8s} {molecule:8s} {label} {angle}")

    failed = [x for x in results if not x[3]]

    print()
    if failed:
        print(f"WARNING: {len(failed)} case(s) failed.")
    else:
        print("ALL 12 CASES COMPLETED SUCCESSFULLY.")

    print()
    print("Results directory:")
    print(ROOT / "results")


if __name__ == "__main__":
    main()