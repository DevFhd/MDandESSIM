
from pathlib import Path
import re
import csv

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "orbital_energies.csv"

HARTREE_TO_EV = 27.211386245988

NUMBER = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[EeDd][-+]?\d+)?"
)

# Match orbital rows, including explicit (HOMO) and (LUMO) labels.
ROW_PATTERN = re.compile(
    rf"^\s*(\d+)\s+"
    rf"({NUMBER})\s+"
    rf"({NUMBER})\s+"
    rf"({NUMBER})"
    rf"\s*(?:\((HOMO|LUMO)\))?",
    re.IGNORECASE,
)


def to_float(value):
    return float(value.replace("D", "E").replace("d", "e"))


def parse_orbitals(output_text):
    """
    Extract explicitly labelled HOMO and LUMO levels.
    Fractional occupations are allowed.
    """

    homo = None
    lumo = None

    for line in output_text.splitlines():
        match = ROW_PATTERN.match(line)

        if not match:
            continue

        label = (match.group(5) or "").upper()

        if label not in ("HOMO", "LUMO"):
            continue

        orbital = {
            "index": int(match.group(1)),
            "occupation": to_float(match.group(2)),
            "energy_eh": to_float(match.group(3)),
            "energy_ev_printed": to_float(match.group(4)),
        }

        if label == "HOMO":
            homo = orbital
        elif label == "LUMO":
            lumo = orbital

    if homo is None or lumo is None:
        return None

    if lumo["index"] <= homo["index"]:
        return None

    # Calculate the gap from Hartree energies to avoid
    # precision loss from the rounded eV column.
    gap_eh = lumo["energy_eh"] - homo["energy_eh"]
    gap_ev = gap_eh * HARTREE_TO_EV

    return {
        "HOMO_index": homo["index"],
        "HOMO_Eh": homo["energy_eh"],
        "HOMO_eV": homo["energy_ev_printed"],
        "LUMO_index": lumo["index"],
        "LUMO_Eh": lumo["energy_eh"],
        "LUMO_eV": lumo["energy_ev_printed"],
        "gap_from_levels_Eh": gap_eh,
        "gap_from_levels_eV": gap_ev,
        "HOMO_occupation": homo["occupation"],
        "LUMO_occupation": lumo["occupation"],
    }


def identify_file(path):
    match = re.fullmatch(
        r"([A-F])_(\d+)deg_xtb\.out",
        path.name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    torsion = match.group(1).upper()
    angle = int(match.group(2))

    parent_names = {p.name.lower() for p in path.parents}

    if "dr3" in parent_names:
        molecule = "DR3"
    elif "oidtbr" in parent_names:
        molecule = "OiDTBR"
    else:
        molecule = path.parent.name

    return molecule, torsion, angle


def main():
    files = sorted(ROOT.rglob("*_xtb.out"))
    results = []

    if not files:
        print(f"No *_xtb.out files found under: {ROOT}")
        return

    for path in files:
        identity = identify_file(path)

        if identity is None:
            print(f"SKIP: Unrecognized filename: {path}")
            continue

        molecule, torsion, angle = identity

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            parsed = parse_orbitals(text)
        except OSError as exc:
            print(f"READ ERROR: {path}: {exc}")
            parsed = None

        row = {
            "molecule": molecule,
            "torsion": torsion,
            "angle_deg": angle,
            "file": str(path.relative_to(ROOT)),
            "status": "OK" if parsed else "NOT_PARSED",
        }

        if parsed:
            row.update(parsed)

            print(
                f"OK: {molecule} {torsion} {angle}° | "
                f"HOMO={parsed['HOMO_eV']:.4f} eV | "
                f"LUMO={parsed['LUMO_eV']:.4f} eV | "
                f"gap={parsed['gap_from_levels_eV']:.6f} eV"
            )
        else:
            print(f"NOT PARSED: {path}")

        results.append(row)

    columns = [
        "molecule", "torsion", "angle_deg", "file", "status",
        "HOMO_index", "HOMO_Eh", "HOMO_eV",
        "LUMO_index", "LUMO_Eh", "LUMO_eV",
        "gap_from_levels_Eh", "gap_from_levels_eV",
        "HOMO_occupation", "LUMO_occupation",
    ]

    with OUTPUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    successful = sum(r["status"] == "OK" for r in results)

    print(f"\nSaved: {OUTPUT}")
    print(f"Files processed: {len(results)}")
    print(f"Successfully parsed: {successful}")

    if successful != len(results):
        print("Check the remaining NOT_PARSED files individually.")


if __name__ == "__main__":
    main()