
#!/usr/bin/env python3
"""
Validated torsional analysis for LAMMPS trajectories.

Outputs:
  validated_dihedral_angles.csv
  per_dihedral_statistics.csv
  molecule_level_statistics.csv
  final_summary_with_CI.csv
  topology_audit.csv
  run_metadata.txt

Assumptions:
  - LAMMPS data file has Atoms and Dihedrals sections.
  - Atom records use the usual "id molecule-id ..." layout.
  - The trajectory has orthorhombic periodic boundaries.
  - The trajectory contains wrapped, scaled, or unwrapped coordinates.

Important:
  - Confidence intervals are bootstrapped across molecules.
  - They do not independently quantify temporal sampling uncertainty.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_TYPES = [51, 61, 67, 80, 167, 186]
METRICS = ["P15", "P30", "Mean_delta", "RMS"]


# ---------------------------------------------------------------------
# LAMMPS data-file parsing
# ---------------------------------------------------------------------

def is_section_header(line):
    """Identify a LAMMPS data-file section header."""
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if re.match(r"^\d+\s+", s):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_ /-]*(?:\s+#.*)?$", s))


def read_lammps_section(path, section_name):
    """Return section header and data lines for a named LAMMPS section."""
    lines = Path(path).read_text(errors="replace").splitlines()
    active = False
    header = None
    records = []

    for line in lines:
        s = line.strip()

        if not active:
            if re.match(
                rf"^{re.escape(section_name)}\s*(?:#.*)?$",
                s,
                re.IGNORECASE,
            ):
                active = True
                header = s
            continue

        if is_section_header(s):
            break

        if not s or s.startswith("#"):
            continue

        records.append(s.split("#", 1)[0].split())

    if header is None:
        raise ValueError(f"Section '{section_name}' not found in {path}")

    return header, records


def parse_topology(data_path, target_types):
    """
    Parse atom molecule IDs and dihedral quartets.

    For standard 'full' and 'molecular' atom styles, column 2 is
    the molecule ID. This is checked against the four atoms in each
    selected dihedral.
    """
    atom_header, atom_rows = read_lammps_section(data_path, "Atoms")
    _, dih_rows = read_lammps_section(data_path, "Dihedrals")

    atom_style = "unspecified"
    match = re.search(r"#\s*(\w+)", atom_header)
    if match:
        atom_style = match.group(1).lower()

    if atom_style in {"atomic", "charge"}:
        raise ValueError(
            f"Atoms header indicates style '{atom_style}', which does not "
            "provide the standard molecule-ID column. Supply a topology "
            "with molecule IDs or adapt parse_topology explicitly."
        )

    molecule_by_atom = {}
    for row in atom_rows:
        if len(row) < 3:
            continue
        try:
            atom_id = int(row[0])
            molecule_id = int(row[1])
        except ValueError:
            continue
        molecule_by_atom[atom_id] = molecule_id

    if not molecule_by_atom:
        raise ValueError("Could not parse atom IDs and molecule IDs.")

    records = []
    for row in dih_rows:
        if len(row) < 6:
            continue
        try:
            values = list(map(int, row[:6]))
        except ValueError:
            continue

        did, dtype, a1, a2, a3, a4 = values
        if dtype not in target_types:
            continue

        quartet = (a1, a2, a3, a4)
        molecule_ids = [molecule_by_atom.get(a) for a in quartet]

        if any(m is None for m in molecule_ids):
            molecule_id = None
            molecule_check = "missing_atom"
        elif len(set(molecule_ids)) != 1:
            molecule_id = None
            molecule_check = "atoms_cross_molecules"
        else:
            molecule_id = molecule_ids[0]
            molecule_check = "ok"

        # Canonical quartet catches duplicates in forward/reverse order.
        canonical = min(quartet, quartet[::-1])

        records.append({
            "dihedral_id": did,
            "type": dtype,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "molecule_id": molecule_id,
            "molecule_check": molecule_check,
            "canonical_quartet": "-".join(map(str, canonical)),
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No target dihedrals found in the topology.")

    return df, molecule_by_atom, atom_style


# ---------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------

def parse_dump_frames(dump_path):
    """Yield timestep, sorted atom IDs, coordinates, box and coordinate mode."""
    with open(dump_path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break
            timestep = int(timestep_line.strip())

            if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected dump format at NUMBER OF ATOMS.")
            n_atoms = int(f.readline().strip())

            box_header = f.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Could not find BOX BOUNDS.")

            header_tokens = box_header.split()[3:]
            if any(x in header_tokens for x in ("xy", "xz", "yz")):
                raise NotImplementedError(
                    "Triclinic box detected. This version deliberately "
                    "stops rather than applying an incorrect orthogonal "
                    "minimum-image correction."
                )

            bounds = []
            for _ in range(3):
                vals = f.readline().split()
                if len(vals) < 2:
                    raise ValueError("Malformed box bounds.")
                bounds.append((float(vals[0]), float(vals[1])))

            lower = np.array([b[0] for b in bounds], dtype=float)
            lengths = np.array([b[1] - b[0] for b in bounds], dtype=float)

            if np.any(lengths <= 0):
                raise ValueError("Invalid box lengths.")

            atom_header = f.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError("Could not find ATOMS header.")

            columns = atom_header.split()[2:]
            col = {name: i for i, name in enumerate(columns)}

            if "id" not in col:
                raise ValueError("Trajectory must contain atom IDs.")

            if all(k in col for k in ("xu", "yu", "zu")):
                coord_cols = ("xu", "yu", "zu")
                mode = "unwrapped"
            elif all(k in col for k in ("x", "y", "z")):
                coord_cols = ("x", "y", "z")
                mode = "wrapped"
            elif all(k in col for k in ("xs", "ys", "zs")):
                coord_cols = ("xs", "ys", "zs")
                mode = "scaled"
            else:
                raise ValueError(
                    "Need xu/yu/zu, x/y/z, or xs/ys/zs coordinates."
                )

            ids = np.empty(n_atoms, dtype=np.int64)
            xyz = np.empty((n_atoms, 3), dtype=float)

            for i in range(n_atoms):
                row = f.readline().split()
                if len(row) < len(columns):
                    raise ValueError("Incomplete atom record in trajectory.")
                ids[i] = int(row[col["id"]])
                xyz[i] = [float(row[col[k]]) for k in coord_cols]

            order = np.argsort(ids)
            ids = ids[order]
            xyz = xyz[order]

            if mode == "scaled":
                xyz = lower[None, :] + xyz * lengths[None, :]

            yield timestep, ids, xyz, lengths, mode


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def planar_deviation(angle_deg):
    """
    Distance to the nearest planar orientation (0 or +/-180 degrees).
    Returns values between 0 and 90 degrees.
    """
    wrapped = (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return np.minimum(np.abs(wrapped), 180.0 - np.abs(wrapped))


def torsion_angle_deg(points):
    """Signed dihedral angle for four sequential Cartesian points."""
    p0, p1, p2, p3 = points

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    norm_b1 = np.linalg.norm(b1)
    if norm_b1 < 1e-12:
        return np.nan

    b1u = b1 / norm_b1
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    if np.linalg.norm(v) < 1e-12 or np.linalg.norm(w) < 1e-12:
        return np.nan

    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def get_torsion(ids, xyz, quartet, box_lengths, mode):
    """Calculate one torsion; apply minimum-image correction if wrapped."""
    idx = np.searchsorted(ids, np.asarray(quartet, dtype=np.int64))

    if np.any(idx >= len(ids)):
        return np.nan
    if not np.array_equal(ids[idx], quartet):
        return np.nan

    points = xyz[idx].copy()

    if mode in ("wrapped", "scaled"):
        # Reconstruct the local chain using minimum-image bond vectors.
        local = np.empty_like(points)
        local[0] = points[0]

        for j in range(1, 4):
            bond = points[j] - points[j - 1]
            bond -= box_lengths * np.round(bond / box_lengths)
            local[j] = local[j - 1] + bond

        points = local

    return torsion_angle_deg(points)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def summarize_angles(angles):
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]

    if len(angles) == 0:
        return {
            "P15": np.nan,
            "P30": np.nan,
            "Mean_delta": np.nan,
            "RMS": np.nan,
            "n_frames_valid": 0,
        }

    delta = planar_deviation(angles)

    return {
        "P15": float(np.mean(delta <= 15.0)),
        "P30": float(np.mean(delta <= 30.0)),
        "Mean_delta": float(np.mean(delta)),
        "RMS": float(np.sqrt(np.mean(delta**2))),
        "n_frames_valid": int(len(delta)),
    }


def bootstrap_ci(values, n_boot=10000, seed=2026):
    """Percentile bootstrap CI across independent molecule-level values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, np.nan, np.nan, len(x)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boot_means[i] = np.mean(sample)

    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(low), float(high), len(x)


def make_summaries(per_dihedral, n_boot, seed):
    """
    First average torsions within each molecule and type.
    Then summarize across molecules, avoiding treating all torsions
    within one molecule as independent molecules.
    """
    valid = per_dihedral[
        per_dihedral["molecule_id"].notna()
    ].copy()

    molecule = (
        valid.groupby(["type", "molecule_id"], as_index=False)[METRICS]
        .mean()
    )

    molecule_path_columns = ["type", "molecule_id"] + METRICS
    molecule = molecule[molecule_path_columns]

    rows = []
    for dtype in TARGET_TYPES:
        sub_d = valid[valid["type"] == dtype]
        sub_m = molecule[molecule["type"] == dtype]

        row = {
            "Type": dtype,
            "N_dihedrals": int(sub_d["dihedral_id"].nunique()),
            "N_molecules": int(sub_m["molecule_id"].nunique()),
        }

        for metric in METRICS:
            # Report spread among dihedral-level estimates descriptively.
            vals_d = sub_d[metric].dropna().to_numpy(float)
            row[f"{metric}_dihedral_mean"] = (
                float(np.mean(vals_d)) if len(vals_d) else np.nan
            )
            row[f"{metric}_dihedral_median"] = (
                float(np.median(vals_d)) if len(vals_d) else np.nan
            )
            row[f"{metric}_dihedral_SD"] = (
                float(np.std(vals_d, ddof=1)) if len(vals_d) > 1 else np.nan
            )

            # Primary estimate: each molecule contributes one value per type.
            vals_m = sub_m[metric].dropna().to_numpy(float)
            mean, low, high, n = bootstrap_ci(
                vals_m,
                n_boot=n_boot,
                seed=seed + dtype + len(metric),
            )
            row[f"{metric}_molecule_mean"] = mean
            row[f"{metric}_molecule_SD"] = (
                float(np.std(vals_m, ddof=1)) if len(vals_m) > 1 else np.nan
            )
            row[f"{metric}_CI95_low"] = low
            row[f"{metric}_CI95_high"] = high
            row[f"{metric}_N_molecules"] = n

        rows.append(row)

    return molecule, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validated molecular torsion analysis from LAMMPS files."
    )
    parser.add_argument("--topology", default="system_150.data")
    parser.add_argument(
        "--trajectory",
        default="traj_300K_first20ns.lammpstrj",
    )
    parser.add_argument("--output-dir", default="validated_torsion_output")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    topology_path = Path(args.topology)
    trajectory_path = Path(args.trajectory)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not topology_path.exists():
        sys.exit(f"ERROR: topology file not found: {topology_path}")
    if not trajectory_path.exists():
        sys.exit(f"ERROR: trajectory file not found: {trajectory_path}")
    if args.bootstrap < 100:
        sys.exit("ERROR: use at least 100 bootstrap replicates.")

    topology, molecule_by_atom, atom_style = parse_topology(
        topology_path, TARGET_TYPES
    )

    # Audit duplicate/reversed quartets within the same torsion type.
    topology["duplicate_quartet"] = topology.duplicated(
        subset=["type", "canonical_quartet"], keep=False
    )

    topology.to_csv(out_dir / "topology_audit.csv", index=False)

    print("Topology audit")
    print("--------------")
    print(f"Atoms with molecule IDs: {len(molecule_by_atom)}")
    print(f"Atoms header: {atom_style}")
    print(f"Selected dihedral records: {len(topology)}")

    for dtype in TARGET_TYPES:
        sub = topology[topology["type"] == dtype]
        n_mol = sub["molecule_id"].nunique(dropna=True)
        n_bad = int((sub["molecule_check"] != "ok").sum())
        n_dup = int(sub["duplicate_quartet"].sum())
        print(
            f"Type {dtype}: records={len(sub)}, molecules={n_mol}, "
            f"invalid molecule assignments={n_bad}, "
            f"records in duplicate quartets={n_dup}"
        )

    if (topology["molecule_check"] != "ok").any():
        bad = topology[topology["molecule_check"] != "ok"]
        bad.to_csv(out_dir / "invalid_dihedrals.csv", index=False)
        raise ValueError(
            "Some selected dihedrals have missing atoms or cross molecule "
            "boundaries. Inspect invalid_dihedrals.csv before proceeding."
        )

    # Do not silently remove duplicate records: stop for review.
    if topology["duplicate_quartet"].any():
        raise ValueError(
            "Duplicate/reversed atom quartets detected. Inspect "
            "topology_audit.csv and resolve whether they are true duplicates "
            "before computing final statistics."
        )

    records = topology.to_dict("records")
    series = {int(r["dihedral_id"]): [] for r in records}

    quartet_by_id = {
        int(r["dihedral_id"]): (
            int(r["a1"]), int(r["a2"]), int(r["a3"]), int(r["a4"])
        )
        for r in records
    }
    type_by_id = {
        int(r["dihedral_id"]): int(r["type"]) for r in records
    }
    molecule_by_dih = {
        int(r["dihedral_id"]): int(r["molecule_id"]) for r in records
    }

    angle_rows = []
    n_frames = 0
    first_step = None
    last_step = None
    modes = set()
    expected_ids = None

    for step, atom_ids, xyz, box_lengths, mode in parse_dump_frames(
        trajectory_path
    ):
        n_frames += 1
        first_step = step if first_step is None else first_step
        last_step = step
        modes.add(mode)

        if expected_ids is None:
            expected_ids = atom_ids
        elif not np.array_equal(expected_ids, atom_ids):
            raise ValueError("Atom IDs differ between trajectory frames.")

        for r in records:
            did = int(r["dihedral_id"])
            angle = get_torsion(
                atom_ids,
                xyz,
                quartet_by_id[did],
                box_lengths,
                mode,
            )
            series[did].append(angle)
            angle_rows.append({
                "timestep": step,
                "dihedral_id": did,
                "type": type_by_id[did],
                "molecule_id": molecule_by_dih[did],
                "angle_deg": angle,
                "planar_deviation_deg": (
                    float(planar_deviation(angle))
                    if np.isfinite(angle) else np.nan
                ),
            })

    if n_frames == 0:
        raise ValueError("No trajectory frames were parsed.")

    angles_df = pd.DataFrame(angle_rows)
    angles_df.to_csv(
        out_dir / "validated_dihedral_angles.csv", index=False
    )

    per_rows = []
    for did, angles in series.items():
        stats = summarize_angles(angles)
        per_rows.append({
            "dihedral_id": did,
            "type": type_by_id[did],
            "molecule_id": molecule_by_dih[did],
            **stats,
        })

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(out_dir / "per_dihedral_statistics.csv", index=False)

    molecule_df, final_summary = make_summaries(
        per_df,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    molecule_df.to_csv(
        out_dir / "molecule_level_statistics.csv", index=False
    )
    final_summary.to_csv(
        out_dir / "final_summary_with_CI.csv", index=False
    )

    metadata = [
        f"Topology: {topology_path.resolve()}",
        f"Trajectory: {trajectory_path.resolve()}",
        f"Frames parsed: {n_frames}",
        f"First timestep: {first_step}",
        f"Last timestep: {last_step}",
        f"Coordinate modes: {', '.join(sorted(modes))}",
        f"Selected types: {TARGET_TYPES}",
        f"Bootstrap replicates: {args.bootstrap}",
        f"Bootstrap seed: {args.seed}",
        "CI method: percentile bootstrap over molecule-level means",
        "CI limitation: temporal autocorrelation is not independently modeled",
        "Duplicate quartet check: passed",
        "Molecule membership check: passed",
    ]
    (out_dir / "run_metadata.txt").write_text(
        "\n".join(metadata) + "\n",
        encoding="utf-8",
    )

    print("\nTrajectory")
    print("----------")
    print(f"Frames: {n_frames}")
    print(f"First timestep: {first_step}")
    print(f"Last timestep: {last_step}")
    print(f"Coordinate mode(s): {', '.join(sorted(modes))}")

    print("\nFinal summary")
    print("-------------")
    print(final_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()