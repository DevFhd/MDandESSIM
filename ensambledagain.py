import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_TYPES = [51, 61, 67, 80, 167, 186]


def parse_lammps_dihedrals(data_path):
    """Read the LAMMPS 'Dihedrals' section: id type atom1 atom2 atom3 atom4."""
    lines = Path(data_path).read_text(errors="replace").splitlines()
    section = None
    records = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^Dihedrals\s*(#.*)?$", stripped, re.I):
            section = "dihedrals"
            continue
        if section == "dihedrals":
            # A new LAMMPS section header ends the current section.
            if re.match(r"^[A-Za-z][A-Za-z0-9_ /-]*(?:\s+#.*)?$", stripped):
                if not re.match(r"^\d+\s+", stripped):
                    break
            parts = stripped.split("#", 1)[0].split()
            if len(parts) >= 6 and all(re.fullmatch(r"-?\d+", x) for x in parts[:6]):
                did, dtype, a1, a2, a3, a4 = map(int, parts[:6])
                records.append((did, dtype, a1, a2, a3, a4))
    if not records:
        raise ValueError(
            f"No dihedral records found in {data_path}. "
            "Check that it is a LAMMPS data file with a 'Dihedrals' section."
        )
    df = pd.DataFrame(records, columns=["dihedral_id", "type", "a1", "a2", "a3", "a4"])
    return df


def parse_dump_frames(dump_path):
    """Yield (timestep, atom_ids, positions_by_id, box_lengths, coord_mode)."""
    with open(dump_path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step_line = f.readline()
            if not step_line:
                break
            timestep = int(step_line.strip())

            if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected LAMMPS dump format after TIMESTEP.")
            n_atoms = int(f.readline().strip())

            box_header = f.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Could not find ITEM: BOX BOUNDS.")
            if any(k in box_header.split()[3:] for k in ("xy", "xz", "yz")):
                raise NotImplementedError(
                    "This script currently supports orthorhombic boxes only; "
                    "the dump appears to use a triclinic box."
                )
            bounds = []
            for _ in range(3):
                vals = f.readline().split()
                if len(vals) < 2:
                    raise ValueError("Malformed box bounds.")
                bounds.append((float(vals[0]), float(vals[1])))
            box_lengths = np.array([hi - lo for lo, hi in bounds], dtype=float)

            atom_header = f.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError("Could not find ITEM: ATOMS.")
            columns = atom_header.split()[2:]
            col_index = {name: i for i, name in enumerate(columns)}

            if "id" not in col_index:
                raise ValueError("Dump ATOMS section must contain an 'id' column.")

            if all(k in col_index for k in ("xu", "yu", "zu")):
                coord_cols = ("xu", "yu", "zu")
                coord_mode = "unwrapped"
            elif all(k in col_index for k in ("x", "y", "z")):
                coord_cols = ("x", "y", "z")
                coord_mode = "wrapped"
            elif all(k in col_index for k in ("xs", "ys", "zs")):
                coord_cols = ("xs", "ys", "zs")
                coord_mode = "scaled"
            else:
                raise ValueError(
                    "No supported coordinates found. Need xu/yu/zu, x/y/z, or xs/ys/zs."
                )

            atom_ids = np.empty(n_atoms, dtype=np.int64)
            positions = np.empty((n_atoms, 3), dtype=float)
            for i in range(n_atoms):
                row = f.readline().split()
                if len(row) < len(columns):
                    raise ValueError("Unexpected end of atom rows in dump.")
                atom_ids[i] = int(row[col_index["id"]])
                positions[i] = [float(row[col_index[k]]) for k in coord_cols]

            if coord_mode == "scaled":
                positions = positions * box_lengths[None, :] + np.array([b[0] for b in bounds])[None, :]

            order = np.argsort(atom_ids)
            atom_ids = atom_ids[order]
            positions = positions[order]
            yield timestep, atom_ids, positions, box_lengths, coord_mode


def planar_deviation(angle_deg):
    """Distance to nearest planar torsion (0 or +/-180), in [0, 90] degrees."""
    wrapped = (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return np.minimum(np.abs(wrapped), 180.0 - np.abs(wrapped))


def torsion_angle_deg(points):
    """
    Signed dihedral angle in degrees from four sequential Cartesian points.
    The sign convention does not affect the planarity-based statistics.
    """
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
    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def get_dihedral_angle(atom_positions, atom_ids, quartet, box_lengths, coord_mode):
    """Calculate one torsion, applying minimum-image corrections when needed."""
    # atom_positions is sorted in the same order as atom_ids.
    idx = np.searchsorted(atom_ids, np.asarray(quartet, dtype=np.int64))
    if np.any(idx >= len(atom_ids)) or not np.array_equal(atom_ids[idx], quartet):
        return np.nan
    xyz = atom_positions[idx].copy()

    if coord_mode in ("wrapped", "scaled"):
        # Build a locally unwrapped chain using minimum-image bond vectors.
        unwrapped = np.empty_like(xyz)
        unwrapped[0] = xyz[0]
        for j in range(1, 4):
            bond = xyz[j] - xyz[j - 1]
            bond -= box_lengths * np.round(bond / box_lengths)
            unwrapped[j] = unwrapped[j - 1] + bond
        xyz = unwrapped

    return torsion_angle_deg(xyz)


def summarize_series(angle_series):
    """Compute time-series metrics for one physical dihedral."""
    angle_series = np.asarray(angle_series, dtype=float)
    angle_series = angle_series[np.isfinite(angle_series)]
    if len(angle_series) == 0:
        return None
    delta = planar_deviation(angle_series)
    return {
        "P15": float(np.mean(delta <= 15.0)),
        "P30": float(np.mean(delta <= 30.0)),
        "Mean_delta": float(np.mean(delta)),
        "RMS": float(np.sqrt(np.mean(delta ** 2))),
        "n_frames_valid": int(len(delta)),
    }


def aggregate_per_type(per_dihedral, ddof=1):
    """Summarize the per-dihedral metrics across dihedrals of each type."""
    rows = []
    metric_map = {
        "P15": "P15",
        "P30": "P30",
        "Mean_delta": "Mean_delta",
        "RMS": "RMS",
    }
    for dtype in TARGET_TYPES:
        sub = per_dihedral[per_dihedral["type"] == dtype]
        row = {"Type": dtype, "N_dihedrals": int(len(sub))}
        for source_col, stem in metric_map.items():
            vals = sub[source_col].dropna().to_numpy(dtype=float)
            row[f"{stem}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{stem}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{stem}_SD"] = float(np.std(vals, ddof=ddof)) if len(vals) > ddof else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def compare_summaries(new_df, old_path, out_path):
    old_path = Path(old_path)
    if not old_path.exists():
        print(f"\nOld summary not found at: {old_path}")
        print("Recomputed summary has still been saved.")
        return

    old = pd.read_csv(old_path)
    old_type = find_column(old, ["Type", "type", "dihedral_type"])
    if old_type is None:
        print("\nCould not identify a Type column in old summary; comparison skipped.")
        print("Old columns:", list(old.columns))
        return
    old = old.copy()
    old[old_type] = pd.to_numeric(old[old_type], errors="coerce")
    old = old.rename(columns={old_type: "Type"})

    # Match common column naming styles.
    aliases = {
        "P15_mean": ["P15_mean", "P15 mean"],
        "P15_median": ["P15_median", "P15 median"],
        "P15_SD": ["P15_SD", "P15_sd", "P15 SD"],
        "P30_mean": ["P30_mean", "P30 mean"],
        "P30_median": ["P30_median", "P30 median"],
        "P30_SD": ["P30_SD", "P30_sd", "P30 SD"],
        "Mean_delta_mean": ["Mean_delta_mean", "Mean_delta_mean_deg"],
        "Mean_delta_median": ["Mean_delta_median", "Mean_delta_median_deg"],
        "Mean_delta_SD": ["Mean_delta_SD", "Mean_delta_sd"],
        "RMS_mean": ["RMS_mean"],
        "RMS_median": ["RMS_median"],
        "RMS_SD": ["RMS_SD", "RMS_sd"],
        "N_dihedrals": ["N_dihedrals", "N", "n_dihedrals"],
    }
    rename = {}
    for standard, choices in aliases.items():
        col = find_column(old, choices)
        if col is not None:
            rename[col] = standard
    old = old.rename(columns=rename)

    merged = new_df.merge(old, on="Type", how="outer", suffixes=("_new", "_old"))
    comparisons = []
    for col in aliases:
        new_col = f"{col}_new" if f"{col}_new" in merged.columns else col
        old_col = f"{col}_old" if f"{col}_old" in merged.columns else None
        if old_col is None or new_col not in merged.columns:
            continue
        a = pd.to_numeric(merged[new_col], errors="coerce")
        b = pd.to_numeric(merged[old_col], errors="coerce")
        diff = a - b
        merged[f"diff_{col}"] = diff
        merged[f"abs_diff_{col}"] = np.abs(diff)
        comparisons.append((col, np.nanmax(np.abs(diff.to_numpy())) if diff.notna().any() else np.nan))

    merged.to_csv(out_path, index=False)
    print(f"\nComparison saved to: {out_path}")
    print("\nMaximum absolute differences by column:")
    for col, value in comparisons:
        print(f"  {col:20s} {value:.8g}")
    if not comparisons:
        print("No matching numeric columns were found. Old columns:", list(old.columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="system_150.data")
    parser.add_argument("--trajectory", default="traj_300K_first20ns.lammpstrj")
    parser.add_argument("--old-summary", default="ensemble_analysis/ensemble_summary.csv")
    parser.add_argument("--output-dir", default="ensemble_recomputed")
    parser.add_argument("--ddof", type=int, default=1, choices=[0, 1],
                        help="SD convention across dihedrals: 1=sample SD (default), 0=population SD.")
    args = parser.parse_args()

    topology_path = Path(args.topology)
    trajectory_path = Path(args.trajectory)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not topology_path.exists():
        sys.exit(f"ERROR: topology file not found: {topology_path}")
    if not trajectory_path.exists():
        sys.exit(f"ERROR: trajectory file not found: {trajectory_path}")

    topology = parse_lammps_dihedrals(topology_path)
    topology = topology[topology["type"].isin(TARGET_TYPES)].copy()
    if topology.empty:
        sys.exit(f"ERROR: none of the target dihedral types {TARGET_TYPES} were found.")

    counts = topology.groupby("type").size().to_dict()
    print("Topology counts by type:")
    for dtype in TARGET_TYPES:
        print(f"  {dtype}: {counts.get(dtype, 0)}")

    # Each type/id gets an angle time series across all available frames.
    records = topology.to_dict("records")
    series = {int(r["dihedral_id"]): [] for r in records}
    type_by_id = {int(r["dihedral_id"]): int(r["type"]) for r in records}
    quartet_by_id = {
        int(r["dihedral_id"]): (int(r["a1"]), int(r["a2"]), int(r["a3"]), int(r["a4"]))
        for r in records
    }

    n_frames = 0
    first_step = None
    last_step = None
    coord_modes = set()
    expected_atom_ids = None

    for timestep, atom_ids, positions, box_lengths, coord_mode in parse_dump_frames(trajectory_path):
        n_frames += 1
        first_step = timestep if first_step is None else first_step
        last_step = timestep
        coord_modes.add(coord_mode)
        if expected_atom_ids is None:
            expected_atom_ids = atom_ids
        elif not np.array_equal(expected_atom_ids, atom_ids):
            raise ValueError("Atom IDs differ between frames; refusing inconsistent analysis.")

        for record in records:
            did = int(record["dihedral_id"])
            angle = get_dihedral_angle(
                positions, atom_ids, quartet_by_id[did], box_lengths, coord_mode
            )
            series[did].append(angle)

    if n_frames == 0:
        sys.exit("ERROR: no trajectory frames were parsed.")

    print(f"\nFrames analyzed: {n_frames}")
    print(f"First/last timestep: {first_step} / {last_step}")
    print(f"Coordinate mode(s): {', '.join(sorted(coord_modes))}")
    print("Note: timestep span alone is not used to infer a continuous trajectory duration.")

    per_rows = []
    for did, angles in series.items():
        stats = summarize_series(angles)
        if stats is None:
            continue
        per_rows.append({
            "dihedral_id": did,
            "type": type_by_id[did],
            **stats,
        })

    per_df = pd.DataFrame(per_rows)
    per_path = out_dir / "per_dihedral_statistics.csv"
    per_df.to_csv(per_path, index=False)

    summary = aggregate_per_type(per_df, ddof=args.ddof)
    # Include number of frames in metadata columns, not as independent samples.
    summary["n_frames"] = n_frames
    summary["first_timestep"] = first_step
    summary["last_timestep"] = last_step

    summary_path = out_dir / "ensemble_summary_recomputed.csv"
    summary.to_csv(summary_path, index=False)

    print("\nRecomputed ensemble summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nPer-dihedral statistics saved to: {per_path}")
    print(f"Recomputed summary saved to: {summary_path}")

    compare_summaries(
        summary,
        args.old_summary,
        out_dir / "summary_comparison.csv",
    )


if __name__ == "__main__":
    main()
