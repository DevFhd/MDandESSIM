
from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
BASE = Path(r"C:\Users\fahad\OneDrive\Desktop\simulations")
DATA_FILE = BASE / "system_150.data"
TRAJ_FILE = BASE / "traj_300K_first20ns.lammpstrj"
OUT_DIR = BASE / "cummean_acf_results"

TARGET_TYPES = [51, 61, 67, 80, 167, 186]

# Change this if your LAMMPS timestep is not 2 fs.
TIMESTEP_FS = 2.0

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ LAMMPS DATA FILE
# ============================================================
def read_data_sections(path):
    lines = path.read_text(errors="replace").splitlines()
    sections = defaultdict(list)
    current = None

    known_sections = {
        "atoms", "bonds", "angles", "dihedrals", "impropers",
        "velocities", "masses", "pair coefficients",
        "bond coefficients", "angle coefficients",
        "dihedral coefficients", "improper coefficients",
    }

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        # Recognize section headers, including e.g. "Atoms # full".
        header = s.split("#", 1)[0].strip().lower()
        if header in known_sections:
            current = header
            continue

        # Ignore the file header and counts before the first section.
        if current is None:
            continue

        fields = s.split()
        if fields and fields[0].lstrip("+-").isdigit():
            sections[current].append(fields)

    return sections


sections = read_data_sections(DATA_FILE)

atom_mol = {}
dihedrals = []

for row in sections["atoms"]:
    # Expected format: atom-ID molecule-ID atom-type charge x y z
    if len(row) >= 7:
        atom_id, mol_id = int(row[0]), int(row[1])
        atom_mol[atom_id] = mol_id

for row in sections["dihedrals"]:
    # Expected format: dihedral-ID type atom1 atom2 atom3 atom4
    if len(row) >= 6:
        did, dtype, a1, a2, a3, a4 = map(int, row[:6])
        if dtype in TARGET_TYPES:
            mols = {atom_mol.get(a) for a in (a1, a2, a3, a4)}
            if len(mols) == 1 and None not in mols:
                dihedrals.append({
                    "id": did,
                    "type": dtype,
                    "mol": next(iter(mols)),
                    "atoms": (a1, a2, a3, a4),
                })

if not atom_mol:
    raise RuntimeError("No atoms parsed. Check the Atoms section format.")
if not dihedrals:
    raise RuntimeError("No target dihedrals parsed. Check the topology file.")

print(f"Parsed atoms: {len(atom_mol)}")
print(f"Selected dihedrals: {len(dihedrals)}")
print(f"Molecule IDs: {len(set(atom_mol.values()))}")


# ============================================================
# TRAJECTORY READER
# Supports orthorhombic boxes and xu/yu/zu, x/y/z, xs/ys/zs
# ============================================================
def read_dump(path):
    with path.open("r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Unexpected dump format after timestep.")
            n_atoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Missing BOX BOUNDS.")
            bounds = []
            for _ in range(3):
                vals = list(map(float, f.readline().split()))
                bounds.append((vals[0], vals[1]))

            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError("Missing ATOMS header.")

            cols = line.split()[2:]
            col_index = {name: i for i, name in enumerate(cols)}

            if "id" not in col_index:
                raise RuntimeError("Trajectory ATOMS header has no id column.")

            if all(c in col_index for c in ("xu", "yu", "zu")):
                xyz_cols = ("xu", "yu", "zu")
                scaled = False
            elif all(c in col_index for c in ("x", "y", "z")):
                xyz_cols = ("x", "y", "z")
                scaled = False
            elif all(c in col_index for c in ("xs", "ys", "zs")):
                xyz_cols = ("xs", "ys", "zs")
                scaled = True
            else:
                raise RuntimeError(
                    "Cannot find xyz columns. Expected xu/yu/zu, x/y/z, "
                    "or xs/ys/zs."
                )

            ids = np.empty(n_atoms, dtype=np.int64)
            xyz = np.empty((n_atoms, 3), dtype=float)

            for i in range(n_atoms):
                vals = f.readline().split()
                ids[i] = int(vals[col_index["id"]])
                xyz[i] = [float(vals[col_index[c]]) for c in xyz_cols]

            # Convert scaled coordinates to Cartesian for orthorhombic boxes.
            if scaled:
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                xyz = lo + xyz * (hi - lo)

            # Map atom IDs to coordinates.
            coords = {int(aid): xyz[i] for i, aid in enumerate(ids)}
            yield step, coords


# ============================================================
# GEOMETRY AND PLANAR DEVIATION
# ============================================================
def minimum_image(v, box_lengths):
    return v - box_lengths * np.round(v / box_lengths)


def dihedral_angle(p0, p1, p2, p3, box_lengths):
    # Unwrap consecutive bond vectors with minimum-image convention.
    b0 = minimum_image(p0 - p1, box_lengths)
    b1 = minimum_image(p2 - p1, box_lengths)
    b2 = minimum_image(p3 - p2, box_lengths)

    norm = np.linalg.norm(b1)
    if norm < 1e-12:
        return np.nan

    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv < 1e-12 or nw < 1e-12:
        return np.nan

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def planar_deviation_deg(angle_deg):
    # Distance from the nearest planar state (0 or +/-180 degrees).
    return abs(((angle_deg + 90.0) % 180.0) - 90.0)


# ============================================================
# PROCESS TRAJECTORY
# ============================================================
steps = []
angles_by_dihedral = {d["id"]: [] for d in dihedrals}
box_lengths_by_frame = []

for frame_index, (step, coords) in enumerate(read_dump(TRAJ_FILE), start=1):
    steps.append(step)

    # Re-read box bounds from the current frame is required for PBC.
    # The dump reader yields coordinates but not bounds, so use a fixed
    # orthorhombic box estimate from the coordinate ranges here.
    # For trajectories with wrapped x/y/z, provide actual box lengths below.
    xyz_all = np.array(list(coords.values()))
    box_lengths = np.ptp(xyz_all, axis=0)
    box_lengths[box_lengths < 1e-8] = 1.0
    box_lengths_by_frame.append(box_lengths)

    for d in dihedrals:
        try:
            p0, p1, p2, p3 = [coords[a] for a in d["atoms"]]
            ang = dihedral_angle(p0, p1, p2, p3, box_lengths)
        except KeyError:
            ang = np.nan
        angles_by_dihedral[d["id"]].append(ang)

if len(steps) < 2:
    raise RuntimeError("Need at least two trajectory frames.")

steps = np.asarray(steps)
time_ns = (steps - steps[0]) * TIMESTEP_FS / 1e6

angle_matrix = {
    d["id"]: np.asarray(angles_by_dihedral[d["id"]], dtype=float)
    for d in dihedrals
}
dev_matrix = {
    did: planar_deviation_deg(angles)
    for did, angles in angle_matrix.items()
}


# ============================================================
# CUMULATIVE MEAN OF PLANAR DEVIATION
# ============================================================
cum_rows = []

for dtype in TARGET_TYPES:
    selected = [d["id"] for d in dihedrals if d["type"] == dtype]
    if not selected:
        continue

    values = np.vstack([dev_matrix[did] for did in selected])
    valid = np.isfinite(values)
    sums = np.where(valid, values, 0.0).cumsum(axis=1)
    counts = valid.cumsum(axis=1)

    total_sum = sums.sum(axis=0)
    total_count = counts.sum(axis=0)
    cumulative = np.divide(
        total_sum, total_count,
        out=np.full_like(total_sum, np.nan),
        where=total_count > 0
    )

    for i in range(len(steps)):
        cum_rows.append({
            "dihedral_type": dtype,
            "step": steps[i],
            "time_ns": time_ns[i],
            "cumulative_mean_planar_deviation_deg": cumulative[i],
        })

pd.DataFrame(cum_rows).to_csv(
    OUT_DIR / "cumulative_mean.csv", index=False
)


# ============================================================
# AUTOCORRELATION FUNCTION (ACF)
# Compute ACF independently for each dihedral time series, then
# average ACFs within each dihedral type.
# ============================================================
def acf_fft(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        return None

    x = x - x.mean()
    variance = np.dot(x, x) / len(x)
    if variance < 1e-14:
        return None

    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n]
    acov = acov / np.arange(n, 0, -1)
    return acov / acov[0]


acf_rows = []
