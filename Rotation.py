from pathlib import Path
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

DR3_FILE = Path(
    r"C:\Users\fahad\OneDrive\Desktop\simulations\Data\DR3f.xyz"
)

OIDTBR_FILE = Path(
    r"C:\Users\fahad\OneDrive\Desktop\simulations\Data\OiDTBRf.xyz"
)

OUTPUT_ROOT = Path(
    r"C:\Users\fahad\OneDrive\Desktop\simulations\xTB_torsion_scan"
)

ANGLES = [0.0, 90.0]

# 1-based atom numbering from your definitions
DIHEDRALS = {
    "DR3": {
        "A": (30, 29, 102, 104),
        "B": (90, 88, 68, 61),
        "C": (105, 107, 127, 129),
        "D": (130, 132, 109, 110),
    },
    "OiDTBR": {
        "E": (18, 17, 15, 14),
        "F": (27, 26, 20, 19),
    },
}

# Covalent radii, Å
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

BOND_TOLERANCE = 0.45


# ============================================================
# XYZ I/O
# ============================================================

def read_xyz(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_atoms = int(lines[0])
    comment = lines[1]

    atoms = []
    coords = []

    for line in lines[2:2 + n_atoms]:
        parts = line.split()

        element = parts[0]
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]

        atoms.append(element)
        coords.append(xyz)

    coords = np.asarray(coords, dtype=float)

    if len(atoms) != n_atoms:
        raise RuntimeError(
            f"{filename}: expected {n_atoms} atoms, found {len(atoms)}"
        )

    return atoms, coords, comment


def write_xyz(filename, atoms, coords, comment):
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")

        for element, xyz in zip(atoms, coords):
            f.write(
                f"{element:2s} "
                f"{xyz[0]:16.8f} "
                f"{xyz[1]:16.8f} "
                f"{xyz[2]:16.8f}\n"
            )


# ============================================================
# BOND DETECTION
# ============================================================

def build_bond_graph(atoms, coords):
    n = len(atoms)
    graph = {i: set() for i in range(n)}

    for i in range(n):
        ri = COVALENT_RADII.get(atoms[i])
        if ri is None:
            raise RuntimeError(
                f"No covalent radius defined for element {atoms[i]}"
            )

        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(atoms[j])

            distance = np.linalg.norm(coords[i] - coords[j])

            cutoff = ri + rj + BOND_TOLERANCE

            if distance <= cutoff:
                graph[i].add(j)
                graph[j].add(i)

    return graph


# ============================================================
# GRAPH OPERATIONS
# ============================================================

def connected_component(graph, start, blocked_edge):
    """
    Return all atoms connected to 'start' after removing blocked_edge.
    blocked_edge is a tuple of zero-based indices (a, b).
    """

    a, b = blocked_edge

    visited = set()
    stack = [start]

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:

            if (
                (current == a and neighbor == b)
                or
                (current == b and neighbor == a)
            ):
                continue

            if neighbor not in visited:
                stack.append(neighbor)

    return visited


# ============================================================
# DIHEDRAL CALCULATION
# ============================================================

def dihedral_angle(coords, quartet):
    """
    Calculate dihedral angle in degrees.

    quartet is 1-based.
    """

    i, j, k, l = [x - 1 for x in quartet]

    p0 = coords[i]
    p1 = coords[j]
    p2 = coords[k]
    p3 = coords[l]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    angle = np.degrees(np.arctan2(y, x))

    return angle


# ============================================================
# ROTATION
# ============================================================

def rotate_points_about_axis(
    coords,
    indices,
    axis_point_1,
    axis_point_2,
    angle_deg
):
    """
    Rotate selected atoms around the axis defined by
    axis_point_1 -> axis_point_2.
    """

    new_coords = coords.copy()

    p1 = coords[axis_point_1]
    p2 = coords[axis_point_2]

    axis = p2 - p1
    axis /= np.linalg.norm(axis)

    theta = np.radians(angle_deg)

    # Rodrigues rotation matrix
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])

    I = np.eye(3)

    R = (
        I
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )

    for idx in indices:
        relative = coords[idx] - p1
        rotated = R @ relative
        new_coords[idx] = p1 + rotated

    return new_coords


def make_torsion_geometry(
    atoms,
    coords,
    quartet,
    target_angle
):
    """
    Rotate one side of the central j-k bond so that the
    requested dihedral angle is reached.
    """

    i, j, k, l = [x - 1 for x in quartet]

    graph = build_bond_graph(atoms, coords)

    # The bond around which we rotate is j-k.
    central_bond = (j, k)

    # Determine which fragment contains l after cutting j-k.
    fragment_k = connected_component(
        graph,
        k,
        central_bond
    )

    # Sanity check:
    if l not in fragment_k:
        raise RuntimeError(
            f"Atom {l + 1} is not in the k-side fragment "
            f"for dihedral {quartet}."
        )

    # Do NOT rotate j itself or the opposite fragment.
    rotating_atoms = sorted(fragment_k)

    current_angle = dihedral_angle(coords, quartet)

    # We need the signed rotation that transforms the
    # current dihedral into the target.
    delta = target_angle - current_angle

    # Wrap to [-180, 180]
    delta = (delta + 180.0) % 360.0 - 180.0

    new_coords = rotate_points_about_axis(
        coords,
        rotating_atoms,
        j,
        k,
        delta
    )

    final_angle = dihedral_angle(
        new_coords,
        quartet
    )

    return new_coords, current_angle, final_angle, rotating_atoms


# ============================================================
# VALIDATION
# ============================================================

def validate_geometry(
    original_coords,
    new_coords,
    rotating_atoms,
    quartet
):
    i, j, k, l = [x - 1 for x in quartet]

    rotating_atoms = set(rotating_atoms)

    # Central bond length must remain unchanged
    original_bond = np.linalg.norm(
        original_coords[j] - original_coords[k]
    )

    new_bond = np.linalg.norm(
        new_coords[j] - new_coords[k]
    )

    bond_error = abs(new_bond - original_bond)

    # Atoms outside rotating fragment should not move
    fixed_indices = [
        idx
        for idx in range(len(original_coords))
        if idx not in rotating_atoms
    ]

    if fixed_indices:
        max_fixed_displacement = np.max(
            np.linalg.norm(
                new_coords[fixed_indices]
                - original_coords[fixed_indices],
                axis=1
            )
        )
    else:
        max_fixed_displacement = 0.0

    return bond_error, max_fixed_displacement


# ============================================================
# MAIN
# ============================================================

def process_molecule(name, xyz_file):

    print()
    print("=" * 80)
    print(f"{name}")
    print("=" * 80)

    atoms, coords, comment = read_xyz(xyz_file)

    print(f"Atoms: {len(atoms)}")
    print(f"XYZ:   {xyz_file}")

    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for site, quartet in DIHEDRALS[name].items():

        print()
        print("-" * 80)
        print(f"{site}: {quartet}")
        print("-" * 80)

        original_angle = dihedral_angle(
            coords,
            quartet
        )

        print(
            f"Original dihedral: "
            f"{original_angle:.6f} deg"
        )

        for target_angle in ANGLES:

            try:
                (
                    new_coords,
                    before,
                    after,
                    rotating_atoms,
                ) = make_torsion_geometry(
                    atoms,
                    coords,
                    quartet,
                    target_angle
                )

                bond_error, fixed_error = validate_geometry(
                    coords,
                    new_coords,
                    rotating_atoms,
                    quartet
                )

                print(
                    f"Target {target_angle:6.1f}° | "
                    f"before {before:9.3f}° | "
                    f"after {after:9.3f}° | "
                    f"rotating atoms {len(rotating_atoms):3d} | "
                    f"bond error {bond_error:.3e} Å | "
                    f"fixed displacement {fixed_error:.3e} Å"
                )

                filename = (
                    output_dir
                    / f"{site}_{int(target_angle):03d}deg.xyz"
                )

                write_xyz(
                    filename,
                    atoms,
                    new_coords,
                    f"{name} {site} torsion={target_angle:.1f} deg"
                )

            except Exception as e:

                print(
                    f"ERROR for {site} at "
                    f"{target_angle}°: {e}"
                )


if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    process_molecule(
        "DR3",
        DR3_FILE
    )

    process_molecule(
        "OiDTBR",
        OIDTBR_FILE
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
