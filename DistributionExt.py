import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals
from scipy.ndimage import gaussian_filter1d

# USER INPUT
DATA_FILE = r"DATA_FILE"
TRAJ_FILE = r"TRAJ_FILE"

DIHEDRAL_TYPES = [80, 67, 51, 69, 5, 61, 167, 134, 186]
NBINS = 180
STRIDE = 1
SMOOTH_WINDOW = 4
UNWRAP = False

def read_dihedrals_grouped_by_type(datafile, target_types):
    target_set = set(target_types)
    dih_dict = {dtype: [] for dtype in target_set}

    read_section = False
    with open(datafile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Dihedrals"):
                read_section = True
                continue
            if read_section and not line[0].isdigit():
                break
            if read_section:
                cols = line.split()
                dtype = int(cols[1])
                if dtype in target_set:
                    a, b, c, d = map(int, cols[2:6])
                    dih_dict[dtype].append([a - 1, b - 1, c - 1, d - 1])

    filtered_dict = {}
    for dtype, indices in dih_dict.items():
        if len(indices) > 0:
            filtered_dict[dtype] = np.array(indices, dtype=int)
            print(f"Type {dtype}: Found {len(indices)} dihedrals")
        else:
            print(f"Type {dtype}: Warning - No dihedrals found!")

    return filtered_dict

def periodic_gaussian_smooth(pdf, sigma):
    if sigma <= 0:
        return pdf
    pad = int(4 * sigma)
    pdf_pad = np.concatenate([pdf[-pad:], pdf, pdf[:pad]])
    pdf_smooth = gaussian_filter1d(pdf_pad, sigma=sigma, mode='nearest')
    return pdf_smooth[pad:-pad]

# LOAD TRAJECTORY
u = mda.Universe(DATA_FILE, TRAJ_FILE, topology_format="DATA", format="LAMMPSDUMP")
dih_by_type = read_dihedrals_grouped_by_type(DATA_FILE, DIHEDRAL_TYPES)

bins = np.linspace(-180.0, 180.0, NBINS + 1)
bin_width = bins[1] - bins[0]
centers = 0.5 * (bins[:-1] + bins[1:])

hists = {dtype: np.zeros(NBINS) for dtype in dih_by_type}
sample_counts = {dtype: 0 for dtype in dih_by_type}

frame_count = 0
for ts in u.trajectory[::STRIDE]:
    frame_count += 1
    if UNWRAP:
        u.atoms.unwrap(compound="fragments")

    pos = u.atoms.positions

    for dtype, dih_idx in dih_by_type.items():
        a = pos[dih_idx[:, 0]]
        b = pos[dih_idx[:, 1]]
        c = pos[dih_idx[:, 2]]
        d = pos[dih_idx[:, 3]]

        angles = np.degrees(calc_dihedrals(a, b, c, d))
        angles = (angles + 180.0) % 360.0 - 180.0

        h, _ = np.histogram(angles, bins=bins)
        hists[dtype] += h
        sample_counts[dtype] += angles.size

for dtype in dih_by_type:
    pdf = hists[dtype] / (sample_counts[dtype] * bin_width)
    pdf_smooth = periodic_gaussian_smooth(pdf, sigma=SMOOTH_WINDOW)

    out_file = f"dihedral_type_{dtype}.dat"
    np.savetxt(out_file, np.column_stack([centers, pdf_smooth]), header="phi(deg)    P(phi)")
    print(f"Saved: {out_file}")

print(f"\nProcessing complete across {frame_count} frames!")
