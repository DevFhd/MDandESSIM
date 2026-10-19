# MDandESSIM

## Molecular Dynamics and Electronic Structure Simulation

### Purpose of This Repository

This repository contains the computational scripts used in the study *Molecular Dynamics and Electronic Structure Investigation of the DR3/O-IDTBR Donor–Acceptor System in Organic Solar Cells*. It is intended to support the reproducibility of the computational workflow and the analysis of the study's results.

The scripts cover molecular dynamics (MD) simulations, dihedral angle analysis, statistical characterization of molecular conformations, electronic structure calculations using GFN2-xTB, and data visualization.

## Repository Contents

| File | Description |
|---|---|
| `ACFandCMEst.py` | Extracts data from molecular dynamics simulations and calculates autocorrelation functions and cumulative mean statistics. |
| `DistributionExt.py` | Extracts dihedral angle data from MD simulations and calculates the probability density distributions for the selected dihedral angles. |
| `DistributionVis.py` | Visualizes the dihedral angle probability density distributions for the six dihedrals. |
| `MDStats.py` | Calculates statistical descriptors of the dihedral angle distributions, including mean deviation, $P(15^\circ)$, $P(30^\circ)$, and confidence intervals. |
| `Rotation.py` | Rotates each selected dihedral in each molecule to generate two conformational conditions: planar $(0^\circ)$ and twisted $(90^\circ)$. |
| `RunxTB.py` | Performs GFN2-xTB calculations for each molecule under both conformational conditions and extracts the corresponding HOMO and LUMO energies. |
| `SimCode.lmp` | Defines the LAMMPS workflow for simulating 150 molecular complexes through three main stages: heating, cooling, and data collection. |
| `xTBVis.py` | Visualizes the electronic structure data extracted from the GFN2-xTB calculations. |

## Computational Workflow

The repository is organized around the following workflow:

1. **Molecular dynamics simulations:** Generate molecular trajectories using LAMMPS.
2. **Conformational analysis:** Extract dihedral angles and characterize their probability distributions.
3. **Statistical analysis:** Evaluate conformational statistics, autocorrelation functions, cumulative means, and confidence intervals.
4. **Electronic structure calculations:** Evaluate HOMO and LUMO energies for planar and twisted conformations using GFN2-xTB.
5. **Visualization:** Generate plots to examine conformational distributions and electronic structure results.

## Reproducibility

The scripts are provided to document the computational procedures and facilitate the reproduction and independent evaluation of the study's results. Reproducing the complete workflow may require the appropriate software packages, input structures, simulation parameters, and molecular dynamics trajectories.
