from pathlib import Path
import re

FILE = Path("molden.input")

text = FILE.read_text(encoding="utf-8", errors="replace")

# ------------------------------------------------------------
# Extract sections
# ------------------------------------------------------------

gto_start = text.find("[GTO]")
mo_start = text.find("[MO]")

if gto_start == -1 or mo_start == -1:
    raise RuntimeError("Could not find [GTO] or [MO] section.")

gto = text[gto_start + len("[GTO]"):mo_start]

# ------------------------------------------------------------
# Find atom blocks
#
# Molden format:
#
#       1 0
#       s ...
#       ...
#       2 0
#       s ...
#       ...
# ------------------------------------------------------------

pattern = re.compile(
    r"(?m)^\s*(\d+)\s+0\s*$"
)

matches = list(pattern.finditer(gto))

print("=" * 70)
print("BASIS FUNCTION INSPECTION")
print("=" * 70)

print("Atom blocks found:", len(matches))

# ------------------------------------------------------------
# Count basis functions for each atom
# ------------------------------------------------------------

# Number of Cartesian basis functions:
# s = 1
# p = 3
# d = 6
# f = 10
# g = 15
#
# xTB Molden should normally use these shells explicitly.

shell_size = {
    "s": 1,
    "p": 3,
    "d": 6,
    "f": 10,
    "g": 15,
}

atom_basis = []

for idx, match in enumerate(matches):

    atom_number = int(match.group(1))

    start = match.end()

    if idx + 1 < len(matches):
        end = matches[idx + 1].start()
    else:
        end = len(gto)

    block = gto[start:end]

    basis_count = 0
    shells = []

    for line in block.splitlines():

        m = re.match(
            r"^\s*([spdfg])\s+\d+",
            line,
            re.IGNORECASE
        )

        if m:
            shell = m.group(1).lower()
            shells.append(shell)
            basis_count += shell_size[shell]

    atom_basis.append(
        (atom_number, basis_count, shells)
    )

# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

total_basis = 0

for atom_number, count, shells in atom_basis:

    total_basis += count

    if atom_number <= 15:
        print(
            f"Atom {atom_number:4d}: "
            f"{count:3d} basis functions   "
            f"shells = {shells}"
        )

print()
print("-" * 70)

print("Atoms represented :", len(atom_basis))
print("Total basis funcs :", total_basis)

print()

# ------------------------------------------------------------
# Compare against Molden MO coefficient count
# ------------------------------------------------------------

mo = text[mo_start:]

blocks = re.split(r"(?m)^Sym=", mo)[1:]

print("MO blocks         :", len(blocks))

if blocks:

    coefficient_pattern = re.compile(
        r"(?m)^\s*\d+\s+[-+0-9.Ee]+"
    )

    first_coeffs = sum(
        bool(coefficient_pattern.match(line))
        for line in blocks[0].splitlines()
    )

    print("Coefficients / MO :", first_coeffs)

    print()

    if total_basis == first_coeffs:
        print("SUCCESS: basis count matches MO coefficient count.")
    else:
        print(
            "WARNING: mismatch:",
            total_basis,
            "basis functions vs",
            first_coeffs,
            "MO coefficients"
        )