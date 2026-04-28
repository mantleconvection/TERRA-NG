#!/usr/bin/env python3
"""Parse matrix sweep log and plot (lat × r) heatmap of throughput."""
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

LOG = sys.argv[1]

# Grid order (matches run_matrix_sweep.sh)
LATERAL = [1, 2, 3, 4, 8]
RADIAL  = [2, 4, 8, 16, 32, 64]

# Parse the log
results = {}  # (lat, r) -> gdofs
with open(LOG) as f:
    txt = f.read()

# Regex: find config header, then the next HIP (id=1) rate
pattern = re.compile(r'--- \((\d+),(\d+),1\) team=\d+ ---\n(.*?)(?=--- \(|=== Done)', re.DOTALL)
for m in pattern.finditer(txt):
    lat, r = int(m.group(1)), int(m.group(2))
    body = m.group(3)
    rates = re.findall(r',1,8,[^,]+,([\d.e+]+)', body)
    if rates:
        results[(lat, r)] = float(rates[-1]) / 1e9
    else:
        results[(lat, r)] = None

# Build matrix (rows=radial, cols=lateral), row 0 = smallest r at bottom
matrix = np.full((len(RADIAL), len(LATERAL)), np.nan)
for j, lat in enumerate(LATERAL):
    for i, r in enumerate(RADIAL):
        v = results.get((lat, r))
        if v is not None:
            matrix[i, j] = v

# Reverse so largest radial at top
matrix_disp = matrix[::-1]
radial_disp = list(reversed(RADIAL))

# --- Plot heatmap ---
fig, ax = plt.subplots(figsize=(12, 9))

# Colormap: red → orange → yellow → green
cmap = mcolors.LinearSegmentedColormap.from_list(
    'perf', ['#d32f2f', '#f57c00', '#fbc02d', '#8bc34a', '#2e7d32'])
vmax = np.nanmax(matrix_disp)
vmin = np.nanmin(matrix_disp[matrix_disp > 0]) if np.any(matrix_disp > 0) else 0
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

masked = np.ma.masked_invalid(matrix_disp)
im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')

# Annotate cells with Gdofs/s
for i in range(len(radial_disp)):
    for j in range(len(LATERAL)):
        v = matrix_disp[i, j]
        if np.isnan(v):
            ax.text(j, i, 'skip', ha='center', va='center',
                    color='gray', fontsize=16, style='italic')
        else:
            color = 'white' if norm(v) < 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=color, fontsize=20, fontweight='bold')

ax.set_xticks(range(len(LATERAL)))
ax.set_xticklabels([f'{l}×{l}' for l in LATERAL], fontsize=20)
ax.set_yticks(range(len(radial_disp)))
ax.set_yticklabels(radial_disp, fontsize=20)

ax.set_xlabel('Lateral tile (lat × lat)', fontsize=24)
ax.set_ylabel('Radial tile (r)', fontsize=24)
ax.set_title('EpsDivDivKerngen throughput (Gdofs/s) — MI250X GCD\n'
             'r_passes=1, wedge split, level 8',
             fontsize=22, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax, shrink=0.9)
cbar.set_label('Gdofs/s', fontsize=20)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
out = 'matrix_heatmap.png'
plt.savefig(out, dpi=150)
print(f'Saved: {out}')
print(f'Peak: {vmax:.2f} Gdofs/s')

# Print the underlying table
print()
print(' ' * 8 + '  '.join(f'{l}x{l:<5}' for l in LATERAL))
for i, r in enumerate(radial_disp):
    row = '  '.join(f'{matrix_disp[i,j]:>7.2f}' if not np.isnan(matrix_disp[i,j]) else '   skip'
                    for j in range(len(LATERAL)))
    print(f'r={r:<5} {row}')
