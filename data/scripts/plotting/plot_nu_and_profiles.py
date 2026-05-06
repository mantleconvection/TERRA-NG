#!/usr/bin/env python3
"""Plot radial T profiles at every 500 timesteps alongside Nusselt number evolution.

Nu values are read from `nu.csv` written by the solver each timestep
(columns: timestep, sim_time, Nu_top_Q1, Nu_top_FV).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import os

# --- Config ---
profile_dir = sys.argv[1] if len(sys.argv) > 1 else "/p/scratch/walberlamovinggeo/boehm2/output_C1_lvl5_fgmres10/radial_profiles"
nu_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(profile_dir.rstrip("/")), "nu.csv")
output_file = sys.argv[3] if len(sys.argv) > 3 else "nu_and_profiles.png"
# Optional 4th arg: figure title.  Default: derive from output filename stem.
title = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(output_file).replace(
    "nu_and_profiles_", "").replace(".png", "").replace("_", " ")
# Optional 5th arg: reference Nu interval as "min,max[,central[,label]]" — drawn
# as a translucent horizontal band on the Nu panel.  Empty string disables.
ref_nu = sys.argv[5] if len(sys.argv) > 5 else ""
step_interval = 500

# --- Load Nusselt from CSV ---
nu_df = pd.read_csv(nu_csv)
timesteps_nu = nu_df["timestep"].to_numpy()
nu_values = nu_df["Nu_top_Q1"].to_numpy()

# --- Collect radial profile files at step_interval ---
profile_steps = list(range(0, max(timesteps_nu) + 1, step_interval))
profile_steps = [s for s in profile_steps if os.path.exists(os.path.join(profile_dir, f"radial_profiles_T_{s}.csv"))]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Left panel: radial T profiles
cmap = cm.viridis
colors = [cmap(i / max(len(profile_steps) - 1, 1)) for i in range(len(profile_steps))]

for step, color in zip(profile_steps, colors):
    fname = os.path.join(profile_dir, f"radial_profiles_T_{step}.csv")
    df = pd.read_csv(fname)
    df = df.sort_values("radius")
    ax1.plot(df["avg"], df["radius"], color=color, label=f"step {step}")

ax1.set_xlabel("Temperature (avg)")
ax1.set_ylabel("Radius")
ax1.set_title("Radial Temperature Profiles")
ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.5))
ax1.grid(True, alpha=0.3)

# Right panel: Nusselt number evolution
ax2.plot(timesteps_nu, nu_values, 'k-', linewidth=0.8)
# Annotate the final Nu value reached by the run.
final_ts, final_nu = int(timesteps_nu[-1]), float(nu_values[-1])
ax2.plot(final_ts, final_nu, 's', color='black', markersize=6,
         markerfacecolor='white', markeredgewidth=1.2, zorder=5)
ax2.annotate(f"final: Nu={final_nu:.4f} @ ts={final_ts}",
             xy=(final_ts, final_nu),
             xytext=(-8, 12), textcoords='offset points',
             ha='right', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='black', alpha=0.85),
             arrowprops=dict(arrowstyle='-', color='black', lw=0.6))
# Mark the profile timesteps (use nearest available Nu sample)
ts_arr = np.asarray(timesteps_nu)
for step, color in zip(profile_steps, colors):
    idx = int(np.searchsorted(ts_arr, step))
    if idx >= len(ts_arr):
        idx = len(ts_arr) - 1
    ax2.axvline(x=step, color=color, alpha=0.5, linewidth=0.8, linestyle='--')
    ax2.plot(ts_arr[idx], nu_values[idx], 'o', color=color, markersize=5)

ax2.set_xlabel("Timestep")
ax2.set_ylabel("Nu_top (Q1)")
ax2.set_title("Nusselt Number Evolution")
ax2.grid(True, alpha=0.3)

# Optional reference Nu band (e.g., from a published benchmark).
if ref_nu:
    parts = ref_nu.split(",")
    nu_min, nu_max = float(parts[0]), float(parts[1])
    nu_central = float(parts[2]) if len(parts) > 2 else None
    ref_label = parts[3] if len(parts) > 3 else "ref. Nu"
    ax2.axhspan(nu_min, nu_max, color="red", alpha=0.15,
                label=f"{ref_label}: {nu_min:.3f}–{nu_max:.3f}")
    if nu_central is not None:
        ax2.axhline(nu_central, color="red", linestyle=":", linewidth=1.0,
                    label=f"{ref_label} central: {nu_central:.3f}")
    ax2.legend(fontsize=8, loc="best")

plt.suptitle(title, fontsize=14)
plt.tight_layout()
plt.savefig(output_file, dpi=200)
print(f"Saved to {output_file}")
