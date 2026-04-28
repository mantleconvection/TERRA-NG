#!/usr/bin/env python3
"""Plot radial T profiles at every 500 timesteps alongside Nusselt number evolution."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import sys
import os

# --- Config ---
profile_dir = sys.argv[1] if len(sys.argv) > 1 else "/p/scratch/walberlamovinggeo/boehm2/output_C1_lvl5_fgmres10/radial_profiles"
log_file = sys.argv[2] if len(sys.argv) > 2 else "/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mc_level5_fgmres10_13660945.out"
output_file = sys.argv[3] if len(sys.argv) > 3 else "nu_and_profiles.png"
step_interval = 500

# --- Parse Nusselt from log file ---
timesteps_nu = []
nu_values = []
with open(log_file) as f:
    for line in f:
        m = re.search(r'Nu_top \(Q1\) = ([0-9.e+-]+)', line)
        if m:
            nu_values.append(float(m.group(1)))

# Nu is output every 100 steps starting at step 0 (index 0 = step 0, index 1 = step 100, ...)
# But first entry is timestep 0, then every 10 timesteps (check the output frequency)
# Actually from the code: output every 10 timesteps for Nu, every 100 for radial profiles
# Let's figure out the timestep mapping
# Nu is printed at timestep 0 and then every 10 timesteps (timestep % 10 == 0)
# But output_frequency=100 for radial profiles
timesteps_nu = [i * 10 for i in range(len(nu_values))]

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
# Mark the profile timesteps
for step, color in zip(profile_steps, colors):
    idx = step // 10
    if idx < len(nu_values):
        ax2.axvline(x=step, color=color, alpha=0.5, linewidth=0.8, linestyle='--')
        ax2.plot(step, nu_values[idx], 'o', color=color, markersize=5)

ax2.set_xlabel("Timestep")
ax2.set_ylabel("Nu_top (Q1)")
ax2.set_title("Nusselt Number Evolution")
ax2.grid(True, alpha=0.3)

plt.suptitle("Level 5, 10 FGMRES, SUPG", fontsize=14)
plt.tight_layout()
plt.savefig(output_file, dpi=200)
print(f"Saved to {output_file}")
