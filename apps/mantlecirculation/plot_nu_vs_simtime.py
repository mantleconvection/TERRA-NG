#!/usr/bin/env python3
"""Plot Nusselt number vs simulated time for level 5 and level 6, 10 FGMRES."""

import re
import matplotlib.pyplot as plt

def extract(logfile):
    nu_vals = []
    sim_times = [0.0]  # Nu at timestep 0 corresponds to t=0
    with open(logfile) as f:
        for line in f:
            m = re.search(r'Nu_top \(Q1\) = ([0-9.e+-]+)', line)
            if m:
                nu_vals.append(float(m.group(1)))
            m2 = re.search(r'Simulated time: ([0-9.e+-]+)', line)
            if m2:
                sim_times.append(float(m2.group(1)))
    # Nu is output every 10 steps. sim_times[i] is after step i.
    # Nu[0] = step 0, Nu[1] = step 10, Nu[2] = step 20, ...
    # sim_times[0] = 0, sim_times[1] = after step 1, sim_times[10] = after step 10
    nu_times = []
    for i in range(len(nu_vals)):
        step = i * 10
        if step < len(sim_times):
            nu_times.append(sim_times[step])
        else:
            break
    nu_vals = nu_vals[:len(nu_times)]
    return nu_times, nu_vals

t5, nu5 = extract('/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mc_level5_fgmres10_13660945.out')
t6, nu6 = extract('/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mc_level6_16gpu_13660992.out')

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t5, nu5, 'b-', linewidth=0.8, label='Level 5 (MT64)')
ax.plot(t6, nu6, 'r-', linewidth=0.8, label='Level 6 (MT128)')

ax.set_xlabel('Simulated time')
ax.set_ylabel('Nu_top (Q1)')
ax.set_title('C1 Benchmark — Nusselt vs Simulated Time (10 FGMRES, SUPG)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 18)

plt.tight_layout()
output = '/p/home/jusers/boehm2/juwels/terraneo/apps/mantlecirculation/nu_vs_simtime.png'
plt.savefig(output, dpi=200)
print(f'Saved to {output}')
