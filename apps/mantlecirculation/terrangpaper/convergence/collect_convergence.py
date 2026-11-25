#!/usr/bin/env python3
"""Collect the run_convergence.sh job outputs into the two paper tables.

Reads every conv_MT<MT>_<profile>_c<cycles>_<jobid>.out below the given
directories (default: the working directory) and prints, per MT, the FGMRES
iteration count from the 100-iteration runs and the relative residual after
the 10-iteration runs, for the Lin and Stotz profiles.
"""
import glob
import os
import re
import sys

dirs = sys.argv[1:] or ["."]
row_re = re.compile(r"DCA\(re-discretized\)\s*\|")
name_re = re.compile(r"conv_MT(\d+)_(lin|stotz)_c(\d+)_\d+\.out$")

iters, resid = {}, {}
for d in dirs:
    for path in glob.glob(os.path.join(d, "**", "conv_MT*_c*_*.out"), recursive=True):
        m = name_re.search(os.path.basename(path))
        if not m:
            continue
        mt, prof, cyc = int(m.group(1)), m.group(2), int(m.group(3))
        last = None
        with open(path) as f:
            for line in f:
                if row_re.search(line):
                    last = line
        if last is None:
            continue
        cols = [c.strip() for c in last.split("|")]
        # ... | asympt_rate | converged | cycles | final_rel_res | krylov | rmu | scheme |
        cycles, final_rel_res = int(cols[-6]), float(cols[-5])
        if cyc >= 100:
            iters[(mt, prof)] = cycles
        else:
            resid[(mt, prof)] = final_rel_res

mts = sorted({mt for mt, _ in list(iters) + list(resid)})
print(f"{'MT':>7} {'Lin iters':>10} {'Stotz iters':>12} {'Lin res@10':>12} {'Stotz res@10':>13}")
for mt in mts:
    def g(tab, p, fmt):
        v = tab.get((mt, p))
        return fmt % v if v is not None else "-"
    print(f"MT{mt:<5} {g(iters,'lin','%d'):>10} {g(iters,'stotz','%d'):>12} "
          f"{g(resid,'lin','%.2e'):>12} {g(resid,'stotz','%.2e'):>13}")
