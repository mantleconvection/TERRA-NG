#!/usr/bin/env python3
"""
Aggregate per-step timings from finished bench_mt SLURM jobs into a CSV.

Reads bench_mt/jobs/manifest.txt (written by submit_bench_mt.py), locates
each cell's stdout (bench_<cell>.o<jobid>), parses the per-timestep log lines,
computes per-step wall-clock times, drops a configurable number of warmup
steps, and emits one row per cell.

Usage:
    python3 collect_bench_mt.py                 # writes bench_mt/results.csv
    python3 collect_bench_mt.py --warmup 5      # change warmup count
    python3 collect_bench_mt.py -o other.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics as stats
import sys
from pathlib import Path

BENCH_DIR  = Path(__file__).resolve().parent
JOB_DIR    = BENCH_DIR / "jobs"
MANIFEST   = JOB_DIR / "manifest.txt"

TS_LINE_RE  = re.compile(
    r"^\[LOG \| rank\s*0\s*\| (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^\]]*\] ### Timestep (\d+) ###"
)
DOFS_LINE_RE = re.compile(
    r"Degrees of freedom in \(T,u,p\) = \((\d+), (\d+), (\d+)\)"
)
FGMRES_RE   = re.compile(r"\| fgmres_solver \|")

def parse_log(log_path: Path, warmup: int) -> dict:
    """Return dict with timing summary or {'error': str} on failure."""
    if not log_path.exists():
        return {"error": f"log missing: {log_path}"}

    timestamps = []   # (timestep_int, datetime)
    dofs       = None

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = TS_LINE_RE.match(line)
            if m:
                t  = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                ts = int(m.group(2))
                timestamps.append((ts, t))
                continue
            if dofs is None:
                d = DOFS_LINE_RE.search(line)
                if d:
                    dofs = (int(d.group(1)), int(d.group(2)), int(d.group(3)))

    if len(timestamps) < warmup + 2:
        return {"error": f"only {len(timestamps)} timesteps logged in {log_path.name}"}

    # per-step durations between consecutive '### Timestep N ###' markers.
    durations = []
    for (ts0, t0), (ts1, t1) in zip(timestamps, timestamps[1:]):
        durations.append((ts1, (t1 - t0).total_seconds()))

    # Drop the first `warmup` durations (often slow due to lazy init,
    # GPU caches, MG factorization etc.).
    measured = [d for (_ts, d) in durations[warmup:]]
    if not measured:
        return {"error": f"no measured timesteps left after warmup={warmup}"}

    return {
        "n_timesteps_logged": len(timestamps),
        "n_measured":         len(measured),
        "warmup_dropped":     min(warmup, len(durations)),
        "step_mean_s":        stats.mean(measured),
        "step_stddev_s":      stats.pstdev(measured) if len(measured) > 1 else 0.0,
        "step_p50_s":         stats.median(measured),
        "step_min_s":         min(measured),
        "step_max_s":         max(measured),
        "dofs_T":             dofs[0] if dofs else None,
        "dofs_u":             dofs[1] if dofs else None,
        "dofs_p":             dofs[2] if dofs else None,
    }

def read_manifest():
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}; run submit_bench_mt.py first")
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            (cell_tag, job_id, mesh_max, level_subd, mesh_min, radial_extra, n_gpus, nodes, tpn) = parts
            rows.append(dict(
                cell_tag=cell_tag, job_id=job_id,
                mesh_max=int(mesh_max),
                level_subdomains=int(level_subd),
                mesh_min=int(mesh_min),
                radial_extra=int(radial_extra),
                n_gpus=int(n_gpus), nodes=int(nodes), tasks_per_node=int(tpn),
            ))
    return rows

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=3,
                   help="number of leading per-step durations to drop (default 3)")
    p.add_argument("-o", "--output", default=str(BENCH_DIR / "results.csv"),
                   help="output CSV path")
    args = p.parse_args(argv)

    rows = read_manifest()

    out_rows = []
    for r in rows:
        log_path = JOB_DIR / f"bench_{r['cell_tag']}.o{r['job_id']}"
        timing   = parse_log(log_path, warmup=args.warmup)

        out = dict(
            cell_tag=r["cell_tag"],
            mt_label=2 ** r["mesh_max"],
            mesh_max=r["mesh_max"],
            level_subdomains=r["level_subdomains"],
            mesh_min=r["mesh_min"],
            radial_extra=r["radial_extra"],
            n_gpus=r["n_gpus"],
            nodes=r["nodes"],
            tasks_per_node=r["tasks_per_node"],
            job_id=r["job_id"],
            **timing,
        )
        out_rows.append(out)
        if "error" in timing:
            print(f"  [skip] {r['cell_tag']} (job {r['job_id']}): {timing['error']}")
        else:
            print(f"  {r['cell_tag']:>14s}  job {r['job_id']}  "
                  f"DoFs(T,u,p)=({timing.get('dofs_T')}, {timing.get('dofs_u')}, {timing.get('dofs_p')})  "
                  f"step_mean={timing['step_mean_s']:.3f}s  measured={timing['n_measured']}")

    # Build superset of all keys (some rows have 'error', others have timing fields).
    fieldnames = []
    for r in out_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main(sys.argv[1:])
