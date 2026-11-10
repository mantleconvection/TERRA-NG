#!/usr/bin/env python3
"""Per-step wall time of each completed LUMI reproduction point, compared with
the published lumi_mt_strong_scaling.csv when given. Per-step time is the
timer_tree_9.json 'timestep' node: root_time / count (count is 9: steps 0..9).
sum_time is the sum across ranks and avg_time the rank mean; do not use them.

usage: collect_lumi.py [OUTROOT] [published.csv]
"""
import json, glob, os, sys, csv
root = sys.argv[1] if len(sys.argv) > 1 else "/scratch/project_465002367/bohmfabi/scal_a3_lumi"
pub = {}
if len(sys.argv) > 2:
    for r in csv.DictReader(open(sys.argv[2])):
        pub[(int(r["mt"]), int(r["gpus"]), r["mode"])] = float(r["s_per_step"])
def node(n, name):
    if n.get("name") == name: return n
    ch = n.get("children"); ch = list(ch.values()) if isinstance(ch, dict) else (ch or [])
    for c in ch:
        r = node(c, name)
        if r: return r
rows = []
for d in sorted(glob.glob(os.path.join(root, "MT*_g*_*"))):
    f = os.path.join(d, "timer_trees", "timer_tree_9.json")
    if not os.path.exists(f): continue
    nm = os.path.basename(d); a = nm.split("_")
    mt, g, mode = int(a[0][2:]), int(a[1][1:]), a[2]
    t = node(json.load(open(f)), "timestep")
    if not t: continue
    rows.append((mt, g, mode, t["root_time"] / t["count"]))
rows.sort()
print("%-6s %6s %-7s %10s %10s %8s" % ("MT", "gcds", "mode", "rerun", "published", "delta"))
for mt, g, mode, sp in rows:
    p = pub.get((mt, g, mode))
    print("%-6d %6d %-7s %10.3f %10s %8s" % (mt, g, mode, sp, ("%.3f" % p) if p else "-", ("%+.1f%%" % (100 * (sp - p) / p)) if p else "-"))
print("\ncompleted points: %d" % len(rows))
