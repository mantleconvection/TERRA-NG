#!/usr/bin/env python
"""Strong-scaling plot of kernel (compute) vs communication time per matvec,
one panel per problem size (rad_level), from the operator timer-tree JSONs."""
import json, glob, re, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"

def find(node, name):
    if node.get("name") == name:
        return node
    for c in node.get("children", []):
        r = find(c, name)
        if r:
            return r
    return None

# data[rad_level] = list of (n_gpus, kernel_s, comm_s, apply_s) per matvec
data = collections.defaultdict(list)
for f in glob.glob(f"{OUT}/STR_l8r*/tts/*.json"):
    m = re.search(r"_np(\d+)_.*_rad(\d+)\.json", f)
    if not m:
        continue
    ng, rad = int(m.group(1)), int(m.group(2))
    root = json.load(open(f))
    ap = find(root, "epsilon_divdiv_apply")
    ke = find(root, "epsilon_divdiv_kernel")
    co = find(root, "epsilon_divdiv_comm")
    if not (ap and ke and co):
        continue
    cnt = ap.get("count") or 1
    if cnt != 20:                         # warmup-free trees only (timed-only)
        print(f"  skip rad{rad} g{ng}: count={cnt} (warmup not excluded)")
        continue
    per = lambda n: n["avg_time"] / cnt   # per-rank avg time per matvec [s]
    if per(co) * 1e3 > 50:                # comm contention outlier (waitall stall)
        print(f"  skip rad{rad} g{ng}: comm={per(co)*1e3:.0f} ms (network contention)")
        continue
    data[rad].append((ng, per(ke), per(co), per(ap)))

rads = sorted(data)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False)
axes = axes.flatten()
for i, rad in enumerate(rads):
    ax = axes[i]
    pts = sorted(data[rad])
    g  = [p[0] for p in pts]
    ke = [p[1] * 1e3 for p in pts]   # ms
    co = [p[2] * 1e3 for p in pts]
    ap = [p[3] * 1e3 for p in pts]
    ax.loglog(g, ke, "o-", color="tab:blue",   label="kernel (compute)")
    ax.loglog(g, co, "s-", color="tab:red",    label="communication")
    ax.loglog(g, ap, "^--", color="0.4",       label="apply (total)", lw=1)
    # ideal strong-scaling reference anchored to the kernel's first point
    ideal = [ke[0] * g[0] / x for x in g]
    ax.loglog(g, ideal, ":", color="tab:blue", alpha=0.5, label="ideal (1/N)")
    dofs_G = {6:0.13,7:0.25,8:0.51,9:1.01,10:2.02,11:4.03}.get(rad, 0)
    ax.set_title(f"rad_level {rad}  —  {dofs_G:.2f} G dofs")
    ax.set_xlabel("GCDs (GPUs)")
    ax.set_ylabel("time per matvec [ms]")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
for j in range(len(rads), len(axes)):
    axes[j].axis("off")
fig.suptitle("EpsDivDiv operator: kernel vs communication strong scaling "
             "(lat=8, lat_sdr=1; radial refinement only)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("kernel_vs_comm_strong_scaling.png", dpi=130)
print("wrote kernel_vs_comm_strong_scaling.png")
# also print the crossover (where comm overtakes kernel) per problem
for rad in rads:
    pts = sorted(data[rad])
    cross = next((p[0] for p in pts if p[2] > p[1]), None)
    print(f"  rad{rad}: comm>kernel from {cross} GCDs" if cross
          else f"  rad{rad}: kernel-bound throughout")
