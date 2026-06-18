#!/usr/bin/env python
"""Strong-scaling plot for the largest problem (rad_level 11, 4.03 G dofs):
kernel (compute) + communication substeps, time per matvec vs #GPUs."""
import json, glob, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find(n, name):
    if n.get("name") == name:
        return n
    for c in n.get("children", []):
        r = find(c, name)
        if r:
            return r

rows = {}
for f in glob.glob("outputs/STR_l8r11_g*/tts/*.json"):
    ng = int(re.search(r"_np(\d+)_", f).group(1))
    root = json.load(open(f))
    ap = find(root, "epsilon_divdiv_apply"); cnt = ap["count"]
    if cnt != 20:           # only warmup-free trees (timed-only); skip stale count=25
        print(f"  skip g{ng}: count={cnt} (not warmup-free yet)")
        continue
    v = lambda nm: (find(root, nm)["avg_time"] / cnt * 1e3) if find(root, nm) else 0.0
    rows[ng] = dict(
        apply=v("epsilon_divdiv_apply"),
        kernel=v("epsilon_divdiv_kernel"),
        comm=v("epsilon_divdiv_comm"),
        waitall=v("ShellBoundaryCommPlan::waitall"),
        # pack/post/unpack/local-exchange buffer ops combined into one line
        bufops=(v("ShellBoundaryCommPlan::pack_remote")
                + v("ShellBoundaryCommPlan::post_isends")
                + v("ShellBoundaryCommPlan::post_irecvs")
                + v("ShellBoundaryCommPlan::unpack_local")
                + v("ShellBoundaryCommPlan::local_comm")),
    )

g = sorted(rows)
def series(k): return [rows[x][k] for x in g]

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.loglog(g, series("apply"),  "k^--", lw=1.5, label="apply (total matvec)")
ax.loglog(g, series("kernel"), "o-", color="tab:blue",   lw=2, label="kernel (compute)")
ax.loglog(g, series("comm"),   "s-", color="tab:red",    lw=2, label="communication (total)")
ax.loglog(g, series("waitall"),"v-", color="tab:orange", label="  └ waitall (MPI wait)")
ax.loglog(g, series("bufops"), "D-", color="tab:green",  label="  └ buffer ops (pack/post/unpack)")
# ideal strong-scaling reference on the kernel
k = series("kernel")
ax.loglog(g, [k[0] * g[0] / x for x in g], ":", color="tab:blue", alpha=0.5, label="ideal 1/N (kernel)")

ax.set_xlabel("GCDs (GPUs)")
ax.set_ylabel("time per matvec [ms]")
ax.set_title("EpsDivDiv strong scaling — rad_level 11 (4.03 G dofs)\n"
             "kernel + communication substeps")
ax.set_xticks(g); ax.set_xticklabels([str(x) for x in g])
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig("rad11_kernel_comm_substeps.png", dpi=140)
print("wrote rad11_kernel_comm_substeps.png")
print(f"{'GCDs':>6} {'apply':>7} {'kernel':>7} {'comm':>6} {'waitall':>8} {'bufops':>7}")
for x in g:
    r = rows[x]
    print(f"{x:6d} {r['apply']:7.2f} {r['kernel']:7.2f} {r['comm']:6.2f} {r['waitall']:8.2f} {r['bufops']:7.2f}")
